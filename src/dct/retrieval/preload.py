"""Preload — assemble session-start context bundle."""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .types import RetrievalConfig, PreloadBundle


# -- token estimator -----------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token per 4 chars. Good enough for budgeting.

    Note: replace with tiktoken once we want exact counts (see DP-36).
    """
    return len(text) // 4


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut text to fit within max_tokens (char/4 estimator)."""
    if _estimate_tokens(text) <= max_tokens:
        return text
    return text[: max_tokens * 4]


# -- anchor loader -------------------------------------------------------------

def _load_anchors(config: RetrievalConfig) -> tuple[str, int]:
    """Read + concat anchor files, respecting token cap. Missing files are skipped."""
    parts: list[str] = []
    for path in config.anchor_paths:
        if not path.exists():
            continue
        parts.append(path.read_text())
    combined = "\n\n".join(parts)
    capped = _truncate_to_tokens(combined, config.preload_anchor_cap)
    return capped, _estimate_tokens(capped)


# -- distilled loader ----------------------------------------------------------

@dataclass(frozen=True)
class DistilledNote:
    channel: str
    session_id: str
    concepts: list[str]
    distilled_at: str
    body: str
    gist: str = ""  # one-line summary from frontmatter; preferred for recent injection


def _parse_distilled(path: Path) -> DistilledNote | None:
    text = path.read_text()
    if not text.startswith("---\n"):
        return None
    try:
        _, fm, body = text.split("---\n", 2)
    except ValueError:
        return None
    try:
        meta = yaml.safe_load(fm) or {}
    except yaml.YAMLError:
        return None
    # FIX (2026-05-27): distillations use `compacted_at`, not `distilled_at`.
    # Fall back to `distilled_at` for legacy notes that use the old key.
    distilled_at = meta.get("compacted_at", "") or meta.get("distilled_at", "")
    # YAML auto-parses ISO8601 timestamps to datetime; normalize back to ISO Z.
    if isinstance(distilled_at, datetime):
        if distilled_at.tzinfo is None:
            distilled_at = distilled_at.replace(tzinfo=timezone.utc)
        distilled_at = distilled_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    return DistilledNote(
        channel=str(meta.get("source", meta.get("source_channel", ""))),
        session_id=str(meta.get("session_id", path.stem)),
        concepts=list(meta.get("concepts") or []),
        distilled_at=str(distilled_at),
        body=body.strip(),
        gist=str(meta.get("gist", "")),
    )


# Incremental per-file note cache (2026-07-16 latency campaign): the old
# design keyed a whole-list cache on the max-mtime across every root — any
# distillation/archive write invalidated it, and a miss re-read AND
# re-parsed every .md file (~2-3k with archives, observed 1.9-14.4s inside
# the cascade). Computing the key itself rglob-stat'ed everything per call.
# Now: a 15s TTL serves the last list with ZERO I/O; on expiry a single
# stat-walk re-parses only new/changed files and sweeps deletions.
_NOTE_CACHE: dict[str, tuple[tuple, "DistilledNote | None"]] = {}
# checked_mono uses time.monotonic() — wall-clock (NTP/manual) rollback must
# not extend the TTL (Codex P1).
_NOTE_LIST = {"checked_mono": float("-inf"), "fast_key": (), "notes": []}
_NOTE_LOCK = threading.Lock()
_NOTE_SCANNING = {"active": False}
_NOTE_SCAN_TTL_S = 15.0
import logging as _logging
_plog = _logging.getLogger(__name__)


def _reset_note_cache() -> None:
    """Test/tooling helper: drop all cached notes and force a fresh scan
    (replaces the old `_DISTILL_CACHE.clear()` reset)."""
    with _NOTE_LOCK:
        _NOTE_CACHE.clear()
        _NOTE_LIST["checked_mono"] = float("-inf")
        _NOTE_LIST["fast_key"] = ()
        _NOTE_LIST["notes"] = []


def _scan_notes(config: RetrievalConfig) -> list[DistilledNote]:
    """Full stat-walk + incremental parse. Runs OUTSIDE _NOTE_LOCK (Codex
    P1: holding a global lock through filesystem walking + YAML parsing
    stalls every concurrent caller). Single-scanner is guaranteed by the
    _NOTE_SCANNING flag; results publish atomically under the lock."""
    all_roots = [config.distill_root] + list(config.archive_roots)
    active_roots = [r for r in all_roots if r.is_dir()]
    new_cache: dict[str, tuple[tuple, "DistilledNote | None"]] = {}
    changed = 0
    for root in active_roots:
        for p in root.rglob("*.md"):
            try:
                fst = p.stat()
            except OSError:
                continue
            if not p.is_file():
                continue
            key = str(p)
            stamp = (fst.st_mtime_ns, fst.st_size)
            hit = _NOTE_CACHE.get(key)
            if hit is not None and hit[0] == stamp:
                new_cache[key] = hit  # unchanged — reuse without re-reading
                continue
            new_cache[key] = (stamp, _parse_distilled(p))
            changed += 1
    if changed:
        _plog.debug("[preload] note scan: %d changed of %d files",
                    changed, len(new_cache))
    notes = [n for _, n in new_cache.values() if n is not None]
    notes.sort(key=lambda n: n.distilled_at, reverse=True)
    with _NOTE_LOCK:
        _NOTE_CACHE.clear()
        _NOTE_CACHE.update(new_cache)  # deletions sweep implicitly
    return notes


def _load_all_distilled(config: RetrievalConfig) -> list[DistilledNote]:
    """All distilled notes from distill_root + archive_roots, newest first.

    FIX (2026-05-27): walks distill_root/**/*.md directly (surface-named
    subdirectories never existed). Phase 2 (2026-05-28): also walks
    config.archive_roots; archives share the frontmatter schema.

    Incremental + stale-while-scanning (2026-07-16): inside a 15s TTL the
    last list serves with no filesystem access at all (the fast key is
    built from configured path strings, not is_dir() probes — Codex P2).
    On expiry exactly one caller scans (stat-walk, re-parse only changed
    files); concurrent callers serve the stale list instead of queueing
    behind the scan. ≤15s staleness is the accepted trade for
    "today/recent" session context.
    """
    fast_key = (str(config.distill_root),
                tuple(str(r) for r in config.archive_roots))
    now_m = time.monotonic()
    with _NOTE_LOCK:
        same_key = _NOTE_LIST["fast_key"] == fast_key
        if same_key and now_m - _NOTE_LIST["checked_mono"] < _NOTE_SCAN_TTL_S:
            return _NOTE_LIST["notes"]
        if same_key and _NOTE_SCANNING["active"]:
            # A scan is in flight — serve stale rather than queue behind it.
            return _NOTE_LIST["notes"]
        _NOTE_SCANNING["active"] = True
    try:
        notes = _scan_notes(config)
        with _NOTE_LOCK:
            _NOTE_LIST["checked_mono"] = time.monotonic()
            _NOTE_LIST["fast_key"] = fast_key
            _NOTE_LIST["notes"] = notes
        return notes
    finally:
        with _NOTE_LOCK:
            _NOTE_SCANNING["active"] = False


# -- today / recent split ------------------------------------------------------

def _start_of_day_utc(ts: float) -> float:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    sod = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return sod.timestamp()


def _parse_iso_to_ts(s: str) -> float:
    s2 = s.replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        return datetime.fromisoformat(s2).timestamp()
    except ValueError:
        return 0.0


def _split_today_and_recent(
    config: RetrievalConfig,
    *,
    now: float,
    last_n: int | None = None,
) -> tuple[list[DistilledNote], dict[str, list[DistilledNote]]]:
    """Split distilled notes into today-aggregate and last-N recent (flat, not per-surface).

    FIX (2026-05-27): Surface-based split replaced with a flat walk over all
    distillations. `recent_by_surface` is preserved as a single-key dict keyed
    "recent" so the PreloadBundle shape and inject.py renderer are unchanged.
    """
    if last_n is None:
        last_n = config.preload_last_n
    start_today = _start_of_day_utc(now)

    all_notes = _load_all_distilled(config)
    today_notes = [n for n in all_notes if _parse_iso_to_ts(n.distilled_at) >= start_today]
    older_notes = [n for n in all_notes if _parse_iso_to_ts(n.distilled_at) < start_today]

    # Use a single "recent" bucket — no surface split (all distillations share
    # the same source). last_n controls how many pre-today sessions to inject.
    recent_by_surface: dict[str, list[DistilledNote]] = {"recent": older_notes[:last_n]}

    today_notes.sort(key=lambda n: n.distilled_at, reverse=True)
    return today_notes, recent_by_surface


# -- public preload ------------------------------------------------------------

def _render_note(n: DistilledNote) -> str:
    # Use gist (1-2 sentence summary) when available — much more token-efficient
    # than injecting the full body. Falls back to body for legacy notes.
    summary = n.gist if n.gist else n.body
    return f"[{n.distilled_at}] {n.session_id}\n{summary}"


def preload(config: RetrievalConfig, *, now: float) -> PreloadBundle:
    """Assemble session-start context bundle.

    - anchors: concat of anchor_paths, capped at preload_anchor_cap
    - today_summaries: distilled notes from today (all surfaces), capped
    - recent_summaries: per-surface distilled notes before today, up to last_n,
      each surface independently capped at preload_surface_cap
    """
    anchors, anchor_tokens = _load_anchors(config)
    today_notes, recent_by_surface = _split_today_and_recent(config, now=now)

    today_text = "\n\n---\n\n".join(_render_note(n) for n in today_notes)
    today_text = _truncate_to_tokens(today_text, config.preload_today_cap)

    # FIX (2026-05-27): iterate the actual keys returned by _split_today_and_recent
    # ("recent") rather than config.surfaces (which produced empty lookups).
    recent_summaries: dict[str, str] = {}
    for bucket, notes in recent_by_surface.items():
        text = "\n\n---\n\n".join(_render_note(n) for n in notes)
        recent_summaries[bucket] = _truncate_to_tokens(text, config.preload_surface_cap)

    total_tokens = (
        anchor_tokens
        + _estimate_tokens(today_text)
        + sum(_estimate_tokens(t) for t in recent_summaries.values())
    )

    return PreloadBundle(
        anchors=anchors,
        today_summaries=today_text,
        recent_summaries=recent_summaries,
        total_tokens=total_tokens,
    )
