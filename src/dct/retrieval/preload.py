"""Preload — assemble session-start context bundle."""
from __future__ import annotations
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


# In-memory cache for the full distillation list, invalidated by distill_root
# directory mtime. A new compaction writes a new file → dir mtime changes →
# next request rebuilds. Cache cleared when it exceeds 8 entries (shouldn't
# happen in practice — key is stable across a daemon run).
_DISTILL_CACHE: dict[tuple, list[DistilledNote]] = {}


def _distill_root_mtime(root: Path) -> float:
    """Max mtime of any .md file under root, for cache invalidation."""
    try:
        return max(
            (p.stat().st_mtime for p in root.rglob("*.md") if p.is_file()),
            default=0.0,
        )
    except OSError:
        return 0.0


def _load_all_distilled(config: RetrievalConfig) -> list[DistilledNote]:
    """Load all distilled notes from distill_root + archive_roots, newest first.

    FIX (2026-05-27): The original _load_distilled() looked for surface-named
    subdirectories (voice/, claude-code/, telegram/, vault/) that never existed.
    All 589+ distillations live at distill_root/<slug>/<slug>.md with a flat
    `source: telegram-dispatch daemon` field — no surface split. This function
    walks distill_root/**/*.md directly and sorts by distilled_at (which maps
    to compacted_at in the frontmatter).

    Phase 2 (2026-05-28): also walks config.archive_roots (e.g.
    vault/compaction-archive/ written by compaction_archive.py). Archive files
    share the same frontmatter schema (compacted_at, gist) so _parse_distilled
    handles them without modification.

    Results are cached in-process, invalidated by combined max-mtime across all
    roots so a fresh archive write is picked up on the next request.
    """
    all_roots = [config.distill_root] + [r for r in config.archive_roots if r.is_dir()]
    active_roots = [r for r in all_roots if r.is_dir()]
    if not active_roots:
        return []

    # Cache key: tuple of (root_str, mtime) for every active root, sorted for stability.
    mtime_parts = tuple(
        (str(r), _distill_root_mtime(r)) for r in sorted(active_roots, key=str)
    )
    cached = _DISTILL_CACHE.get(mtime_parts)
    if cached is not None:
        return cached

    notes: list[DistilledNote] = []
    for root in active_roots:
        for p in root.rglob("*.md"):
            if not p.is_file():
                continue
            n = _parse_distilled(p)
            if n is not None:
                notes.append(n)
    notes.sort(key=lambda n: n.distilled_at, reverse=True)

    if len(_DISTILL_CACHE) > 8:
        _DISTILL_CACHE.clear()
    _DISTILL_CACHE[mtime_parts] = notes
    return notes


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
