"""Frozen benchmark builder — select & freeze the versioned question set.

Pipeline:
  1. Pull verbatim user questions from utility.jsonl followup excerpts (line-anchored).
  2. For each candidate, run a RETRIEVAL-ONLY pass (no reply, no judge — cheap)
     to get cascade_count under the current live config.
  3. Select the richness band (cascade_count in [lo, hi]) — a no-LLM proxy for
     the match_rate middle-band: empty retrieval can't discriminate between lever
     settings, saturated retrieval is already maxed. The discriminating questions
     live in between.
  4. Freeze ~50 as an immutable, versioned JSON asset. A version is written ONCE;
     a change is a NEW version (v2), never an in-place edit.

NOTE: cascade_count is a PROXY for discriminating power. True match_rate would
require an LLM reply per candidate (thousands of calls). This proxy is cheap and
directionally correct: it selects questions where retrieval is non-trivial and
varies with lever settings.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from dct.retrieval.service import build_config, run

UTILITY_JSONL = Path.home() / "example-stack" / "dynamic-context-traversal" / "logs" / "utility.jsonl"
DEFAULT_ASSET_DIR = (
    Path.home() / "example-stack" / "dynamic-context-traversal" / "benchmark"
)

_MIN_Q_LEN = 25

# Richness band defaults — tuned so empty/saturated are excluded.
DEFAULT_LO = 3
DEFAULT_HI = 40


def assign_id(question: str) -> str:
    """Deterministic content-hash id (stable across runs)."""
    return "q_" + hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:12]


def load_candidates(n: int, path: Path = UTILITY_JSONL) -> list[dict[str, Any]]:
    """Distinct non-trivial followup questions with line-anchored provenance."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    lines = path.read_text().splitlines()
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") != "followup":
            continue
        q = (r.get("excerpt") or "").strip()
        if len(q) < _MIN_Q_LEN or q in seen:
            continue
        seen.add(q)
        out.append({
            "id": assign_id(q),
            "question": q,
            "source_ref": f"utility.jsonl#L{lineno}",
        })
        if len(out) >= n:
            break
    return out


def measure_richness(candidates: list[dict[str, Any]], config=None) -> list[dict[str, Any]]:
    """Retrieval-only pass per candidate → attach cascade_count. No LLM calls."""
    cfg = config if config is not None else build_config()
    for c in candidates:
        try:
            out = run(c["question"], config_override=cfg)
            c["cascade_count"] = int(out.get("cascade_count", 0))
        except Exception:
            c["cascade_count"] = 0
    return candidates


def filter_richness_band(
    candidates: list[dict[str, Any]], lo: int = DEFAULT_LO, hi: int = DEFAULT_HI
) -> list[dict[str, Any]]:
    """Keep candidates whose cascade_count is in [lo, hi] (the discriminating band)."""
    return [c for c in candidates if lo <= c.get("cascade_count", 0) <= hi]


def freeze(
    selected: list[dict[str, Any]],
    asset_path: Path,
    *,
    version: int,
    lo: int = DEFAULT_LO,
    hi: int = DEFAULT_HI,
) -> Path:
    """Write the immutable versioned asset. Refuses to overwrite an existing version."""
    asset_path = Path(asset_path)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selection": {
            "criterion": "retrieval richness band (cascade_count proxy for match_rate middle-band)",
            "richness_lo": lo,
            "richness_hi": hi,
            "source": "utility.jsonl followup excerpts (verbatim user turns)",
            "note": "cascade_count is a no-LLM proxy for discriminating power; "
                    "context-light (no preceding-turn window recoverable from logs)",
        },
        "count": len(selected),
        "questions": selected,
    }
    # Durable atomic publish (Codex #1/#2/#3/#4/#5):
    #   1. mkstemp() — unique, exclusive temp file in the SAME directory (no
    #      predictable path, no symlink-follow, no O_TRUNC race).
    #   2. Full write loop (POSIX short-write safe) + fsync — data durable on
    #      disk BEFORE the final path exists.
    #   3. os.link() — no-overwrite atomic publish; fails FileExistsError if the
    #      version is already frozen (race-safe immutability).
    #   4. fsync the parent directory so the new dir entry is durable too.
    #   5. temp always unlinked, even on write/fsync failure (no leak).
    import tempfile

    body = json.dumps(payload, indent=2).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        dir=str(asset_path.parent), prefix=asset_path.name + ".", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        # POSIX-safe full write (os.write may short-write). Defensively raise on
        # a 0-byte write to avoid an infinite freeze loop. Keep mkstemp's 0600
        # DURING the write so partial user-excerpt contents aren't world-readable
        # through the temp path; widen to 0644 only after the full payload lands.
        view = memoryview(body)
        while view:
            written = os.write(fd, view)
            if written == 0:
                raise OSError("os.write returned 0 — refusing to spin")
            view = view[written:]
        os.fchmod(fd, 0o644)  # restore 0644 readability contract, post-write
        os.fsync(fd)
        os.close(fd)
        fd = -1
        try:
            os.link(str(tmp), str(asset_path))  # atomic, fails if dest exists
        except FileExistsError:
            raise FileExistsError(
                f"Benchmark version already frozen at {asset_path}. "
                "A version is immutable — bump to a new version instead."
            )
        # Durably persist the new directory entry.
        dir_fd = os.open(str(asset_path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if fd != -1:
            os.close(fd)
        if tmp.exists():
            os.unlink(tmp)
    return asset_path


def build_v1(
    target_count: int = 50,
    candidate_pool: int = 3000,
    lo: int = DEFAULT_LO,
    hi: int = DEFAULT_HI,
    asset_dir: Path = DEFAULT_ASSET_DIR,
) -> tuple[Path, dict[str, Any]]:
    """End-to-end: load → measure richness → band-filter → sample → freeze v1."""
    cands = load_candidates(candidate_pool)
    measure_richness(cands)

    # Guard against a global retrieval failure silently freezing a biased asset:
    # if almost everything scored 0, retrieval was likely broken, not the
    # questions. (Codex diff-audit finding #7.)
    nonzero = sum(1 for c in cands if c.get("cascade_count", 0) > 0)
    if cands and nonzero / len(cands) < 0.2:
        raise RuntimeError(
            f"only {nonzero}/{len(cands)} candidates retrieved any concepts — "
            "retrieval likely failed globally; refusing to freeze a biased asset"
        )

    banded = filter_richness_band(cands, lo=lo, hi=hi)
    selected = banded[:target_count]
    if len(selected) < target_count:
        raise RuntimeError(
            f"only {len(selected)} in-band candidates (need {target_count}) — "
            "widen the pool or the richness band before freezing"
        )
    asset_path = asset_dir / "pdct-questions-v1.json"
    freeze(selected, asset_path, version=1, lo=lo, hi=hi)
    # Histogram for sanity.
    from collections import Counter
    hist = Counter(c["cascade_count"] for c in cands)
    summary = {
        "candidates_scored": len(cands),
        "in_band": len(banded),
        "selected": len(selected),
        "richness_histogram": dict(sorted(hist.items())),
    }
    return asset_path, summary
