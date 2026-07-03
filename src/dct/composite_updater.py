# dynamic-context-traversal/src/dct/composite_updater.py
"""PDCT P1.5 — composite_update appender for async era_judge results.

Called by the P1.3b runner after commit_result() succeeds. Reads the
matching 'turn' row from utility.jsonl by turn_id, recomputes composite
with all 4 legs (era_judge now populated), appends a kind=composite_update
row.

pdct_report resolves per-turn composite by scanning utility.jsonl in file
order and taking the LAST row for each turn_id where kind is 'turn' or
'composite_update'. File order is the tie-break (not timestamp), because
rows are append-only and composite_update always follows its turn row.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


def append_composite_update(
    logs_dir: Path,
    turn_id: str,
    era_judge_score: int,
) -> bool:
    """Recompute composite with era_judge and append composite_update row.

    Args:
        logs_dir: directory containing utility.jsonl
        turn_id: the turn to update (must exist as a 'turn' row)
        era_judge_score: 1–5 ordinal from judge result

    Returns:
        True if row was appended successfully.
        False if turn_id not found, score invalid, or write failed.
    """
    if not isinstance(era_judge_score, int) or isinstance(era_judge_score, bool):
        log.warning("append_composite_update: invalid era_judge_score %r", era_judge_score)
        return False
    if era_judge_score < 1 or era_judge_score > 5:
        log.warning("append_composite_update: era_judge_score out of range %r", era_judge_score)
        return False

    util_path = logs_dir / "utility.jsonl"
    if not util_path.exists():
        log.warning("append_composite_update: utility.jsonl not found at %s", util_path)
        return False

    # Find last 'turn' row for this turn_id
    turn_row: dict | None = None
    try:
        with util_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "turn" and r.get("turn_id") == turn_id:
                    turn_row = r  # keep scanning — last match wins
    except OSError as e:
        log.warning("append_composite_update: read failed: %s", e)
        return False

    if turn_row is None:
        log.warning("append_composite_update: no 'turn' row found for turn_id=%s", turn_id)
        return False

    # 2026-06-10 guard: if the turn had NO retrieval signal at all (all three
    # synchronous legs null — i.e. a no_concepts/ablation skip that slipped
    # into the judge queue before the daemon-side dct_context gate existed),
    # skip the row entirely. With era_judge now carrying live weight, writing
    # a composite from era_judge alone on an empty-context turn would inject
    # a meaningless floor score into lever research. Defense for the existing
    # queue backlog; new enqueues are gated at the daemon.
    if all(
        turn_row.get(k) is None
        for k in ("match_rate", "cosine_score", "self_rating")
    ):
        log.info(
            "append_composite_update: skipping empty-context turn %s "
            "(all synchronous legs null)", turn_id,
        )
        return False

    # Pass the RAW 1-5 era_judge score. compute_composite() normalizes 1-5 → [0,1]
    # itself (composite.py). Pre-normalizing here double-normalizes (5 → 1.0 → 0.0)
    # and silently zeroes the leg the moment era_judge gets nonzero live weight.
    # Currently harmless (live weight 0.0) but a latent landmine. (Codex #2,
    # build #56 diff-audit — same class of bug fixed in research/runner.py.)
    # Recompute composite with all 4 legs
    try:
        from dct.composite import compute_composite
        comp_result = compute_composite({
            "match_rate": turn_row.get("match_rate"),
            "cosine_score": turn_row.get("cosine_score"),
            "self_rating": turn_row.get("self_rating"),
            "era_judge": era_judge_score,
        })
    except Exception as e:
        log.warning("append_composite_update: composite compute failed: %s", e)
        return False

    # Append composite_update row
    update_row = {
        "kind": "composite_update",
        "schema_version": 1,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "turn_id": turn_id,
        # Copy topic metadata from turn row so --topic filtering works
        "thread_id": turn_row.get("thread_id"),
        "topic_id": turn_row.get("topic_id") or turn_row.get("thread_id"),
        "chat_id": turn_row.get("chat_id"),
        "pdct_utility_composite": comp_result.score,
        "composite_legs_used": comp_result.legs_used,
        "composite_legs_missing": comp_result.legs_missing,
        "era_judge_score": era_judge_score,
        # Telemetry only — the normalized [0,1] value for readability. NOT fed to
        # compute_composite (which normalizes the raw score itself). Computed
        # locally here so the log row keeps the human-readable normalized value.
        "era_judge_norm": round((era_judge_score - 1) / 4.0, 4),
    }
    try:
        util_path.parent.mkdir(parents=True, exist_ok=True)
        with util_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(update_row) + "\n")
        return True
    except OSError as e:
        log.warning("append_composite_update: write failed: %s", e)
        return False


__all__ = ["append_composite_update"]
