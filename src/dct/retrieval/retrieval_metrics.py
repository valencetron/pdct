"""Pure metric primitives for PDCT retrieval evaluation.

No LLM, no live cascade — these operate on already-fetched result rows and
question dicts so they are cheap to unit-test and reusable across the eval
harness and the diagnostic scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence


def gold_ids(question: dict) -> set:
    """All identifiers that should resolve to the gold doc for a question."""
    ids: set = set()
    sid = question.get("source_distillation_id")
    if sid:
        ids.add(sid)
    sp = question.get("source_path")
    if sp:
        ids.add(Path(sp).stem)
    return ids


def _row_id(r: Any) -> Optional[str]:
    """Extract id from a dataclass/object row (.id) or a dict row (['id'])."""
    if isinstance(r, dict):
        return r.get("id")
    return getattr(r, "id", None)


def gold_rank(rows: Sequence, gold: set) -> Optional[int]:
    """0-based rank of the first row whose id is in gold, else None.

    Accepts both object rows (DistillationRow with .id) and dict rows
    (CLI JSON output)."""
    for i, r in enumerate(rows):
        if _row_id(r) in gold:
            return i
    return None


def recall_at_k(rows: Sequence, gold: set, k: int) -> Optional[bool]:
    """True/False if gold doc is in top-k; None when the question has no gold
    (e.g. negative/abstain questions) so callers can exclude it from recall."""
    if not gold:
        return None
    rank = gold_rank(rows, gold)
    return rank is not None and rank < k
