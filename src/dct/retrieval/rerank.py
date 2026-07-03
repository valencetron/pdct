"""Cross-encoder reranking for memory retrieval.

The bi-encoder/concept channels are good at "about the same topic" but
cannot distinguish topical neighbors from documents that actually answer
the question (e.g. five different "stale brief" incidents). A cross-encoder
reads (query, doc) jointly and fixes the final ordering.

Model: cross-encoder/ms-marco-MiniLM-L-12-v2 — ~9s one-time load,
~0.35s for 25 pairs on this machine.

Public API:
    rerank(query, candidates) -> candidates reordered
        candidates: list of (id, doc_text, prior_score)
        returns list of (id, blended_score) sorted desc

Graceful: any failure returns the prior ordering unchanged.
"""
from __future__ import annotations

import math
from typing import Any

# L-12 swap 2026-06-12: cold canary r@1 0.40→0.667, r@5 0.833→0.933,
# p50 3.4s→4.4s. See docs/reports/2026-06-11-benchmark-sweep-report.md §7.
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
_MODEL: Any = None

# Blend weights: cross-encoder dominates final ordering, prior keeps
# graph/temporal evidence from being erased entirely.
_CE_WEIGHT = 0.7
_PRIOR_WEIGHT = 0.3


def _get_model() -> Any:
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder
        # device="cpu": see vec_index.py — MPS deadlocks on long runs.
        _MODEL = CrossEncoder(_MODEL_NAME, device="cpu")
    return _MODEL


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(
    query: str,
    candidates: list[tuple[str, str, float]],
) -> list[tuple[str, float]]:
    """Reorder candidates by blended cross-encoder + prior score.

    Args:
      query: the user's retrieval query.
      candidates: (id, doc_text, prior_score 0..1) tuples.

    Returns [(id, blended_score)] sorted descending. On any failure,
    returns the input order with prior scores untouched.
    """
    fallback = [(cid, prior) for cid, _, prior in candidates]
    if not query.strip() or len(candidates) < 2:
        return fallback
    try:
        model = _get_model()
        pairs = [(query, text[:1500] if text else cid)
                 for cid, text, _ in candidates]
        raw = model.predict(pairs, show_progress_bar=False)
        # Min-max normalize CE scores within the pool. Raw ms-marco logits
        # sigmoid to tiny absolute values (~0.0002-0.005 on this corpus),
        # which would let the prior term swamp a 20x relative CE preference.
        # What matters is the CE's *ranking* of this pool, not its absolute
        # calibration against MS MARCO.
        ces = [_sigmoid(float(x)) for x in raw]
        lo, hi = min(ces), max(ces)
        span = (hi - lo) or 1.0
        out = []
        for (cid, _, prior), ce in zip(candidates, ces):
            ce_n = (ce - lo) / span
            out.append((cid, _CE_WEIGHT * ce_n + _PRIOR_WEIGHT * prior))
        out.sort(key=lambda t: -t[1])
        return out
    except Exception:
        return fallback
