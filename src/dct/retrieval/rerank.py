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
import threading
from typing import Any

# L-12 swap 2026-06-12: cold canary r@1 0.40→0.667, r@5 0.833→0.933,
# p50 3.4s→4.4s. See docs/reports/2026-06-11-benchmark-sweep-report.md §7.
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
_MODEL: Any = None
# SHARED with vec_index — concurrent construction of even *different*
# sentence-transformers models corrupts via shared torch/transformers
# internals (live: rerank warm raced the boot warmer's bge load and died
# with meta tensor, 2026-07-16 17:17). One process-wide construction lock.
from dct.retrieval.vec_index import _MODEL_LOCK

# Blend weights: cross-encoder dominates final ordering, prior keeps
# graph/temporal evidence from being erased entirely.
_CE_WEIGHT = 0.7
_PRIOR_WEIGHT = 0.3


def _get_model() -> Any:
    """BLOCKING loader — warmers/batch only; request path uses
    get_model_if_ready(). Locked + probe-validated construction: concurrent
    CrossEncoder construction fails with meta-tensor corruption on
    torch 2.8 + s-t 5.1.2 (see vec_index._get_model, 2026-07-16)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from sentence_transformers import CrossEncoder
            # device="cpu": see vec_index.py — MPS deadlocks on long runs.
            model = CrossEncoder(_MODEL_NAME, device="cpu")
            # Validate BEFORE caching — raced construction fails here
            # instead of poisoning the singleton.
            model.predict([("warmup", "probe")], show_progress_bar=False)
            _MODEL = model
        return _MODEL


_WARM_THREAD: Any = None
_WARM_SPAWN_LOCK = threading.Lock()
_WARM_FAIL_TS = 0.0
_WARM_FAIL_COOLDOWN_S = 60.0


def get_model_if_ready() -> Any | None:
    """Request-path accessor: warm model or None; never constructs. The
    rerank is a quality bonus — skipping it beats blowing the cascade
    budget on a ~9s model load. Kicks a background warm on miss (Codex P1:
    without this, processes lacking a boot warmer never rerank at all)."""
    if _MODEL is not None:
        return _MODEL
    ensure_warm_async()
    return None


def ensure_warm_async() -> None:
    """Debounced background warm — mirrors vec_index.ensure_warm_async."""
    global _WARM_THREAD, _WARM_FAIL_TS
    if _MODEL is not None:
        return
    import time as _time
    with _WARM_SPAWN_LOCK:
        if _MODEL is not None:
            return
        if _WARM_THREAD is not None and _WARM_THREAD.is_alive():
            return
        if _time.time() - _WARM_FAIL_TS < _WARM_FAIL_COOLDOWN_S:
            return
        def _warm():
            global _WARM_FAIL_TS
            try:
                _get_model()
            except Exception as e:  # noqa: BLE001
                _WARM_FAIL_TS = _time.time()
                import logging
                logging.getLogger(__name__).warning(
                    "[rerank] background model warm failed: %s", e)
        t = threading.Thread(target=_warm, name="rerank-warm", daemon=True)
        _WARM_THREAD = t
        t.start()


def reset_model() -> None:
    """Clear a poisoned singleton so the next warm constructs fresh."""
    global _MODEL
    with _MODEL_LOCK:
        _MODEL = None


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
        model = get_model_if_ready()
        if model is None:
            return fallback  # not warm yet — never construct on request path
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
    except Exception as exc:
        if "meta tensor" in str(exc):
            # Poisoned model from a raced construction — clear so the next
            # background warm builds a clean one (2026-07-16).
            reset_model()
        return fallback
