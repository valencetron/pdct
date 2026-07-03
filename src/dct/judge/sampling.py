"""Deterministic-hash turn sampling for the judge (P1.3a).

A turn is sampled if and only if hash(turn_id) % 100 < rate*100. This
guarantees:
- Reproducibility: same turn_id → same decision across runs.
- No flipping mid-day: if we sampled this turn once, we'd sample it again
  if it ever re-played.
- Cap-friendly: the sampling decision is independent of daily counters,
  so the queue's daily cap can reject overflow without breaking the
  determinism contract.
"""
from __future__ import annotations

import hashlib


def should_sample_turn(turn_id: str | None, rate: float = 0.25) -> bool:
    """Return True iff turn_id should be sampled at the given rate.

    Args:
        turn_id: The PDCT turn identifier. Empty/None → False.
        rate: Sampling rate in [0, 1]. Out-of-range values clamp.
    """
    if not turn_id:
        return False
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    h = hashlib.sha256(turn_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 10000  # finer granularity than %100
    return bucket < int(rate * 10000)


__all__ = ["should_sample_turn"]
