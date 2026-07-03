# dynamic-context-traversal/src/dct/composite.py
"""PDCT P1.5 — availability-normalized weighted composite score.

Combines match_rate, cosine_score, self_rating, and era_judge into a
single [0,1] score. Legs with None values (or weight=0) are excluded;
the sum of remaining weights is the denominator (availability-normalized).

Weight design: constants target the 4-leg final state (sum=1.0 when
era_judge=0.3 in P1.3b). During P1.5, active legs sum to 0.7 — this is
intentional; availability normalization keeps the composite in [0,1].

era_judge weight is 0.0 until P1.3b ships — set it to 0.3 there
(and reduce others proportionally if needed).

Usage:
    from dct.composite import compute_composite
    result = compute_composite({
        "match_rate": 0.4,
        "cosine_score": 0.7,
        "self_rating": "useful",
        "era_judge": None,
    })
    # result.score: float | None
    # result.legs_used: list[str]
    # result.legs_missing: list[str]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Weights target the 4-leg final state (all four sum to 1.0 at P1.3b).
# 2026-06-10: era_judge activated at 0.3 — the P1.3b pipeline (queue, worker,
# runner, composite_updater) had been live for weeks but this weight was never
# flipped, so every judge score contributed 0 to the composite. Activated as
# part of the composite-null/judge-calibration fix (PDCT audit, Jun 10).
WEIGHTS: dict[str, float] = {
    "match_rate":   0.2,
    "cosine_score": 0.3,
    "self_rating":  0.2,
    "era_judge":    0.3,
}

# Map self_rating string values → [0, 1]. Applied after strip().lower().
SELF_RATING_MAP: dict[str, float] = {
    "useful":  1.0,
    "partial": 0.75,
    "noise":   0.25,
    "absent":  0.0,
}

_LEG_ORDER = ("match_rate", "cosine_score", "self_rating", "era_judge")


@dataclass(frozen=True)
class CompositeResult:
    """Result of compute_composite()."""
    score: Optional[float]               # None if zero legs contribute
    legs_used: list[str] = field(default_factory=list)
    legs_missing: list[str] = field(default_factory=list)
    weights_used: dict[str, float] = field(default_factory=dict)


def _normalize_leg(name: str, value: object) -> Optional[float]:
    """Convert a raw leg value to a finite [0, 1] float, or None if missing/invalid.

    Rules per leg:
      self_rating  — map via SELF_RATING_MAP after strip().lower(); None if unknown
      era_judge    — (score−1)/4, clamped to [0,1]; 1–5 expected but handles out-of-range
      match_rate   — numeric, validated finite, clamped to [0,1]
      cosine_score — numeric, validated finite, clamped to [0,1]
    """
    if value is None:
        return None

    if name == "self_rating":
        if not isinstance(value, str):
            return None
        return SELF_RATING_MAP.get(value.strip().lower())

    if name == "era_judge":
        try:
            v = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        # 1–5 ordinal → [0,1], clamped for out-of-range inputs
        return max(0.0, min(1.0, (v - 1.0) / 4.0))

    # match_rate, cosine_score — already expected to be [0,1]
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return max(0.0, min(1.0, v))


def compute_composite(
    legs: dict[str, object],
    weights: Optional[dict[str, float]] = None,
) -> CompositeResult:
    """Compute availability-normalized weighted mean over provided legs.

    Args:
        legs: dict mapping leg name → raw value. Unknown keys are ignored.
        weights: override WEIGHTS (for testing / P1.3b weight update).
                 Defaults to module-level WEIGHTS.

    Returns:
        CompositeResult with score=None if no legs contribute (all missing
        or all have weight=0).
    """
    w = weights if weights is not None else WEIGHTS
    used: list[str] = []
    missing: list[str] = []
    weights_used: dict[str, float] = {}
    numerator = 0.0
    denominator = 0.0

    for name in _LEG_ORDER:
        leg_weight = w.get(name, 0.0)
        if leg_weight == 0.0:
            # Reserved / disabled leg — structurally missing
            missing.append(name)
            continue
        raw = legs.get(name)
        normalized = _normalize_leg(name, raw)
        if normalized is None:
            missing.append(name)
        else:
            used.append(name)
            weights_used[name] = leg_weight
            numerator += leg_weight * normalized
            denominator += leg_weight

    if denominator == 0.0:
        return CompositeResult(score=None, legs_used=[], legs_missing=missing, weights_used={})

    return CompositeResult(
        score=numerator / denominator,
        legs_used=used,
        legs_missing=missing,
        weights_used=weights_used,
    )


__all__ = ["compute_composite", "CompositeResult", "WEIGHTS", "SELF_RATING_MAP", "_normalize_leg"]
