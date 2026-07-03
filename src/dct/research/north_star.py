"""North Star veto — fail-closed, advisory.

Given before/after criteria values, compute per-criterion deltas.

- FAIL-CLOSED: a missing REQUIRED criterion BLOCKS the recommendation (never
  silently skipped). A blocked candidate cannot be recommended.
- ADVISORY VETO: a material regression (delta < -epsilon) on any criterion
  recommends a veto. Alex makes the final call — this does not auto-act.

Which criteria are sandbox-computable vs system-level is documented per run;
the veto only judges criteria actually present in BOTH baseline and candidate.
"""
from __future__ import annotations

from typing import Any, Optional


def veto_check(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    required: Optional[list[str]] = None,
    epsilon: float = 0.05,
) -> dict[str, Any]:
    """Compare candidate vs baseline criteria.

    Returns:
      blocked       — True if a required criterion is missing (fail-closed)
      vetoed        — True if any criterion regressed beyond epsilon (advisory)
      per_criterion — {criterion: delta} for every criterion present in both
      regressions   — {criterion: delta} for the ones that triggered the veto
      missing       — required criteria absent from candidate or baseline
    """
    required = required or []

    missing = [
        c for c in required
        if c not in candidate or c not in baseline
    ]
    blocked = len(missing) > 0

    per_criterion: dict[str, float] = {}
    for c in set(candidate) & set(baseline):
        per_criterion[c] = candidate[c] - baseline[c]

    regressions = {
        c: d for c, d in per_criterion.items() if d < -abs(epsilon)
    }
    vetoed = len(regressions) > 0

    return {
        "blocked": blocked,
        "vetoed": vetoed,
        "per_criterion": per_criterion,
        "regressions": regressions,
        "missing": missing,
        "epsilon": epsilon,
        "detail": (
            f"BLOCKED — missing required criteria: {missing}"
            if blocked
            else (
                f"VETO (advisory) — regressions: {regressions}"
                if vetoed
                else "no material regression; OK to recommend"
            )
        ),
    }
