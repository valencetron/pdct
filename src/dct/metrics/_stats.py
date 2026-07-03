"""Wilson score interval — small no-deps stats helper for prelim metrics."""
from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95%-CI for binomial proportion k/n.

    Returns (lo, hi). For n==0 returns (0.0, 1.0) — uninformative but defined.
    For k==n or k==0 returns proper one-sided-ish bounds (Wilson handles
    boundaries gracefully).

    References: Wilson 1927, "Probable Inference, the Law of Succession,
    and Statistical Inference."
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))
