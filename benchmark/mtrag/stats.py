"""Bootstrap confidence intervals and paired significance for Leg-2 arms.

All resampling is deterministic given `seed` so reported CIs are reproducible.
We use the percentile bootstrap for per-arm means and a PAIRED bootstrap for
arm-vs-arm deltas (resample query indices once per replicate, apply the same
indices to both arms — this respects the within-query correlation between arms
that share one adapter, which an unpaired test would ignore and overstate).
"""
from __future__ import annotations
import random


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of `values`.
    Returns (mean, lo, hi). Empty input -> (0,0,0)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return _mean(values), lo, hi


def paired_bootstrap_delta(a, b, n_boot=10000, alpha=0.05, seed=0):
    """Paired analysis for mean(a) - mean(b), where a[i] and b[i] are the two
    arms' scores on the SAME query i.

    Returns dict with:
      - delta, lo, hi : observed delta + percentile-bootstrap 95% CI
      - p             : two-sided p from a PAIRED SIGN-FLIP PERMUTATION test
                        (the valid null: under H0 the per-query difference sign
                        is exchangeable, so randomly negate each diff and count
                        how often |permuted mean| >= |observed mean|).
      - significant   : CI excludes 0
    CI and p are computed by separate, correct procedures (bootstrap for the CI,
    permutation for the p) — the bootstrap distribution is centered on the
    observed delta and is NOT a valid null, so it is never used for p."""
    n = len(a)
    if len(b) != n:
        raise ValueError(f"paired arms must be equal length: {len(a)} vs {len(b)}")
    if n == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0,
                "significant": False, "n": 0}
    diffs = [ai - bi for ai, bi in zip(a, b)]
    obs = sum(diffs) / n
    rng = random.Random(seed)
    # percentile bootstrap for the CI
    boot = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boot.append(sum(diffs[i] for i in idx) / n)
    boot.sort()
    lo = boot[int((alpha / 2) * n_boot)]
    hi = boot[int((1 - alpha / 2) * n_boot) - 1]
    # paired sign-flip permutation test for the p-value (valid null at 0)
    rng_p = random.Random(seed + 1)
    abs_obs = abs(obs)
    ge = 0
    for _ in range(n_boot):
        s = sum(d if rng_p.random() < 0.5 else -d for d in diffs) / n
        if abs(s) >= abs_obs - 1e-12:
            ge += 1
    p = (ge + 1) / (n_boot + 1)  # add-one smoothing, never exactly 0
    return {"delta": round(obs, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "p": round(p, 4), "significant": (lo > 0 or hi < 0), "n": n}


def unpaired_bootstrap_delta(a, b, n_boot=10000, alpha=0.05, seed=0):
    """Two-independent-sample analysis for mean(a) - mean(b), where a and b are
    DIFFERENT, independent groups (e.g. late-turn vs early-turn queries — not
    the same items, so pairing would be invalid).

    Returns dict with:
      - delta, lo, hi : observed delta + percentile-bootstrap 95% CI (each group
                        resampled independently with replacement)
      - p             : two-sided p from a LABEL-PERMUTATION test (the valid
                        null: pool a+b, shuffle group labels, count how often
                        |permuted delta| >= |observed delta|).
      - significant   : CI excludes 0"""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0,
                "significant": False, "n_a": na, "n_b": nb}
    obs = _mean(a) - _mean(b)
    rng = random.Random(seed)
    # percentile bootstrap for the CI
    boot = []
    for _ in range(n_boot):
        da = sum(a[rng.randrange(na)] for _ in range(na)) / na
        db = sum(b[rng.randrange(nb)] for _ in range(nb)) / nb
        boot.append(da - db)
    boot.sort()
    lo = boot[int((alpha / 2) * n_boot)]
    hi = boot[int((1 - alpha / 2) * n_boot) - 1]
    # label-permutation test for the p-value (valid two-sample null)
    pool = list(a) + list(b)
    rng_p = random.Random(seed + 1)
    abs_obs = abs(obs)
    ge = 0
    for _ in range(n_boot):
        rng_p.shuffle(pool)
        pa = sum(pool[:na]) / na
        pb = sum(pool[na:]) / nb
        if abs(pa - pb) >= abs_obs - 1e-12:
            ge += 1
    p = (ge + 1) / (n_boot + 1)
    return {"delta": round(obs, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "p": round(p, 4), "significant": (lo > 0 or hi < 0),
            "n_a": na, "n_b": nb}
