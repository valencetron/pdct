"""Paired-delta sweep — the experiment that produces a finding.

Grid a lever across its clamp range (incumbent always an arm). For each
question, delta = candidate − incumbent composite (paired: nets out question
difficulty + LLM jitter). Aggregate paired deltas with a DETERMINISTIC,
seeded bootstrap CI. A candidate WINS only if its paired-delta CI is entirely
above 0.

In-process only: each arm is a RetrievalConfig built by replacing the swept
lever on the incumbent config; run_cell uses config_override → no file write.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import replace
from typing import Any, Optional

from dct.research.runner import run_cell  # patched in tests
from dct.retrieval.overrides import LEVER_SPEC
from dct.retrieval.service import build_config

log = logging.getLogger(__name__)

_CI_ALPHA = 0.05  # 95% CI
_BOOTSTRAP_N = 2000


def build_grid(lever: str, incumbent: float, n: int = 5) -> list:
    """Evenly spaced grid across the lever's clamp range, incumbent always included."""
    spec = LEVER_SPEC.get(lever)
    if not spec or spec.get("min") is None:
        raise ValueError(f"lever {lever} is not a numeric swept lever")
    lo, hi = spec["min"], spec["max"]
    is_int = spec["type"] == "int"
    points = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    if is_int:
        points = [int(round(p)) for p in points]
        incumbent = int(incumbent)
    grid = sorted(set(points) | {incumbent})
    return grid


def paired_deltas(
    incumbent_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> list[float]:
    """Per-question candidate−incumbent composite deltas. Missing pairs dropped.

    When a question has multiple replicates, the mean composite is used.
    """
    def by_q(rows):
        agg = defaultdict(list)
        for r in rows:
            c = r.get("composite")
            if c is not None:
                agg[r["question"]].append(float(c))
        return {q: sum(v) / len(v) for q, v in agg.items()}

    inc = by_q(incumbent_rows)
    cand = by_q(candidate_rows)
    deltas = []
    for q in inc:
        if q in cand:
            deltas.append(cand[q] - inc[q])
    return deltas


def bootstrap_ci(
    deltas: list[float], *, seed: int = 42, n_boot: int = _BOOTSTRAP_N, alpha: float = _CI_ALPHA
) -> dict[str, Any]:
    """Deterministic seeded bootstrap CI of the mean paired delta."""
    if not deltas:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return {"mean": sum(deltas) / n, "lo": lo, "hi": hi, "n": n}


# Minimum surviving paired questions before a winner can be declared. Guards
# against a degenerate CI computed from 1-2 surviving cells (most failed) being
# crowned a winner. (Codex diff-audit finding #3.)
MIN_PAIRS_FOR_WINNER = 10


def is_winner(ci: dict[str, Any], min_pairs: int = MIN_PAIRS_FOR_WINNER) -> bool:
    """Winner only if the paired-delta CI is entirely above 0 AND enough
    questions actually paired (not a degenerate CI from a handful of survivors)."""
    if ci.get("n", 0) < min_pairs:
        return False
    return ci.get("lo", 0.0) > 0.0


def sweep_lever(
    lever: str,
    questions: list[str],
    *,
    grid: Optional[list] = None,
    incumbent: Optional[float] = None,
    replicates: int = 1,
    n_grid: int = 5,
    seed: int = 42,
    base_config=None,
    min_pairs: int = MIN_PAIRS_FOR_WINNER,
    return_rows: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[Any, list[dict[str, Any]]]]:
    """Run every arm over all questions, pair vs incumbent, rank by CI.

    Deterministic arm ordering (sorted grid). Each arm runs in-process via
    config_override. Returns per-arm CIs + the winner (if any).
    """
    base = base_config if base_config is not None else build_config()
    if incumbent is None:
        incumbent = getattr(base, lever)
    if grid is None:
        grid = build_grid(lever, incumbent, n=n_grid)
    # A custom grid that omits the incumbent would KeyError on arm_rows[incumbent]
    # after burning expensive LLM cells. Fail fast with a clear message.
    # (Codex diff-audit finding #5.)
    if incumbent not in grid:
        raise ValueError(
            f"incumbent {incumbent} not in grid {grid} — the incumbent must "
            "always be an arm so candidates can be paired against it"
        )

    # Run all arms (deterministic order). Collect rows per arm.
    arm_rows: dict[Any, list[dict[str, Any]]] = {}
    for arm in sorted(grid):
        cfg = replace(base, **{lever: arm})
        rows: list[dict[str, Any]] = []
        for q in questions:
            rows.extend(run_cell(q, cfg, replicates=replicates, arm_label=str(arm)))
        arm_rows[arm] = rows

    inc_rows = arm_rows[incumbent]
    arms_out = {}
    winner = None
    best_lo = 0.0
    for arm, rows in arm_rows.items():
        if arm == incumbent:
            arms_out[arm] = {"ci": {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": len(rows)},
                             "is_incumbent": True}
            continue
        deltas = paired_deltas(inc_rows, rows)
        ci = bootstrap_ci(deltas, seed=seed)
        won = is_winner(ci, min_pairs=min_pairs)
        arms_out[arm] = {"ci": ci, "is_winner": won}
        if won and ci["lo"] > best_lo:
            best_lo = ci["lo"]
            winner = arm

    result = {
        "lever": lever,
        "incumbent": incumbent,
        "grid": sorted(grid),
        "arms": arms_out,
        "winner": winner,
        "seed": seed,
        "replicates": replicates,
        "n_questions": len(questions),
    }
    if return_rows:
        return result, arm_rows
    return result
