"""Ablation comparison panel — `python -m dct.metrics ablation`.

Joins logs/measurement.jsonl with logs/utility.jsonl on turn_id and
parent_turn_id. Splits by pdct_skipped_reason ∈ {none, ablation},
reports correction-rate (next user) with Wilson 95% CIs, retrieval
token cost, and skip-reason breakdown.

Spec: §Stage 3 CLI section.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from dct.retrieval.measurement import get_logs_dir

from ._io import days_ago, iter_rows_jsonl
from ._stats import wilson_ci

# Recommended minimum sample per arm for ±0.10 95%-CI half-width at p~0.5
N_MIN_RECOMMENDED = 80


def _format_rate_ci(k: int, n: int) -> str:
    if n == 0:
        return "n/a (n=0)"
    rate = k / n
    lo, hi = wilson_ci(k, n)
    return f"{rate:.3f} [{lo:.3f}, {hi:.3f}]"


def _safe_mean(vals: list) -> float:
    nums = [v for v in vals if isinstance(v, (int, float))]
    return (sum(nums) / len(nums)) if nums else 0.0


def run(*, days: int = 7) -> int:
    logs_dir = get_logs_dir()
    measurement_path = logs_dir / "measurement.jsonl"
    utility_path = logs_dir / "utility.jsonl"
    since = days_ago(days)

    measurements = list(
        iter_rows_jsonl(measurement_path, since=since, kind="turn_measurement")
    )
    followups = list(
        iter_rows_jsonl(utility_path, since=since, kind="followup")
    )

    # Index followups by parent_turn_id (one per parent — last wins)
    followup_by_parent: dict[str, dict] = {}
    for f in followups:
        ptid = f.get("parent_turn_id")
        if ptid:
            followup_by_parent[ptid] = f

    n_total = len(measurements)

    # Split arms
    pdct_on = [m for m in measurements if m.get("pdct_skipped_reason") == "none"]
    ablation = [m for m in measurements if m.get("pdct_skipped_reason") == "ablation"]

    print(
        f"PDCT ablation · last {days} days · "
        f"n_total={n_total} (PDCT-on={len(pdct_on)}, Ablation={len(ablation)})"
    )
    if n_total == 0:
        return 0

    # Skip-reason breakdown
    skip_counter = Counter(
        str(m.get("pdct_skipped_reason", "?")) for m in measurements
    )
    print()
    print("Skip-reason breakdown:")
    for reason, ct in sorted(skip_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<14} {ct}")

    # Correction-rate per arm
    def _arm_correction(arm: list[dict]) -> tuple[int, int]:
        n = len(arm)
        corrections = 0
        for m in arm:
            tid = m.get("turn_id")
            f = followup_by_parent.get(tid)
            if f and f.get("rating") == "correction":
                corrections += 1
        return corrections, n

    on_corr, on_n = _arm_correction(pdct_on)
    ab_corr, ab_n = _arm_correction(ablation)

    print()
    print(f"  {'metric':<28} {'PDCT-on':<32} {'Ablation':<32}")
    print(f"  {'n turns':<28} {on_n:<32} {ab_n:<32}")
    print(
        f"  {'correction rate ± 95%':<28} "
        f"{_format_rate_ci(on_corr, on_n):<32} "
        f"{_format_rate_ci(ab_corr, ab_n):<32}"
    )

    # Token cost
    on_retrieval = _safe_mean([m.get("retrieval_context_chars", 0) for m in pdct_on])
    ab_retrieval = _safe_mean([m.get("retrieval_context_chars", 0) for m in ablation])
    print(
        f"  {'mean retrieval_chars':<28} "
        f"{on_retrieval:<32.0f} {ab_retrieval:<32.0f}"
    )

    on_out = _safe_mean([m.get("output_chars", 0) for m in pdct_on])
    ab_out = _safe_mean([m.get("output_chars", 0) for m in ablation])
    print(
        f"  {'mean output_chars':<28} "
        f"{on_out:<32.0f} {ab_out:<32.0f}"
    )

    on_conv = _safe_mean([m.get("conversation_length", 0) for m in pdct_on])
    ab_conv = _safe_mean([m.get("conversation_length", 0) for m in ablation])
    print(
        f"  {'mean conversation_length':<28} "
        f"{on_conv:<32.0f} {ab_conv:<32.0f}"
    )

    # Sample-size warning
    warnings = []
    if 0 < on_n < N_MIN_RECOMMENDED:
        warnings.append(f"PDCT-on n={on_n} below n_min={N_MIN_RECOMMENDED}")
    if 0 < ab_n < N_MIN_RECOMMENDED:
        warnings.append(f"Ablation n={ab_n} below n_min={N_MIN_RECOMMENDED}")
    if warnings:
        print()
        print("⚠  Sample-size warning:")
        for w in warnings:
            print(f"   {w}")
        print(f"   Continue soak until both arms reach n>={N_MIN_RECOMMENDED} for ±0.10 CI.")

    return 0
