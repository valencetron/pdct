"""Token-cost panel — `python -m dct.metrics tokens [--days N]`.

Reads $PDCT_LOGS_DIR/measurement.jsonl (turn_measurement rows) and prints
mean/p50/p95 of identity_anchor_chars, retrieval_context_chars,
total_injected_chars, total_injected_tokens_est, prompt_total_chars,
cascade_latency_ms, plus skip-reason breakdown.

Spec: §Stage 1.
"""
from __future__ import annotations

import statistics
from collections import Counter

from dct.retrieval.measurement import get_logs_dir

from ._io import days_ago, iter_rows_jsonl

_FIELDS = [
    ("identity_anchor_chars", "identity_anchor_chars"),
    ("retrieval_context_chars", "retrieval_chars"),
    ("total_injected_chars", "total_injected_chars"),
    ("total_injected_tokens_est", "tokens_est (total)"),
    ("prompt_total_chars", "prompt_total_chars"),
    ("cascade_latency_ms", "cascade_latency_ms"),
    ("output_chars", "output_chars"),
]


def _pct(values: list[int | float], p: float) -> int | float:
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def run(*, days: int = 7) -> int:
    logs_dir = get_logs_dir()
    path = logs_dir / "measurement.jsonl"
    since = days_ago(days)
    rows = list(iter_rows_jsonl(path, since=since, kind="turn_measurement"))

    n = len(rows)
    print(f"PDCT token cost · last {days} days · n={n} turns")
    if n == 0:
        return 0

    # Skip-reason breakdown
    skip_counter: Counter[str] = Counter(
        str(r.get("pdct_skipped_reason", "?")) for r in rows
    )
    print()
    print("Skip-reason breakdown:")
    for reason, ct in sorted(skip_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<14}{ct}")
    print()

    # Aggregate panel — split by skip-reason bucket would be nice but Stage-1
    # keeps it simple: aggregate over all turns. The ablation CLI splits.
    print(f"  {'metric':<28} {'mean':>8}   {'p50':>8}   {'p95':>8}")
    for field, label in _FIELDS:
        vals = [r.get(field) for r in rows if isinstance(r.get(field), (int, float))]
        if not vals:
            continue
        mean = statistics.mean(vals)
        p50 = _pct(vals, 50)
        p95 = _pct(vals, 95)
        print(f"  {label:<28} {mean:>8.0f}   {p50:>8.0f}   {p95:>8.0f}")

    return 0
