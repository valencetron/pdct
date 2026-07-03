"""Utility (surface-reuse rate) panel — `python -m dct.metrics utility`.

Reads $PDCT_LOGS_DIR/utility.jsonl (kind=turn rows). Splits by
pdct_skipped_reason (none → PDCT-on, ablation → Ablation), reports
aggregate match-rate with Wilson 95% CIs, hop split for PDCT-on, and
top never-matched concepts.

Spec: §Stage 2.
"""
from __future__ import annotations

from collections import Counter

from dct.retrieval.measurement import get_logs_dir

from ._io import UTILITY_MATCH_SCHEMA_MIN, days_ago, iter_rows_jsonl
from ._stats import wilson_ci


def _aggregate(rows: list[dict]) -> dict:
    """Σ eligible / Σ matched + by_hop sums."""
    total_eligible = 0
    total_matched = 0
    by_hop: dict[int, dict[str, int]] = {}
    for r in rows:
        total_eligible += int(r.get("concepts_eligible", 0) or 0)
        total_matched += int(r.get("concepts_matched", 0) or 0)
        bh = r.get("by_hop") or {}
        if not isinstance(bh, dict):
            continue
        for hop_key, b in bh.items():
            try:
                hop = int(hop_key)
            except (ValueError, TypeError):
                continue
            slot = by_hop.setdefault(hop, {"eligible": 0, "matched": 0})
            slot["eligible"] += int(b.get("eligible", 0) or 0)
            slot["matched"] += int(b.get("matched", 0) or 0)
    return {
        "total_eligible": total_eligible,
        "total_matched": total_matched,
        "by_hop": by_hop,
    }


def _format_rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    rate = k / n
    lo, hi = wilson_ci(k, n)
    return f"{rate:.3f} [{lo:.3f}, {hi:.3f}]"


def _never_matched(rows: list[dict], top_n: int = 10) -> list[tuple[str, int]]:
    """Concepts that appeared as injected but never as matched. Returns [(concept, n_injections)]."""
    inj_count: Counter[str] = Counter()
    matched_count: Counter[str] = Counter()
    for r in rows:
        for c in r.get("injected_concepts", []) or []:
            inj_count[c] += 1
        for c in r.get("matched_concepts", []) or []:
            matched_count[c] += 1
    never = [
        (c, n) for c, n in inj_count.most_common()
        if matched_count.get(c, 0) == 0
    ]
    return never[:top_n]


def run(*, days: int = 7) -> int:
    logs_dir = get_logs_dir()
    path = logs_dir / "utility.jsonl"
    since = days_ago(days)
    # Era-gate: only schema>=6 rows have node_kinds-aware eligibility/match
    # semantics; mixing pre-6 rows corrupts the aggregate (Code/Concept split).
    all_rows = list(iter_rows_jsonl(
        path, since=since, kind="turn",
        min_schema=UTILITY_MATCH_SCHEMA_MIN,
    ))

    n = len(all_rows)
    pdct_on = [r for r in all_rows if r.get("pdct_skipped_reason") == "none"]
    ablation = [r for r in all_rows if r.get("pdct_skipped_reason") == "ablation"]

    print(
        f"PDCT surface-reuse · last {days} days · "
        f"n={n} (PDCT-on={len(pdct_on)}, Ablation={len(ablation)})"
    )
    if n == 0:
        return 0

    agg_on = _aggregate(pdct_on)
    agg_ab = _aggregate(ablation)

    print()
    print(f"  {'metric':<20} {'PDCT-on':<32} {'Ablation':<32}")
    print(f"  {'total eligible':<20} {agg_on['total_eligible']:<32} {agg_ab['total_eligible']:<32}")
    print(f"  {'total matched':<20} {agg_on['total_matched']:<32} {agg_ab['total_matched']:<32}")
    print(
        f"  {'rate ± Wilson 95%':<20} "
        f"{_format_rate(agg_on['total_matched'], agg_on['total_eligible']):<32} "
        f"{_format_rate(agg_ab['total_matched'], agg_ab['total_eligible']):<32}"
    )

    # Hop split — PDCT-on only (ablation by_hop is None)
    if agg_on["by_hop"]:
        print()
        print("PDCT-on by hop:")
        for hop in sorted(agg_on["by_hop"].keys()):
            b = agg_on["by_hop"][hop]
            print(
                f"  hop-{hop}: {_format_rate(b['matched'], b['eligible'])} "
                f"({b['matched']}/{b['eligible']})"
            )

    # Never-matched concepts (PDCT-on only)
    nm = _never_matched(pdct_on)
    if nm:
        print()
        print("Top never-matched concepts (PDCT-on):")
        for concept, ct in nm:
            print(f"  {concept:<40} n={ct}")

    return 0
