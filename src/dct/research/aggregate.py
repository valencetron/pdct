# src/dct/research/aggregate.py
"""Pure aggregation layer — bridges sweep_lever's paired-delta CI output to the
write_report exp dict. No I/O, no LLM calls. Fully unit-tested with synthetic rows.

Row contract (from runner.run_cell — CONFIRM in Task 0):
  row["composite"] : float | None   — the FINAL composite scalar for that replicate
  row["legs"]      : dict[str,float] — per-judge-leg scores
  row["question"]  : str
  row["arm_label"] : str

D2 (double-transform guard): we mean the FINAL composite scalar across rows. We NEVER
aggregate per-leg then re-run compute_composite — that is the #56 double-transform bug.
Per-leg numbers are aggregated separately (aggregate_legs) for display/veto ONLY.
"""
from __future__ import annotations

from typing import Any, Optional


def aggregate_composite(rows: list[dict[str, Any]]) -> Optional[float]:
    """Mean of the final composite scalar across rows. None if no scored rows."""
    vals = [r["composite"] for r in rows if r.get("composite") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)

from dct.composite import _normalize_leg
from dct.research import BENCHMARK_WEIGHTS

# Only the canonical benchmark legs are aggregated/coverage-checked. Unknown keys in
# row["legs"] (telemetry, future legs) are IGNORED rather than silently flowing into
# the veto via _normalize_leg's permissive 0-1 numeric fallback (Codex r4 #6).
_CANONICAL_LEGS = frozenset(BENCHMARK_WEIGHTS.keys())


def aggregate_legs(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of each CANONICAL leg independently across rows, NORMALIZED to [0,1] first.

    Codex #3: legs arrive on mixed scales (era_judge raw 1-5, others 0-1). We normalize
    each value via dct.composite._normalize_leg BEFORE averaging so the veto's epsilon is
    scale-invariant. Codex r4 #6: only BENCHMARK_WEIGHTS legs are aggregated; unknown
    keys are ignored. Values that normalize to None are skipped. ABSOLUTE values for the
    veto baseline — NEVER recomposed into a composite (D2/D3)."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in rows:
        for leg, raw in (r.get("legs") or {}).items():
            if leg not in _CANONICAL_LEGS:
                continue
            val = _normalize_leg(leg, raw)
            if val is None:
                continue
            sums[leg] = sums.get(leg, 0.0) + val
            counts[leg] = counts.get(leg, 0) + 1
    return {leg: sums[leg] / counts[leg] for leg in sums}


def leg_coverage(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Fraction of rows in which each CANONICAL leg has a VALID (non-None after
    normalization) value. Codex r3 P1: presence of a leg KEY is not the same as the
    judge leg actually working — 1 valid era_judge out of 50 rows still yields an
    aggregate. decide() uses this to veto low-coverage legs. Only canonical legs are
    counted (Codex r4 #6). Denominator is total rows; valid in k of n -> k/n."""
    n = len(rows)
    if n == 0:
        return {}
    valid: dict[str, int] = {}
    for r in rows:
        for leg, raw in (r.get("legs") or {}).items():
            if leg not in _CANONICAL_LEGS:
                continue
            if _normalize_leg(leg, raw) is not None:
                valid[leg] = valid.get(leg, 0) + 1
    return {leg: valid[leg] / n for leg in valid}


def _mean_composite_by_question(rows: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in rows:
        c = r.get("composite")
        if c is None:
            continue
        q = r["question"]
        sums[q] = sums.get(q, 0.0) + c
        counts[q] = counts.get(q, 0) + 1
    return {q: sums[q] / counts[q] for q in sums}


def top_moving_questions(
    incumbent_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    k: int = 5,
) -> list[dict[str, Any]]:
    """Per-question mean composite delta (candidate - incumbent), ranked by |delta|.
    Only questions present in BOTH arms are compared."""
    inc = _mean_composite_by_question(incumbent_rows)
    cand = _mean_composite_by_question(candidate_rows)
    deltas = [
        {"question": q, "delta": cand[q] - inc[q]}
        for q in inc.keys() & cand.keys()
    ]
    # Deterministic tie order (Codex r4 #9): primary by |delta| desc, secondary by
    # question text asc — stable across runs for reproducible reports.
    deltas.sort(key=lambda d: (-abs(d["delta"]), d["question"]))
    return deltas[:k]


def summarize(
    sweep_result: dict[str, Any],
    arm_rows: dict[Any, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Assemble the aggregation summary from a sweep_result + the raw per-arm rows.

    D1 (no-winner): when winner is None, after_composite == before_composite ==
    incumbent, per_leg_delta is empty. We NEVER fabricate a delta from a losing arm.
    """
    incumbent = sweep_result["incumbent"]
    winner = sweep_result.get("winner")
    inc_rows = arm_rows[incumbent]

    before = aggregate_composite(inc_rows)
    incumbent_legs = aggregate_legs(inc_rows)
    incumbent_coverage = leg_coverage(inc_rows)

    if winner is None:
        return {
            "lever": sweep_result["lever"],
            "incumbent": incumbent,
            "winner": None,
            "before_composite": before,
            "after_composite": before,          # D1: no fabricated movement
            "incumbent_legs": incumbent_legs,
            "candidate_legs": dict(incumbent_legs),  # Codex #7: copy, don't alias
            "incumbent_coverage": incumbent_coverage,
            "candidate_coverage": dict(incumbent_coverage),
            "per_leg_delta": {},
            "top_moving_questions": [],
        }

    cand_rows = arm_rows[winner]
    after = aggregate_composite(cand_rows)
    candidate_legs = aggregate_legs(cand_rows)
    candidate_coverage = leg_coverage(cand_rows)
    # Codex r2 #2: delta domain is the UNION of both arms' legs, not just the
    # candidate's. A leg present in only one arm is reported as None (missing),
    # never as a fake delta-vs-0.0. The veto (decide) separately blocks on missing
    # required legs.
    all_legs = set(candidate_legs) | set(incumbent_legs)
    per_leg_delta: dict[str, Any] = {}
    for leg in all_legs:
        if leg in candidate_legs and leg in incumbent_legs:
            per_leg_delta[leg] = candidate_legs[leg] - incumbent_legs[leg]
        else:
            per_leg_delta[leg] = None  # present in only one arm — not a real delta
    return {
        "lever": sweep_result["lever"],
        "incumbent": incumbent,
        "winner": winner,
        "before_composite": before,
        "after_composite": after,
        "incumbent_legs": incumbent_legs,
        "candidate_legs": candidate_legs,
        "incumbent_coverage": incumbent_coverage,
        "candidate_coverage": candidate_coverage,
        "per_leg_delta": per_leg_delta,
        "top_moving_questions": top_moving_questions(inc_rows, cand_rows, k=5),
    }


from dct.research import north_star


def _canonical_required_legs() -> list[str]:
    from dct.research import BENCHMARK_WEIGHTS
    return [leg for leg, w in BENCHMARK_WEIGHTS.items() if w > 0]


def decide(
    summary: dict[str, Any],
    sweep_result: dict[str, Any],
    *,
    margin_floor: float,
    epsilon: float = 0.05,
    required_legs: Optional[list[str]] = None,
    min_coverage: float = 0.8,
) -> dict[str, Any]:
    """Promotion policy. PROMOTE only if ALL hold:
      1. there is a CI winner (summary['winner'] is not None),
      2. every REQUIRED leg has adequate coverage (>= min_coverage valid rows) in
         BOTH arms — presence of a leg KEY is not "the judge worked" (Codex r3 P1);
         required legs come from the canonical weighted set, NOT incumbent-observed
         legs (Codex r3 P1#2), so total judge failure can't quietly drop the baseline,
      3. no leg regressed past epsilon (north_star veto on NORMALIZED legs),
      4. the PAIRED-DELTA CI LOWER BOUND (sweep_result.arms[winner].ci.lo) exceeds
         margin_floor — confidently above the operational floor, not just mean>floor
         (Codex r2 #3). This is the paired statistic that selected the winner (Codex
         #2), NOT the unpaired composite difference.

    Otherwise HOLD. Returns verdict + machine reason + human sentence.
    """
    lever = summary["lever"]
    winner = summary.get("winner")
    required = required_legs if required_legs is not None else _canonical_required_legs()

    # Guard 0: threshold sanity (Codex Build #58 diff-audit F2). A non-finite
    # margin_floor (nan/inf) makes the `lo <= margin_floor` promote-gate degenerate
    # (`x <= nan` is always False), silently fail-OPEN, and promote. Reject it outright.
    import math as _math0
    if not (isinstance(margin_floor, (int, float)) and not isinstance(margin_floor, bool)
            and _math0.isfinite(margin_floor)):
        return {
            "verdict": "HOLD",
            "reason": "bad_margin_floor",
            "north_star": None,
            "sentence": (
                f"Refusing to evaluate `{lever}` — margin_floor={margin_floor!r} is not a "
                f"finite real; a non-finite threshold would make the promote-gate fail open."
            ),
        }

    # Guard 1: CI winner exists (D1)
    if winner is None:
        return {
            "verdict": "HOLD",
            "reason": "no_winner",
            "north_star": None,
            "sentence": f"No candidate cleared the bar on `{lever}` — holding incumbent.",
        }

    # Guard 1b: CI CONTRACT validity FIRST (Codex r6 #5/#6) — before coverage/veto, so a
    # broken sweep result is reported as malformed_ci, not masked as low_coverage. Reject
    # bool (ints in Python), non-finite mean/lo, winner mismatch, and insufficient pairs.
    import math as _math
    from dct.research.sweep import MIN_PAIRS_FOR_WINNER

    def _finite_real(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and _math.isfinite(x)

    # NOTE: we validate the WINNER arm's ci only. In sweep_lever, the winner (a
    # candidate arm) records ci["n"] = number of paired questions; the incumbent arm's
    # ci["n"] is len(rows) and is NOT paired — do not extend this check to all arms
    # uniformly or it will trip over the incumbent's different n semantics (Codex r7 #4).
    ci = (sweep_result.get("arms", {}).get(winner, {}) or {}).get("ci", {})
    n = ci.get("n")
    _lo, _mean, _hi = ci.get("lo"), ci.get("mean"), ci.get("hi")
    # Codex Build #58 diff-audit F1: validate hi AND the ordering invariant
    # lo <= mean <= hi. A contradictory CI (e.g. hi < lo) must fail-closed, not promote
    # on a stale lo. All three bounds must be finite reals; bounds must be monotone.
    _bounds_ok = (
        _finite_real(_lo) and _finite_real(_mean) and _finite_real(_hi)
        and _lo <= _mean <= _hi
    )
    if (not _bounds_ok
            or sweep_result.get("winner") != winner
            or not isinstance(n, int) or isinstance(n, bool) or n < MIN_PAIRS_FOR_WINNER):
        return {
            "verdict": "HOLD",
            "reason": "malformed_ci",
            "north_star": None,
            "sentence": (
                f"`{lever}`->{winner} won but the paired-delta CI is missing/malformed, "
                f"winner-mismatched, or under-paired (ci={ci!r}) — refusing to promote "
                f"on a broken result contract."
            ),
        }

    # Guard 2: required-leg COVERAGE in BOTH arms (Codex r3 P1).
    inc_cov = summary.get("incumbent_coverage", {})
    cand_cov = summary.get("candidate_coverage", {})
    low_cov = {
        leg: {"incumbent": inc_cov.get(leg, 0.0), "candidate": cand_cov.get(leg, 0.0)}
        for leg in required
        if inc_cov.get(leg, 0.0) < min_coverage or cand_cov.get(leg, 0.0) < min_coverage
    }
    if low_cov:
        return {
            "verdict": "HOLD",
            "reason": "low_leg_coverage",
            "north_star": None,
            "sentence": (
                f"`{lever}`->{winner} BLOCKED — required judge-leg(s) below "
                f"{min_coverage:.0%} coverage: {low_cov}. The judge effectively failed "
                f"for these legs; cannot promote on a degenerate baseline."
            ),
        }

    # Guard 3: no leg cratered (D4 + Codex r2 #1) — NORMALIZED absolute legs.
    ns = north_star.veto_check(
        summary["candidate_legs"], summary["incumbent_legs"],
        required=required, epsilon=epsilon,
    )
    if ns["blocked"]:
        return {
            "verdict": "HOLD",
            "reason": "leg_missing",
            "north_star": ns,
            "sentence": (
                f"`{lever}`->{winner} is BLOCKED — required judge-leg(s) missing from "
                f"the candidate ({ns['missing']}); cannot promote. {ns['detail']}"
            ),
        }
    if ns["vetoed"]:
        return {
            "verdict": "HOLD",
            "reason": "leg_regression",
            "north_star": ns,
            "sentence": (
                f"`{lever}`->{winner} won on composite but a judge-leg regressed "
                f"({ns['regressions']}) — holding. {ns['detail']}"
            ),
        }

    # Guard 4: confidently worth the move — PAIRED-DELTA CI LOWER BOUND (Codex r2 #3).
    lo = ci["lo"]
    mean = ci["mean"]
    if lo <= margin_floor:
        return {
            "verdict": "HOLD",
            "reason": "margin_below_floor",
            "north_star": ns,
            "sentence": (
                f"`{lever}`->{winner} won statistically (paired mean +{mean:.4f}) but the "
                f"CI lower bound +{lo:.4f} is not confidently above the margin floor "
                f"({margin_floor}) — not worth flipping."
            ),
        }

    return {
        "verdict": "PROMOTE",
        "reason": "all_guards_pass",
        "north_star": ns,
        "sentence": (
            f"Flipping `{lever}` {summary['incumbent']}->{winner} lifts composite by a "
            f"paired-delta +{mean:.4f} (CI lo +{lo:.4f} > floor {margin_floor}), no "
            f"judge-leg regressed. Recommend PROMOTE."
        ),
    }


def build_exp(
    summary: dict[str, Any],
    verdict: dict[str, Any],
    *,
    trigger: str,
) -> dict[str, Any]:
    """Merge summary + verdict into the exp dict write_report consumes."""
    return {
        "lever": summary["lever"],
        "trigger": trigger,
        "incumbent": summary["incumbent"],
        "winner": summary.get("winner"),
        "before_composite": summary["before_composite"],
        "after_composite": summary["after_composite"],
        "per_leg_delta": summary["per_leg_delta"],
        "top_moving_questions": summary["top_moving_questions"],
        "north_star": verdict.get("north_star"),
        "verdict": verdict["sentence"],          # human sentence (report body)
        "verdict_label": verdict["verdict"],     # machine PROMOTE/HOLD (Codex r4 #10)
        "reason": verdict.get("reason"),         # machine reason code
    }
