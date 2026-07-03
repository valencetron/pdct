"""Calibration — the go/no-go gate for the benchmark.

Question: is a plausible lever effect resolvable above LLM run-to-run jitter?

We have NO repeated-seed history, so jitter is measured empirically by running
the same questions multiple times at ONE fixed setting and taking the
within-question stdev of the composite.

resolvable() answers: given that noise, would a candidate effect of size E
clear a paired-bootstrap CI with n_questions paired deltas and R replicates?

Codex finding #6: use the EMPIRICAL paired-bootstrap logic that the real sweep
uses, not a parametric power formula. The standard error of a paired mean delta
is approximately stdev_of_paired_delta / sqrt(n). With R replicates averaged
per question, the per-question noise shrinks by sqrt(R). The paired delta's
stdev is bounded by sqrt(2) * within_question_stdev (two independent arms).
An effect is resolvable when it exceeds ~2 standard errors (95% CI clears 0).
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

# Decision constant: 95% CI clears 0 at ~1.96 SE. Use 2.0 for a small safety margin.
_CI_Z = 2.0


def measure_jitter(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Within-question stdev of the composite, averaged across questions.

    rows: replicate rows from run_cell at a SINGLE fixed setting, each with
    "question" and "composite".
    """
    by_q: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        c = r.get("composite")
        if c is None:
            continue
        by_q[r["question"]].append(float(c))

    # Only questions with >=2 successful replicates yield a jitter estimate.
    # n_questions MUST count those, not all keys — otherwise one-replicate
    # questions inflate the count and a sd=0.0 (from zero measurable jitter)
    # spuriously PASSes. (Codex diff-audit finding #2.)
    measurable = {q: v for q, v in by_q.items() if len(v) >= 2}
    stdevs = [statistics.pstdev(v) for v in measurable.values()]
    n_reps = max((len(v) for v in by_q.values()), default=0)
    return {
        "n_questions": len(measurable),
        "n_questions_total": len(by_q),
        "n_replicates": n_reps,
        "mean_within_question_stdev": statistics.mean(stdevs) if stdevs else 0.0,
        "max_within_question_stdev": max(stdevs) if stdevs else 0.0,
        "per_question_stdev": {q: statistics.pstdev(v) for q, v in measurable.items()},
    }


def standard_error(within_question_stdev: float, n_questions: int, replicates: int) -> float:
    """SE of the paired mean delta given the noise model.

    paired-delta stdev ≈ sqrt(2) * within_question_stdev (two arms),
    reduced by sqrt(R) per-question averaging, then sqrt(n_questions) for the mean.
    """
    if n_questions <= 0 or replicates <= 0:
        return float("inf")
    paired_sd = math.sqrt(2.0) * within_question_stdev / math.sqrt(replicates)
    return paired_sd / math.sqrt(n_questions)


def resolvable(
    within_question_stdev: float,
    candidate_effect: float,
    n_questions: int,
    replicates: int,
) -> bool:
    """True if a candidate_effect would clear a 95% paired CI given the noise."""
    se = standard_error(within_question_stdev, n_questions, replicates)
    return candidate_effect >= _CI_Z * se


def min_replicates(
    within_question_stdev: float,
    candidate_effect: float,
    n_questions: int,
    max_replicates: int = 5,
) -> int | None:
    """Smallest R in [1, max_replicates] that resolves the effect, or None."""
    for r in range(1, max_replicates + 1):
        if resolvable(within_question_stdev, candidate_effect, n_questions, r):
            return r
    return None


def verdict(
    rows: list[dict[str, Any]],
    *,
    candidate_effect: float = 0.08,
    n_questions: int = 50,
    max_replicates: int = 5,
) -> dict[str, Any]:
    """Go/no-go: can the benchmark resolve a candidate_effect at R≤max_replicates?

    PASS → recommended_R = the smallest R that resolves the effect.
    FAIL → jitter swamps the signal; surface to Alex before building the sweep.
    """
    j = measure_jitter(rows)
    sd = j["mean_within_question_stdev"]

    # FAIL-CLOSED on insufficient data. A jitter of 0.0 computed from zero (or
    # one-replicate) questions is NOT stability — it's missing data, and it
    # would spuriously "resolve" any effect. Require at least 2 questions that
    # each have >=2 successful replicates before trusting the verdict.
    min_questions_required = 2
    if j["n_questions"] < min_questions_required:
        return {
            "pass": False,
            "recommended_R": max_replicates,
            "mean_within_question_stdev": sd,
            "max_within_question_stdev": j["max_within_question_stdev"],
            "candidate_effect": candidate_effect,
            "n_questions": n_questions,
            "se_at_recommended_R": float("inf"),
            "insufficient_data": True,
            "detail": (
                f"INSUFFICIENT DATA — only {j['n_questions']} question(s) had "
                f">=2 successful replicates (need {min_questions_required}). "
                "Cannot measure jitter. Likely cause: LLM calls failed "
                "(rate limit / auth). Re-run when calls succeed."
            ),
        }

    r = min_replicates(sd, candidate_effect, n_questions, max_replicates)
    passed = r is not None
    return {
        "pass": passed,
        "recommended_R": r if passed else max_replicates,
        "mean_within_question_stdev": sd,
        "max_within_question_stdev": j["max_within_question_stdev"],
        "candidate_effect": candidate_effect,
        "n_questions": n_questions,
        "se_at_recommended_R": standard_error(sd, n_questions, r or max_replicates),
        "detail": (
            f"jitter(sd)={sd:.4f}; effect={candidate_effect}; "
            + (
                f"resolvable at R={r} (n={n_questions})"
                if passed
                else f"NOT resolvable at R<={max_replicates} (n={n_questions}) — "
                "increase replicates/questions or accept a coarser claim"
            )
        ),
    }
