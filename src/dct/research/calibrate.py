"""Spike CLI — measure LLM jitter on real questions, print the go/no-go verdict.

    python -m dct.research.calibrate --questions 8 --replicates 5

Pulls verbatim user questions from utility.jsonl `followup` excerpts (the real,
re-askable user turns), runs run_cell at a SINGLE fixed setting (current live
config) for R replicates each, measures within-question jitter, and prints the
calibration verdict. This is the GO/NO-GO for the whole build: if jitter swamps
a ~0.08 lever effect at R<=5, STOP before building the sweep.

Makes REAL LLM calls (reply + judge). Token cost: ~questions * replicates * 2.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dct.research import calibration
from dct.research.runner import run_cell
from dct.retrieval.service import build_config

log = logging.getLogger(__name__)

UTILITY_JSONL = Path.home() / "example-stack" / "dynamic-context-traversal" / "logs" / "utility.jsonl"

# Minimum length so we skip trivial followups like "yes" / "ok".
_MIN_Q_LEN = 25


def load_questions(n: int, path: Path = UTILITY_JSONL) -> list[str]:
    """Pull n distinct, non-trivial user questions from utility.jsonl followups."""
    seen: set[str] = set()
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") != "followup":
            continue
        q = (r.get("excerpt") or "").strip()
        if len(q) < _MIN_Q_LEN or q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= n:
            break
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--effect", type=float, default=0.08, help="candidate lever effect to resolve")
    ap.add_argument("--n-benchmark", type=int, default=50, help="planned benchmark size")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        print("NO QUESTIONS LOADED — check utility.jsonl followups")
        return 2

    print(f"=== CALIBRATION SPIKE: {len(questions)} questions x {args.replicates} replicates ===")
    for i, q in enumerate(questions):
        print(f"  Q{i+1}: {q[:70]}")

    cfg = build_config()  # single FIXED setting (current live config)
    all_rows = []
    for i, q in enumerate(questions):
        rows = run_cell(q, cfg, replicates=args.replicates, arm_label="fixed")
        composites = [r["composite"] for r in rows if r.get("composite") is not None]
        print(f"  Q{i+1} composites: {[round(c, 3) for c in composites]}")
        all_rows.extend(rows)

    v = calibration.verdict(
        all_rows, candidate_effect=args.effect, n_questions=args.n_benchmark
    )

    print("\n=== VERDICT ===")
    print(f"  mean within-question stdev (jitter): {v['mean_within_question_stdev']:.4f}")
    print(f"  max  within-question stdev:          {v['max_within_question_stdev']:.4f}")
    print(f"  candidate effect to resolve:         {v['candidate_effect']}")
    print(f"  planned benchmark size:              {v['n_questions']}")
    verdict_word = "PASS ✅ — build the sweep" if v["pass"] else "FAIL ⛔ — STOP, surface to Alex"
    print(f"  GO/NO-GO: {verdict_word}")
    print(f"  recommended replicates R: {v['recommended_R']}")
    print(f"  detail: {v['detail']}")
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
