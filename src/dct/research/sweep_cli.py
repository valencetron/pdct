"""Top-level lever sweep driver.

    python3 -m dct.research.sweep_cli --lever cascade_score_floor --replicates 3

Chains: load frozen asset -> pre-flight jitter guard -> sweep_lever -> aggregate
.summarize -> aggregate.decide -> write_report. Prints the verdict + report path.
DOES NOT mutate live config — applying a promoted lever is a separate build.

Makes REAL LLM calls when run live (reply + judge per arm x question x replicate).
The test suite monkeypatches run_cell + sweep_lever_with_rows so no LLM is hit.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

from dct.research import aggregate, calibration, report
from dct.research.runner import run_cell
from dct.research.sweep import sweep_lever
from dct.retrieval.service import build_config

log = logging.getLogger(__name__)

DEFAULT_ASSET = (
    Path.home() / "example-stack" / "dynamic-context-traversal"
    / "benchmark" / "pdct-questions-v1.json"
)
# Jitter ceiling from the #56 calibration spike — max acceptable pre-flight
# within-question stdev. The guard compares the live stdev against this before
# spending the expensive sweep.
DEFAULT_JITTER_CEILING = 0.02
# Operational margin floor — the paired-delta CI LOWER BOUND must exceed this for a
# PROMOTE. A DIFFERENT statistic from the jitter ceiling (Codex r2 #4).
DEFAULT_MARGIN_FLOOR = 0.02


def load_asset_questions(asset_path: Path) -> list[str]:
    data = json.loads(Path(asset_path).read_text())
    qs = [q["question"] for q in data.get("questions", [])]
    if not qs:
        raise RuntimeError(f"no questions in frozen asset {asset_path}")
    return qs


def sweep_lever_with_rows(lever: str, questions: list[str], **kw) -> tuple[dict, dict]:
    """Thin wrapper: sweep_lever with return_rows=True (added in Task 6.5) returns
    (result_dict, arm_rows) in a single pass — no double LLM cost. Indirected through
    this module-level function so the smoke test can monkeypatch it."""
    return sweep_lever(lever, questions, return_rows=True, **kw)


def run_sweep(
    *,
    lever: str,
    asset_path: Path = DEFAULT_ASSET,
    vault_root: Optional[Path] = None,
    replicates: int = 3,
    jitter_ceiling: float = DEFAULT_JITTER_CEILING,
    margin_floor: float = DEFAULT_MARGIN_FLOOR,
) -> int:
    # jitter_ceiling = max acceptable pre-flight within-question stdev.
    # margin_floor   = operational paired-delta CI-lower-bound threshold to promote.
    # These are DIFFERENT statistics (Codex r2 #4) — keep them separate.

    # Threshold sanity FIRST (Codex Build #58 diff-audit F2). A non-finite ceiling or
    # floor (nan/inf) makes the comparison gates degenerate and fail OPEN: `x > nan` is
    # always False (jitter never trips) and `lo <= nan` is always False (always
    # promotes). Reject before spending a single LLM call. rc=4 = bad-threshold.
    import math as _math
    for _name, _val in (("jitter_ceiling", jitter_ceiling), ("margin_floor", margin_floor)):
        if not (isinstance(_val, (int, float)) and not isinstance(_val, bool)
                and _math.isfinite(_val)):
            log.error("[sweep] %s=%r is not a finite real — refusing to run; a "
                      "non-finite threshold makes the gate fail open.", _name, _val)
            return 4

    # Validate the lever BEFORE any LLM call (Codex r4 #8 / r5 #2) — must be a KNOWN
    # NUMERIC swept lever with a finite min/max, not just present in LEVER_SPEC (which
    # also holds boolean levers like cascade_heat_enabled that build_grid can't sweep).
    # Probe build_grid: it raises/returns degenerate for non-numeric or unbounded specs.
    from dct.retrieval.overrides import LEVER_SPEC
    from dct.research.sweep import build_grid
    if lever not in LEVER_SPEC:
        log.error("[sweep] unknown lever %r — valid levers: %s", lever, sorted(LEVER_SPEC))
        return 3
    try:
        incumbent_probe = getattr(build_config(), lever)
        probe_grid = build_grid(lever, incumbent_probe, n=5)
    except Exception as e:  # noqa: BLE001
        log.error("[sweep] lever %r is not a sweepable numeric lever: %s", lever, e)
        return 3
    if not probe_grid or len(set(probe_grid)) < 2:
        log.error("[sweep] lever %r produced a degenerate grid %r (needs a finite "
                  "numeric min/max range) — cannot sweep.", lever, probe_grid)
        return 3

    questions = load_asset_questions(asset_path)

    # Pre-flight jitter guard (D5 + Codex #4): reuse the EXISTING calibration logic,
    # which groups by question and fail-closes unless >=2 questions have repeated
    # successful replicates. Run on the first 2 asset questions x replicates at the
    # incumbent setting — do NOT reinvent a flat-list check.
    cfg = build_config()
    guard_rows = []
    for q in questions[:2]:
        guard_rows.extend(run_cell(q, cfg, replicates=max(replicates, 2),
                                   arm_label="jitter-guard"))
    j = calibration.measure_jitter(guard_rows)
    # FAIL-CLOSED: measure_jitter returns mean_within_question_stdev=0.0 (NOT None)
    # when there is no measurable data — a 0.0 from missing replicates is NOT
    # stability. Require >=2 questions each with >=2 successful replicates (j[
    # "n_questions"] counts only measurable ones) AND stdev within ceiling.
    if j["n_questions"] < 2 or j["mean_within_question_stdev"] > jitter_ceiling:
        log.error("[sweep] jitter guard TRIPPED (measurable questions=%d, "
                  "within-question stdev=%.4f vs ceiling %.3f) — aborting before the "
                  "expensive sweep; CI would be calibrated to stale/insufficient noise.",
                  j["n_questions"], j["mean_within_question_stdev"], jitter_ceiling)
        return 2

    # Expensive sweep.
    sweep_result, arm_rows = sweep_lever_with_rows(
        lever, questions, replicates=replicates, base_config=cfg
    )

    summary = aggregate.summarize(sweep_result, arm_rows)
    verdict = aggregate.decide(summary, sweep_result, margin_floor=margin_floor)
    exp = aggregate.build_exp(summary, verdict, trigger="manual sweep")

    path = report.write_report(exp, vault_root=vault_root)
    print(f"\n=== VERDICT: {verdict['verdict']} ===")
    print(verdict["sentence"])
    print(f"report: {path}")
    if verdict["verdict"] == "PROMOTE":
        print("\nNOTE: live config NOT changed. Surface to Alex for the apply decision "
              "(deploy controller is a separate build).")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", required=True)
    ap.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--jitter-ceiling", type=float, default=DEFAULT_JITTER_CEILING)
    ap.add_argument("--margin-floor", type=float, default=DEFAULT_MARGIN_FLOOR)
    ap.add_argument("--vault-root", type=Path, default=None,
                    help="override the Obsidian vault root (for dry runs / CI)")
    args = ap.parse_args()
    return run_sweep(
        lever=args.lever, asset_path=args.asset, vault_root=args.vault_root,
        replicates=args.replicates, jitter_ceiling=args.jitter_ceiling,
        margin_floor=args.margin_floor,
    )


if __name__ == "__main__":
    raise SystemExit(main())
