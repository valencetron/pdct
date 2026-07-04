"""Drift watchdog + convergence maintenance (Build 106, Task 3).

Statistics (Codex plan-audit #9 — defined, not vibes):
    * trailing window: last WINDOW baseline Tier 1 scores
    * drift: current score below trailing mean minus
      max(NOISE_BAND, 1 sample stddev)
    * hysteresis: DRIFT_CONSECUTIVE consecutive drift readings required to
      reopen (no flapping)
    * minimum MIN_OBSERVATIONS scores before drift can fire (cold start)
    * "queue exhausted" is distinct from "converged"

The watchdog never experiments. It re-scores the CURRENT live config against
the Tier 1 reference benchmark, appends to history, and — when drift is
confirmed — reopens experimentation by clearing the converged flag and
resetting the rejection counter. State shares ``tune/state.json`` with engine.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

from dct.tuning import engine

WINDOW = 10
MIN_OBSERVATIONS = 5
DRIFT_CONSECUTIVE = 2
NOISE_BAND = engine.NOISE_BAND


def check_drift(history: list[float], current: float) -> tuple[bool, str]:
    """Pure drift predicate over a trailing window (no hysteresis here)."""
    window = [s for s in history[-WINDOW:] if isinstance(s, (int, float))]
    if len(window) < MIN_OBSERVATIONS:
        return False, f"insufficient_observations ({len(window)}/{MIN_OBSERVATIONS})"
    mean = statistics.fmean(window)
    stdev = statistics.stdev(window) if len(window) >= 2 else 0.0
    threshold = mean - max(NOISE_BAND, stdev)
    if current < threshold:
        return True, (f"drift: current={current:.4f} < mean={mean:.4f} "
                      f"- max(band,stdev)={max(NOISE_BAND, stdev):.4f}")
    return False, "within_band"


def run_watchdog(
    *,
    tier1_fn: Callable[[Optional[dict]], dict] = None,
    now: Optional[float] = None,
) -> dict:
    """One watchdog pass. Returns a status dict; never raises.

    Reopening requires DRIFT_CONSECUTIVE consecutive drift readings persisted
    in state (hysteresis).
    """
    from dct.tuning.harness import run_reference_benchmark
    tier1_fn = tier1_fn or (lambda co: run_reference_benchmark(co))

    td = engine.tune_dir()
    state_path = td / "state.json"
    state = engine._load_json(state_path, {
        "baseline_t1": None, "baseline_t2": None, "done": [],
        "consecutive_rejections": 0, "converged": False, "history": [],
    })

    try:
        res = tier1_fn(None)
        score = engine._t1_score(res)
    except Exception as e:  # noqa: BLE001
        return {"action": "error", "error": f"{type(e).__name__}: {e}"}
    if score is None:
        return {"action": "error", "error": "tier1 benchmark unavailable"}

    history = list(state.get("history") or [])
    drifted, why = check_drift(history, score)

    streak = int(state.get("drift_streak") or 0)
    streak = streak + 1 if drifted else 0
    state["drift_streak"] = streak

    reopened = False
    if streak >= DRIFT_CONSECUTIVE and state.get("converged"):
        state["converged"] = False
        state["consecutive_rejections"] = 0
        state["drift_streak"] = 0
        reopened = True

    history.append(score)
    state["history"] = history[-20:]
    state["baseline_t1"] = score
    engine._save_json(state_path, state)

    row = {
        "ts": now or time.time(), "kind": "watchdog", "score": score,
        "drifted": drifted, "why": why, "streak": streak, "reopened": reopened,
    }
    with (td / "ledger.jsonl").open("a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"action": "reopened" if reopened else "scored", **row}
