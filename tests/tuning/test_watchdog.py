"""Watchdog tests — drift stats, hysteresis, reopen, cold start (Task 3)."""
import json

import pytest

from dct.tuning import engine, watchdog


@pytest.fixture(autouse=True)
def _tune_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_RUNTIME_DIR", str(tmp_path / "runtime"))
    yield tmp_path


def _t1(score):
    return lambda co: {"status": "ok", "recall_at5": score,
                       "jaccard_concept_ab_mean": 0.4}


def _seed_state(**kw):
    st = {"baseline_t1": 0.9, "baseline_t2": None, "done": [],
          "consecutive_rejections": 0, "converged": False, "history": []}
    st.update(kw)
    engine._save_json(engine.tune_dir() / "state.json", st)
    return st


def test_check_drift_insufficient_observations():
    drifted, why = watchdog.check_drift([0.9, 0.9], 0.1)
    assert not drifted and "insufficient" in why


def test_check_drift_fires_below_band():
    hist = [0.90, 0.91, 0.90, 0.89, 0.90, 0.91]
    drifted, _ = watchdog.check_drift(hist, 0.80)
    assert drifted


def test_check_drift_within_band_ok():
    hist = [0.90, 0.91, 0.90, 0.89, 0.90, 0.91]
    drifted, why = watchdog.check_drift(hist, 0.895)
    assert not drifted and why == "within_band"


def test_hysteresis_requires_consecutive_drifts():
    _seed_state(converged=True, history=[0.78] * 6)  # _t1_score(0.9-recall fixture) = 0.78
    r1 = watchdog.run_watchdog(tier1_fn=_t1(0.5))
    assert r1["drifted"] and not r1["reopened"] and r1["streak"] == 1
    # second consecutive drift reading -> reopen. History now contains the 0.5,
    # but window mean still high enough that 0.5 drifts again.
    r2 = watchdog.run_watchdog(tier1_fn=_t1(0.5))
    assert r2["reopened"]
    state = engine._load_json(engine.tune_dir() / "state.json", {})
    assert state["converged"] is False
    assert state["consecutive_rejections"] == 0


def test_good_reading_resets_streak():
    _seed_state(converged=True, history=[0.78] * 6, drift_streak=1)
    r = watchdog.run_watchdog(tier1_fn=_t1(0.9))
    assert not r["drifted"] and r["streak"] == 0 and not r["reopened"]


def test_watchdog_never_raises():
    def boom(co):
        raise RuntimeError("no")
    r = watchdog.run_watchdog(tier1_fn=boom)
    assert r["action"] == "error"


def test_watchdog_appends_ledger():
    _seed_state(history=[0.78] * 6)
    watchdog.run_watchdog(tier1_fn=_t1(0.9))
    rows = [json.loads(l) for l in
            (engine.tune_dir() / "ledger.jsonl").read_text().splitlines()]
    assert rows and rows[-1]["kind"] == "watchdog"
