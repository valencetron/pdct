"""Engine core tests — shadow isolation, batch CAS, lock, tier gates (Task 2)."""
import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from dct.retrieval import overrides as ov
from dct.tuning import engine


@pytest.fixture(autouse=True)
def _tune_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_RUNTIME_DIR", str(tmp_path / "runtime"))
    # runtime_dir() reads env at call time via _env_path
    yield tmp_path


def _mk_tier1(scores):
    """tier1_fn returning canned harness results; records calls."""
    calls = []

    def fn(co):
        calls.append(co)
        s = scores.pop(0) if scores else scores_last[0]
        scores_last[0] = s
        return {"status": "ok", "recall_at5": s, "jaccard_concept_ab_mean": 0.4}

    scores_last = [None]
    fn.calls = calls
    return fn


GOOD = dict(
    graph_nodes_fn=lambda: 1000,
    log_rows_fn=lambda: 100,
    health_fn=lambda: (True, ""),
)


# ── verdict logic ─────────────────────────────────────────────────────────

def test_verdict_tier1_regression_rejects():
    v, r = engine.decide_verdict(baseline_t1=0.9, cand_t1=0.8,
                                 baseline_t2=0.5, cand_t2=0.9,
                                 tier2_available=True)
    assert v == "reject" and r == "tier1_regression"


def test_verdict_tier2_abstained_never_promotes():
    v, r = engine.decide_verdict(baseline_t1=0.9, cand_t1=0.95,
                                 baseline_t2=None, cand_t2=None,
                                 tier2_available=False)
    assert v == "reject" and r == "tier2_abstained"


def test_verdict_promotes_on_tier2_improvement():
    v, r = engine.decide_verdict(baseline_t1=0.9, cand_t1=0.9,
                                 baseline_t2=0.5, cand_t2=0.6,
                                 tier2_available=True)
    assert v == "promote" and r == "tier2_improved"


def test_verdict_noise_band_rejects():
    v, r = engine.decide_verdict(baseline_t1=0.9, cand_t1=0.9,
                                 baseline_t2=0.5, cand_t2=0.51,
                                 tier2_available=True)
    assert v == "reject" and r == "tier2_no_improvement"


# ── floors (cold start) ───────────────────────────────────────────────────

def test_tier2_floor_graph_nodes():
    ok, why = engine.tier2_floors_met(graph_nodes_fn=lambda: 10,
                                      log_rows_fn=lambda: 999)
    assert not ok and "graph_nodes" in why


def test_tier2_floor_log_rows():
    ok, why = engine.tier2_floors_met(graph_nodes_fn=lambda: 999,
                                      log_rows_fn=lambda: 3)
    assert not ok and "log_rows" in why


# ── batch override transaction ────────────────────────────────────────────

def test_write_overrides_batch_atomic(tmp_path):
    p = str(tmp_path / "ov.json")
    ov.write_overrides_batch({"cascade_depth": 3, "cascade_decay": 0.5}, path=p)
    got = ov.load_overrides(p)
    assert got == {"cascade_depth": 3, "cascade_decay": 0.5}


def test_write_overrides_batch_invalid_aborts_whole_batch(tmp_path):
    p = str(tmp_path / "ov.json")
    ov.write_override("cascade_top_k", 40, path=p)
    with pytest.raises(ValueError):
        ov.write_overrides_batch({"cascade_depth": 3, "bogus_lever": 1}, path=p)
    assert ov.load_overrides(p) == {"cascade_top_k": 40}  # untouched


def test_write_overrides_batch_none_deletes(tmp_path):
    p = str(tmp_path / "ov.json")
    ov.write_override("cascade_depth", 3, path=p)
    ov.write_overrides_batch({"cascade_depth": None, "cascade_decay": 0.5}, path=p)
    assert ov.load_overrides(p) == {"cascade_decay": 0.5}


# ── shadow isolation ──────────────────────────────────────────────────────

def test_evaluation_never_writes_overrides(tmp_path, monkeypatch):
    """A full evaluated tick with a REJECT verdict must not touch the live file."""
    live = tmp_path / "live-ov.json"
    live.write_text("{}")
    before = live.stat().st_mtime_ns
    writes = []

    r = engine.run_tick(
        tier1_fn=_mk_tier1([0.9, 0.5]),  # baseline 0.9, candidate 0.5 -> reject
        tier2_fn=lambda co: 0.5,
        apply_batch=lambda ch: writes.append(ch),
        **GOOD,
    )
    assert r.action == "evaluated" and r.verdict == "reject"
    assert writes == []
    assert live.stat().st_mtime_ns == before


def test_promotion_applies_batch_once():
    writes = []
    r = engine.run_tick(
        tier1_fn=_mk_tier1([0.9, 0.9]),
        tier2_fn=lambda co: 0.5 if not co else 0.9,  # candidate improves
        apply_batch=lambda ch: writes.append(ch),
        **GOOD,
    )
    assert r.verdict == "promote", r
    assert writes == [{"cascade_depth": 3}]  # first queue move


def test_health_gate_blocks_promotion():
    writes = []
    r = engine.run_tick(
        tier1_fn=_mk_tier1([0.9, 0.9]),
        tier2_fn=lambda co: 0.5 if not co else 0.9,
        graph_nodes_fn=lambda: 1000, log_rows_fn=lambda: 100,
        health_fn=lambda: (False, "index broken"),
        apply_batch=lambda ch: writes.append(ch),
    )
    assert r.verdict == "reject" and "health_gate" in r.reason
    assert writes == []


# ── convergence ───────────────────────────────────────────────────────────

def test_converges_after_k_rejections():
    for i in range(engine.CONVERGE_AFTER):
        r = engine.run_tick(
            tier1_fn=_mk_tier1([0.9, 0.5] if i == 0 else [0.5]),
            tier2_fn=lambda co: 0.5,
            apply_batch=lambda ch: None,
            **GOOD,
        )
    assert r.converged
    r2 = engine.run_tick(tier1_fn=_mk_tier1([0.9]), tier2_fn=lambda co: 0.5,
                         apply_batch=lambda ch: None, **GOOD)
    assert r2.action == "idle" and r2.converged


def test_error_in_tier1_returns_error_not_raise():
    def boom(co):
        raise RuntimeError("kaboom")
    r = engine.run_tick(tier1_fn=boom, tier2_fn=lambda co: 0.5,
                        apply_batch=lambda ch: None, **GOOD)
    assert r.action == "error" and "kaboom" in r.note


# ── lock contention (Codex #3) ────────────────────────────────────────────

def _hold_lock(runtime_dir, started, release):
    os.environ["PDCT_RUNTIME_DIR"] = runtime_dir
    import fcntl
    from dct.tuning.engine import tune_dir
    lf = open(tune_dir() / "tick.lock", "w")
    fcntl.flock(lf, fcntl.LOCK_EX)
    started.set()
    release.wait(10)
    fcntl.flock(lf, fcntl.LOCK_UN)


def test_concurrent_tick_exits_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_RUNTIME_DIR", str(tmp_path / "rt2"))
    ctx = multiprocessing.get_context("fork")
    started, release = ctx.Event(), ctx.Event()
    p = ctx.Process(target=_hold_lock,
                    args=(str(tmp_path / "rt2"), started, release))
    p.start()
    try:
        assert started.wait(10)
        r = engine.run_tick(tier1_fn=_mk_tier1([0.9]), tier2_fn=lambda co: 0.5,
                            apply_batch=lambda ch: None, **GOOD)
        assert r.action == "busy"
    finally:
        release.set()
        p.join(10)
