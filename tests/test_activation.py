from pathlib import Path

from dct.activation import ActivationEngine, DecayConfig
from dct.event_log import EventLog
from dct.events import Event, EventOp, EventSource


def _ev(ts: float, *concepts: str) -> Event:
    return Event(
        ts=ts,
        source=EventSource.TELEGRAM,
        op=EventOp.WRITE,
        concepts=list(concepts),
    )


def test_empty_engine_has_zero_heat_everywhere() -> None:
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=60.0))
    assert eng.heat("anything", now=100.0) == 0.0


def test_single_event_at_now_has_unit_heat() -> None:
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=60.0))
    eng.consume(_ev(100.0, "alpha"))
    assert eng.heat("alpha", now=100.0) == 1.0


def test_heat_decays_by_half_after_one_half_life() -> None:
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=60.0))
    eng.consume(_ev(0.0, "alpha"))
    assert eng.heat("alpha", now=60.0) == 0.5


def test_heat_decays_to_quarter_after_two_half_lives() -> None:
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=60.0))
    eng.consume(_ev(0.0, "alpha"))
    assert eng.heat("alpha", now=120.0) == 0.25


def test_reactivation_flares_back_to_unit() -> None:
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=60.0))
    eng.consume(_ev(0.0, "alpha"))
    # Heat is 0.5 at t=60.
    eng.consume(_ev(60.0, "alpha"))
    assert eng.heat("alpha", now=60.0) == 1.0


def test_unseen_concept_stays_cold() -> None:
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=60.0))
    eng.consume(_ev(0.0, "alpha"))
    assert eng.heat("beta", now=0.0) == 0.0


def test_consume_multiple_concepts_in_one_event() -> None:
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=60.0))
    eng.consume(_ev(0.0, "alpha", "beta", "gamma"))
    assert eng.heat("alpha", now=0.0) == 1.0
    assert eng.heat("beta", now=0.0) == 1.0
    assert eng.heat("gamma", now=0.0) == 1.0


def test_radius_zero_hops_only_affects_named_concepts() -> None:
    graph = {"alpha": ["beta"], "beta": ["alpha"]}
    eng = ActivationEngine(
        config=DecayConfig(
            half_life_seconds=60.0,
            radius_hop_cap=0,
            radius_falloff=0.5,
        )
    )
    eng.set_graph(graph)
    eng.consume(_ev(0.0, "alpha"))
    assert eng.heat("alpha", now=0.0) == 1.0
    assert eng.heat("beta", now=0.0) == 0.0


def test_radius_one_hop_propagates_with_falloff() -> None:
    graph = {"alpha": ["beta"], "beta": ["alpha"]}
    eng = ActivationEngine(
        config=DecayConfig(
            half_life_seconds=60.0,
            radius_hop_cap=1,
            radius_falloff=0.5,
        )
    )
    eng.set_graph(graph)
    eng.consume(_ev(0.0, "alpha"))
    assert eng.heat("alpha", now=0.0) == 1.0
    assert eng.heat("beta", now=0.0) == 0.5


def test_radius_respects_hop_cap() -> None:
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
    eng = ActivationEngine(
        config=DecayConfig(
            half_life_seconds=60.0,
            radius_hop_cap=2,
            radius_falloff=0.5,
        )
    )
    eng.set_graph(graph)
    eng.consume(_ev(0.0, "a"))
    assert eng.heat("a", now=0.0) == 1.0
    assert eng.heat("b", now=0.0) == 0.5
    assert eng.heat("c", now=0.0) == 0.25
    assert eng.heat("d", now=0.0) == 0.0  # beyond hop cap


def test_radius_uses_shortest_path() -> None:
    # a--b, a--c, b--d, c--d  (d is 2 hops from a through either b or c)
    graph = {
        "a": ["b", "c"],
        "b": ["a", "d"],
        "c": ["a", "d"],
        "d": ["b", "c"],
    }
    eng = ActivationEngine(
        config=DecayConfig(
            half_life_seconds=60.0,
            radius_hop_cap=3,
            radius_falloff=0.5,
        )
    )
    eng.set_graph(graph)
    eng.consume(_ev(0.0, "a"))
    assert eng.heat("d", now=0.0) == 0.25  # 0.5^2, not 0.5^3


def test_replay_from_event_log_reconstructs_state(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)
    log.append(_ev(0.0, "alpha"))
    log.append(_ev(0.0, "beta"))
    log.append(_ev(60.0, "alpha"))  # reactivation

    eng = ActivationEngine.replay(log, config=DecayConfig(half_life_seconds=60.0))

    assert eng.heat("alpha", now=60.0) == 1.0  # just reactivated
    assert eng.heat("beta", now=60.0) == 0.5   # 60s old, 60s half-life


def test_replay_is_independent_of_file_order(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)
    # Write out of order; EventLog sorts on read.
    log.append(_ev(60.0, "alpha"))
    log.append(_ev(0.0, "alpha"))
    log.append(_ev(0.0, "beta"))

    eng = ActivationEngine.replay(log, config=DecayConfig(half_life_seconds=60.0))
    assert eng.heat("alpha", now=60.0) == 1.0
    assert eng.heat("beta", now=60.0) == 0.5


def test_radius_heat_still_decays_temporally() -> None:
    graph = {"alpha": ["beta"], "beta": ["alpha"]}
    eng = ActivationEngine(
        config=DecayConfig(
            half_life_seconds=60.0,
            radius_hop_cap=1,
            radius_falloff=0.5,
        )
    )
    eng.set_graph(graph)
    eng.consume(_ev(0.0, "alpha"))
    # At t=60: alpha temporal 0.5; beta = alpha heat * falloff = 0.5 * 0.5 = 0.25.
    assert eng.heat("alpha", now=60.0) == 0.5
    assert eng.heat("beta", now=60.0) == 0.25


def test_snapshot_returns_nonzero_heat_concepts() -> None:
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=30.0))
    eng.consume(_ev(0.0, "alpha", "beta"))
    eng.consume(_ev(30.0, "gamma"))

    snap = eng.snapshot(now=30.0, min_heat=0.01)
    assert snap["alpha"] == 0.5
    assert snap["beta"] == 0.5
    assert snap["gamma"] == 1.0


def test_snapshot_filters_below_threshold() -> None:
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=1.0))
    eng.consume(_ev(0.0, "alpha"))
    # At t=20, heat = 0.5^20 ≈ 9.5e-7.
    snap = eng.snapshot(now=20.0, min_heat=0.01)
    assert "alpha" not in snap


def test_snapshot_sorted_desc_by_heat() -> None:
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=60.0))
    eng.consume(_ev(0.0, "old"))
    eng.consume(_ev(60.0, "new"))
    snap = eng.snapshot(now=60.0, min_heat=0.01)
    assert list(snap.keys()) == ["new", "old"]


def test_last_seen_ts_returns_none_for_unknown_concept():
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=3600))
    assert eng.last_seen_ts("nope") is None


def test_last_seen_ts_returns_max_ts_after_ignitions():
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=3600))
    eng.consume(_ev(100.0, "alpha"))
    eng.consume(_ev(200.0, "alpha"))
    eng.consume(_ev(150.0, "alpha"))
    assert eng.last_seen_ts("alpha") == 200.0


def test_activation_consume_skips_feedback_event():
    """R2.7: FEEDBACK is meta-signal; must NOT activate concepts in snapshot."""
    from dct.events import Event, EventOp, EventSource
    eng = ActivationEngine(config=DecayConfig(half_life_seconds=3600.0))
    eng.consume(Event(ts=1000.0, source=EventSource.TELEGRAM, op=EventOp.WRITE,
                      concepts=["a", "b"]))
    eng.consume(Event(ts=1010.0, source=EventSource.TELEGRAM, op=EventOp.FEEDBACK,
                      concepts=["c", "d"],
                      metadata={"path": ["c", "d"], "multipliers": [4],
                                "useful_concept": "d", "thread_id": "92"}))
    assert eng.last_seen_ts("a") == 1000.0
    assert eng.last_seen_ts("b") == 1000.0
    assert eng.last_seen_ts("c") is None
    assert eng.last_seen_ts("d") is None
