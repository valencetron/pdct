"""Tests for dct.heat — concept graph builder + point-in-time heat snapshot.

heat.py is the server-side API for Mission Control's DCT heat visualization.
It wraps ActivationEngine.snapshot and adds a co-occurrence graph builder so
the browser can render nodes + edges without re-reading events.jsonl itself.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from dct.event_log import EventLog
from dct.events import Event, EventOp, EventSource


def _write_log(tmp_path: Path, events: list[Event]) -> Path:
    log_path = tmp_path / "events.jsonl"
    with log_path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev.to_dict()) + "\n")
    return log_path


def _evt(ts: float, concepts: list[str]) -> Event:
    return Event(
        ts=ts,
        source=EventSource.CLAUDE_CODE,
        op=EventOp.READ,
        concepts=concepts,
        metadata={},
    )


# ── build_concept_graph ─────────────────────────────────────────────────


def test_concept_graph_counts_nodes_by_occurrence(tmp_path):
    from dct.heat import build_concept_graph

    log_path = _write_log(tmp_path, [
        _evt(1000.0, ["alpha", "beta"]),
        _evt(1010.0, ["alpha", "gamma"]),
        _evt(1020.0, ["alpha"]),
    ])
    cg = build_concept_graph(EventLog(log_path))
    assert cg.nodes == {"alpha": 3, "beta": 1, "gamma": 1}


def test_concept_graph_edges_are_canonical_and_counted(tmp_path):
    """Edges are unordered pairs; we store them with source<target and count co-occurrences."""
    from dct.heat import build_concept_graph

    log_path = _write_log(tmp_path, [
        _evt(1000.0, ["alpha", "beta"]),
        _evt(1010.0, ["beta", "alpha"]),   # same pair, reversed
        _evt(1020.0, ["alpha", "gamma"]),
    ])
    cg = build_concept_graph(EventLog(log_path))
    edge_map = {(a, b): n for (a, b, n) in cg.edges}
    assert edge_map == {("alpha", "beta"): 2, ("alpha", "gamma"): 1}


def test_concept_graph_singleton_event_has_no_edges(tmp_path):
    from dct.heat import build_concept_graph

    log_path = _write_log(tmp_path, [_evt(1000.0, ["solo"])])
    cg = build_concept_graph(EventLog(log_path))
    assert cg.nodes == {"solo": 1}
    assert cg.edges == []


def test_concept_graph_dedupes_within_event(tmp_path):
    """If the same concept appears twice in one event's concept list, count once."""
    from dct.heat import build_concept_graph

    log_path = _write_log(tmp_path, [_evt(1000.0, ["alpha", "alpha", "beta"])])
    cg = build_concept_graph(EventLog(log_path))
    assert cg.nodes == {"alpha": 1, "beta": 1}
    edge_map = {(a, b): n for (a, b, n) in cg.edges}
    assert edge_map == {("alpha", "beta"): 1}


def test_concept_graph_empty_log(tmp_path):
    from dct.heat import build_concept_graph

    log_path = tmp_path / "empty.jsonl"
    log_path.write_text("")
    cg = build_concept_graph(EventLog(log_path))
    assert cg.nodes == {}
    assert cg.edges == []


# ── compute_heat_at ─────────────────────────────────────────────────────


def test_heat_recent_event_is_hot(tmp_path):
    from dct.heat import compute_heat_at

    log_path = _write_log(tmp_path, [_evt(1000.0, ["alpha"])])
    heat = compute_heat_at(log_path, ts=1000.0, half_life=100.0)
    assert heat["alpha"] == pytest.approx(1.0)


def test_heat_decays_by_half_life(tmp_path):
    from dct.heat import compute_heat_at

    log_path = _write_log(tmp_path, [_evt(1000.0, ["alpha"])])
    heat_now = compute_heat_at(log_path, ts=1100.0, half_life=100.0)
    assert heat_now["alpha"] == pytest.approx(0.5, rel=1e-4)

    heat_later = compute_heat_at(log_path, ts=1200.0, half_life=100.0)
    assert heat_later["alpha"] == pytest.approx(0.25, rel=1e-4)


def test_heat_filters_below_min_heat_threshold(tmp_path):
    """Concepts whose heat falls below min_heat drop out of the dict."""
    from dct.heat import compute_heat_at

    log_path = _write_log(tmp_path, [_evt(1000.0, ["alpha"])])
    heat = compute_heat_at(log_path, ts=2000.0, half_life=100.0, min_heat=0.1)
    assert "alpha" not in heat  # after 10 half-lives, far below 0.1


def test_heat_ignores_events_after_ts(tmp_path):
    """Heat at time T should only reflect events at or before T."""
    from dct.heat import compute_heat_at

    log_path = _write_log(tmp_path, [
        _evt(1000.0, ["alpha"]),
        _evt(2000.0, ["beta"]),  # after our snapshot time
    ])
    heat = compute_heat_at(log_path, ts=1500.0, half_life=1000.0)
    assert "alpha" in heat
    assert "beta" not in heat


def test_heat_blast_radius_warms_neighbors(tmp_path):
    """With hop_cap=1, a direct neighbor of an ignited concept should get partial heat."""
    from dct.heat import compute_heat_at

    log_path = _write_log(tmp_path, [
        _evt(1000.0, ["alpha", "beta"]),  # builds graph edge alpha-beta
        _evt(2000.0, ["alpha"]),           # ignite alpha hot
    ])
    heat = compute_heat_at(log_path, ts=2000.0, half_life=1000.0, hop_cap=1)
    assert heat["alpha"] == pytest.approx(1.0)
    # beta hasn't been ignited since ts=1000.0 (half-life ago → 0.5 base).
    # Via blast radius from alpha's ignition at 2000 with falloff 0.5, beta gets 0.5.
    # Engine picks the max, so beta == 0.5
    assert heat["beta"] == pytest.approx(0.5, rel=1e-3)


# ── time_range ──────────────────────────────────────────────────────────


def test_time_range_returns_min_and_max(tmp_path):
    from dct.heat import time_range

    log_path = _write_log(tmp_path, [
        _evt(1000.0, ["a"]),
        _evt(3000.0, ["b"]),
        _evt(2000.0, ["c"]),
    ])
    tr = time_range(EventLog(log_path))
    assert tr == (1000.0, 3000.0)


def test_time_range_empty_log_returns_none(tmp_path):
    from dct.heat import time_range

    log_path = tmp_path / "empty.jsonl"
    log_path.write_text("")
    assert time_range(EventLog(log_path)) == (None, None)


# ── CLI: python -m dct.heat ─────────────────────────────────────────────


def _run_cli(monkeypatch, capsys, *argv: str) -> dict:
    from dct import heat

    monkeypatch.setattr("sys.argv", ["dct.heat", *argv])
    heat.main()
    return json.loads(capsys.readouterr().out)


def test_cli_mode_graph_emits_nodes_and_edges(tmp_path, monkeypatch, capsys):
    log_path = _write_log(tmp_path, [
        _evt(1000.0, ["alpha", "beta"]),
        _evt(1010.0, ["alpha", "gamma"]),
    ])
    out = _run_cli(monkeypatch, capsys, "--log", str(log_path), "--mode", "graph")
    node_ids = {n["id"] for n in out["nodes"]}
    assert node_ids == {"alpha", "beta", "gamma"}
    edge_keys = {(e["source"], e["target"]) for e in out["edges"]}
    assert edge_keys == {("alpha", "beta"), ("alpha", "gamma")}


def test_cli_mode_heat_requires_ts(tmp_path, monkeypatch):
    from dct import heat

    log_path = _write_log(tmp_path, [_evt(1000.0, ["x"])])
    monkeypatch.setattr("sys.argv", ["dct.heat", "--log", str(log_path), "--mode", "heat"])
    with pytest.raises(SystemExit):
        heat.main()


def test_cli_mode_heat_emits_per_concept_heat(tmp_path, monkeypatch, capsys):
    log_path = _write_log(tmp_path, [_evt(1000.0, ["alpha"])])
    out = _run_cli(monkeypatch, capsys,
                   "--log", str(log_path), "--mode", "heat",
                   "--ts", "1000.0", "--half-life", "100")
    assert out["ts"] == 1000.0
    assert out["heat"]["alpha"] == pytest.approx(1.0)


def test_cli_mode_range_emits_ts_min_and_max(tmp_path, monkeypatch, capsys):
    log_path = _write_log(tmp_path, [
        _evt(1000.0, ["a"]),
        _evt(3000.0, ["b"]),
    ])
    out = _run_cli(monkeypatch, capsys, "--log", str(log_path), "--mode", "range")
    assert out["ts_min"] == 1000.0
    assert out["ts_max"] == 3000.0


# ── Track B: feedback multipliers + topic_id + ignore_feedback ────────────

def _make_log(tmp_path: Path, events: list[Event]) -> EventLog:
    p = tmp_path / "events_fb.jsonl"
    with p.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev.to_dict()) + "\n")
    return EventLog(p)


def _edge_weight(g, a, b):
    for x, y, w in g.edges:
        if {x, y} == {a, b}:
            return w
    return 0


def test_feedback_event_applies_path_multipliers(tmp_path):
    from dct.heat import build_concept_graph
    log = _make_log(tmp_path, [
        Event(ts=1.0, source=EventSource.TELEGRAM, op=EventOp.FEEDBACK,
              concepts=["consciousness", "phenomenology"],
              metadata={
                  "thread_id": "92",
                  "useful_concept": "phenomenology",
                  "path": ["consciousness", "memory", "phenomenology"],
                  "multipliers": [4, 5],
              }),
    ])
    g = build_concept_graph(log)
    assert _edge_weight(g, "consciousness", "memory") == 4
    assert _edge_weight(g, "memory", "phenomenology") == 5


def test_feedback_event_combines_with_co_occurrence(tmp_path):
    from dct.heat import build_concept_graph
    log = _make_log(tmp_path, [
        Event(ts=1.0, source=EventSource.TELEGRAM, op=EventOp.WRITE,
              concepts=["a", "b"]),
        Event(ts=2.0, source=EventSource.TELEGRAM, op=EventOp.FEEDBACK,
              concepts=["a", "b"],
              metadata={"thread_id": "92", "useful_concept": "b",
                        "path": ["a", "b"], "multipliers": [4]}),
    ])
    g = build_concept_graph(log)
    assert _edge_weight(g, "a", "b") == 1 + 4


def test_feedback_does_not_increment_node_counts(tmp_path):
    from dct.heat import build_concept_graph
    log = _make_log(tmp_path, [
        Event(ts=1.0, source=EventSource.TELEGRAM, op=EventOp.WRITE,
              concepts=["a", "b"]),
        Event(ts=2.0, source=EventSource.TELEGRAM, op=EventOp.FEEDBACK,
              concepts=["a", "b"],
              metadata={"path": ["a", "b"], "multipliers": [4],
                        "thread_id": "92", "useful_concept": "b"}),
    ])
    g = build_concept_graph(log)
    assert g.nodes.get("a") == 1
    assert g.nodes.get("b") == 1


def test_feedback_malformed_metadata_is_skipped(tmp_path):
    from dct.heat import build_concept_graph
    log = _make_log(tmp_path, [
        Event(ts=1.0, source=EventSource.TELEGRAM, op=EventOp.FEEDBACK,
              concepts=["a", "b"], metadata={}),
    ])
    g = build_concept_graph(log)
    assert _edge_weight(g, "a", "b") == 0


def test_topic_id_filter_applies_only_to_feedback(tmp_path):
    from dct.heat import build_concept_graph
    log = _make_log(tmp_path, [
        Event(ts=1.0, source=EventSource.TELEGRAM, op=EventOp.WRITE,
              concepts=["a", "b"], metadata={"thread_id": "92"}),
        Event(ts=2.0, source=EventSource.TELEGRAM, op=EventOp.WRITE,
              concepts=["a", "b"], metadata={"thread_id": "260"}),
        Event(ts=3.0, source=EventSource.TELEGRAM, op=EventOp.FEEDBACK,
              concepts=["a", "b"],
              metadata={"thread_id": "92", "path": ["a", "b"],
                        "multipliers": [4], "useful_concept": "b"}),
        Event(ts=4.0, source=EventSource.TELEGRAM, op=EventOp.FEEDBACK,
              concepts=["a", "b"],
              metadata={"thread_id": "260", "path": ["a", "b"],
                        "multipliers": [10], "useful_concept": "b"}),
    ])
    g_92 = build_concept_graph(log, topic_id="92")
    assert _edge_weight(g_92, "a", "b") == 2 + 4
    g_260 = build_concept_graph(log, topic_id="260")
    assert _edge_weight(g_260, "a", "b") == 2 + 10
    g_all = build_concept_graph(log, topic_id=None)
    assert _edge_weight(g_all, "a", "b") == 2 + 4 + 10


def test_ignore_feedback_strips_all_feedback_at_read_time(tmp_path):
    from dct.heat import build_concept_graph
    log = _make_log(tmp_path, [
        Event(ts=1.0, source=EventSource.TELEGRAM, op=EventOp.WRITE,
              concepts=["a", "b"]),
        Event(ts=2.0, source=EventSource.TELEGRAM, op=EventOp.FEEDBACK,
              concepts=["a", "b"],
              metadata={"thread_id": "92", "path": ["a", "b"],
                        "multipliers": [10], "useful_concept": "b"}),
    ])
    g_with = build_concept_graph(log, ignore_feedback=False)
    g_without = build_concept_graph(log, ignore_feedback=True)
    assert _edge_weight(g_with, "a", "b") == 1 + 10
    assert _edge_weight(g_without, "a", "b") == 1


def test_ignore_feedback_default_false_for_back_compat(tmp_path):
    from dct.heat import build_concept_graph
    log = _make_log(tmp_path, [
        Event(ts=1.0, source=EventSource.TELEGRAM, op=EventOp.WRITE,
              concepts=["a", "b"]),
    ])
    g = build_concept_graph(log)
    assert _edge_weight(g, "a", "b") == 1


def test_compute_heat_at_skips_feedback_events(tmp_path):
    """R3.10: feedback events must not contribute to heat."""
    from dct.heat import compute_heat_at
    p = tmp_path / "heat_fb.jsonl"
    with p.open("w") as f:
        f.write(json.dumps(Event(ts=1000.0, source=EventSource.TELEGRAM,
                                 op=EventOp.WRITE, concepts=["a", "b"]).to_dict()) + "\n")
        f.write(json.dumps(Event(ts=1010.0, source=EventSource.TELEGRAM,
                                 op=EventOp.FEEDBACK, concepts=["c", "d"],
                                 metadata={"path": ["c", "d"], "multipliers": [4],
                                           "useful_concept": "d", "thread_id": "92"}).to_dict()) + "\n")
    heat = compute_heat_at(p, ts=1020.0, half_life=3600.0)
    assert "a" in heat
    assert "b" in heat
    assert "c" not in heat
    assert "d" not in heat


# ---------------------------------------------------------------------------
# Track C — Directed Transitions (Claim 2b)
# ---------------------------------------------------------------------------

def test_build_concept_graph_returns_transitions(tmp_path):
    """ConceptGraph must have a transitions field populated from traversal events."""
    from dct.heat import build_concept_graph
    from dct.events import EventOp, EventSource

    log = _make_log(tmp_path, [
        Event(ts=1.0, source=EventSource.TELEGRAM, op=EventOp.TRAVERSAL,
              concepts=["memory", "obsidian", "daemon"],
              metadata={"role": "assistant", "chat_id": "1", "thread_id": "t1",
                        "turn_index": "1", "extraction_source": "cascade"}),
    ])
    cg = build_concept_graph(log)
    assert hasattr(cg, "transitions"), "ConceptGraph missing transitions field"
    # memory→obsidian and obsidian→daemon must be recorded
    assert cg.transitions.get(("memory", "obsidian"), 0) >= 1
    assert cg.transitions.get(("obsidian", "daemon"), 0) >= 1
    # Directed: memory→obsidian != obsidian→memory (reverse not present)
    assert cg.transitions.get(("obsidian", "memory"), 0) == 0


def test_directed_transitions_asymmetry(tmp_path):
    """transitions[(a,b)] and transitions[(b,a)] accumulate independently."""
    from dct.heat import build_concept_graph
    from dct.events import EventOp, EventSource

    log = _make_log(tmp_path, [
        # Two traversals a→b, one traversal b→a → (a,b)=2 vs (b,a)=1
        Event(ts=1.0, source=EventSource.TELEGRAM, op=EventOp.TRAVERSAL,
              concepts=["alpha", "beta"],
              metadata={"role": "assistant", "chat_id": "1", "thread_id": "t",
                        "turn_index": "1", "extraction_source": "cascade"}),
        Event(ts=2.0, source=EventSource.TELEGRAM, op=EventOp.TRAVERSAL,
              concepts=["alpha", "beta"],
              metadata={"role": "assistant", "chat_id": "1", "thread_id": "t",
                        "turn_index": "2", "extraction_source": "cascade"}),
        Event(ts=3.0, source=EventSource.TELEGRAM, op=EventOp.TRAVERSAL,
              concepts=["beta", "alpha"],
              metadata={"role": "assistant", "chat_id": "1", "thread_id": "t",
                        "turn_index": "3", "extraction_source": "cascade"}),
    ])
    cg = build_concept_graph(log)
    assert cg.transitions[("alpha", "beta")] == 2
    assert cg.transitions[("beta", "alpha")] == 1
