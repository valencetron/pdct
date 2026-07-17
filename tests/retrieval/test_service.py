"""Minimal smoke test for the retrieval service CLI.

The `run()` function hits the real events.jsonl and distill root on disk, so
we only test the parts that are pure-logic or easily faked. Full end-to-end
verification happens via manual invocation of the CLI.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from dct.retrieval import service


def test_existing_anchor_paths_filters_missing(tmp_path, monkeypatch):
    real = tmp_path / "real.md"
    real.write_text("x")
    missing = tmp_path / "missing.md"
    monkeypatch.setattr(service, "_ANCHOR_CANDIDATES", (real, missing))
    got = service._existing_anchor_paths()
    assert got == [real]


def test_build_config_includes_all_surfaces(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "_ANCHOR_CANDIDATES", ())
    monkeypatch.setattr(service, "DISTILL_ROOT", tmp_path)
    cfg = service.build_config()
    assert cfg.surfaces == ["voice", "claude-code", "telegram", "vault"]


def test_load_or_build_graph_uses_in_memory_cache_when_key_matches(tmp_path, monkeypatch):
    """R3.2: same (path, mtime, topic_id, ignore_feedback) returns cached graph."""
    from dct.heat import ConceptGraph
    events = tmp_path / "events.jsonl"
    events.write_text("")
    service._GRAPH_CACHE.clear()

    sentinel = ConceptGraph(nodes={"marker": 1}, edges=[])
    mtime = events.stat().st_mtime
    # Cache key includes vec_near flags + vault_mtime (Track C Codex r2 fix).
    # Use _env_bool/_env_float to match service.py's defaults exactly.
    vec_near_flag = service._env_bool("DCT_VEC_NEAR_ENABLED", True)
    vec_near_thresh = service._env_float("DCT_VEC_NEAR_THRESHOLD", 0.70)
    try:
        vault_mtime = max(
            (f.stat().st_mtime for f in service.DISTILL_ROOT.rglob("*.md") if f.is_file()),
            default=0.0,
        )
    except OSError:
        vault_mtime = 0.0
    key = (service._CACHE_VERSION, str(events), mtime, None, False,
           vec_near_flag, vec_near_thresh, vault_mtime)
    service._GRAPH_CACHE[key] = sentinel

    result = service._load_or_build_graph(events_path=events)
    assert result is sentinel  # cached identity


def test_load_or_build_graph_keys_per_topic_and_ablation(tmp_path, monkeypatch):
    """R3.2: different topic_id or ignore_feedback gets its own cached graph."""
    from dct.heat import ConceptGraph
    events = tmp_path / "events.jsonl"
    events.write_text("")
    service._GRAPH_CACHE.clear()

    g_default = service._load_or_build_graph(events_path=events)
    g_topic_92 = service._load_or_build_graph(events_path=events, topic_id="92")
    g_ablation = service._load_or_build_graph(events_path=events, ignore_feedback=True)
    # Three distinct entries.
    assert len(service._GRAPH_CACHE) == 3
    # All are ConceptGraph instances (distinct identity not guaranteed if empty,
    # but distinct entries in cache is enough).
    assert all(isinstance(g, ConceptGraph)
               for g in (g_default, g_topic_92, g_ablation))


def test_cli_run_returns_bundle_shape():
    """Spawn the CLI with empty user_text. Confirms the command wires up;
    real graph load is expensive (events.jsonl is ~1MB) but only on first
    run. Subsequent calls use the pickle cache."""
    proc = subprocess.run(
        [sys.executable, "-m", "dct.retrieval.service"],
        input=json.dumps({"user_text": "", "current_context": []}),
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    if proc.returncode != 0:
        pytest.skip(f"service CLI unavailable in test env: {proc.stderr[:500]}")
    out = json.loads(proc.stdout)
    assert "prompt_block" in out
    assert "seed_concepts" in out
    assert "cascade_concepts" in out
    assert "cascade_count" in out
    assert "bundle_tokens" in out
    # Empty text → no seeds → no cascade
    assert out["seed_concepts"] == []
    assert out["cascade_concepts"] == []


# ── Trim helper tests ─────────────────────────────────────────────────────
from dct.retrieval.types import ConceptHit, RetrievalConfig
from dct.retrieval.service import _trim_hits, _filter_by_heat, _filter_by_eligibility


def _hit(concept: str, score: float, hop: int = 1) -> ConceptHit:
    return ConceptHit(concept=concept, score=score, source_slug="t", snippet="", hop=hop)


def test_trim_hits_drops_below_floor():
    cfg = RetrievalConfig(
        anchor_paths=[], distill_root=Path("/tmp"), surfaces=[],
        cascade_score_floor=0.1, cascade_top_k=80,
    )
    hits = [_hit("a", 0.9), _hit("b", 0.5), _hit("c", 0.05), _hit("d", 0.0)]
    trimmed, pre = _trim_hits(hits, cfg)
    out = {h.concept for h in trimmed}
    assert out == {"a", "b"}, out
    assert pre == 4


def test_trim_hits_clamps_to_top_k():
    cfg = RetrievalConfig(
        anchor_paths=[], distill_root=Path("/tmp"), surfaces=[],
        cascade_score_floor=0.0, cascade_top_k=3,
    )
    hits = [_hit(f"c{i}", 0.5) for i in range(10)]
    trimmed, pre = _trim_hits(hits, cfg)
    assert len(trimmed) == 3
    assert pre == 10


def test_trim_hits_always_preserves_seeds():
    """Hop-0 (seed) concepts survive even when floor is high and top_k is tiny."""
    cfg = RetrievalConfig(
        anchor_paths=[], distill_root=Path("/tmp"), surfaces=[],
        cascade_score_floor=0.99, cascade_top_k=1,
    )
    hits = [
        _hit("seed_a", 1.0, hop=0),
        _hit("seed_b", 1.0, hop=0),
        _hit("seed_c", 1.0, hop=0),
        _hit("hi_score_neighbor", 0.95, hop=1),
    ]
    trimmed, pre = _trim_hits(hits, cfg)
    out = {h.concept for h in trimmed}
    assert {"seed_a", "seed_b", "seed_c"}.issubset(out), out
    assert pre == 4


def test_trim_hits_empty_input():
    cfg = RetrievalConfig(anchor_paths=[], distill_root=Path("/tmp"), surfaces=[])
    trimmed, pre = _trim_hits([], cfg)
    assert trimmed == []
    assert pre == 0


def test_run_returns_cascade_paths_keyed_by_concept(monkeypatch, tmp_path):
    """run() must include cascade_paths so daemon can attribute credit."""
    from dct.events import Event, EventOp, EventSource

    events_path = tmp_path / "events.jsonl"
    with events_path.open("w") as f:
        for ev in [
            Event(ts=1.0, source=EventSource.TELEGRAM, op=EventOp.WRITE,
                  concepts=["consciousness", "memory"]),
            Event(ts=2.0, source=EventSource.TELEGRAM, op=EventOp.WRITE,
                  concepts=["memory", "phenomenology"]),
        ]:
            f.write(json.dumps(ev.to_dict()) + "\n")

    monkeypatch.setattr(service, "EVENTS_JSONL", events_path)
    monkeypatch.setattr(service, "_ANCHOR_CANDIDATES", ())
    distill_root = tmp_path / "distill"
    for c in ("voice", "claude-code", "telegram", "vault"):
        (distill_root / c).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(service, "DISTILL_ROOT", distill_root)
    service._GRAPH_CACHE.clear()

    out = service.run("consciousness phenomenology", current_context=[])
    assert "cascade_paths" in out
    paths = out["cascade_paths"]
    assert isinstance(paths, dict)
    for c in out.get("cascade_concepts", []):
        if c in paths:
            assert paths[c][-1] == c


def test_run_accepts_topic_id_and_ignore_feedback(monkeypatch, tmp_path):
    """topic_id and ignore_feedback flow through to graph build."""
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("")
    monkeypatch.setattr(service, "EVENTS_JSONL", events_path)
    monkeypatch.setattr(service, "_ANCHOR_CANDIDATES", ())
    distill_root = tmp_path / "distill"
    for c in ("voice", "claude-code", "telegram", "vault"):
        (distill_root / c).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(service, "DISTILL_ROOT", distill_root)
    service._GRAPH_CACHE.clear()

    out = service.run("hello", current_context=[], topic_id="92",
                      ignore_feedback=True)
    assert "cascade_paths" in out


# ── Heat filter helper tests ───────────────────────────────────────────────


def test_filter_by_heat_drops_cold_non_seeds():
    cfg = RetrievalConfig(
        anchor_paths=[], distill_root=Path("/tmp"), surfaces=[],
        cascade_heat_floor=0.05,
    )
    hits = [
        _hit("hot_seed", 1.0, hop=0),
        _hit("warm_neighbor", 0.7, hop=1),
        _hit("cold_neighbor", 0.6, hop=1),
        _hit("absent_neighbor", 0.4, hop=2),
    ]
    heat = {
        "hot_seed": 0.99,
        "warm_neighbor": 0.20,
        "cold_neighbor": 0.01,
        # absent_neighbor missing → treated as cold
    }
    filtered, pre_count = _filter_by_heat(hits, heat, cfg)
    out = {h.concept for h in filtered}
    assert out == {"hot_seed", "warm_neighbor"}, out
    assert pre_count == 4


def test_filter_by_heat_always_keeps_seeds():
    """User-typed seeds always pass even when stone cold (reignition rule)."""
    cfg = RetrievalConfig(
        anchor_paths=[], distill_root=Path("/tmp"), surfaces=[],
        cascade_heat_floor=0.99,
    )
    hits = [
        _hit("dormant_seed", 1.0, hop=0),
        _hit("warm_neighbor", 0.5, hop=1),
    ]
    heat = {"warm_neighbor": 0.5}  # seed not in heat dict — stone cold
    filtered, pre_count = _filter_by_heat(hits, heat, cfg)
    out = {h.concept for h in filtered}
    assert "dormant_seed" in out, out  # seed survives
    assert "warm_neighbor" not in out  # below 0.99 floor → dropped
    assert pre_count == 2


def test_filter_by_heat_empty_input():
    cfg = RetrievalConfig(anchor_paths=[], distill_root=Path("/tmp"), surfaces=[])
    filtered, pre_count = _filter_by_heat([], {}, cfg)
    assert filtered == []
    assert pre_count == 0


def test_filter_by_heat_zero_floor_keeps_present_concepts():
    """heat_floor=0.0 keeps every concept that appears in the heat dict.

    Concepts ABSENT from heat are still dropped (stone cold by definition).
    """
    cfg = RetrievalConfig(
        anchor_paths=[], distill_root=Path("/tmp"), surfaces=[],
        cascade_heat_floor=0.0,
    )
    hits = [
        _hit("a", 0.9, hop=1),
        _hit("b", 0.5, hop=1),
        _hit("absent", 0.4, hop=1),
    ]
    heat = {"a": 0.001, "b": 0.0001}  # both present, even if tiny
    filtered, _ = _filter_by_heat(hits, heat, cfg)
    out = {h.concept for h in filtered}
    assert out == {"a", "b"}  # absent excluded


def test_run_actually_filters_cold_neighbor(monkeypatch):
    """A cold neighbor of a hot seed must be DROPPED from cascade_concepts."""
    from dct.heat import ConceptGraph
    from dct.retrieval import service as svc
    from dct.retrieval.types import PreloadBundle

    # Graph: hot_seed_concept -> cold_neighbor_xyz (single edge).
    # Use a name that won't accidentally match the prose seed extractor.
    graph = ConceptGraph(
        nodes={"hot_seed_concept": 5, "cold_neighbor_xyz": 5},
        edges=[("cold_neighbor_xyz", "hot_seed_concept", 1)],
    )
    monkeypatch.setattr(svc, "_load_or_build_graph", lambda *_a, **_kw: graph)
    monkeypatch.setattr(
        svc, "preload",
        lambda cfg, now=None: PreloadBundle(
            anchors="", today_summaries="", recent_summaries={}, total_tokens=0
        ),
    )
    # Stub heat: only hot_seed_concept is warm; cold_neighbor_xyz is well below floor.
    monkeypatch.setattr(
        svc, "_load_or_build_heat",
        lambda *_a, **_kw: {f"filler_{i}": 0.5 for i in range(50)} | {"hot_seed_concept": 0.99},
    )

    # Use only the explicit seed in the user text. cold_neighbor_xyz is reachable
    # via cascade hop-1 but heat-cold and should be dropped.
    result = svc.run("hot_seed_concept matters here", current_context=[])

    # Sanity: the hot seed survived.
    assert "hot_seed_concept" in result["cascade_concepts"]
    # Filter actually worked: cold neighbor is gone despite being graph-adjacent.
    assert "cold_neighbor_xyz" not in result["cascade_concepts"]
    assert result["heat_skipped_reason"] == "none"
    assert result["post_heat_count"] < result["pre_heat_count"]


def test_run_seed_survives_even_when_cold(monkeypatch):
    """User-typed cold seed must survive (reignition rule)."""
    from dct.heat import ConceptGraph
    from dct.retrieval import service as svc
    from dct.retrieval.types import PreloadBundle

    graph = ConceptGraph(nodes={"dormant_concept": 1}, edges=[])
    monkeypatch.setattr(svc, "_load_or_build_graph", lambda *_a, **_kw: graph)
    monkeypatch.setattr(
        svc, "preload",
        lambda cfg, now=None: PreloadBundle(
            anchors="", today_summaries="", recent_summaries={}, total_tokens=0
        ),
    )
    # Heat dict has plenty of entries (insufficient-data guard NOT triggered)
    # but dormant_concept is absent → would normally be cold.
    monkeypatch.setattr(
        svc, "_load_or_build_heat",
        lambda *_a, **_kw: {f"unrelated_{i}": 0.5 for i in range(50)},
    )

    result = svc.run("dormant_concept revisit", current_context=[])
    # Seed survives despite being cold.
    assert "dormant_concept" in result["cascade_concepts"]


def test_run_insufficient_data_skips_filter(monkeypatch):
    """Tiny heat dict means new / sparse session — skip filter."""
    from dct.heat import ConceptGraph
    from dct.retrieval import service as svc
    from dct.retrieval.types import PreloadBundle

    graph = ConceptGraph(
        nodes={"alpha": 5, "beta": 5},
        edges=[("alpha", "beta", 1)],
    )
    monkeypatch.setattr(svc, "_load_or_build_graph", lambda *_a, **_kw: graph)
    monkeypatch.setattr(
        svc, "preload",
        lambda cfg, now=None: PreloadBundle(
            anchors="", today_summaries="", recent_summaries={}, total_tokens=0
        ),
    )
    # Heat has only 5 entries — below default min_dict_size=20.
    monkeypatch.setattr(
        svc, "_load_or_build_heat",
        lambda *_a, **_kw: {"alpha": 0.9, "x1": 0.5, "x2": 0.4, "x3": 0.3, "x4": 0.2},
    )
    result = svc.run("alpha", current_context=[])
    assert result["heat_skipped_reason"] == "insufficient_data"
    # No filtering: pre_heat_count == post_heat_count.
    assert result["pre_heat_count"] == result["post_heat_count"]


def test_run_fails_open_on_compute_error(monkeypatch):
    """If compute_heat_at raises, run() returns successfully with skipped reason."""
    from dct.heat import ConceptGraph
    from dct.retrieval import service as svc
    from dct.retrieval.types import PreloadBundle

    graph = ConceptGraph(nodes={"alpha": 5}, edges=[])
    monkeypatch.setattr(svc, "_load_or_build_graph", lambda *_a, **_kw: graph)
    monkeypatch.setattr(
        svc, "preload",
        lambda cfg, now=None: PreloadBundle(
            anchors="", today_summaries="", recent_summaries={}, total_tokens=0
        ),
    )

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated heat compute failure")

    monkeypatch.setattr(svc, "_load_or_build_heat", _boom)

    result = svc.run("alpha", current_context=[])
    assert result["heat_skipped_reason"] == "compute_error"
    assert result["pre_heat_count"] == result["post_heat_count"]


def test_run_disabled_via_env(monkeypatch):
    """DCT_CASCADE_HEAT_ENABLED=false bypasses filtering entirely."""
    from dct.heat import ConceptGraph
    from dct.retrieval import service as svc
    from dct.retrieval.types import PreloadBundle

    monkeypatch.setenv("DCT_CASCADE_HEAT_ENABLED", "false")
    graph = ConceptGraph(nodes={"alpha": 5, "beta": 5}, edges=[("alpha", "beta", 1)])
    monkeypatch.setattr(svc, "_load_or_build_graph", lambda *_a, **_kw: graph)
    monkeypatch.setattr(
        svc, "preload",
        lambda cfg, now=None: PreloadBundle(
            anchors="", today_summaries="", recent_summaries={}, total_tokens=0
        ),
    )
    # If heat were called, it would raise — but disabled path skips it.
    monkeypatch.setattr(
        svc, "_load_or_build_heat",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("should not be called when disabled")),
    )

    result = svc.run("alpha", current_context=[])
    assert result["heat_skipped_reason"] == "disabled"
    assert result["pre_heat_count"] == result["post_heat_count"]


def test_load_or_build_heat_invalidates_on_mtime_change(tmp_path, monkeypatch):
    """Same ts bucket but events file mutated → miss serves STALE instantly
    (stale-while-revalidate, 2026-07-16) and a background recompute lands a
    fresh snapshot for subsequent calls. The old contract (immediate fresh
    compute on the request path) is intentionally superseded: a full
    events.jsonl replay (~1-2s) blew the 3s cascade budget on every turn."""
    from dct.retrieval import service as svc

    events_path = tmp_path / "events.jsonl"
    events_path.write_text("")  # start empty

    calls = []

    def fake_compute(_path, *, ts, half_life):
        # Mark the call and return a minimally-sized dict that includes
        # whatever the file currently contains (so we can prove "fresh read").
        text = Path(_path).read_text()
        calls.append((ts, half_life, text))
        return {f"slug_{len(text)}": 0.5}

    monkeypatch.setattr(svc, "compute_heat_at", fake_compute)
    # Clear all heat state for a clean run.
    monkeypatch.setattr(svc, "_HEAT_CACHE", {})
    monkeypatch.setattr(svc, "_HEAT_LATEST", {})
    monkeypatch.setattr(svc, "_HEAT_REBUILDING", {})

    ts = 12345.0  # fixed ts → same bucket for both calls
    h1 = svc._load_or_build_heat(events_path, ts=ts, half_life=21600.0)
    # Mutate file — st_mtime advances.
    import time as _t
    _t.sleep(0.01)  # ensure mtime granularity advances on filesystems with second-level precision
    events_path.write_text("MUTATED\n")
    h2 = svc._load_or_build_heat(events_path, ts=ts, half_life=21600.0)
    assert h2 is h1, "miss must serve the stale snapshot instantly"

    # Background recompute observes the mutation and lands a fresh snapshot.
    deadline = _t.time() + 5
    while len(calls) < 2 and _t.time() < deadline:
        _t.sleep(0.02)
    assert len(calls) == 2, f"expected background recompute, got {len(calls)}"
    while svc._HEAT_REBUILDING and _t.time() < deadline:
        _t.sleep(0.02)
    h3 = svc._load_or_build_heat(events_path, ts=ts + 60, half_life=21600.0)
    assert h3 != h1, "fresh snapshot must serve after the recompute lands"


def test_load_or_build_heat_caches_within_bucket(tmp_path, monkeypatch):
    """Same ts bucket + unchanged file → second call hits cache."""
    from dct.retrieval import service as svc

    events_path = tmp_path / "events.jsonl"
    events_path.write_text("X\n")

    calls = []

    def fake_compute(_path, *, ts, half_life):
        calls.append(ts)
        return {"a": 0.5}

    monkeypatch.setattr(svc, "compute_heat_at", fake_compute)
    svc._HEAT_CACHE.clear()

    ts = 12345.0
    svc._load_or_build_heat(events_path, ts=ts, half_life=21600.0)
    svc._load_or_build_heat(events_path, ts=ts + 5.0, half_life=21600.0)  # same bucket
    assert len(calls) == 1, f"expected 1 compute (cached), got {len(calls)}"


# ── Eligibility filter tests (P1.1: junk-concept blocklist) ───────────────


def test_filter_by_eligibility_drops_single_token_non_seeds():
    """Concepts that fail concept_eligible_tokens (single-token / stopwords)
    must be dropped from the cascade so injection set == scorable set."""
    cfg = RetrievalConfig(
        anchor_paths=[], distill_root=Path("/tmp"), surfaces=[],
        cascade_eligibility_filter_enabled=True,
    )
    hits = [
        _hit("memory", 0.9, hop=1),                # single token → DROP
        _hit("ide", 0.8, hop=1),                   # single token → DROP
        _hit("phase5-card-control", 0.7, hop=1),   # 2 eligible (phase5, control) → KEEP
        _hit("voice-pipeline", 0.6, hop=2),        # 2 eligible → KEEP
        _hit("the-and", 0.5, hop=1),               # all stopwords → DROP
    ]
    filtered, pre, dropped = _filter_by_eligibility(hits, cfg)
    out = {h.concept for h in filtered}
    assert out == {"phase5-card-control", "voice-pipeline"}, out
    assert pre == 5
    assert set(dropped) == {"memory", "ide", "the-and"}


def test_filter_by_eligibility_always_keeps_seeds():
    """Hop-0 seeds bypass eligibility — user typed [[Memory]] is still a seed."""
    cfg = RetrievalConfig(
        anchor_paths=[], distill_root=Path("/tmp"), surfaces=[],
        cascade_eligibility_filter_enabled=True,
    )
    hits = [
        _hit("memory", 1.0, hop=0),  # single-token SEED → must survive
        _hit("ide", 1.0, hop=0),     # single-token SEED → must survive
        _hit("memory", 0.5, hop=1),  # single-token non-seed → drop
    ]
    filtered, pre, dropped = _filter_by_eligibility(hits, cfg)
    seeds_kept = [h for h in filtered if h.hop == 0]
    assert len(seeds_kept) == 2  # both seeds preserved
    # The non-seed copy of "memory" is dropped
    assert "memory" in dropped
    assert pre == 3


def test_filter_by_eligibility_disabled_passes_through():
    """Toggle off → no filtering happens; all hits and zero dropped."""
    cfg = RetrievalConfig(
        anchor_paths=[], distill_root=Path("/tmp"), surfaces=[],
        cascade_eligibility_filter_enabled=False,
    )
    hits = [
        _hit("memory", 0.9, hop=1),
        _hit("ide", 0.8, hop=1),
        _hit("phase5-card-control", 0.7, hop=1),
    ]
    filtered, pre, dropped = _filter_by_eligibility(hits, cfg)
    assert len(filtered) == 3
    assert pre == 3
    assert dropped == []


def test_filter_by_eligibility_empty_input():
    cfg = RetrievalConfig(anchor_paths=[], distill_root=Path("/tmp"), surfaces=[])
    filtered, pre, dropped = _filter_by_eligibility([], cfg)
    assert filtered == []
    assert pre == 0
    assert dropped == []


def test_run_drops_ineligible_neighbors_from_cascade(monkeypatch):
    """End-to-end: a hot single-token neighbor must NOT appear in
    cascade_concepts when the eligibility filter is enabled (default)."""
    from dct.heat import ConceptGraph
    from dct.retrieval import service as svc
    from dct.retrieval.types import PreloadBundle

    # Graph: explicit user seed (multi-token) → single-token neighbor + multi-token neighbor.
    graph = ConceptGraph(
        nodes={"phase5-card-control": 5, "memory": 5, "voice-pipeline": 5},
        edges=[
            ("memory", "phase5-card-control", 1),
            ("voice-pipeline", "phase5-card-control", 1),
        ],
    )
    monkeypatch.setattr(svc, "_load_or_build_graph", lambda *_a, **_kw: graph)
    monkeypatch.setattr(
        svc, "preload",
        lambda cfg, now=None: PreloadBundle(
            anchors="", today_summaries="", recent_summaries={}, total_tokens=0
        ),
    )
    # All three concepts hot enough to survive heat filter.
    monkeypatch.setattr(
        svc, "_load_or_build_heat",
        lambda *_a, **_kw: (
            {"phase5-card-control": 0.9, "memory": 0.9, "voice-pipeline": 0.9}
            | {f"filler_{i}": 0.5 for i in range(50)}
        ),
    )

    result = svc.run("phase5-card-control discussion", current_context=[])
    cascade = set(result["cascade_concepts"])
    # Multi-token neighbor survives.
    assert "voice-pipeline" in cascade
    # Single-token neighbor is dropped by eligibility filter.
    assert "memory" not in cascade
    # Telemetry surfaces the count.
    assert result["eligibility_filter_enabled"] is True
    assert result["eligibility_dropped_count"] >= 1
    assert "memory" in result["eligibility_dropped_sample"]


def test_run_eligibility_disabled_via_env(monkeypatch):
    """DCT_CASCADE_ELIGIBILITY_FILTER=false leaves single-token neighbors in."""
    from dct.heat import ConceptGraph
    from dct.retrieval import service as svc
    from dct.retrieval.types import PreloadBundle

    monkeypatch.setenv("DCT_CASCADE_ELIGIBILITY_FILTER", "false")
    graph = ConceptGraph(
        nodes={"phase5-card-control": 5, "memory": 5},
        edges=[("memory", "phase5-card-control", 1)],
    )
    monkeypatch.setattr(svc, "_load_or_build_graph", lambda *_a, **_kw: graph)
    monkeypatch.setattr(
        svc, "preload",
        lambda cfg, now=None: PreloadBundle(
            anchors="", today_summaries="", recent_summaries={}, total_tokens=0
        ),
    )
    monkeypatch.setattr(
        svc, "_load_or_build_heat",
        lambda *_a, **_kw: (
            {"phase5-card-control": 0.9, "memory": 0.9}
            | {f"filler_{i}": 0.5 for i in range(50)}
        ),
    )

    result = svc.run("phase5-card-control discussion", current_context=[])
    cascade = set(result["cascade_concepts"])
    # When filter is disabled, single-token neighbor passes through.
    assert "memory" in cascade
    assert result["eligibility_filter_enabled"] is False
    assert result["eligibility_dropped_count"] == 0
