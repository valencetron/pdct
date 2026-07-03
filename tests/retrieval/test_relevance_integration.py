"""Integration smoke: service.run() honors relevance policy end-to-end."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dct.retrieval import service
from dct.retrieval.relevance import _RULES_CACHE


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch):
    _RULES_CACHE.clear()
    service._GRAPH_CACHE.clear()
    service._HEAT_CACHE.clear()
    yield
    _RULES_CACHE.clear()
    service._GRAPH_CACHE.clear()
    service._HEAT_CACHE.clear()


def _seed_events(events_path: Path) -> None:
    """Tiny synthetic event log with two strongly co-occurring families:
    'family-time' / 'weekend' and 'exampleco-labs-buildout' / 'buildout-power'
    / 'buildout-permits'. Cross-edge: one event mentions both, so cascade
    from 'family-time' reaches buildout neighbors."""
    def _ev(ts, concepts):
        # Use op=write (READ/WRITE drive the co-occurrence graph; TURN/TRAVERSAL
        # are skipped by build_concept_graph).
        return json.dumps({
            "ts": ts,
            "source": "telegram",
            "op": "write",
            "concepts": concepts,
            "metadata": {"topic_id": "t1"},
        })
    events_path.write_text("\n".join([
        _ev(1000.0, ["exampleco-labs-buildout", "buildout-power", "family-time", "weekend"]),
        _ev(1100.0, ["exampleco-labs-buildout", "buildout-permits"]),
        _ev(1200.0, ["family-time", "weekend"]),
        _ev(1300.0, ["family-time", "buildout-power", "weekend"]),
    ]) + "\n")


def test_service_run_no_snapshot_no_filter(tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    _seed_events(events)
    monkeypatch.setattr(service, "EVENTS_JSONL", events)
    monkeypatch.setenv("DCT_RELEVANCE_ENABLED", "0")

    result = service.run("[[exampleco-labs-buildout]]", current_context=[])
    assert result["relevance_rule_id"] == ""
    assert result["relevance_dropped_count"] == 0
    assert result["cascade_top_k_effective"] == result["top_k"]
    assert result["cascade_score_floor_effective"] == result["score_floor"]


def test_service_run_dry_run_logs_but_does_not_filter(tmp_path, monkeypatch):
    """Dry-run contract: rule_id stamped, dropped_count > 0 (proves filter
    SAW concepts to drop), but cascade still contains those concepts
    (proves filter did NOT apply). Codex r2 P2 #4 — without the
    dropped_count assertion, a broken implementation reporting zero
    would pass."""
    events = tmp_path / "events.jsonl"
    _seed_events(events)
    monkeypatch.setattr(service, "EVENTS_JSONL", events)
    monkeypatch.setenv("DCT_CASCADE_HEAT_ENABLED", "0")

    # Phase 1: baseline (filter off) — establish that buildout-* concepts
    # exist in the cascade for [[family-time]] seed.
    monkeypatch.setenv("DCT_RELEVANCE_ENABLED", "0")
    baseline = service.run(
        "[[family-time]]",
        current_context=[],
        now_snapshot={"cell_key": "sun.morning", "workday_status": "Weekend"},
        surface="telegram",
    )
    buildout_in_baseline = [c for c in baseline["cascade_concepts"] if c.startswith("buildout-")]
    assert buildout_in_baseline, (
        "fixture must produce buildout-* concepts for this dry-run test"
    )

    # Phase 2: dry-run on. Same seed; cascade UNCHANGED but telemetry
    # reports rule_id + nonzero dropped_count.
    service._GRAPH_CACHE.clear()
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "rules": [{
            "id": "weekend-personal",
            "match": {"day_of_week": ["sun"]},
            "policy": {"denied_concept_prefixes": ["buildout-"]},
        }],
    }))
    monkeypatch.setenv("DCT_RELEVANCE_ENABLED", "1")
    monkeypatch.setenv("DCT_RELEVANCE_DRY_RUN", "1")
    monkeypatch.setenv("DCT_RELEVANCE_RULES_PATH", str(rules_path))

    result = service.run(
        "[[family-time]]",
        current_context=[],
        now_snapshot={"cell_key": "sun.morning", "workday_status": "Weekend"},
        surface="telegram",
    )
    assert result["relevance_rule_id"] == "weekend-personal"
    # CRITICAL: dry-run must report would-drop count > 0 — this is the
    # contract that lets us observe rules in launchd before flipping live.
    assert result["relevance_dropped_count"] >= len(buildout_in_baseline), (
        f"dry-run reported drop count {result['relevance_dropped_count']} but "
        f"baseline had {len(buildout_in_baseline)} denied concepts; filter "
        f"is silently no-op"
    )
    # And cascade output is UNCHANGED — buildout-* survive in dry-run.
    buildout_in_result = [c for c in result["cascade_concepts"] if c.startswith("buildout-")]
    assert buildout_in_result, (
        "dry-run must NOT actually drop concepts; buildout-* should still "
        "appear in cascade_concepts (only telemetry reports the would-drop)"
    )


def test_service_run_live_mode_drops_denied_prefixes(tmp_path, monkeypatch):
    """Two-phase: prove fixture produces buildout-* concepts; then enable
    filter and assert they're gone."""
    events = tmp_path / "events.jsonl"
    _seed_events(events)
    monkeypatch.setattr(service, "EVENTS_JSONL", events)

    # Phase 1: filter disabled. Disable heat filter too — synthetic events
    # have ts ~1300 (epoch), which is decades cold. Heat would drop them all.
    monkeypatch.setenv("DCT_RELEVANCE_ENABLED", "0")
    monkeypatch.setenv("DCT_CASCADE_HEAT_ENABLED", "0")
    baseline = service.run(
        "[[family-time]]",
        current_context=[],
        now_snapshot={"cell_key": "sun.morning", "workday_status": "Weekend"},
        surface="telegram",
    )
    buildout_in_baseline = [c for c in baseline["cascade_concepts"] if c.startswith("buildout-")]
    assert buildout_in_baseline, (
        "fixture must produce at least one buildout-* concept; otherwise "
        "the live-filter test passes vacuously"
    )

    # Phase 2: filter enabled.
    service._GRAPH_CACHE.clear()
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "rules": [{
            "id": "weekend-personal",
            "match": {"day_of_week": ["sun"]},
            "policy": {"denied_concept_prefixes": ["buildout-"]},
        }],
    }))
    monkeypatch.setenv("DCT_RELEVANCE_ENABLED", "1")
    monkeypatch.setenv("DCT_RELEVANCE_DRY_RUN", "0")
    monkeypatch.setenv("DCT_RELEVANCE_RULES_PATH", str(rules_path))

    result = service.run(
        "[[family-time]]",
        current_context=[],
        now_snapshot={"cell_key": "sun.morning", "workday_status": "Weekend"},
        surface="telegram",
    )
    assert result["relevance_rule_id"] == "weekend-personal"
    for c in result["cascade_concepts"]:
        assert not c.startswith("buildout-"), f"denied concept survived: {c}"
    assert result["relevance_dropped_count"] >= len(buildout_in_baseline)


def test_service_run_disabled_short_circuits(tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    _seed_events(events)
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "rules": [{"id": "always", "match": {}, "policy": {"denied_concept_prefixes": ["family"]}}],
    }))
    monkeypatch.setattr(service, "EVENTS_JSONL", events)
    monkeypatch.setenv("DCT_RELEVANCE_ENABLED", "0")
    monkeypatch.setenv("DCT_RELEVANCE_RULES_PATH", str(rules_path))

    result = service.run(
        "[[family-time]]",
        current_context=[],
        now_snapshot={"cell_key": "sun.morning"},
        surface="telegram",
    )
    assert result["relevance_rule_id"] == ""
    assert result["relevance_dropped_count"] == 0
