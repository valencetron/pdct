from pathlib import Path
import pytest
from dct.retrieval.types import ConceptHit, PreloadBundle, RetrievalConfig


def test_concept_hit_is_frozen():
    hit = ConceptHit(concept="consciousness", score=1.0, source_slug="seed", snippet="", hop=0)
    with pytest.raises((AttributeError, Exception)):
        hit.score = 2.0  # type: ignore[misc]


def test_concept_hit_fields():
    hit = ConceptHit(concept="c", score=0.3, source_slug="heat", snippet="s", hop=2)
    assert hit.concept == "c"
    assert hit.score == 0.3
    assert hit.source_slug == "heat"
    assert hit.snippet == "s"
    assert hit.hop == 2


def test_preload_bundle_fields(tmp_path):
    bundle = PreloadBundle(
        anchors="anchor text",
        today_summaries="today",
        recent_summaries={"voice": "vr", "telegram": "tr"},
        total_tokens=42,
    )
    assert bundle.anchors == "anchor text"
    assert bundle.today_summaries == "today"
    assert bundle.recent_summaries["voice"] == "vr"
    assert bundle.total_tokens == 42


def test_retrieval_config_defaults(tmp_path):
    cfg = RetrievalConfig(
        anchor_paths=[tmp_path / "CLAUDE.md"],
        distill_root=tmp_path / "distill",
        surfaces=["voice", "claude-code", "telegram", "vault"],
    )
    assert cfg.cascade_depth == 2
    assert cfg.cascade_decay == 0.4
    assert cfg.cascade_budget_ms == 800
    assert cfg.cascade_token_cap == 10_000
    assert cfg.cascade_score_floor == 0.05
    assert cfg.cascade_top_k == 80
    # 5_000 -> 32_000 (2026-07-27 SOUL.md truncation incident, commit 2769ee8).
    # pdct-public keeps 5_000 — this assertion intentionally diverges.
    assert cfg.preload_anchor_cap == 32_000
    assert cfg.anchor_structure_aware is True
    assert cfg.anchor_tier_order == ("inviolable", "core", "journal")
    assert cfg.preload_today_cap == 5_000
    assert cfg.preload_surface_cap == 5_000
    assert cfg.preload_last_n == 10  # bumped from 3 (2026-05-27, distill loader fix)


def test_concept_hit_default_path_is_empty_list():
    h = ConceptHit(concept="x", score=1.0, source_slug="seed", snippet="", hop=0)
    assert h.path == []
    assert isinstance(h.path, list)


def test_concept_hit_explicit_path_preserved():
    h = ConceptHit(
        concept="phenomenology",
        score=0.42,
        source_slug="hop-2",
        snippet="",
        hop=2,
        path=["consciousness", "memory", "phenomenology"],
    )
    assert h.path == ["consciousness", "memory", "phenomenology"]
    assert h.path[-1] == h.concept


def test_concept_hit_path_invariant_when_nonempty():
    """When path is set, last element must equal concept (cascade contract)."""
    h = ConceptHit(
        concept="b", score=1.0, source_slug="hop-1", snippet="", hop=1,
        path=["a", "b"],
    )
    if h.path:
        assert h.path[-1] == h.concept


def test_concept_hit_two_instances_with_same_path_are_distinct_lists():
    """Default factory must produce a fresh list per instance, not a shared one."""
    h1 = ConceptHit(concept="a", score=1.0, source_slug="seed", snippet="", hop=0)
    h2 = ConceptHit(concept="b", score=1.0, source_slug="seed", snippet="", hop=0)
    assert h1.path is not h2.path


def test_default_cascade_heat_enabled():
    cfg = RetrievalConfig(anchor_paths=[], distill_root=Path("/tmp"), surfaces=[])
    assert cfg.cascade_heat_enabled is True


def test_default_cascade_heat_floor():
    cfg = RetrievalConfig(anchor_paths=[], distill_root=Path("/tmp"), surfaces=[])
    assert cfg.cascade_heat_floor == 0.01


def test_default_cascade_heat_half_life_s():
    cfg = RetrievalConfig(anchor_paths=[], distill_root=Path("/tmp"), surfaces=[])
    assert cfg.cascade_heat_half_life_s == 21600.0


def test_default_cascade_heat_min_dict_size():
    cfg = RetrievalConfig(anchor_paths=[], distill_root=Path("/tmp"), surfaces=[])
    assert cfg.cascade_heat_min_dict_size == 20
