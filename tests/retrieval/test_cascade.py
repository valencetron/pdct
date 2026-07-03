"""Cascade retrieval tests."""
from __future__ import annotations

import pytest

from dct.retrieval.cascade import _neighbors_of, cascade
from dct.retrieval.types import RetrievalConfig, ConceptHit


class _FakeGraph:
    """Minimal stand-in matching ConceptGraph.edges shape."""
    def __init__(self, edges: list[tuple[str, str, int]]):
        self.edges = edges


def test_neighbors_of_returns_directed_symmetric():
    g = _FakeGraph([("a", "b", 3), ("a", "c", 1), ("d", "e", 5)])
    result = _neighbors_of(g, "a")
    assert result == {"b": 3, "c": 1}


def test_neighbors_of_reverse_edge():
    g = _FakeGraph([("x", "y", 4)])
    assert _neighbors_of(g, "y") == {"x": 4}


def test_neighbors_of_missing_concept():
    g = _FakeGraph([("a", "b", 1)])
    assert _neighbors_of(g, "z") == {}


def test_cascade_hop_zero_returns_seed(config: RetrievalConfig):
    g = _FakeGraph([])
    hits = cascade(
        seed_concepts=["consciousness"],
        graph=g,
        heat={"consciousness": 1.0},
        config=config,
    )
    assert any(h.concept == "consciousness" and h.hop == 0 and h.score == 1.0 for h in hits)


def test_cascade_hop_one_direct_neighbors(config: RetrievalConfig):
    g = _FakeGraph([
        ("consciousness", "phenomenology", 4),
        ("consciousness", "memory", 2),
    ])
    hits = cascade(
        seed_concepts=["consciousness"],
        graph=g,
        heat={},
        config=config,
    )
    concepts = {h.concept: h for h in hits}
    assert concepts["consciousness"].hop == 0
    assert concepts["phenomenology"].hop == 1
    assert concepts["memory"].hop == 1
    assert concepts["phenomenology"].score > concepts["memory"].score


def test_cascade_hop_two_with_decay(config: RetrievalConfig):
    g = _FakeGraph([
        ("consciousness", "phenomenology", 4),
        ("phenomenology", "qualia", 4),
    ])
    hits = cascade(
        seed_concepts=["consciousness"],
        graph=g,
        heat={},
        config=config,
    )
    concepts = {h.concept: h for h in hits}
    assert concepts["qualia"].hop == 2
    assert concepts["qualia"].score == pytest.approx(concepts["phenomenology"].score * 0.3)


def test_cascade_depth_one_skips_hop_two(config: RetrievalConfig):
    g = _FakeGraph([
        ("a", "b", 3),
        ("b", "c", 3),
    ])
    cfg = RetrievalConfig(
        anchor_paths=config.anchor_paths,
        distill_root=config.distill_root,
        surfaces=config.surfaces,
        cascade_depth=1,
    )
    hits = cascade(seed_concepts=["a"], graph=g, heat={}, config=cfg)
    concepts = {h.concept for h in hits}
    assert "b" in concepts
    assert "c" not in concepts


def test_cascade_dedup_against_current_context(config: RetrievalConfig):
    g = _FakeGraph([
        ("consciousness", "phenomenology", 4),
        ("consciousness", "memory", 2),
    ])
    hits = cascade(
        seed_concepts=["consciousness"],
        graph=g,
        heat={},
        config=config,
        current_context={"phenomenology"},
    )
    concepts = {h.concept for h in hits}
    assert "phenomenology" not in concepts
    assert "memory" in concepts
    assert "consciousness" in concepts


def test_cascade_skips_seed_if_in_context(config: RetrievalConfig):
    g = _FakeGraph([])
    hits = cascade(
        seed_concepts=["x"],
        graph=g,
        heat={},
        config=config,
        current_context={"x"},
    )
    assert hits == []


def test_cascade_respects_budget_ms(config: RetrievalConfig):
    g = _FakeGraph([
        ("a", "b", 3),
        ("b", "c", 3),
    ])
    cfg = RetrievalConfig(
        anchor_paths=config.anchor_paths,
        distill_root=config.distill_root,
        surfaces=config.surfaces,
        cascade_depth=2,
        cascade_budget_ms=0,
    )
    hits = cascade(seed_concepts=["a"], graph=g, heat={}, config=cfg)
    concepts = {h.concept for h in hits}
    assert "a" in concepts
    assert "c" not in concepts


def test_cascade_seed_path_is_self(config: RetrievalConfig):
    g = _FakeGraph([])
    hits = cascade(seed_concepts=["consciousness"], graph=g, heat={}, config=config)
    seed = next(h for h in hits if h.concept == "consciousness")
    assert seed.path == ["consciousness"]
    assert seed.hop == 0


def test_cascade_hop_one_path(config: RetrievalConfig):
    g = _FakeGraph([("consciousness", "memory", 3)])
    hits = cascade(seed_concepts=["consciousness"], graph=g, heat={}, config=config)
    h = next(h for h in hits if h.concept == "memory")
    assert h.path == ["consciousness", "memory"]
    assert h.hop == 1


def test_cascade_hop_two_path_full_trajectory(config: RetrievalConfig):
    g = _FakeGraph([
        ("consciousness", "memory", 3),
        ("memory", "phenomenology", 4),
    ])
    hits = cascade(seed_concepts=["consciousness"], graph=g, heat={}, config=config)
    h = next(h for h in hits if h.concept == "phenomenology")
    assert h.path == ["consciousness", "memory", "phenomenology"]
    assert h.hop == 2


def test_cascade_multi_seed_higher_weight_wins_path(config: RetrievalConfig):
    """Two seeds reach the same node; higher-edge-weight path wins."""
    g = _FakeGraph([
        ("consciousness", "phenomenology", 4),
        ("memory", "phenomenology", 1),
    ])
    hits = cascade(
        seed_concepts=["consciousness", "memory"], graph=g, heat={}, config=config,
    )
    h = next(h for h in hits if h.concept == "phenomenology")
    assert h.path == ["consciousness", "phenomenology"]


def test_cascade_best_score_path_wins(config: RetrievalConfig):
    """A weak hop-1 path is replaced by a stronger hop-2 path through a peer.

    Track B (R3 fix): credit-assignment goes to the strongest trajectory,
    not the shallowest one. Edge weights:
      a→b = 1   (weak direct path)
      a→c = 4   (strong)
      c→b = 4   (strong)
    The hop-2 path a→c→b wins because c→b is strong; the weak a→b is
    overwritten. Seeds remain immutable (covered by separate test).
    """
    g = _FakeGraph([
        ("a", "b", 1),
        ("a", "c", 4),
        ("c", "b", 4),
    ])
    hits = cascade(seed_concepts=["a"], graph=g, heat={}, config=config)
    b = next(h for h in hits if h.concept == "b")
    assert b.hop == 2
    assert b.path == ["a", "c", "b"]


def test_cascade_seed_never_replaced(config: RetrievalConfig):
    """Seeds (hop=0, score=1.0) are not replaced even if a path scores higher.

    This is the only exception to best-score-globally — seeds are the user's
    intent, not derived hits.
    """
    g = _FakeGraph([
        ("a", "b", 5),
        ("b", "a", 5),  # would create a hop-2 path back to seed
    ])
    hits = cascade(seed_concepts=["a"], graph=g, heat={}, config=config)
    a = next(h for h in hits if h.concept == "a")
    assert a.hop == 0
    assert a.path == ["a"]
    assert a.score == 1.0
    assert a.source_slug == "seed"


def test_cascade_seed_collision_path_is_self(config: RetrievalConfig):
    """If a seed is also a hop-1 neighbor of another seed, seed wins."""
    g = _FakeGraph([("a", "b", 5)])
    hits = cascade(seed_concepts=["a", "b"], graph=g, heat={}, config=config)
    a = next(h for h in hits if h.concept == "a")
    b = next(h for h in hits if h.concept == "b")
    assert a.hop == 0 and a.path == ["a"]
    assert b.hop == 0 and b.path == ["b"]


# ---------------------------------------------------------------------------
# Track C — Directed transitions biasing (Claim 2b)
# ---------------------------------------------------------------------------

def test_transitions_bias_next_hop_score():
    """When transitions[(a,b)] > transitions[(a,c)], b scores higher than c."""
    from dct.heat import ConceptGraph

    # Graph: a connects to both b and c with equal edge weight (10)
    # Transitions: a→b has count 5, a→c has count 1
    # With biasing, b should score higher than c
    edges = [("a", "b", 10), ("a", "c", 10)]
    nodes = {"a": 3, "b": 2, "c": 2}
    transitions = {("a", "b"): 5, ("a", "c"): 1}
    import pathlib
    graph = ConceptGraph(nodes=nodes, edges=edges, transitions=transitions)
    config = RetrievalConfig(
        anchor_paths=[], distill_root=pathlib.Path("/tmp"), surfaces=[],
        cascade_transitions_enabled=True,
        cascade_transitions_bias=0.5,
    )
    hits = cascade(["a"], graph=graph, heat={}, config=config)
    hit_map = {h.concept: h.score for h in hits}

    assert "b" in hit_map and "c" in hit_map, "both b and c must be retrieved"
    assert hit_map["b"] > hit_map["c"], (
        f"b (transitions=5) should score higher than c (transitions=1): "
        f"b={hit_map['b']:.4f} c={hit_map['c']:.4f}"
    )


def test_transitions_disabled_gives_equal_scores():
    """With cascade_transitions_enabled=False, equal-weight edges score equally."""
    from dct.heat import ConceptGraph
    import pathlib

    edges = [("a", "b", 10), ("a", "c", 10)]
    nodes = {"a": 3, "b": 2, "c": 2}
    transitions = {("a", "b"): 100, ("a", "c"): 1}  # huge imbalance but disabled
    graph = ConceptGraph(nodes=nodes, edges=edges, transitions=transitions)
    config = RetrievalConfig(
        anchor_paths=[], distill_root=pathlib.Path("/tmp"), surfaces=[],
        cascade_transitions_enabled=False,
    )
    hits = cascade(["a"], graph=graph, heat={}, config=config)
    hit_map = {h.concept: h.score for h in hits}

    assert abs(hit_map.get("b", 0) - hit_map.get("c", 0)) < 1e-9, (
        "With transitions disabled, equal-weight edges should score equally"
    )


# ── Build #60: behavioral proof that depth + decay control traversal ──
from tests.retrieval._graph_helpers import chain_graph


def _core_config(decay, depth):
    return RetrievalConfig(
        anchor_paths=[], distill_root="/tmp", surfaces=["voice"],
        cascade_decay=decay, cascade_depth=depth, cascade_score_floor=0.0,
    )


def test_cascade_depth_controls_reach():
    """depth=1 stops at hop-1; depth=2 reaches hop-2 node."""
    g = chain_graph(["a", "b", "c"])  # c is hop-2
    reach1 = {h.concept for h in cascade(seed_concepts=["a"], graph=g, heat={}, config=_core_config(0.5, 1))}
    reach2 = {h.concept for h in cascade(seed_concepts=["a"], graph=g, heat={}, config=_core_config(0.5, 2))}
    assert "c" not in reach1
    assert "c" in reach2


def test_cascade_decay_controls_hop2_score():
    """Higher decay -> higher hop-2 score (compounding less attenuation)."""
    g = chain_graph(["a", "b", "c"])
    c_lo = next(h.score for h in cascade(seed_concepts=["a"], graph=g, heat={}, config=_core_config(0.2, 3)) if h.concept == "c")
    c_hi = next(h.score for h in cascade(seed_concepts=["a"], graph=g, heat={}, config=_core_config(0.8, 3)) if h.concept == "c")
    assert c_lo < c_hi
