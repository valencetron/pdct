"""Tests for VEC_NEAR embedding index (Track C Claim 3)."""
from __future__ import annotations
import json
import pytest
from pathlib import Path


def _make_distill(tmp_path: Path, slug: str, concepts: list[str], body: str) -> Path:
    f = tmp_path / f"{slug}.md"
    concepts_yaml = "\n".join(f"  - {c}" for c in concepts)
    f.write_text(f"---\nslug: {slug}\nconcepts:\n{concepts_yaml}\n---\n\n{body}\n")
    return f


def test_build_vec_near_edges_returns_list(tmp_path):
    """build_vec_near_edges must return a list of 4-tuples (a, b, weight, 'vec_near')."""
    from dct.retrieval.vec_index import build_vec_near_edges

    _make_distill(tmp_path, "memory-obsidian", ["memory", "obsidian"],
                  "Memory stored in Obsidian vault for context retrieval and knowledge management.")
    _make_distill(tmp_path, "memory-vault", ["memory", "vault"],
                  "Vault-based memory system for context retrieval and persistent knowledge.")
    _make_distill(tmp_path, "voice-pipeline", ["voice", "retell"],
                  "Voice pipeline uses Retell AI for phone calls and text to speech.")

    edges = build_vec_near_edges(tmp_path, threshold=0.5)
    assert isinstance(edges, list)
    for e in edges:
        assert len(e) == 4, f"Expected 4-tuple, got: {e}"
        a, b, w, etype = e
        assert isinstance(a, str) and isinstance(b, str)
        assert isinstance(w, int) and w >= 1
        assert etype == "vec_near"


def test_build_vec_near_edges_threshold_filters(tmp_path):
    """High threshold should return fewer edges than low threshold."""
    from dct.retrieval.vec_index import build_vec_near_edges

    for i in range(5):
        _make_distill(tmp_path, f"doc-{i}", [f"concept-{i}"],
                      f"Document {i} about topic {i} with some content about knowledge retrieval.")

    low = build_vec_near_edges(tmp_path, threshold=0.3)
    high = build_vec_near_edges(tmp_path, threshold=0.98)
    assert len(low) >= len(high), "higher threshold should yield fewer or equal edges"


def test_build_vec_near_edges_empty_vault(tmp_path):
    """Empty vault returns empty edge list."""
    from dct.retrieval.vec_index import build_vec_near_edges
    edges = build_vec_near_edges(tmp_path, threshold=0.7)
    assert edges == []


def test_build_vec_near_edges_no_concepts_skipped(tmp_path):
    """Distillations with no concepts are skipped."""
    from dct.retrieval.vec_index import build_vec_near_edges

    # One file with concepts, one without
    f1 = tmp_path / "with-concepts.md"
    f1.write_text("---\nconcepts:\n  - memory\n---\nMemory content.")
    f2 = tmp_path / "no-concepts.md"
    f2.write_text("# Just a doc\n\nNo frontmatter concepts here.")

    # Single doc with concepts — can't make edges
    edges = build_vec_near_edges(tmp_path, threshold=0.5)
    assert edges == []


def test_build_vec_near_edges_actually_produces_edges(tmp_path):
    pytest.importorskip("sentence_transformers")
    """Two highly similar documents must produce at least one VEC_NEAR edge.

    Uses a very low threshold (0.3) and two nearly identical texts to
    ensure at least one edge is produced — validates the full pipeline
    (model load → encode → cosine → edge construction), not just shape.
    """
    from dct.retrieval.vec_index import build_vec_near_edges

    # Two very similar texts — same words, different concept slugs
    _make_distill(tmp_path, "context-retrieval-a", ["context-retrieval-a"],
                  "Context retrieval from memory vault using graph traversal and concept matching.")
    _make_distill(tmp_path, "context-retrieval-b", ["context-retrieval-b"],
                  "Context retrieval from memory store using graph traversal and concept lookup.")

    edges = build_vec_near_edges(tmp_path, threshold=0.3)
    assert len(edges) >= 1, (
        "Two nearly identical distillations must produce at least one VEC_NEAR edge "
        f"at threshold=0.3. Got: {edges}"
    )
    # Validate structure
    a, b, w, etype = edges[0]
    assert etype == "vec_near"
    assert w >= 1


def test_vec_near_edges_participate_in_cascade(tmp_path):
    """VEC_NEAR edges added to typed_edges must be traversable by cascade().

    Codex r2 P0 fix: without the _build_adj fix, VEC_NEAR edges only exist
    in typed_edges metadata and are never walked.
    """
    import pathlib
    from dct.retrieval.cascade import cascade
    from dct.retrieval.types import RetrievalConfig
    from dct.heat import ConceptGraph

    # Graph: a→b exists only as VEC_NEAR (not in edges list)
    # Without the P0 fix, b would never be reached from seed a.
    nodes = {"a": 1, "b": 1}
    edges = []  # No CO_OCCUR edges at all
    typed_edges = [("a", "b", 7, "vec_near")]
    transitions: dict = {}
    graph = ConceptGraph(nodes=nodes, edges=edges, transitions=transitions, typed_edges=typed_edges)

    config = RetrievalConfig(
        anchor_paths=[], distill_root=pathlib.Path("/tmp"), surfaces=[],
        cascade_vec_near_enabled=True,
        cascade_transitions_enabled=False,
    )
    hits = cascade(["a"], graph=graph, heat={}, config=config)
    hit_concepts = {h.concept for h in hits}

    assert "b" in hit_concepts, (
        "VEC_NEAR edge a→b must make b reachable from seed a. "
        "If this fails, _build_adj() is not merging typed_edges."
    )
