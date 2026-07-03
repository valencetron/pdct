from pathlib import Path
from unittest.mock import patch

from dct.retrieval.related import related_distillations, RelatedRef
from dct.retrieval.distill_index import DistillationRef
from dct.heat import ConceptGraph


def _make_graph() -> ConceptGraph:
    # 3 concepts, fully connected, equal weights
    return ConceptGraph(
        nodes={"a": 5, "b": 5, "c": 5},
        edges=[("a", "b", 3), ("a", "c", 2), ("b", "c", 4)],
    )


def _make_index() -> dict[str, DistillationRef]:
    return {
        "self":  DistillationRef(id="self",  path=Path("/x/self.md"),  date="", title="Self",  concepts=["a"]),
        "near":  DistillationRef(id="near",  path=Path("/x/near.md"),  date="", title="Near",  concepts=["b"]),
        "far":   DistillationRef(id="far",   path=Path("/x/far.md"),   date="", title="Far",   concepts=["c"]),
        "alt":   DistillationRef(id="alt",   path=Path("/x/alt.md"),   date="", title="Alt",   concepts=["a", "b"]),
    }


def test_related_excludes_self() -> None:
    idx = _make_index()
    with patch("dct.retrieval.related._load_graph", return_value=_make_graph()):
        rows = related_distillations("self", k=3, index=idx)
    ids = [r.id for r in rows]
    assert "self" not in ids


def test_related_returns_top_k_by_score() -> None:
    idx = _make_index()
    with patch("dct.retrieval.related._load_graph", return_value=_make_graph()):
        rows = related_distillations("self", k=2, index=idx)
    assert len(rows) == 2
    # Scores monotonic non-increasing
    assert rows[0].score >= rows[1].score
    # All RelatedRef
    assert all(isinstance(r, RelatedRef) for r in rows)


def test_related_handles_unknown_id() -> None:
    idx = _make_index()
    with patch("dct.retrieval.related._load_graph", return_value=_make_graph()):
        rows = related_distillations("missing", k=3, index=idx)
    assert rows == []


def test_related_handles_distillation_with_no_concepts() -> None:
    idx = _make_index()
    idx["empty"] = DistillationRef(id="empty", path=Path("/x/empty.md"),
                                    date="", title="Empty", concepts=[])
    with patch("dct.retrieval.related._load_graph", return_value=_make_graph()):
        rows = related_distillations("empty", k=3, index=idx)
    assert rows == []
