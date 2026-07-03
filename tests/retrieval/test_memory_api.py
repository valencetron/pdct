from pathlib import Path
from unittest.mock import patch

import pytest

from dct.retrieval.memory_api import query_memory, DistillationRow
from dct.retrieval.distill_index import DistillationRef
from dct.heat import ConceptGraph
from dct.retrieval.types import ConceptHit


def _idx() -> dict[str, DistillationRef]:
    return {
        "alpha": DistillationRef(id="alpha", path=Path("/x/alpha.md"),
                                  date="2026-04-20", title="Alpha",
                                  concepts=["consciousness", "philosophy"],
                                  gist="Deep dive on consciousness."),
        "beta":  DistillationRef(id="beta",  path=Path("/x/beta.md"),
                                  date="2026-04-21", title="Beta",
                                  concepts=["bioelectricity"],
                                  gist="Bioelectricity notes."),
        "gamma": DistillationRef(id="gamma", path=Path("/x/gamma.md"),
                                  date="2026-04-22", title="Gamma",
                                  concepts=["unrelated"],
                                  gist=""),
    }


def _hits(*pairs) -> list[ConceptHit]:
    return [ConceptHit(concept=c, score=s, source_slug="t", snippet="", hop=0)
            for c, s in pairs]


def test_query_memory_string_seed_returns_rows():
    with patch("dct.retrieval.memory_api.build_index", return_value=_idx()), \
         patch("dct.retrieval.memory_api._cascade_for_seed",
               return_value=_hits(("consciousness", 1.0), ("philosophy", 0.7))), \
         patch("dct.retrieval.memory_api.telemetry.log_call"):
        rows = query_memory("consciousness")
    assert len(rows) >= 1
    assert rows[0].id == "alpha"
    assert rows[0].source == "graph"
    assert all(isinstance(r, DistillationRow) for r in rows)


def test_query_memory_list_seed_dedupes():
    with patch("dct.retrieval.memory_api.build_index", return_value=_idx()), \
         patch("dct.retrieval.memory_api._cascade_for_seed",
               side_effect=[_hits(("consciousness", 1.0)), _hits(("bioelectricity", 1.0))]), \
         patch("dct.retrieval.memory_api.telemetry.log_call"):
        rows = query_memory(["consciousness", "bioelectricity"])
    ids = [r.id for r in rows]
    assert "alpha" in ids and "beta" in ids
    # dedup: each id appears at most once
    assert len(ids) == len(set(ids))


def test_query_memory_falls_back_to_ripgrep_when_cascade_empty(tmp_path):
    # Real files for ripgrep to find
    root = tmp_path / "distillations"
    root.mkdir()
    (root / "match.md").write_text("---\ntitle: Match\n---\n\nthe rare keyword appears here\n")
    (root / "miss.md").write_text("---\ntitle: Miss\n---\n\ntotally different content\n")
    fallback_idx = {
        "match": DistillationRef(id="match", path=root / "match.md", date="",
                                  title="Match", concepts=[], gist=""),
        "miss":  DistillationRef(id="miss",  path=root / "miss.md",  date="",
                                  title="Miss",  concepts=[], gist=""),
    }
    with patch("dct.retrieval.memory_api.build_index", return_value=fallback_idx), \
         patch("dct.retrieval.memory_api._cascade_for_seed", return_value=[]), \
         patch("dct.retrieval.memory_api.telemetry.log_call"):
        rows = query_memory("rare keyword")
    assert any(r.id == "match" for r in rows)
    assert all(r.source == "fallback" for r in rows if r.id == "match")


def test_query_memory_returns_empty_when_no_matches():
    empty_idx = {
        "x": DistillationRef(id="x", path=Path("/nonexistent.md"),
                              date="", title="X", concepts=[], gist=""),
    }
    with patch("dct.retrieval.memory_api.build_index", return_value=empty_idx), \
         patch("dct.retrieval.memory_api._cascade_for_seed", return_value=[]), \
         patch("dct.retrieval.memory_api._ripgrep_fallback", return_value=[]), \
         patch("dct.retrieval.memory_api.telemetry.log_call"):
        rows = query_memory("nothing matches")
    assert rows == []


def test_query_memory_logs_telemetry():
    with patch("dct.retrieval.memory_api.build_index", return_value=_idx()), \
         patch("dct.retrieval.memory_api._cascade_for_seed",
               return_value=_hits(("consciousness", 1.0))), \
         patch("dct.retrieval.memory_api.telemetry.log_call") as mock_log:
        query_memory("consciousness", _surface="voice")
    assert mock_log.called
    kwargs = mock_log.call_args.kwargs
    assert kwargs["surface"] == "voice"
    assert kwargs["fn"] == "query_memory"
    assert kwargs["seed"] == "consciousness"
    assert kwargs["used_fallback"] is False


from dct.retrieval.memory_api import read_memory, MemoryRead


def test_read_memory_returns_full_body(tmp_path):
    p = tmp_path / "alpha.md"
    p.write_text("---\ntitle: Alpha\nconcepts: [consciousness]\n---\n\n## Summary\nBody here.\n")
    idx = {"alpha": DistillationRef(id="alpha", path=p, date="2026-04-20",
                                     title="Alpha", concepts=["consciousness"], gist="")}
    with patch("dct.retrieval.memory_api.build_index", return_value=idx), \
         patch("dct.retrieval.memory_api.related_distillations", return_value=[]), \
         patch("dct.retrieval.memory_api.telemetry.log_call"):
        result = read_memory("alpha")
    assert isinstance(result, MemoryRead)
    assert result.id == "alpha"
    assert result.date == "2026-04-20"
    assert "Body here." in result.body
    assert result.related_distillations == []


def test_read_memory_includes_related(tmp_path):
    from dct.retrieval.related import RelatedRef
    p = tmp_path / "alpha.md"
    p.write_text("---\ntitle: Alpha\n---\n\nbody\n")
    idx = {"alpha": DistillationRef(id="alpha", path=p, date="",
                                     title="Alpha", concepts=["x"], gist="")}
    rels = [RelatedRef(id="other", title="Other", score=0.8)]
    with patch("dct.retrieval.memory_api.build_index", return_value=idx), \
         patch("dct.retrieval.memory_api.related_distillations", return_value=rels), \
         patch("dct.retrieval.memory_api.telemetry.log_call"):
        result = read_memory("alpha")
    assert len(result.related_distillations) == 1
    assert result.related_distillations[0].id == "other"


def test_read_memory_raises_for_unknown_id():
    with patch("dct.retrieval.memory_api.build_index", return_value={}), \
         patch("dct.retrieval.memory_api.telemetry.log_call"):
        with pytest.raises(KeyError):
            read_memory("ghost")
