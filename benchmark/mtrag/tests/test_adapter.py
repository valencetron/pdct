from benchmark.mtrag import build_graph, adapter


def _g():
    passages = [
        {"id": "p1", "title": "", "text": "roth ira retirement account tax free withdrawals income"},
        {"id": "p2", "title": "", "text": "roth ira contribution limits and income phase out rules"},
        {"id": "p3", "title": "", "text": "mortgage refinance interest rate points closing costs"},
    ]
    return build_graph.build(passages, top_k=5)


def test_rank_prefers_concept_passages():
    g = _g()
    A = adapter.PassageAdapter(g)
    ranked = A.rank({"roth": 1.0, "ira": 0.8}, top_n=3)
    ids = [pid for pid, _ in ranked]
    assert ids[0] in ("p1", "p2") and "p3" not in ids[:1]


def test_empty_activation_returns_empty():
    g = _g()
    A = adapter.PassageAdapter(g)
    assert A.rank({}, top_n=3) == []


def test_fallback_when_no_inverted_index_hit():
    # Use a concept that is NOT a graph node / not in the inverted index, but
    # whose tokens DO appear in passage text -> forces the BM25-over-all fallback.
    g = _g()
    A = adapter.PassageAdapter(g)
    concept = "withdrawals"
    assert concept not in g.concept_to_passages, "precondition: not in inverted index"
    assert any("withdrawals" in txt for txt in g.passage_text.values()), \
        "precondition: token present in some passage"
    ranked = A.rank({concept: 1.0}, top_n=3)
    assert ranked  # non-empty ONLY achievable via the fallback path
    assert "p1" in [pid for pid, _ in ranked]  # p1 contains 'withdrawals'
