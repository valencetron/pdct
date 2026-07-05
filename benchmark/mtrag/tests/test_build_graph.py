from benchmark.mtrag import build_graph


def test_build_graph_shapes():
    passages = [
        {"id": "p1", "title": "", "text": "roth ira retirement account tax free withdrawals income"},
        {"id": "p2", "title": "", "text": "roth ira contribution limits and income phase out rules"},
        {"id": "p3", "title": "", "text": "mortgage refinance interest rate points closing costs"},
        {"id": "p4", "title": "", "text": "dividend yield payout ratio income investing strategy"},
    ]
    g = build_graph.build(passages, top_k=5)
    assert g.graph.nodes
    assert isinstance(g.graph.edges, list)
    for a, b, w in g.graph.edges:
        assert a < b and isinstance(w, int)
    assert any(pids for pids in g.concept_to_passages.values())
    sample = next(iter(g.concept_to_passages.values()))
    assert sample <= {"p1", "p2", "p3", "p4"}
