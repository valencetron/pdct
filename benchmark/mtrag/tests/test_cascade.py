from benchmark.mtrag import build_graph, cascade


def _graph():
    passages = [
        {"id": "p1", "title": "", "text": "roth ira retirement account tax free withdrawals income limits"},
        {"id": "p2", "title": "", "text": "roth ira contribution income phase out rules conversion"},
        {"id": "p3", "title": "", "text": "mortgage refinance interest rate points closing costs escrow"},
        {"id": "p4", "title": "", "text": "mortgage payment principal interest taxes insurance amortization"},
        {"id": "p5", "title": "", "text": "dividend yield payout ratio income investing strategy stocks"},
    ]
    return build_graph.build(passages, top_k=6)


def test_reset_and_turn_returns_activation():
    g = _graph()
    cc = cascade.MtragCascade(g)
    cc.reset()
    r = cc.turn("tell me about roth ira contributions")
    assert isinstance(r["activation"], dict) and r["activation"]


def test_path_dependence_same_final_query():
    g = _graph()
    closer = "summarize what we covered"
    a = cascade.MtragCascade(g)
    a.reset()
    for t in ["roth ira contributions", "income phase out rules", closer]:
        ra = a.turn(t)
    b = cascade.MtragCascade(g)
    b.reset()
    for t in ["mortgage refinance rate", "closing costs escrow", closer]:
        rb = b.turn(t)
    ca, cb = set(ra["activation"]), set(rb["activation"])
    assert ca != cb
