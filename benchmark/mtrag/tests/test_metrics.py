import math
from benchmark.mtrag import metrics


def test_jaccard_and_divergence():
    assert abs(metrics.jaccard({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-9
    assert abs(metrics.divergence({"a", "b"}, {"b", "c"}) - 2 / 3) < 1e-9
    assert metrics.divergence(set(), set()) == 0.0


def test_recall_at_k():
    assert metrics.recall_at_k(["p1", "p2", "p3", "p4"], {"p3", "p9"}, 5) == 0.5
    assert metrics.recall_at_k(["p1", "p2", "p3", "p4"], {"p3", "p9"}, 2) == 0.0


def test_ndcg_at_k():
    assert abs(metrics.ndcg_at_k(["p1", "p2"], {"p1", "p2"}, 2) - 1.0) < 1e-9
    assert abs(metrics.ndcg_at_k(["x", "p1"], {"p1"}, 2) - (1 / math.log2(3))) < 1e-9


def test_permutation_null_excludes_reference_and_is_exhaustive():
    def play(turns):  # set depends only on first middle element
        return {turns[1]}
    middle = ["m1", "m2", "m3"]
    null, n = metrics.permutation_null_divergence(play, middle, ["op"], ["cl"])
    assert n == 5  # 3! - 1 reference
    assert 1.0 in null and 0.0 in null
