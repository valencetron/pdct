from dataclasses import dataclass

from dct.retrieval.retrieval_metrics import gold_ids, gold_rank, recall_at_k


@dataclass
class _Row:
    id: str


def _q(sid=None, path=None):
    q = {}
    if sid:
        q["source_distillation_id"] = sid
    if path:
        q["source_path"] = path
    return q


def test_gold_ids_from_distillation_id():
    assert gold_ids(_q(sid="abc-123")) == {"abc-123"}


def test_gold_ids_includes_path_stem():
    q = _q(sid="abc-123", path="/a/b/2026-04-27-1518-topic.md")
    assert gold_ids(q) == {"abc-123", "2026-04-27-1518-topic"}


def test_gold_ids_empty_when_no_source():
    assert gold_ids(_q()) == set()


def test_gold_rank_finds_first_match():
    rows = [_Row("x"), _Row("gold"), _Row("y")]
    assert gold_rank(rows, {"gold"}) == 1


def test_gold_rank_none_when_absent():
    rows = [_Row("x"), _Row("y")]
    assert gold_rank(rows, {"gold"}) is None


def test_gold_rank_handles_dict_rows():
    rows = [{"id": "x"}, {"id": "gold"}]
    assert gold_rank(rows, {"gold"}) == 1


def test_recall_at_k_counts_within_k():
    rows = [_Row("a"), _Row("b"), _Row("gold")]
    assert recall_at_k(rows, {"gold"}, k=5) is True
    assert recall_at_k(rows, {"gold"}, k=2) is False
    assert recall_at_k(rows, set(), k=5) is None  # no gold = not applicable
