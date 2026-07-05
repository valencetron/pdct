import pytest

from benchmark.mtrag import ingest

# MTRAG data is fetched, not shipped (see fetch_mtrag.py). Skip cleanly on a
# fresh clone so the suite is green pre-fetch.
if not (ingest.DATA / "fiqa.jsonl.zip").exists():
    pytest.skip("MTRAG data not fetched — run: python -m benchmark.mtrag.fetch_mtrag",
                allow_module_level=True)


def test_load_fiqa_passages_nonempty():
    passages = ingest.load_passages("fiqa", limit=500)
    assert len(passages) > 100
    p = passages[0]
    assert p["id"] and isinstance(p["text"], str) and len(p["text"]) > 0


def test_load_conversations_by_corpus():
    convos = ingest.load_conversations(corpus="fiqa")
    assert 10 <= len(convos) <= 60
    c = convos[0]
    assert any(m["speaker"] == "user" for m in c["messages"])


def test_load_qrels():
    qrels = ingest.load_qrels("fiqa")
    assert len(qrels) > 50
    any_gold = next(iter(qrels.values()))
    assert isinstance(any_gold, set) and len(any_gold) >= 1


def test_load_questions_variant():
    rows = ingest.load_retrieval_tasks("fiqa", "questions")
    assert rows and "_id" in rows[0] and "text" in rows[0]


def test_load_passages_includes_all_gold():
    passages, missing = ingest.load_passages_with_gold("fiqa")
    assert isinstance(missing, set)
    # report-only: FiQA passage corpus should contain all gold ids
    assert len(missing) == 0, f"{len(missing)} gold passages absent from corpus"
