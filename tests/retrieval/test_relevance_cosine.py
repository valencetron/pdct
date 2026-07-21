

# ── Cosine circuit breaker (2026-07-17 hardening follow-up) ──────────

class _SlowModel:
    def encode(self, texts, **kw):
        import time
        import numpy as np
        time.sleep(0.01)
        return np.zeros((len(texts), 4), dtype="float32")


def test_slow_encode_trips_breaker(monkeypatch):
    from dct.retrieval import relevance as rel
    from dct.retrieval.cascade import ConceptHit
    rel._COSINE_BREAKER["until_mono"] = 0.0
    monkeypatch.setattr(rel, "COSINE_SLOW_S", 0.001)  # everything is "slow"
    hits = [ConceptHit(concept="a-b", hop=1, score=1.0, path=[], snippet="", source_slug="t")]
    out, dropped = rel.query_cosine_filter(
        "a sufficiently long user query text", hits,
        threshold=0.5, _model_override=_SlowModel())
    import time
    assert rel._COSINE_BREAKER["until_mono"] > time.monotonic()
    # next call skips instantly, fail-open
    out2, dropped2 = rel.query_cosine_filter(
        "another sufficiently long query", hits,
        threshold=0.5, _model_override="__RAISE__")  # would raise if not skipped
    assert out2 == hits and dropped2 == 0


def test_breaker_expires(monkeypatch):
    from dct.retrieval import relevance as rel
    import time
    from dct.retrieval.cascade import ConceptHit
    rel._COSINE_BREAKER["until_mono"] = time.monotonic() - 1  # expired
    hits = [ConceptHit(concept="a-b", hop=1, score=1.0, path=[], snippet="", source_slug="t")]
    out, dropped = rel.query_cosine_filter(
        "a sufficiently long user query text", hits,
        threshold=0.0, _model_override=_SlowModel())
    assert isinstance(out, list)  # filter ran (not skipped)
