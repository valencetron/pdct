

# ── Cosine circuit breaker (2026-07-17 hardening follow-up) ──────────

import pytest


@pytest.fixture(autouse=True)
def _isolate_concept_store(tmp_path, monkeypatch):
    """Every cosine test uses a private temp store, never the production
    on-disk cache (Codex diff r1 test-gap: fakes must not touch prod).

    Also stub the off-process query encode to None by default (WS1 Phase 2):
    these tests must be hermetic whether or not a real embed service happens
    to be listening on the default socket. Tests exercising the off-proc path
    override _encode_query explicitly."""
    from dct.retrieval import relevance as rel
    from dct.retrieval import concept_embeddings as ce
    store = ce.ConceptEmbeddingStore(path=tmp_path / "iso.npz")
    monkeypatch.setattr(rel, "_concept_store", lambda: store)
    monkeypatch.setattr(rel, "_encode_query", lambda text: None)


class _SlowModel:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import time
        import numpy as np
        time.sleep(0.01)
        v = np.zeros((len(texts), 384), dtype="float32")
        v[:, 2] = 1.0  # valid 384-D, finite, normalized
        return v


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


# ── WS1 Phase 1: concept-embedding cache wiring (2026-07-21) ──────────

def test_cosine_filter_encodes_query_only_when_concepts_cached(tmp_path, monkeypatch):
    import numpy as np
    from dct.retrieval import relevance as rel
    from dct.retrieval import concept_embeddings as ce
    from dct.retrieval.cascade import ConceptHit
    rel._COSINE_BREAKER["until_mono"] = 0.0

    class _CountingModel:
        def __init__(self): self.encoded = []
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            self.encoded.append(list(texts))
            v = np.zeros((len(texts), 384), dtype="float32"); v[:, 2] = 1.0
            return v

    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    model = _CountingModel()
    monkeypatch.setattr(rel, "_concept_store", lambda: store)
    hits = [ConceptHit(concept="a-b", hop=1, score=1.0, path=[], snippet="", source_slug="t"),
            ConceptHit(concept="c-d", hop=1, score=1.0, path=[], snippet="", source_slug="t")]
    # warm the concept texts (simulates a prior turn / the warmer)
    store.get_many([ce.concept_text("a-b"), ce.concept_text("c-d")], model=model)
    model.encoded.clear()
    out, dropped = rel.query_cosine_filter(
        "a sufficiently long user query", hits, threshold=-1.0, _model_override=model)
    assert len(model.encoded) == 1 and len(model.encoded[0]) == 1  # query only
    assert out == hits


def test_cached_path_equals_fresh_path(tmp_path, monkeypatch):
    """Codex r1 #11: cached and uncached paths must yield IDENTICAL keep/drop.
    Uses a deterministic model with known cosines and a discriminating threshold.
    """
    import numpy as np
    from dct.retrieval import relevance as rel
    from dct.retrieval import concept_embeddings as ce
    from dct.retrieval.cascade import ConceptHit
    rel._COSINE_BREAKER["until_mono"] = 0.0

    # Deterministic model: query -> [1,0,0]; "a b" (relevant) -> [1,0,0] (cos 1);
    # "c d" (irrelevant) -> [0,1,0] (cos 0). Threshold 0.5 keeps a-b, drops c-d.
    class _DetModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            out = []
            for t in texts:
                v = np.zeros(384, dtype="float32")
                v[1] = 1.0 if t.startswith("c d") else 0.0
                v[0] = 0.0 if t.startswith("c d") else 1.0
                out.append(v)
            return np.array(out, dtype="float32")

    hits = [ConceptHit(concept="a-b", hop=1, score=1.0, path=[], snippet="", source_slug="t"),
            ConceptHit(concept="c-d", hop=1, score=1.0, path=[], snippet="", source_slug="t")]

    # FRESH path: empty store, everything a miss.
    store_fresh = ce.ConceptEmbeddingStore(path=tmp_path / "fresh.npz")
    monkeypatch.setattr(rel, "_concept_store", lambda: store_fresh)
    out_fresh, drop_fresh = rel.query_cosine_filter(
        "the relevant query text here", hits, threshold=0.5, _model_override=_DetModel())

    # CACHED path: pre-warm the concepts, then run.
    store_warm = ce.ConceptEmbeddingStore(path=tmp_path / "warm.npz")
    store_warm.get_many([ce.concept_text("a-b"), ce.concept_text("c-d")], model=_DetModel())
    monkeypatch.setattr(rel, "_concept_store", lambda: store_warm)
    out_warm, drop_warm = rel.query_cosine_filter(
        "the relevant query text here", hits, threshold=0.5, _model_override=_DetModel())

    assert [h.concept for h in out_fresh] == [h.concept for h in out_warm] == ["a-b"]
    assert drop_fresh == drop_warm == 1


# ── WS1 Phase 2: off-process query encode wiring (2026-07-21) ─────────

def _hit(concept):
    from dct.retrieval.cascade import ConceptHit
    return ConceptHit(concept=concept, hop=1, score=1.0, path=[], snippet="",
                      source_slug="t")


class _CountingModel:
    def __init__(self):
        self.encoded = []

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np
        self.encoded.append(list(texts))
        v = np.zeros((len(texts), 384), dtype="float32")
        v[:, 2] = 1.0
        return v


def test_offproc_query_encode_skips_inproc_model(tmp_path, monkeypatch):
    """Happy path: query served off-proc + concepts cached ⇒ the in-process
    model.encode is NEVER called (proves the encode left the daemon process)."""
    import numpy as np
    from dct.retrieval import relevance as rel
    from dct.retrieval import concept_embeddings as ce
    rel._COSINE_BREAKER["until_mono"] = 0.0

    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    model = _CountingModel()
    monkeypatch.setattr(rel, "_concept_store", lambda: store)
    store.get_many([ce.concept_text("a-b"), ce.concept_text("c-d")], model=model)
    model.encoded.clear()

    qv = np.zeros(384, dtype="float32"); qv[2] = 1.0  # cos 1 with cached concepts
    monkeypatch.setattr(rel, "_encode_query", lambda text: qv)

    out, dropped = rel.query_cosine_filter(
        "a sufficiently long user query", [_hit("a-b"), _hit("c-d")],
        threshold=0.5, _model_override=model)
    assert model.encoded == []                       # no in-process encode at all
    assert [h.concept for h in out] == ["a-b", "c-d"]
    assert dropped == 0


def test_fallback_to_inproc_when_service_down(tmp_path, monkeypatch):
    """Never worse than Phase 1: off-proc encode returns None ⇒ the query is
    encoded in-process (exactly one encode, one text)."""
    from dct.retrieval import relevance as rel
    from dct.retrieval import concept_embeddings as ce
    rel._COSINE_BREAKER["until_mono"] = 0.0

    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    model = _CountingModel()
    monkeypatch.setattr(rel, "_concept_store", lambda: store)
    store.get_many([ce.concept_text("a-b"), ce.concept_text("c-d")], model=model)
    model.encoded.clear()

    monkeypatch.setattr(rel, "_encode_query", lambda text: None)  # service down

    out, dropped = rel.query_cosine_filter(
        "a sufficiently long user query", [_hit("a-b"), _hit("c-d")],
        threshold=-1.0, _model_override=model)
    assert len(model.encoded) == 1 and len(model.encoded[0]) == 1  # query only
    assert out == [_hit("a-b"), _hit("c-d")] or [h.concept for h in out] == ["a-b", "c-d"]


def test_no_local_model_when_service_up_and_concepts_cached(tmp_path, monkeypatch):
    """Codex #2 (critical): with the query off-proc AND every concept cached,
    the filter runs even when the daemon's local model is COLD — it never even
    queries get_model_if_ready. Under Phase 1 this turn would have SKIPPED."""
    import numpy as np
    from dct.retrieval import relevance as rel
    from dct.retrieval import vec_index
    from dct.retrieval import concept_embeddings as ce
    rel._COSINE_BREAKER["until_mono"] = 0.0

    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    monkeypatch.setattr(rel, "_concept_store", lambda: store)
    store.get_many([ce.concept_text("a-b"), ce.concept_text("c-d")],
                   model=_CountingModel())  # warm as the nightly warmer would

    called = {"n": 0}
    def _cold():
        called["n"] += 1
        return None
    monkeypatch.setattr(vec_index, "get_model_if_ready", _cold)

    qv = np.zeros(384, dtype="float32"); qv[2] = 1.0
    monkeypatch.setattr(rel, "_encode_query", lambda text: qv)

    out, dropped = rel.query_cosine_filter(
        "a sufficiently long user query", [_hit("a-b"), _hit("c-d")], threshold=0.5)
    assert called["n"] == 0                          # local model never queried
    assert [h.concept for h in out] == ["a-b", "c-d"]
    assert dropped == 0
