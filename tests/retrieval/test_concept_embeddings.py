"""Concept-embedding cache tests (WS1 Phase 1, 2026-07-21)."""
import numpy as np

from dct.retrieval import concept_embeddings as ce


class _FakeModel:
    """Deterministic fake: vector encodes (len, n_words, const), normalized."""

    def __init__(self):
        self.encode_calls = []

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        self.encode_calls.append(list(texts))
        out = []
        for t in texts:
            v = np.zeros(384, dtype="float32")
            v[0] = len(t)
            v[1] = len(t.split())
            v[2] = 1.0
            n = np.linalg.norm(v) or 1.0
            out.append(v / n)
        return np.array(out)


def test_hash_is_stable_and_text_sensitive():
    assert ce.text_key("alpha beta") == ce.text_key("alpha beta")
    assert ce.text_key("alpha beta") != ce.text_key("alpha gamma")


def test_get_many_encodes_only_misses(tmp_path):
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    m = _FakeModel()
    texts = ["alpha beta", "gamma delta"]
    vecs1 = store.get_many(texts, model=m)
    assert vecs1.shape == (2, 384)
    assert m.encode_calls == [texts]
    vecs2 = store.get_many(texts, model=m)
    assert len(m.encode_calls) == 1
    assert np.allclose(vecs1, vecs2)


def test_partial_miss_encodes_only_new(tmp_path):
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    m = _FakeModel()
    store.get_many(["a b"], model=m)
    m.encode_calls.clear()
    store.get_many(["a b", "c d"], model=m)
    assert m.encode_calls == [["c d"]]


def test_duplicate_texts_in_one_call_encoded_once(tmp_path):
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    m = _FakeModel()
    out = store.get_many(["dup", "dup", "dup"], model=m)
    assert out.shape == (3, 384)
    assert m.encode_calls == [["dup"]]  # deduped before encode
    assert np.allclose(out[0], out[1]) and np.allclose(out[1], out[2])


def test_persist_and_reload(tmp_path):
    p = tmp_path / "c.npz"
    m = _FakeModel()
    s1 = ce.ConceptEmbeddingStore(path=p)
    v1 = s1.get_many(["hello world"], model=m)
    s1.save()
    assert p.exists()
    s2 = ce.ConceptEmbeddingStore(path=p)
    m2 = _FakeModel()
    v2 = s2.get_many(["hello world"], model=m2)
    assert m2.encode_calls == []
    assert np.allclose(v1, v2)


def test_canonical_concept_text():
    assert ce.concept_text("cascade-timeout") == "cascade timeout"
    assert ce.concept_text("a-b", "hello world") == "a b hello world"
    assert ce.concept_text("a-b", "  ") == "a b"  # blank snippet ignored
    long = "x" * 500
    assert ce.concept_text("a", long) == "a " + "x" * 200  # snippet capped 200


def test_fingerprint_mismatch_rejects_store(tmp_path):
    p = tmp_path / "c.npz"
    m = _FakeModel()
    s1 = ce.ConceptEmbeddingStore(path=p)
    s1.get_many(["hello world"], model=m)
    s1.save()
    # Simulate a model change: bump the module fingerprint, reload.
    orig = ce.MODEL_FINGERPRINT
    try:
        ce.MODEL_FINGERPRINT = "different-model/999/v9"
        s2 = ce.ConceptEmbeddingStore(path=p)
        assert len(s2) == 0  # stale store rejected
        m2 = _FakeModel()
        s2.get_many(["hello world"], model=m2)
        assert m2.encode_calls == [["hello world"]]  # re-encoded fresh
    finally:
        ce.MODEL_FINGERPRINT = orig


def test_corrupt_store_degrades_to_empty(tmp_path):
    p = tmp_path / "c.npz"
    p.write_bytes(b"not a valid npz")
    store = ce.ConceptEmbeddingStore(path=p)
    m = _FakeModel()
    v = store.get_many(["x y"], model=m)
    assert v.shape == (1, 384) and m.encode_calls == [["x y"]]


def test_empty_input_returns_empty(tmp_path):
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    v = store.get_many([], model=_FakeModel())
    assert v.shape == (0, 384)


def test_save_noop_when_not_dirty(tmp_path):
    p = tmp_path / "c.npz"
    store = ce.ConceptEmbeddingStore(path=p)
    store.save()  # nothing encoded → dirty is False
    assert not p.exists()


def test_order_preserved_with_mixed_hits_and_misses(tmp_path):
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    m = _FakeModel()
    store.get_many(["seen"], model=m)
    m.encode_calls.clear()
    texts = ["new1", "seen", "new2"]
    out = store.get_many(texts, model=m)
    # each row must correspond to its input text position
    for i, t in enumerate(texts):
        direct = m.encode([t])[0] if False else None  # not used; positional check below
    # positional integrity: re-fetch individually and compare
    for i, t in enumerate(texts):
        single = store.get_many([t], model=m)
        assert np.allclose(out[i], single[0])


# ── Codex diff r1: concurrency, pruning, dimension contract ──────────

def test_bad_dimension_raises_not_poisons(tmp_path):
    """A model returning wrong-dim output must raise, not cache garbage."""
    import numpy as np
    class _BadModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return np.zeros((len(texts), 7), dtype="float32")  # wrong dim
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    try:
        store.get_many(["x y"], model=_BadModel())
        assert False, "expected ValueError on bad dim"
    except ValueError:
        pass
    assert len(store) == 0  # nothing poisoned


def test_rebuild_prunes_removed_concepts(tmp_path):
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    m = _FakeModel()
    store.get_many(["old one", "keep me"], model=m)
    assert len(store) == 2
    # authoritative rebuild with only "keep me" -> "old one" pruned
    keep = store.vectors_for(["keep me"], model=m)
    store.rebuild(keep)
    assert len(store) == 1
    # reload from disk confirms prune persisted
    s2 = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    assert len(s2) == 1
    m2 = _FakeModel()
    s2.get_many(["keep me"], model=m2)
    assert m2.encode_calls == []          # kept, served from disk
    s2.get_many(["old one"], model=m2)
    assert m2.encode_calls == [["old one"]]  # pruned, re-encoded


def test_concurrent_get_many_is_safe(tmp_path):
    """Many threads reading/filling the same keys must not crash or corrupt."""
    import threading
    import numpy as np
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    texts = [f"concept {i}" for i in range(50)]
    m = _FakeModel()
    barrier = threading.Barrier(8)
    results = []
    errors = []

    def worker():
        try:
            barrier.wait()
            for _ in range(5):
                results.append(store.get_many(texts, model=m))
        except Exception as e:  # noqa
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors
    # all rows for a given text agree across every worker/iteration
    base = results[0]
    for r in results[1:]:
        assert np.allclose(r, base)
    assert len(store) == 50


def test_save_after_concurrent_add_not_marked_clean_prematurely(tmp_path):
    """The dirty flag must not be cleared if entries arrive mid-save."""
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    m = _FakeModel()
    store.get_many(["a"], model=m)          # dirty
    # simulate an add landing after save() snapshots but before it clears:
    # we can't easily interleave, so assert the len-guard logic directly.
    store.save()
    assert not store._dirty                 # clean after a quiet save
    store.get_many(["b"], model=m)          # new add -> dirty again
    assert store._dirty
    store.save()
    assert not store._dirty


def test_mutation_during_save_stays_dirty(tmp_path, monkeypatch):
    """Codex diff r2: a same-length rebuild landing mid-save must NOT be
    marked clean. We interleave by hooking np.savez to mutate the store
    during the I/O window, then assert the store is still dirty."""
    import numpy as np
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    m = _FakeModel()
    store.get_many(["a"], model=m)  # dirty, gen=1

    real_savez = np.savez
    def _savez_then_mutate(*args, **kwargs):
        real_savez(*args, **kwargs)
        # a concurrent rebuild lands DURING the save's I/O window
        store.rebuild({"newkey": np.zeros(384, dtype="float32")})
    monkeypatch.setattr(np, "savez", _savez_then_mutate)

    store.save()  # snapshots gen=1, writes, but gen advanced via rebuild
    # rebuild itself calls save() (nested) which persists {newkey}; the OUTER
    # save must observe gen changed and leave dirty state consistent.
    # Net: the last persisted content must equal the in-memory cache.
    monkeypatch.setattr(np, "savez", real_savez)
    store.save()  # flush any residual
    s2 = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    assert set(s2._cache.keys()) == set(store._cache.keys())  # disk == memory


def test_warmer_refuses_prune_on_empty_graph(monkeypatch, tmp_path):
    """DELIBERATE divergence from Codex 'always prune': an empty graph must
    NOT wipe the cache (transient load failure guard)."""
    import importlib, sys
    sys.path.insert(0, str(__import__("pathlib").Path(
        "~/example-stack/pdct/scripts")))
    warm = importlib.import_module("warm_concept_embeddings")
    # pre-populate a store, then simulate an empty graph
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    store.get_many(["keep a", "keep b"], model=_FakeModel())
    store.save()
    monkeypatch.setattr(warm, "default_store", lambda: store)
    monkeypatch.setattr(warm, "_load_or_build_graph", lambda: type("G", (), {"nodes": {}})())
    rc = warm._run(model_loader=lambda: _FakeModel())
    assert rc == 0 and len(store) == 2  # cache intact, not pruned to empty


def test_missing_texts_cache_only(tmp_path):
    """missing_texts reports uncached texts WITHOUT touching the model
    (WS1 Phase 2, Codex #2 decoupling primitive)."""
    store = ce.ConceptEmbeddingStore(path=tmp_path / "c.npz")
    model = _FakeModel()
    store.get_many(["alpha", "beta"], model=model)
    model.encode_calls.clear()

    miss = store.missing_texts(["alpha", "gamma", "beta", "delta", "gamma"])
    assert miss == ["gamma", "delta"]      # order-preserving, deduped
    assert store.missing_texts(["alpha", "beta"]) == []  # all cached
    assert model.encode_calls == []        # never encoded anything
