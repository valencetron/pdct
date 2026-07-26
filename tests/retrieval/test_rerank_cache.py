"""Cross-encoder result cache — proves the cache is score-identical and that
repeat (query, doc) pairs skip inference. The score-identity is the whole
safety argument: caching cannot change ranking or recall."""
from dct.retrieval import rerank


class _FakeCE:
    """Deterministic CrossEncoder stand-in: score = f(query, text), and it
    counts how many pairs it was actually asked to score."""

    def __init__(self):
        self.calls = 0
        self.pairs_scored = 0

    def predict(self, pairs, show_progress_bar=False):
        self.calls += 1
        self.pairs_scored += len(pairs)
        return [float((hash((q, t)) % 1000) - 500) / 100.0 for q, t in pairs]


def _install(monkeypatch, fake):
    monkeypatch.setattr(rerank, "get_model_if_ready", lambda: fake)
    rerank.reset_cache()


def test_repeat_call_is_identical_and_scores_zero_new_pairs(monkeypatch):
    fake = _FakeCE()
    _install(monkeypatch, fake)
    q = "why did the cosine breaker trip under load"
    cands = [
        ("a", "distillation about the cosine breaker exile window", 0.4),
        ("b", "distillation about the retell voice pipeline", 0.6),
        ("c", "distillation about the embedding cache warmer", 0.2),
    ]
    out1 = rerank.rerank(q, cands)
    after_first = fake.pairs_scored
    out2 = rerank.rerank(q, cands)
    assert out1 == out2                      # byte-identical result
    assert after_first == 3                  # first call scored all 3
    assert fake.pairs_scored == after_first  # second call scored 0 new pairs


def test_partial_overlap_scores_only_new_pairs(monkeypatch):
    fake = _FakeCE()
    _install(monkeypatch, fake)
    q = "same query different pool"
    base = [("a", "alpha", 0.5), ("b", "bravo", 0.5)]
    rerank.rerank(q, base)
    assert fake.pairs_scored == 2
    rerank.rerank(q, base + [("c", "charlie", 0.5)])
    assert fake.pairs_scored == 3            # only +1 (c), a/b served cached


def test_different_query_does_not_reuse(monkeypatch):
    fake = _FakeCE()
    _install(monkeypatch, fake)
    cands = [("a", "alpha", 0.5), ("b", "bravo", 0.5)]
    rerank.rerank("query one", cands)
    rerank.rerank("query two", cands)
    assert fake.pairs_scored == 4            # CE pairs are query-joint; no reuse


def test_cached_equals_freshly_computed(monkeypatch):
    fake = _FakeCE()
    _install(monkeypatch, fake)
    q = "identity"
    cands = [("a", "alpha", 0.3), ("b", "bravo", 0.7),
             ("c", "charlie", 0.5), ("d", "delta", 0.1)]
    warm = rerank.rerank(q, cands)   # second time will be cache-served
    served = rerank.rerank(q, cands)
    rerank.reset_cache()
    fresh = rerank.rerank(q, cands)  # recomputed from scratch
    assert warm == served == fresh


def test_reset_model_clears_cache(monkeypatch):
    fake = _FakeCE()
    _install(monkeypatch, fake)
    cands = [("a", "alpha", 0.5), ("b", "bravo", 0.5)]
    rerank.rerank("q", cands)
    assert fake.pairs_scored == 2
    rerank.reset_model()             # a model swap must invalidate cached scores
    rerank.rerank("q", cands)
    assert fake.pairs_scored == 4    # recomputed after reset


def test_nul_key_framing_no_collision(monkeypatch):
    fake = _FakeCE()
    _install(monkeypatch, fake)
    # ("a","b\0c") and ("a\0b","c") are DISTINCT CE inputs that would collide
    # under naive query+"\0"+text framing. Length-prefixing keeps them apart.
    rerank.rerank("a", [("d1", "b\x00c", 0.5), ("d2", "filler", 0.5)])
    n = fake.pairs_scored
    rerank.rerank("a\x00b", [("d1", "c", 0.5), ("d2", "filler2", 0.5)])
    assert fake.pairs_scored == n + 2   # both fresh misses; no key collision


def test_reset_during_predict_is_not_cached(monkeypatch):
    fake = _FakeCE()
    _install(monkeypatch, fake)
    q = "generation guard"
    cands = [("a", "alpha", 0.5), ("b", "bravo", 0.5)]
    orig = fake.predict

    def predict_then_reset(pairs, show_progress_bar=False):
        out = orig(pairs, show_progress_bar=show_progress_bar)
        rerank.reset_cache()             # a reset lands mid-"prediction"
        return out

    monkeypatch.setattr(fake, "predict", predict_then_reset)
    out = rerank.rerank(q, cands)
    assert len(out) == 2                 # this call still returns a correct result
    assert len(rerank._CE_CACHE) == 0    # but pre-reset logits were NOT written
    monkeypatch.setattr(fake, "predict", orig)
    before = fake.pairs_scored
    rerank.rerank(q, cands)
    assert fake.pairs_scored == before + 2   # next call recomputes; no stale reuse
