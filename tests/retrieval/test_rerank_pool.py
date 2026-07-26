"""Cross-encoder candidate-pool config (memory_api pool-size constants).

Covers the shipped defaults, the `_env_int` parser, and the invariant that a
low `DCT_RERANK_POOL` override can never starve the prior channel below _TOP_K
(the Codex P1 guard on the 2026-07-22 pool trim).

The pool sizes are module-level constants read from the environment at import,
so these tests reload memory_api under controlled env. The `pool_env` fixture
OWNS the four pool env vars for the duration of a test: it snapshots their
original values, lets the test set them + reload, then restores the exact
originals and reloads again — so a process that already had (say)
DCT_RERANK_POOL set for a real deployment never leaks a module/env mismatch
into later tests (Codex r2 P1)."""
import importlib
import os

import pytest

from dct.retrieval import memory_api as M

_POOL_ENV = ("DCT_RERANK_POOL", "DCT_RERANK_SEM", "DCT_RERANK_TEXT", "DCT_RERANK_BM25")


@pytest.fixture
def pool_env():
    saved = {k: os.environ.get(k) for k in _POOL_ENV}

    def _apply(**overrides):
        # Any pool var not named is cleared, so a pre-configured deployment env
        # can't perturb the assertion; named vars are set (None also clears).
        for k in _POOL_ENV:
            if k in overrides:
                v = overrides[k]
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = str(v)
            else:
                os.environ.pop(k, None)
        return importlib.reload(M)

    try:
        yield _apply
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(M)   # module back in sync with the restored env


def test_default_pool_sizes(pool_env):
    m = pool_env()   # all pool vars cleared -> shipped defaults
    # 2026-07-22 sweep winner: latency floor + recall max.
    assert (m._RERANK_POOL, m._SEM_POOL, m._TEXT_POOL, m._BM25_POOL) == (6, 4, 2, 2)
    assert m._RERANK_POOL >= m._TOP_K


def test_env_int_parser():
    # Pure parser: exercised without reloading the module.
    os.environ["DCT_TMP_POOL"] = "7"
    try:
        assert M._env_int("DCT_TMP_POOL", 3) == 7
        os.environ["DCT_TMP_POOL"] = "not-an-int"   # invalid -> default
        assert M._env_int("DCT_TMP_POOL", 3) == 3
        os.environ["DCT_TMP_POOL"] = "-5"            # clamped to >= 0
        assert M._env_int("DCT_TMP_POOL", 3) == 0
        del os.environ["DCT_TMP_POOL"]               # unset -> default
        assert M._env_int("DCT_TMP_POOL", 3) == 3
    finally:
        os.environ.pop("DCT_TMP_POOL", None)


@pytest.mark.parametrize("bad", ["0", "1", "4"])
def test_prior_pool_floored_at_top_k(pool_env, bad):
    m = pool_env(DCT_RERANK_POOL=bad)
    assert m._RERANK_POOL == m._TOP_K   # never below k, whatever the override


def test_supplementary_channels_env_override(pool_env):
    m = pool_env(DCT_RERANK_SEM=9, DCT_RERANK_BM25=0)   # 0 = disable channel
    assert m._SEM_POOL == 9
    assert m._BM25_POOL == 0
