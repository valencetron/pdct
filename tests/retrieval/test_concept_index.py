"""Concept inverted index — proves the candidate set is an EXHAUSTIVE superset
of every doc that could fuzzy-match a cascade concept. That property is the
whole safety argument: gating the fuzzy match on candidates is byte-identical
to scanning all docs, just O(candidates) instead of O(N)."""
import itertools
import pytest
from dct.retrieval import memory_api as M


@pytest.fixture(autouse=True)
def _clear_index_cache():
    """The inverted index is a module global; clear it around every test so
    one test's cached corpus can't silently satisfy another."""
    M._concept_idx_cache = None
    yield
    M._concept_idx_cache = None


class _Ref:
    def __init__(self, rid, concepts):
        self.id = rid
        self.concepts = concepts
        self.title = ""
        self.gist = ""
        self.date = ""
        self.path = None


def test_candidate_set_is_exhaustive_superset():
    # Vocabulary spanning exact, prefix/suffix, token-overlap, short tokens.
    vocab = ["ayan", "ayan-iep-meeting", "ayan-name-correction", "iep",
             "meeting", "pdct", "pdct-latency", "latency-campaign", "voice",
             "voice-pipeline", "retell", "ab", "ab-test", "cosine",
             "cosine-breaker", "embed-service", "service"]
    index = {f"d{i}": _Ref(f"d{i}", [c]) for i, c in enumerate(vocab)}
    t2d = M._concept_token_index(index)
    holder = {r.concepts[0]: rid for rid, r in index.items()}
    checked = 0
    for cc, rc in itertools.product(vocab, vocab):
        if M._concept_match_strength(cc, rc) > 0:
            checked += 1
            assert holder[rc] in M._candidate_ids(cc, t2d), (
                f"EXHAUSTIVENESS VIOLATED: strength({cc!r},{rc!r})>0 "
                f"but its doc is not a candidate")
    assert checked > 8  # sanity: real matches were exercised, incl. short tokens


def test_multi_concept_doc_indexed_on_every_token():
    index = {"d1": _Ref("d1", ["ayan-iep-meeting", "voice-pipeline"])}
    t2d = M._concept_token_index(index)
    for tok in ("ayan", "iep", "meeting", "voice", "pipeline",
                "ayan-iep-meeting", "voice-pipeline"):
        assert "d1" in t2d.get(tok, set()), f"token {tok!r} did not index d1"


def test_no_shared_token_scores_zero_and_is_not_a_candidate():
    index = {"d1": _Ref("d1", ["orchard-pruning"])}
    t2d = M._concept_token_index(index)
    assert M._concept_match_strength("bronze-casting", "orchard-pruning") == 0.0
    assert "d1" not in M._candidate_ids("bronze-casting", t2d)


def test_concept_edit_same_id_rebuilds():
    # Same doc ID + count, concepts EDITED — the index must rebuild, or a
    # long-lived process serves stale tokens and drops valid matches.
    t1 = M._concept_token_index({"d1": _Ref("d1", ["orchard-pruning"])})
    assert "d1" in t1.get("orchard", set())
    t2 = M._concept_token_index({"d1": _Ref("d1", ["bronze-casting"])})
    assert "d1" in t2.get("bronze", set())   # new concept's token resolves
    assert "orchard" not in t2               # old token is gone (rebuilt)


def test_index_rebuilds_on_corpus_change():
    M._concept_idx_cache = None
    idx1 = {"d1": _Ref("d1", ["alpha"])}
    a = M._concept_token_index(idx1)
    assert M._concept_token_index(idx1) is a  # same corpus -> cached
    idx2 = {"d1": _Ref("d1", ["alpha"]), "d2": _Ref("d2", ["beta-gamma"])}
    c = M._concept_token_index(idx2)
    assert c is not a                          # changed -> rebuilt
    assert "d2" in c.get("beta", set()) and "d2" in c.get("gamma", set())
