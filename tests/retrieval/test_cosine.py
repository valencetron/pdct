"""Tests for dct.retrieval.cosine — TF-IDF cosine similarity scorer.

Spec: PDCT v2 P1.2.

The metric pairs the cascade block against the reply text. Higher cosine
means more *rare-term overlap*; lower means topically distant. Tests
focus on the contract:

  * empty inputs → None (undefined)
  * identical text → 1.0 (perfect cosine of identical vectors)
  * disjoint vocabularies → 0.0 (no shared tokens)
  * shared rare words score higher than shared common words
  * stopwords/short tokens excluded
  * paraphrase ranks higher than unrelated text
"""
from __future__ import annotations

import pytest

from dct.retrieval.cosine import cosine_score, _tokenize


# ── _tokenize basics ───────────────────────────────────────────────────


def test_tokenize_drops_stopwords_and_short():
    # 'the', 'and' are stopwords; 'a' is too short.
    out = _tokenize("the cat and a dog")
    assert "the" not in out
    assert "and" not in out
    assert "a" not in out
    # 'cat' and 'dog' are 3 chars (>= MIN_TOKEN_LEN=3) and not stopwords
    assert "cat" in out
    assert "dog" in out


def test_tokenize_lowercases():
    out = _tokenize("PHASE5 Cascade RETRIEVAL")
    assert out == ["phase5", "cascade", "retrieval"]


def test_tokenize_handles_punctuation():
    out = _tokenize("phase5: cascade-retrieval, voice/pipeline.")
    # alpha-num boundaries split everywhere, lowercased
    assert "phase5" in out
    assert "cascade" in out
    assert "retrieval" in out
    assert "voice" in out
    assert "pipeline" in out


def test_tokenize_empty_input():
    assert _tokenize("") == []
    assert _tokenize(None) == []  # type: ignore[arg-type]


# ── cosine_score contract ──────────────────────────────────────────────


def test_cosine_empty_returns_none():
    """Either side empty → None (undefined, distinct from 0.0)."""
    assert cosine_score("", "anything") is None
    assert cosine_score("anything", "") is None
    assert cosine_score("", "") is None


def test_cosine_only_stopwords_returns_none():
    """A side that tokenizes to empty after stopword filtering → None."""
    # "and the are" → all stopwords → no eligible tokens
    assert cosine_score("and the are", "phase5 cascade") is None


def test_cosine_identical_text_is_one():
    """Identical inputs give cosine 1.0 (vectors are colinear)."""
    text = "phase5 cascade retrieval voice pipeline"
    score = cosine_score(text, text)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_cosine_disjoint_vocab_is_zero():
    """No shared eligible tokens → cosine 0.0."""
    score = cosine_score("alpha beta gamma", "delta epsilon zeta")
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cosine_paraphrase_beats_unrelated():
    """A reply that shares topic words ranks higher than one that doesn't."""
    cascade = "voice pipeline retell twilio sip telephony"
    paraphrase_reply = "the retell pipeline handles twilio inbound calls"
    unrelated_reply  = "options spreads volatility delta gamma vega theta"
    score_para = cosine_score(cascade, paraphrase_reply)
    score_unrel = cosine_score(cascade, unrelated_reply)
    assert score_para is not None
    assert score_unrel is not None
    assert score_para > score_unrel


def test_cosine_score_in_unit_interval():
    """Output must always be in [0, 1] for non-empty inputs."""
    pairs = [
        ("phase5 cascade", "phase5 cascade"),                 # identical
        ("phase5 cascade", "voice pipeline"),                 # disjoint
        ("phase5 cascade voice", "voice pipeline phase5"),    # partial
        ("a" * 50, "b" * 50),                                  # short, no overlap (single tokens)
    ]
    for c, r in pairs:
        score = cosine_score(c, r)
        if score is not None:
            assert 0.0 <= score <= 1.0 + 1e-9, (c, r, score)


def test_cosine_partial_overlap_is_between_zero_and_one():
    """Some shared rare tokens, some not → cosine strictly between 0 and 1."""
    cascade = "phase5 cascade retrieval voice pipeline retell"
    reply   = "phase5 retrieval is the cascade design we picked"
    score = cosine_score(cascade, reply)
    assert score is not None
    assert 0.0 < score < 1.0


def test_cosine_does_not_credit_stopwords():
    """Sharing only stopwords/short tokens contributes nothing."""
    # Both sides have only stopwords + short fillers + ONE shared rare token.
    # Compared to a control that shares zero rare tokens, the rare-token
    # version should score strictly higher.
    a_with    = cosine_score("the and is alpha", "the and is alpha")
    a_without = cosine_score("the and is alpha", "the and is omega")
    assert a_with is not None
    assert a_without is not None
    assert a_with > a_without
    # And the "without" case is essentially zero (no rare-term overlap).
    assert a_without == pytest.approx(0.0, abs=1e-9)


def test_cosine_realistic_pdct_block():
    """Smoke-test on a realistic PDCT block + reply pairing."""
    cascade = (
        "## Anchors\n"
        "voice pipeline retell twilio inbound\n"
        "## Today\n"
        "phase5 cascade retrieval shipped\n"
        "## Jogged\n"
        "[[phase5-cascade-retrieval]] [[voice-pipeline]] [[telegram-dispatch]]"
    )
    reply = (
        "shipped phase5 cascade retrieval today; voice pipeline still on retell"
    )
    score = cosine_score(cascade, reply)
    assert score is not None
    # Shared rare tokens: phase5, cascade, retrieval, voice, pipeline, retell — six.
    # Cosine on this pairing should be clearly positive (>0.2).
    assert score > 0.20, f"expected meaningful cosine, got {score}"


def test_cosine_symmetric():
    """cosine_score(a, b) == cosine_score(b, a)."""
    a = "phase5 cascade voice pipeline"
    b = "voice retrieval phase5 cascade today"
    assert cosine_score(a, b) == pytest.approx(cosine_score(b, a), abs=1e-9)
