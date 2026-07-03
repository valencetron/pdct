"""Lexical-overlap cosine similarity between cascade context and assistant reply.

⚠️ Honest naming (Codex review #5): the file is called "cosine" because
that's the math, but this is *not* corpus-IDF TF-IDF. With a two-document
IDF table, a token that appears in both documents gets IDF=1.0 and a
token in only one gets ~1.4. That ratio is too soft to genuinely
down-weight project boilerplate (e.g. "daemon", "cascade", "phase5")
that recurs in both the cascade block and Claude's replies. Stopwords
ARE filtered (so "the/and/is" don't pollute), but mid-frequency
project terms can still inflate the score.

This is fine for a v1 because:
  * Stopword filtering already kills the worst offenders.
  * The metric is monotonic: shared rare tokens DO push the score up;
    we just don't push them up as hard as a real corpus IDF would.
  * The drop-in slot for sentence-transformers / corpus-IDF is the
    same call signature — same args in, same scalar out.

Replace with embedding cosine (P5 follow-up) when paraphrase resilience
or honest IDF matters. Until then: read this score as "lexical overlap
weighted toward rare-across-the-pair," not "TF-IDF."


Adds a softer lexical-overlap signal alongside `score_turn_utility`'s
verbatim word-boundary regex. Same tokenization rules as
`dct.retrieval.utility` so the two metrics are commensurate. This is
NOT paraphrase-aware — the math is purely lexical TF-IDF on shared
adjacent vocabulary; it only catches paraphrase to the extent that
re-worded text reuses some of the same word stems. Real paraphrase
resilience needs sentence embeddings (P5 follow-up).

This is the *cheap* signal in the composite-score stack:

    composite = α·judge + β·cosine + γ·match_rate + δ·self_rate
                                  ↑
                            (this module)

Free, deterministic, zero-dep, ~20ms on a 5kb block + 2kb reply. Catches
"voice pipeline ↔ Retell stack"-style paraphrase only weakly because we
don't carry semantic embeddings. For real paraphrase resilience, swap
the bag-of-words vector for a sentence-transformers embedding (planned
P5/follow-up — same call signature, different vector source).

Spec: PDCT v2 P1.2 — `pdct-v2-honest-metrics-...-1150b1` card.

Notes on construction:
  * Reply IDF and cascade IDF use the *same* token frequency baseline:
    a smoothed log( (N+1) / (df+1) ) over the union vocabulary of the
    two documents only, biased toward "rare across this turn." We don't
    have a corpus-wide IDF table, so this approximation is the cheapest
    thing that still down-weights filler tokens like "the", "and",
    "list", "files", "code" without an external corpus.
  * Returns None when either side has no eligible tokens — caller logs
    null and skips the row in aggregation. Distinct from "0.0 cosine"
    which means "two non-empty texts share zero rare words."
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from .utility import STOPWORDS, MIN_TOKEN_LEN

# Re-tokenize at the *word* level (not concept-slug level). Word-boundary
# regex matches alpha-num runs, which is what we want for prose comparison.
_WORD_RE = re.compile(r"[a-z0-9]+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase, extract alpha-num tokens, drop short + stopwords.

    Uses the same MIN_TOKEN_LEN and STOPWORDS as `utility.py` so the
    two metrics see the same vocabulary. This makes downstream
    comparisons (cosine vs match_rate) interpretable: a token that
    can't appear in match_rate also can't appear in cosine.
    """
    if not text:
        return []
    return [
        t for t in _WORD_RE.findall(text.lower())
        if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS
    ]


def _tf_vector(tokens: list[str]) -> Counter[str]:
    """Term-frequency Counter from a token list."""
    return Counter(tokens)


def _idf_for_pair(
    a: Counter[str], b: Counter[str],
) -> dict[str, float]:
    """Tiny two-document IDF.

    For each term that appears in either document, df is 1 or 2. We use
    smoothed IDF = log( (N+1) / (df+1) ) + 1.0 with N=2.

      - df=1 (rare across the pair) → log(3/2)+1 ≈ 1.405
      - df=2 (in both docs)         → log(3/3)+1 = 1.000

    The constant +1.0 prevents the pure log term from flipping sign when
    df ≥ N, which would otherwise make terms in both documents *negatively
    weighted* and break the cosine sign convention.
    """
    N = 2
    vocab = set(a) | set(b)
    return {
        t: math.log((N + 1) / ((1 if a.get(t, 0) else 0) + (1 if b.get(t, 0) else 0) + 1)) + 1.0
        for t in vocab
    }


def _tfidf(tf: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    return {t: c * idf[t] for t, c in tf.items()}


def _cosine(u: dict[str, float], v: dict[str, float]) -> float:
    if not u or not v:
        return 0.0
    common = set(u) & set(v)
    dot = sum(u[t] * v[t] for t in common)
    nu = math.sqrt(sum(x * x for x in u.values()))
    nv = math.sqrt(sum(x * x for x in v.values()))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


def cosine_score(cascade_text: str, reply_text: str) -> Optional[float]:
    """Return TF-IDF cosine in [0.0, 1.0], or None if undefined.

    None when either side tokenizes to empty (no eligible tokens) — the
    metric is undefined for the empty document. Caller should log null,
    not 0.0, so aggregation excludes the row.

    Args:
      cascade_text: the PDCT block injected into the prompt — typically
                    the dynamic-section content (Today + Recent + Jogged).
                    Anchors should be excluded by the caller; they're
                    boilerplate and would inflate cosine by their bulk.
      reply_text:   the assistant's full reply for this turn.

    Returns:
      float in [0.0, 1.0]: 1.0 means identical eligible-token TF-IDF
      vectors (which essentially never happens with real prose); 0.0
      means zero overlap on rare terms; values 0.05–0.30 are typical
      for "topic-relevant but freshly worded" replies.
    """
    a = _tokenize(cascade_text)
    b = _tokenize(reply_text)
    if not a or not b:
        return None
    tf_a = _tf_vector(a)
    tf_b = _tf_vector(b)
    idf = _idf_for_pair(tf_a, tf_b)
    return _cosine(_tfidf(tf_a, idf), _tfidf(tf_b, idf))
