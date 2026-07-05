"""Passage/query -> concept slugs via TF-IDF n-gram keyphrase extraction."""
from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z-]+")


def slugify(s: str) -> str:
    return _SLUG_RE.sub("-", s.lower()).strip("-")


class CorpusExtractor:
    def __init__(self, docs: list[str], top_k: int = 8, ngram=(1, 3),
                 min_df: int = 2, max_df: float = 0.5):
        self.top_k = top_k
        n = len(docs)
        _min_df = min_df if n > 5 else 1
        _max_df = max_df if n > 5 else 1.0
        self.vec = TfidfVectorizer(
            ngram_range=ngram, stop_words="english",
            min_df=_min_df, max_df=_max_df, token_pattern=r"[A-Za-z][A-Za-z-]+")
        self.vec.fit(docs)
        self._vocab = self.vec.get_feature_names_out()
        self._vocab_set = set(self._vocab)

    def extract(self, text: str, top_k: int | None = None) -> list[str]:
        k = top_k or self.top_k
        row = self.vec.transform([text]).tocoo()
        ranked = sorted(zip(row.col, row.data), key=lambda x: -x[1])
        seen, out = set(), []
        for col, _w in ranked:
            slug = slugify(self._vocab[col])
            if slug and slug not in seen:
                seen.add(slug)
                out.append(slug)
            if len(out) >= k:
                break
        return out


def extract_query_concepts(text: str, extractor: "CorpusExtractor | None" = None,
                           top_k: int = 6) -> list[str]:
    """Query-side seed extraction. If a fitted extractor is given, use its TF-IDF
    ranking; else fall back to slugged content tokens (length>2, deduped)."""
    if extractor is not None:
        cs = extractor.extract(text, top_k=top_k)
        if cs:
            return cs
    seen, out = set(), []
    for tok in _TOKEN.findall(text.lower()):
        if len(tok) <= 2:
            continue
        s = slugify(tok)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= top_k:
            break
    return out
