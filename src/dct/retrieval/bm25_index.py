"""BM25 full-text channel for the union rerank pool.

Measured motivation (2026-06-11 baseline run): plain Okapi BM25 over full
distillation bodies scored recall@1 0.70 on the canary while PDCT's pool
MISSED two targets entirely (the prior's _text_match_boost only sees
title/gist/concepts; rare-token grep needs distinctive identifiers). BM25
catches "many medium-rare words" questions — exactly what those two were.

Pure-python Okapi BM25 (k1=1.5, b=0.75), no deps. Index is cached at module
level and rebuilt when the distillation index stamp changes (same approach
as embed_index mtime stamping, but cheap enough to rebuild in-process:
~1k docs tokenize in <2s, amortized to zero on warm calls).
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter

from dct.retrieval.distill_index import DistillationRef

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_K1 = 1.5
_B = 0.75

_lock = threading.Lock()
_cache: dict | None = None  # {stamp, ids, doc_freqs, doc_lens, avgdl, idf}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _full_text(ref: DistillationRef) -> str:
    parts = [ref.title, ref.gist, " ".join(ref.concepts)]
    try:
        raw = ref.path.read_text(encoding="utf-8", errors="ignore")
        if raw.startswith("---"):
            end = raw.find("---", 3)
            if end != -1:
                raw = raw[end + 3:]
        parts.append(raw)
    except OSError:
        pass
    return "\n".join(p for p in parts if p)


def _stamp(index: dict[str, DistillationRef]) -> tuple:
    return (len(index), hash(frozenset(index.keys())))


def _build(index: dict[str, DistillationRef]) -> dict:
    ids = sorted(index.keys())
    docs = [_tokenize(_full_text(index[rid])) for rid in ids]
    doc_freqs = [Counter(d) for d in docs]
    doc_lens = [len(d) for d in docs]
    avgdl = sum(doc_lens) / max(1, len(docs))
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    return {
        "stamp": _stamp(index), "ids": ids, "doc_freqs": doc_freqs,
        "doc_lens": doc_lens, "avgdl": avgdl, "idf": idf,
    }


def bm25_top(
    query_text: str,
    index: dict[str, DistillationRef],
    k: int = 10,
) -> list[str]:
    """Top-k distillation ids by BM25 over full text. [] on any failure."""
    global _cache
    if not query_text.strip() or not index:
        return []
    try:
        with _lock:
            if _cache is None or _cache["stamp"] != _stamp(index):
                _cache = _build(index)
            c = _cache
        q = _tokenize(query_text)
        scores: list[tuple[float, str]] = []
        for i, freqs in enumerate(c["doc_freqs"]):
            s = 0.0
            dl = c["doc_lens"][i]
            for t in q:
                tf = freqs.get(t)
                if not tf:
                    continue
                s += c["idf"].get(t, 0.0) * tf * (_K1 + 1) / (
                    tf + _K1 * (1 - _B + _B * dl / c["avgdl"]))
            if s > 0:
                scores.append((s, c["ids"][i]))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [rid for _, rid in scores[:k]]
    except Exception:
        return []
