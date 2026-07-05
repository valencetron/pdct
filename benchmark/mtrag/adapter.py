"""Rank MTRAG passages from a cascade activation field.
score(passage) = sum over activated concepts c near passage of
                 activation[c] * bm25_affinity(c, passage).
Candidate set = union of inverted-index passages for activated concepts.
Codex P0-2: if that candidate set is empty, fall back to BM25 over ALL passages
using the concept slugs as the query (guarantees non-empty ranking on vague
non-standalone turns)."""
from __future__ import annotations
import math
import re
from collections import defaultdict
from benchmark.mtrag.build_graph import MtragGraph

_TOK = re.compile(r"[a-z0-9]+")


def _toks(s: str) -> list[str]:
    return _TOK.findall(s.lower())


class PassageAdapter:
    def __init__(self, g: MtragGraph, k1: float = 1.5, b: float = 0.75):
        self.g = g
        self.k1 = k1
        self.b = b
        self._df: dict[str, int] = defaultdict(int)
        self._len: dict[str, int] = {}
        self._tf: dict[str, dict[str, int]] = {}
        for pid, text in g.passage_text.items():
            tf: dict[str, int] = defaultdict(int)
            for t in _toks(text):
                tf[t] += 1
            self._tf[pid] = tf
            self._len[pid] = sum(tf.values()) or 1
            for t in tf:
                self._df[t] += 1
        self._N = max(len(self._tf), 1)
        self._avgdl = (sum(self._len.values()) / self._N) if self._tf else 1.0

    def _bm25(self, concept: str, pid: str) -> float:
        score = 0.0
        tf = self._tf.get(pid, {})
        for term in concept.split("-"):
            f = tf.get(term, 0)
            if f == 0:
                continue
            df = self._df.get(term, 0) or 1
            idf = math.log(1 + (self._N - df + 0.5) / (df + 0.5))
            denom = f + self.k1 * (1 - self.b + self.b * self._len[pid] / self._avgdl)
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def rank(self, activation: dict[str, float], top_n: int = 10,
             max_concepts: int = 12, max_candidates: int = 4000):
        if not activation:
            return []
        # Rank only the strongest activated concepts (the cascade's real signal)
        # — bounds the candidate set so ranking stays fast on a 61k-passage corpus
        # and matches retrieval reality (weak tail activations don't drive recall).
        top = dict(sorted(activation.items(), key=lambda kv: -kv[1])[:max_concepts])
        activation = top
        cand: set[str] = set()
        for c in activation:
            cand |= self.g.concept_to_passages.get(c, set())
            if len(cand) > max_candidates:
                break
        scores: dict[str, float] = defaultdict(float)
        for pid in cand:
            for c, w in activation.items():
                if pid in self.g.concept_to_passages.get(c, set()):
                    aff = self._bm25(c, pid) or 1.0  # presence floor
                    scores[pid] += w * aff
        # Codex P0-2 fallback: empty candidate set -> BM25 over ALL passages.
        if not scores:
            for pid in self._tf:
                s = 0.0
                for c, w in activation.items():
                    s += w * self._bm25(c, pid)
                if s > 0:
                    scores[pid] = s
        return sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
