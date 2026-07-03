"""Retrieval baselines for benchmarking PDCT against standard RAG approaches.

Each baseline implements the same interface: rank(query) -> list[doc_id]
over the SAME corpus (the distillation vault, via distill_index.build_index).
This isolates the retrieval strategy as the only variable.

Baselines:
  bm25       — Okapi BM25 over full doc text (keyword-only, the lexical floor)
  vector     — bge-small cosine top-k (naive vector RAG, the LangChain default)
  hybrid_rrf — BM25 + vector fused with Reciprocal Rank Fusion (industry standard)
  pdct       — the full PDCT pipeline (graph cascade + multi-channel + CE rerank)

Usage:
    PYTHONPATH=src .venv/bin/python benchmark/baselines.py [--systems bm25,vector,...]

Output: benchmark/.v3-work/baselines-<ts>.json + markdown table to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dct.retrieval.distill_index import build_index  # noqa: E402
from dct.retrieval.distill_index import DistillationRef  # noqa: E402

CANARY = ROOT / "benchmark" / "canary-v1.json"
WORK = ROOT / "benchmark" / ".v3-work"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


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


# ---------------------------------------------------------------- BM25

class BM25:
    """Okapi BM25, standard parameters (k1=1.5, b=0.75). No deps."""

    def __init__(self, ids: list[str], docs: list[list[str]],
                 k1: float = 1.5, b: float = 0.75):
        self.ids = ids
        self.k1, self.b = k1, b
        self.doc_freqs = [Counter(d) for d in docs]
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = sum(self.doc_lens) / max(1, len(docs))
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def rank(self, query: str, k: int = 5) -> list[str]:
        q = _tokenize(query)
        scores = []
        for i, freqs in enumerate(self.doc_freqs):
            s = 0.0
            dl = self.doc_lens[i]
            for t in q:
                if t not in freqs:
                    continue
                tf = freqs[t]
                s += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((s, self.ids[i]))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [rid for s, rid in scores[:k] if s > 0]


# ---------------------------------------------------------------- Vector

class VectorRAG:
    """Naive dense retrieval: bge-small cosine top-k over the same doc text
    PDCT's embed_index uses. This is the 'chunk+embed+top-k' default."""

    def __init__(self, index: dict[str, DistillationRef]):
        from dct.retrieval import embed_index as ei
        self._ei = ei
        self.index = index

    def rank(self, query: str, k: int = 5) -> list[str]:
        scores = self._ei.semantic_scores(query, self.index)
        top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [rid for rid, _ in top]


# ---------------------------------------------------------------- Hybrid RRF

class HybridRRF:
    """BM25 + vector fused via Reciprocal Rank Fusion (k=60, the standard)."""

    def __init__(self, bm25: BM25, vec: VectorRAG, rrf_k: int = 60):
        self.bm25, self.vec, self.rrf_k = bm25, vec, rrf_k

    def rank(self, query: str, k: int = 5) -> list[str]:
        fused: dict[str, float] = {}
        for rank_list in (self.bm25.rank(query, 25), self.vec.rank(query, 25)):
            for pos, rid in enumerate(rank_list):
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (self.rrf_k + pos + 1)
        top = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [rid for rid, _ in top]


# ---------------------------------------------------------------- PDCT

class PDCT:
    def rank(self, query: str, k: int = 5) -> list[str]:
        from dct.retrieval.memory_api import query_memory
        rows = query_memory(query, _surface="baseline-bench")
        return [r.id for r in rows][:k]


# ---------------------------------------------------------------- harness

def evaluate(system, questions: list[dict]) -> dict:
    hit1 = hit5 = 0
    mrr = 0.0
    lats: list[float] = []
    misses = []
    for q in questions:
        t0 = time.monotonic()
        ids = system.rank(q["question"], k=5)
        lats.append(time.monotonic() - t0)
        target = q["source_distillation_id"]
        if ids and ids[0] == target:
            hit1 += 1
        if target in ids:
            hit5 += 1
            mrr += 1.0 / (ids.index(target) + 1)
        else:
            misses.append(q["id"])
    n = len(questions)
    lats.sort()
    return {
        "n": n,
        "recall_at_1": round(hit1 / n, 3),
        "recall_at_5": round(hit5 / n, 3),
        "mrr_at_5": round(mrr / n, 3),
        "latency_p50_s": round(lats[len(lats) // 2], 3),
        "latency_p95_s": round(lats[min(n - 1, int(n * 0.95))], 3),
        "misses": misses,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="bm25,vector,hybrid_rrf,pdct")
    args = ap.parse_args()
    wanted = [s.strip() for s in args.systems.split(",")]

    canary = json.loads(CANARY.read_text())
    questions = [q for q in canary["questions"] if q["category"] != "negative"]

    index = build_index()
    ids = sorted(index.keys())
    print(f"corpus: {len(ids)} distillations · {len(questions)} positive questions",
          file=sys.stderr)

    systems: dict[str, object] = {}
    if {"bm25", "hybrid_rrf"} & set(wanted):
        docs = [_tokenize(_full_text(index[rid])) for rid in ids]
        bm25 = BM25(ids, docs)
        systems["bm25"] = bm25
    if {"vector", "hybrid_rrf"} & set(wanted):
        vec = VectorRAG(index)
        vec.rank("warmup", k=1)  # load model before timing
        systems["vector"] = vec
    if "hybrid_rrf" in wanted:
        systems["hybrid_rrf"] = HybridRRF(systems["bm25"], systems["vector"])  # type: ignore
    if "pdct" in wanted:
        p = PDCT()
        p.rank("warmup", k=1)
        systems["pdct"] = p

    results = {}
    for name in wanted:
        if name not in systems:
            continue
        print(f"running {name}…", file=sys.stderr)
        results[name] = evaluate(systems[name], questions)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = WORK / f"baselines-{ts}.json"
    out.write_text(json.dumps({
        "ts": ts, "corpus_size": len(ids), "canary": canary["version"],
        "results": results,
    }, indent=2))

    # markdown table
    cols = ["recall_at_1", "recall_at_5", "mrr_at_5", "latency_p50_s", "latency_p95_s"]
    print("\n| system | " + " | ".join(c.replace("_at_", "@") for c in cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for name, r in results.items():
        print(f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
