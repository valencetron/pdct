"""Evaluate the PDCT rerank pipeline (pool-trim + B5 + cross-encoder + channels)
on MTRAG passages, the way PDCT actually works: BM25 retrieves a candidate pool,
PDCT reranks it, scored vs gold qrels. Shareable, at-scale, conversational, and
it exercises this session's optimizations (the MTRAG-native ranker doesn't).

Arms:
  BM25             — candidate order (baseline)
  PDCT             — full rerank on the RAW query
  PDCT+rewrite     — full rerank on an LLM-rewritten, concept-dense query
                     (--rewrite; tests whether tuning the incoming query to hit
                      concepts better improves retrieval — Alex's idea)

Usage: PYTHONPATH=src:. python benchmark/mtrag/run_memapi_eval.py --corpus fiqa
       [--limit-q N] [--pool 100] [--rewrite]
"""
import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

_D = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_D / "src")); sys.path.insert(0, str(_D))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from benchmark.mtrag import ingest, keyphrase          # noqa: E402
from dct.retrieval.distill_index import DistillationRef  # noqa: E402
from dct.retrieval import memory_api as M               # noqa: E402
from dct.retrieval.types import ConceptHit              # noqa: E402

_TOK = re.compile(r"[a-z0-9]+")


def _clean(t): return t.replace("|user|:", " ").replace("|agent|:", " ").strip()
def _tok(t): return _TOK.findall(t.lower())


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.d = [_tok(t) for t in texts]
        self.dl = [len(x) for x in self.d]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 1.0
        self.k1, self.b = k1, b
        self.tf = [Counter(x) for x in self.d]
        df = Counter()
        for x in self.d:
            df.update(set(x))
        N = len(self.d)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def top(self, query, k):
        q = _tok(query)
        sc = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            dl = self.dl[i]
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf.get(w, 0.0) * (f * (self.k1 + 1)) / \
                        (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                sc.append((i, s))
        sc.sort(key=lambda x: -x[1])
        return [i for i, _ in sc[:k]]


# --- query rewrite (Alex's idea: tune the incoming query to hit concepts) -----
_REWRITE_PROMPT = (
    "You optimize search queries for a concept-based retrieval system. Rewrite the "
    "question into a concise, information-dense search query that surfaces the key "
    "entities, concepts, and domain terms needed to find the answer. Preserve every "
    "specific detail (names, numbers, conditions). Do NOT answer. Return ONLY the "
    "rewritten query.\n\nQuestion: {q}\n\nRewritten query:")


def rewrite_query(text, client, cache):
    if text in cache:
        return cache[text]
    from dct.llm import resolve_model_id
    resp = client.messages.create(
        model=resolve_model_id("haiku"), max_tokens=200,
        messages=[{"role": "user", "content": _REWRITE_PROMPT.format(q=text)}])
    out = "".join(getattr(b, "text", "") for b in resp.content).strip()
    cache[text] = out or text
    return cache[text]


def recall_mrr(ranked, gold, k):
    rec = 1.0 if (set(ranked[:k]) & gold) else 0.0
    rr = next((1.0 / (i + 1) for i, p in enumerate(ranked) if p in gold), 0.0)
    return rec, rr


def score_row(ranked, gold):
    return (recall_mrr(ranked, gold, 1)[0], recall_mrr(ranked, gold, 5)[0],
            recall_mrr(ranked, gold, 5)[1])


def rank_pdct(query, index):
    qc = keyphrase.extract_query_concepts(query) or []
    hits = [ConceptHit(concept=c, score=1.0, source_slug="query", snippet=query, hop=0)
            for c in qc]
    return [r.id for r in M._aggregate([hits], index, query_text=query)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="fiqa")
    ap.add_argument("--limit-q", type=int, default=None)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--rewrite", action="store_true")
    args = ap.parse_args()

    from dct.retrieval import rerank, vec_index
    rerank._get_model(); vec_index._get_model()

    passages = ingest.load_passages(args.corpus)
    texts = [_clean(p["text"]) for p in passages]
    ids = [str(p["id"]) for p in passages]
    print(f"[bm25] indexing {len(passages)} passages…", file=sys.stderr)
    bm25 = BM25(texts)
    dummy = Path(tempfile.mkdtemp(prefix="memapi-", dir="/tmp")) / "empty.md"
    dummy.write_text("---\n---\n", encoding="utf-8")

    qrels = ingest.load_qrels(args.corpus)
    questions = [q for q in ingest.load_retrieval_tasks(args.corpus, "questions")
                 if q["_id"] in qrels]
    if args.limit_q:
        questions = questions[:args.limit_q]

    client, cache, cache_fp = None, {}, Path(f"/tmp/mtrag-rewrite-{args.corpus}.json")
    if args.rewrite:
        from dct.llm import _client_factory
        client = _client_factory()
        cache = json.loads(cache_fp.read_text()) if cache_fp.exists() else {}
    print(f"[bm25] ready; scoring {len(questions)} questions (rewrite={args.rewrite})…",
          file=sys.stderr)

    arms = {"bm25": [], "pdct": []}
    if args.rewrite:
        arms["pdct_rw"] = []
    for n, q in enumerate(questions):
        gold = qrels[q["_id"]]
        text = _clean(q["text"])
        cand = bm25.top(text, args.pool)
        cand_ids = [ids[i] for i in cand]
        index = {}
        for i in cand:
            c = keyphrase.extract_query_concepts(texts[i]) or []
            index[ids[i]] = DistillationRef(id=ids[i], path=dummy, date="",
                                            title=texts[i][:80], concepts=list(c)[:12],
                                            gist=texts[i][:1400])
        arms["bm25"].append(score_row(cand_ids, gold))
        arms["pdct"].append(score_row(rank_pdct(text, index), gold))
        if args.rewrite:
            rw = rewrite_query(text, client, cache)
            # candidate pool stays BM25(raw) so the ONLY variable is the rerank query
            arms["pdct_rw"].append(score_row(rank_pdct(rw, index), gold))
            if n < 2:
                print(f"  rw: {text[:55]!r} -> {rw[:70]!r}", file=sys.stderr)
        if n and n % 40 == 0:
            print(f"  …{n}/{len(questions)}", file=sys.stderr)
            if args.rewrite:
                cache_fp.write_text(json.dumps(cache))

    if args.rewrite:
        cache_fp.write_text(json.dumps(cache))

    def avg(rows, j): return round(sum(r[j] for r in rows) / len(rows), 4) if rows else 0.0
    print(f"\n=== MTRAG rerank eval: {args.corpus} (n={len(questions)}, pool={args.pool}) ===")
    print(f"{'arm':14} {'recall@1':>9} {'recall@5':>9} {'mrr@5':>7}")
    for name in ("bm25", "pdct", "pdct_rw"):
        if name in arms:
            r = arms[name]
            print(f"{name:14} {avg(r,0):>9} {avg(r,1):>9} {avg(r,2):>7}")


if __name__ == "__main__":
    main()
