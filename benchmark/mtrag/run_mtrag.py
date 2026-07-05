"""MTRAG generalization driver.

Leg 1 — path-dependence: same-content/different-ORDER divergence within each
conversation vs an exhaustive permutation null. Isolates order from content.

Leg 2 — Recall/nDCG vs qrels gold, sliced by standalone x depth, across three
arms that share ONE adapter so differences are purely seeds/path:
  - pdct     : replay cumulative USER turns through MtragCascade (path memory).
  - lastturn : single-turn cascade seeded only from the final user turn.
  - rewrite  : single-turn cascade seeded from the provided LLM rewrite text.
Cost: pdct/lastturn make 0 LLM calls; rewrite makes 1 (the rewrite itself).
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict

from benchmark.mtrag import ingest, build_graph, cascade, adapter, metrics, join, stats

MIN_MIDDLE = 3
MAX_MIDDLE = 4
CLOSER = "summarize the key points we covered"
LATE_TURN_THRESHOLD = 4
TOP_N = 10
K = 5

_GRAPH_CACHE: dict = {}


def _get_graph_adapter(corpus: str):
    if corpus not in _GRAPH_CACHE:
        import pickle
        from pathlib import Path
        cache_fp = Path(__file__).resolve().parent / "results" / f"_graph_{corpus}.pkl"
        if cache_fp.exists():
            with open(cache_fp, "rb") as fh:
                g, A = pickle.load(fh)
        else:
            passages, missing = ingest.load_passages_with_gold(corpus)
            if missing:
                print(f"WARNING: {len(missing)} gold passages absent from corpus "
                      f"(recall capped)", file=sys.stderr)
            g = build_graph.build(passages, top_k=8)
            A = adapter.PassageAdapter(g)
            try:
                with open(cache_fp, "wb") as fh:
                    pickle.dump((g, A), fh)
            except Exception as e:
                print(f"(graph cache write skipped: {e})", file=sys.stderr)
        _GRAPH_CACHE[corpus] = (g, A)
    return _GRAPH_CACHE[corpus]


def _user_turns(c):
    return [m["text"] for m in c["messages"] if m["speaker"] == "user"]


# ---------------------------------------------------------------- Leg 1

def _play_passages(turns, g, A, top_n=TOP_N):
    """Final-turn RANKED PASSAGE set. Uses activation WEIGHTS (order-dependent),
    not concept-set membership (which is order-invariant under max-deposit +
    floor — measuring it would mask the path effect on retrieval)."""
    cc = cascade.MtragCascade(g)
    cc.reset()
    passages = set()
    for t in turns:
        r = cc.turn(t)
        passages = {pid for pid, _ in A.rank(r["activation"], top_n=top_n)}
    return passages


def _eligible(convos):
    return [c for c in convos if len(_user_turns(c)) >= MIN_MIDDLE + 1]


def leg1(corpus="fiqa", n_convos=27, seed=0):
    g, A = _get_graph_adapter(corpus)
    convos = _eligible(ingest.load_conversations(corpus=corpus))[:n_convos]
    rng = random.Random(seed)
    per_convo = []
    real_divs, null_p95s, exceed = [], [], 0
    for c in convos:
        ut = _user_turns(c)
        opener, middle = ut[0], ut[1:1 + MAX_MIDDLE]
        if len(middle) < MIN_MIDDLE:
            continue
        play = lambda turns: _play_passages(turns, g, A)
        # bounded distinct-permutation search (Codex P1: avoid infinite loop when
        # all middle turns are identical — then no distinct perm exists).
        permB = middle[:]
        for _ in range(50):
            rng.shuffle(permB)
            if permB != middle:
                break
        if permB == middle:
            # degenerate: identical middle turns, reorder is a no-op → skip convo
            continue
        ref_c = play([opener] + middle + [CLOSER])
        b_c = play([opener] + permB + [CLOSER])
        real = metrics.divergence(ref_c, b_c)
        # MAX_MIDDLE=4 -> 4!-1 = 23 distinct non-reference perms; max_perms=24
        # keeps the null EXHAUSTIVE for every eligible conversation.
        null, n_perms = metrics.permutation_null_divergence(
            play, middle, [opener], [CLOSER], max_perms=24, seed=seed)
        p95 = metrics.percentile(null, 95)
        real_divs.append(real)
        null_p95s.append(p95)
        if real > p95:
            exceed += 1
        per_convo.append({"opener": opener[:60], "real_divergence": round(real, 4),
                          "null_p95": round(p95, 4), "n_perms": n_perms,
                          "middle_len": len(middle)})
    k = max(len(per_convo), 1)
    return {
        "leg": 1, "corpus": corpus, "n_convos_used": len(per_convo),
        "mean_real_divergence": round(sum(real_divs) / k, 4),
        "mean_null_p95": round(sum(null_p95s) / k, 4),
        "frac_exceed_own_null_p95": round(exceed / k, 3),
        "per_convo": per_convo,
    }


# ---------------------------------------------------------------- Leg 2

def slice_key(is_standalone: bool, turn_idx: int):
    return (("standalone" if is_standalone else "non_standalone"),
            ("late" if turn_idx >= LATE_TURN_THRESHOLD else "early"))


def _rank_ids(activation, A):
    return [pid for pid, _ in A.rank(activation, top_n=TOP_N)]


def _pdct_ranking(user_lines, g, A):
    cc = cascade.MtragCascade(g)
    cc.reset()
    act = {}
    for t in user_lines:
        r = cc.turn(t)
        act = r["activation"]
    return _rank_ids(act, A)


def _single_turn_ranking(text, g, A):
    cc = cascade.MtragCascade(g)
    cc.reset()
    r = cc.turn(text)
    return _rank_ids(r["activation"], A)


def leg2(corpus="fiqa", n_convos=27, seed=0):
    g, A = _get_graph_adapter(corpus)
    qrels = ingest.load_qrels(corpus)
    questions = {r["_id"]: r["text"] for r in ingest.load_retrieval_tasks(corpus, "questions")}
    lastturn = {r["_id"]: r["text"] for r in ingest.load_retrieval_tasks(corpus, "lastturn")}
    rewrite = {r["_id"]: r["text"] for r in ingest.load_retrieval_tasks(corpus, "rewrite")}
    convos = ingest.load_conversations(corpus=corpus)
    sidx = join.build_standalone_index(convos)

    # Codex P1: scope the query set to the first n_convos conversations.
    # A query belongs to a conversation if its first user line matches that
    # conversation's normalized first user message.
    allowed_first = {join._norm(_user_turns(c)[0]) for c in convos[:n_convos]
                     if _user_turns(c)}

    # accumulators: slice -> arm -> list of (recall, ndcg)
    acc: dict = defaultdict(lambda: defaultdict(list))
    # diagnostics
    nonempty: dict = defaultdict(lambda: defaultdict(int))
    counts: dict = defaultdict(lambda: defaultdict(int))
    skipped = defaultdict(int)
    rewrite_calls = 0
    seen_qids = set()

    for qid, qtext in questions.items():
        if qid in seen_qids:
            skipped["duplicate_qid"] += 1
            continue
        seen_qids.add(qid)
        ul = join.split_user_turns(qtext)
        if ul and join._norm(ul[0]) not in allowed_first:
            skipped["out_of_scope_convo"] += 1
            continue
        gold = qrels.get(qid)
        if not gold:
            skipped["no_gold"] += 1
            continue
        _cid, turn = join.parse_qid(qid)
        sb = join.standalone_for(qtext, turn, sidx)
        if sb is None:
            skipped["unjoined_standalone"] += 1
            continue
        sk = slice_key(sb, turn)
        user_lines = join.split_user_turns(qtext)
        if not user_lines:
            skipped["empty_user_lines"] += 1
            continue

        # PDCT arm — cumulative replay, 0 LLM calls
        pdct_r = _pdct_ranking(user_lines, g, A)
        # lastturn arm — final user turn only, 0 LLM calls
        lt_text = lastturn.get(qid, user_lines[-1])
        lt_r = _single_turn_ranking(lt_text, g, A)
        # rewrite arm — provided rewrite text, 1 LLM call (precomputed by MTRAG)
        rw_text = rewrite.get(qid)
        if rw_text is None:
            skipped["no_rewrite_variant"] += 1
            rw_r = []
        else:
            rewrite_calls += 1
            rw_r = _single_turn_ranking(rw_text, g, A)

        for arm, ranked in (("pdct", pdct_r), ("lastturn", lt_r), ("rewrite", rw_r)):
            acc[sk][arm].append((metrics.recall_at_k(ranked, gold, K),
                                 metrics.ndcg_at_k(ranked, gold, K)))
            counts[sk][arm] += 1
            if ranked:
                nonempty[sk][arm] += 1

    # summarize — per-arm means with bootstrap CIs, plus PAIRED deltas vs gold.
    out_slices = {}
    for sk, arms in acc.items():
        recalls = {arm: [r for r, _ in v] for arm, v in arms.items()}
        ndcgs = {arm: [n for _, n in v] for arm, v in arms.items()}
        arm_summ = {}
        for arm, v in arms.items():
            rm, rlo, rhi = stats.bootstrap_ci(recalls[arm], seed=seed)
            nm, nlo, nhi = stats.bootstrap_ci(ndcgs[arm], seed=seed)
            arm_summ[arm] = {
                "recall@5": round(rm, 4), "recall@5_ci": [round(rlo, 4), round(rhi, 4)],
                "ndcg@5": round(nm, 4), "ndcg@5_ci": [round(nlo, 4), round(nhi, 4)],
                "n": len(v),
                "nonempty_rate": round(nonempty[sk][arm] / max(counts[sk][arm], 1), 3),
            }
        # paired deltas (recall@5): pdct vs each baseline on the SAME queries
        deltas = {}
        if "pdct" in recalls:
            for base in ("lastturn", "rewrite"):
                if base in recalls and len(recalls[base]) == len(recalls["pdct"]):
                    deltas[f"pdct_minus_{base}"] = stats.paired_bootstrap_delta(
                        recalls["pdct"], recalls[base], seed=seed)
        out_slices["__".join(sk)] = {"arms": arm_summ, "recall_deltas": deltas}
    return {
        "leg": 2, "corpus": corpus,
        "slices": out_slices,
        "headline_slice": "non_standalone__late",
        "cost": {"pdct_llm_calls": 0, "lastturn_llm_calls": 0,
                 "rewrite_llm_calls": rewrite_calls},
        "skipped": dict(skipped),
    }


def leg2_dump(corpus="fiqa", n_convos=27):
    """Run the cascade ONCE and write raw per-query records to JSONL so every
    secondary analysis (path memory, MRR, rewrite-divergence, depth curves,
    ablations) can be computed offline in <1s without re-paying cascade cost.
    Record = {qid, cid, turn, depth, standalone, gold, pdct, lastturn, rewrite}
    where pdct/lastturn/rewrite are the TOP_N ranked passage-id lists."""
    from pathlib import Path
    g, A = _get_graph_adapter(corpus)
    qrels = ingest.load_qrels(corpus)
    questions = {r["_id"]: r["text"] for r in ingest.load_retrieval_tasks(corpus, "questions")}
    lastturn = {r["_id"]: r["text"] for r in ingest.load_retrieval_tasks(corpus, "lastturn")}
    rewrite = {r["_id"]: r["text"] for r in ingest.load_retrieval_tasks(corpus, "rewrite")}
    convos = ingest.load_conversations(corpus=corpus)
    sidx = join.build_standalone_index(convos)
    allowed_first = {join._norm(_user_turns(c)[0]) for c in convos[:n_convos]
                     if _user_turns(c)}
    out_fp = Path(__file__).resolve().parent / "results" / f"leg2_records_{corpus}.jsonl"
    n = 0
    seen = set()
    with open(out_fp, "w") as fh:
        for qid, qtext in questions.items():
            if qid in seen:
                continue
            seen.add(qid)
            ul = join.split_user_turns(qtext)
            if not ul or join._norm(ul[0]) not in allowed_first:
                continue
            gold = qrels.get(qid)
            if not gold:
                continue
            cid, turn = join.parse_qid(qid)
            sb = join.standalone_for(qtext, turn, sidx)
            if sb is None:
                continue
            rw_text = rewrite.get(qid)
            rec = {
                "qid": qid, "cid": cid, "turn": turn,
                "depth": "late" if turn >= LATE_TURN_THRESHOLD else "early",
                "standalone": bool(sb),
                "n_user_turns": len(ul),
                "gold": sorted(gold),
                "pdct": _pdct_ranking(ul, g, A),
                "lastturn": _single_turn_ranking(lastturn.get(qid, ul[-1]), g, A),
                # match leg2 arm-inclusion: a present (even empty-string)
                # rewrite variant is evaluated; only a MISSING variant is None.
                "rewrite": (_single_turn_ranking(rw_text, g, A)
                            if rw_text is not None else None),
            }
            fh.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} records -> {out_fp}", file=sys.stderr)
    return {"records": n, "path": str(out_fp)}


def leg2x(corpus="fiqa", n_convos=27, seed=0):
    """Secondary PDCT signal analysis (no extra LLM cost) — runs the same
    cascade as leg2 but captures per-query rankings and derives:
      - path_memory: how much PDCT's ranking diverges from lastturn (1 - top-K
        rank overlap). 0 => path memory inert; >0 => carried activation moves
        retrieval. Reported overall and BY DEPTH (early vs late) to test the
        thesis that path memory grows with conversation depth.
      - pdct_vs_rewrite_overlap: does the 0-call PDCT ranking approximate the
        165-call rewrite ranking? (cost-efficiency story)
      - mrr: first-relevant rank per arm (PDCT may surface gold higher even at
        equal recall@5)."""
    g, A = _get_graph_adapter(corpus)
    qrels = ingest.load_qrels(corpus)
    questions = {r["_id"]: r["text"] for r in ingest.load_retrieval_tasks(corpus, "questions")}
    lastturn = {r["_id"]: r["text"] for r in ingest.load_retrieval_tasks(corpus, "lastturn")}
    rewrite = {r["_id"]: r["text"] for r in ingest.load_retrieval_tasks(corpus, "rewrite")}
    convos = ingest.load_conversations(corpus=corpus)
    sidx = join.build_standalone_index(convos)
    allowed_first = {join._norm(_user_turns(c)[0]) for c in convos[:n_convos]
                     if _user_turns(c)}

    # per-depth accumulators
    pm_by_depth = defaultdict(list)          # depth -> [1 - overlap(pdct,lastturn)]
    pr_overlap_by_depth = defaultdict(list)  # depth -> [overlap(pdct,rewrite)]
    mrr_acc = defaultdict(list)              # arm -> [mrr]
    pm_all, pr_all = [], []
    seen = set()
    for qid, qtext in questions.items():
        if qid in seen:
            continue
        seen.add(qid)
        ul = join.split_user_turns(qtext)
        if not ul or join._norm(ul[0]) not in allowed_first:
            continue
        gold = qrels.get(qid)
        if not gold:
            continue
        _cid, turn = join.parse_qid(qid)
        depth = "late" if turn >= LATE_TURN_THRESHOLD else "early"

        pdct_r = _pdct_ranking(ul, g, A)
        lt_r = _single_turn_ranking(lastturn.get(qid, ul[-1]), g, A)
        rw_text = rewrite.get(qid)
        rw_r = _single_turn_ranking(rw_text, g, A) if rw_text else []

        pm = 1.0 - metrics.rank_overlap_at_k(pdct_r, lt_r, K)
        pm_all.append(pm)
        pm_by_depth[depth].append(pm)
        if rw_r:
            ov = metrics.rank_overlap_at_k(pdct_r, rw_r, K)
            pr_all.append(ov)
            pr_overlap_by_depth[depth].append(ov)
        mrr_acc["pdct"].append(metrics.mrr(pdct_r, gold))
        mrr_acc["lastturn"].append(metrics.mrr(lt_r, gold))
        if rw_r:
            mrr_acc["rewrite"].append(metrics.mrr(rw_r, gold))

    def _m(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    def _ci(xs):
        if not xs:
            return [0.0, 0.0]
        _, lo, hi = stats.bootstrap_ci(xs, seed=seed)
        return [round(lo, 4), round(hi, 4)]

    return {
        "leg": "2x", "corpus": corpus,
        "path_memory_vs_lastturn": {
            "overall_mean_divergence": _m(pm_all), "ci": _ci(pm_all),
            "by_depth": {d: {"mean": _m(v), "ci": _ci(v), "n": len(v)} for d, v in pm_by_depth.items()},
            "interpretation": ("fraction of top-5 retrieved passages PDCT changes "
                               "vs single-turn lastturn — nonzero => path memory "
                               "actively reshapes retrieval"),
        },
        "pdct_approximates_rewrite": {
            "overall_mean_top5_overlap": _m(pr_all),
            "by_depth": {d: {"mean": _m(v), "n": len(v)} for d, v in pr_overlap_by_depth.items()},
            "interpretation": ("top-5 overlap between 0-call PDCT and 165-call "
                               "rewrite — high => PDCT reproduces expensive "
                               "rewrite retrieval for free"),
        },
        "mrr": {arm: {"mean": _m(v), "n": len(v)} for arm, v in mrr_acc.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", default="1")
    ap.add_argument("--corpus", default="fiqa")
    ap.add_argument("--n-convos", type=int, default=27)
    args = ap.parse_args()
    if args.leg == "1":
        print(json.dumps(leg1(args.corpus, args.n_convos), indent=2))
    elif args.leg == "2":
        print(json.dumps(leg2(args.corpus, args.n_convos), indent=2))
    elif args.leg == "2x":
        print(json.dumps(leg2x(args.corpus, args.n_convos), indent=2))
    elif args.leg == "2dump":
        print(json.dumps(leg2_dump(args.corpus, args.n_convos), indent=2))


if __name__ == "__main__":
    main()
