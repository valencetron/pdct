"""Offline secondary analysis over cached leg2 records (no cascade cost).

Mines paper-candidate signals beyond recall@5:
  A. Path-memory magnitude & depth curve (PDCT vs lastturn ranking divergence)
  B. PDCT distinctness from expensive rewrite (top-K overlap, by depth)
  C. MRR / first-relevant rank, paired vs lastturn (bootstrap significance)
  D. Win/tie/loss rate per query (recall@5) — distributional, not just mean
  E. "Path-memory PAYS OFF" conditional: on queries where PDCT diverges from
     lastturn, does recall go UP, DOWN, or stay flat? (the money question)
  F. Turn-index regression: does |PDCT-lastturn divergence| grow with turn?
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict
from benchmark.mtrag import metrics, stats

RES = Path(__file__).resolve().parent / "results"
K = 5


def load(corpus="fiqa"):
    fp = RES / f"leg2_records_{corpus}.jsonl"
    return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]


def _recall(ranked, gold):
    return metrics.recall_at_k(ranked, set(gold), K)


def analyze(corpus="fiqa", seed=0):
    recs = load(corpus)
    out = {"corpus": corpus, "n_records": len(recs)}
    if not recs:
        out["empty"] = True
        return out

    # ---- A. path memory magnitude + depth -------------------------------
    pm = defaultdict(list)
    for r in recs:
        d = 1.0 - metrics.rank_overlap_at_k(r["pdct"], r["lastturn"], K)
        pm["all"].append(d)
        pm[r["depth"]].append(d)
    out["A_path_memory"] = {
        k: {"mean": round(sum(v)/len(v), 4), "ci": _ci(v, seed), "n": len(v)}
        for k, v in pm.items()
    }
    # unpaired early-vs-late significance (Welch-ish via bootstrap of difference)
    if pm["early"] and pm["late"]:
        out["A_depth_diff"] = stats.unpaired_bootstrap_delta(
            pm["late"], pm["early"], seed=seed)

    # ---- B. distinctness from rewrite -----------------------------------
    rb = defaultdict(list)
    for r in recs:
        if r["rewrite"] is not None:  # present variant (possibly empty ranking)
            ov = metrics.rank_overlap_at_k(r["pdct"], r["rewrite"], K)
            rb["all"].append(ov); rb[r["depth"]].append(ov)
    out["B_rewrite_overlap"] = {
        k: {"mean": round(sum(v)/len(v), 4), "n": len(v)} for k, v in rb.items()
    }

    # ---- C. MRR paired vs lastturn --------------------------------------
    mrr_p = [metrics.mrr(r["pdct"], set(r["gold"])) for r in recs]
    mrr_l = [metrics.mrr(r["lastturn"], set(r["gold"])) for r in recs]
    out["C_mrr"] = {
        "pdct": round(sum(mrr_p)/len(mrr_p), 4),
        "lastturn": round(sum(mrr_l)/len(mrr_l), 4),
        "pdct_minus_lastturn": stats.paired_bootstrap_delta(mrr_p, mrr_l, seed=seed),
    }

    # ---- D. win/tie/loss on recall@5 vs lastturn ------------------------
    w = t = l = 0
    for r in recs:
        dp = _recall(r["pdct"], r["gold"]) - _recall(r["lastturn"], r["gold"])
        if dp > 1e-9: w += 1
        elif dp < -1e-9: l += 1
        else: t += 1
    out["D_winloss_vs_lastturn"] = {"win": w, "tie": t, "loss": l,
                                    "win_rate_excl_ties": round(w/max(w+l, 1), 3)}

    # ---- E. does path memory PAY OFF? -----------------------------------
    # split queries by whether PDCT diverged from lastturn; compare recall.
    moved, still = [], []
    moved_gain = []
    for r in recs:
        div = 1.0 - metrics.rank_overlap_at_k(r["pdct"], r["lastturn"], K)
        rp = _recall(r["pdct"], r["gold"]); rl = _recall(r["lastturn"], r["gold"])
        if div > 1e-9:
            moved.append(rp); moved_gain.append(rp - rl)
        else:
            still.append(rp)
    out["E_path_memory_payoff"] = {
        "moved_queries": {"n": len(moved),
                          "pdct_recall_mean": round(sum(moved)/len(moved), 4) if moved else 0,
                          "mean_recall_gain_vs_lastturn": round(sum(moved_gain)/len(moved_gain), 4) if moved_gain else 0,
                          "gain_ci": _ci(moved_gain, seed)},
        "unmoved_queries": {"n": len(still)},
        "interpretation": ("on queries where path memory actually changed the "
                           "ranking, did recall improve? gain_ci excluding 0 => "
                           "path memory helps WHEN IT FIRES"),
    }

    # ---- F. turn-index correlation of divergence ------------------------
    xs = [r["turn"] for r in recs]
    ys = [1.0 - metrics.rank_overlap_at_k(r["pdct"], r["lastturn"], K) for r in recs]
    out["F_turn_divergence_corr"] = {"pearson_r": round(_pearson(xs, ys), 4),
                                     "n": len(xs)}
    return out


def _ci(xs, seed=0):
    if not xs:
        return [0.0, 0.0]
    _, lo, hi = stats.bootstrap_ci(xs, seed=seed)
    return [round(lo, 4), round(hi, 4)]


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    return cov / ((vx*vy) ** 0.5) if vx > 0 and vy > 0 else 0.0


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "fiqa"
    res = analyze(corpus)
    (RES / f"leg2_analysis_{corpus}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
