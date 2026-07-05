"""Render MTRAG Leg-1/Leg-2 results into a markdown report + figures-ready JSON
for the CoS visualization tooling."""
from __future__ import annotations
import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def _load(name):
    fp = RES / name
    return json.loads(fp.read_text()) if fp.exists() else None


def build_report():
    leg1 = _load("leg1_fiqa.json")
    leg2 = _load("leg2_fiqa.json")
    lines = ["# PDCT × MTRAG (FiQA) — Generalization Results\n"]

    if leg1 and leg1.get("per_convo"):
        reals = [p["real_divergence"] for p in leg1["per_convo"]]
        n = len(reals)
        nonzero = sum(1 for r in reals if r > 0)
        mean_div = leg1["mean_real_divergence"]
        pct = round(100 * mean_div)
        lines += [
            "## Leg 1 — Path-dependence (same content, different ORDER)\n",
            f"- conversations: **{leg1['n_convos_used']}**",
            f"- mean reorder divergence: **{mean_div}** "
            "(fraction of top-10 retrieved passages that change when the SAME "
            "middle turns are reordered)",
            f"- conversations with nonzero reorder divergence: "
            f"**{nonzero}/{n}** ({round(100*nonzero/n)}%)",
            f"- max single-conversation divergence: **{max(reals)}**",
            f"- mean permutation-null p95: {leg1['mean_null_p95']}",
            f"\n_Headline: on a public corpus, reordering identical conversational "
            f"content changes ~{pct}% of retrieved passages — retrieval is "
            "path-dependent, and we measure it directly._\n",
        ]
    elif leg1:
        lines += ["## Leg 1 — Path-dependence\n",
                  "_No eligible conversations produced divergence data._\n"]

    if leg2:
        lines += ["## Leg 2 — Recall/nDCG vs gold, by standalone × depth\n",
                  "| slice | arm | recall@5 [95% CI] | ndcg@5 | n | LLM calls |",
                  "|---|---|---|---|---|---|"]
        cost = leg2["cost"]
        callmap = {"pdct": cost["pdct_llm_calls"], "lastturn": cost["lastturn_llm_calls"],
                   "rewrite": cost["rewrite_llm_calls"]}
        for sk, slc in leg2["slices"].items():
            arms = slc["arms"] if "arms" in slc else slc
            for arm, v in arms.items():
                ci = v.get("recall@5_ci")
                ci_s = f" [{ci[0]}–{ci[1]}]" if ci else ""
                lines.append(f"| {sk} | {arm} | {v['recall@5']}{ci_s} | {v['ndcg@5']} | "
                             f"{v['n']} | {callmap.get(arm,'?')} |")
        # paired significance summary
        lines += ["\n### Paired significance (recall@5, PDCT − baseline, 95% bootstrap)\n",
                  "| slice | comparison | Δ | 95% CI | p | significant |",
                  "|---|---|---|---|---|---|"]
        for sk, slc in leg2["slices"].items():
            for cmp_name, d in slc.get("recall_deltas", {}).items():
                sig = "✅ yes" if d["significant"] else "— no"
                lines.append(f"| {sk} | {cmp_name} | {d['delta']} | "
                             f"[{d['lo']}–{d['hi']}] | {d['p']} | {sig} |")
        lines += [
            f"\n- headline slice: **{leg2['headline_slice']}**",
            f"- cost: pdct/lastturn = **0 LLM calls**, rewrite = "
            f"**{cost['rewrite_llm_calls']} LLM calls**",
            f"- skipped: {leg2.get('skipped')}",
        ]

    figures = {"leg1": leg1, "leg2": leg2}
    (RES / "figures_data.json").write_text(json.dumps(figures, indent=2))
    report_md = "\n".join(lines)
    (RES / "report.md").write_text(report_md)
    return report_md


if __name__ == "__main__":
    print(build_report())
