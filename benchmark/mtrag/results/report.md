# PDCT × MTRAG (FiQA) — Generalization Results

## Leg 1 — Path-dependence (same content, different ORDER)

- conversations: **27**
- mean reorder divergence: **0.3343** (fraction of top-10 retrieved passages that change when the SAME middle turns are reordered)
- conversations with nonzero reorder divergence: **18/27** (67%)
- max single-conversation divergence: **0.9474**
- mean permutation-null p95: 0.5729

_Headline: on a public corpus, reordering identical conversational content changes ~33% of retrieved passages — retrieval is path-dependent, and we measure it directly._

## Leg 2 — Recall/nDCG vs gold, by standalone × depth

| slice | arm | recall@5 [95% CI] | ndcg@5 | n | LLM calls |
|---|---|---|---|---|---|
| standalone__early | pdct | 0.0909 [0.0152–0.1818] | 0.077 | 22 | 0 |
| standalone__early | lastturn | 0.0682 [0.0–0.1515] | 0.063 | 22 | 0 |
| standalone__early | rewrite | 0.0682 [0.0–0.1515] | 0.063 | 22 | 165 |
| non_standalone__early | pdct | 0.0435 [0.0072–0.0942] | 0.0545 | 46 | 0 |
| non_standalone__early | lastturn | 0.058 [0.0145–0.1087] | 0.0624 | 46 | 0 |
| non_standalone__early | rewrite | 0.054 [0.0159–0.1011] | 0.0599 | 46 | 165 |
| non_standalone__late | pdct | 0.0464 [0.0215–0.0773] | 0.0419 | 97 | 0 |
| non_standalone__late | lastturn | 0.0407 [0.0162–0.0708] | 0.0363 | 97 | 0 |
| non_standalone__late | rewrite | 0.0548 [0.0273–0.0887] | 0.0544 | 97 | 165 |

### Paired significance (recall@5, PDCT − baseline, 95% bootstrap)

| slice | comparison | Δ | 95% CI | p | significant |
|---|---|---|---|---|---|
| standalone__early | pdct_minus_lastturn | 0.0227 | [0.0–0.0682] | 0.7116 | — no |
| standalone__early | pdct_minus_rewrite | 0.0227 | [0.0–0.0682] | 0.7116 | — no |
| non_standalone__early | pdct_minus_lastturn | -0.0145 | [-0.0362–0.0] | 0.2488 | — no |
| non_standalone__early | pdct_minus_rewrite | -0.0105 | [-0.05–0.0326] | 0.6096 | — no |
| non_standalone__late | pdct_minus_lastturn | 0.0057 | [-0.0153–0.0271] | 0.5936 | — no |
| non_standalone__late | pdct_minus_rewrite | -0.0084 | [-0.033–0.0146] | 0.4826 | — no |

- headline slice: **non_standalone__late**
- cost: pdct/lastturn = **0 LLM calls**, rewrite = **165 LLM calls**
- skipped: {'unjoined_standalone': 15}