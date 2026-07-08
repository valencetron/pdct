# PDCT × MTRAG — Cross-Corpus Results (Leg 2 secondary analysis)

Three MTRAG passage corpora, all retrieval performed at **zero LLM cost** for PDCT
and last-turn arms (rewrite arm shown elsewhere for cost contrast). p-values are from
**label-permutation tests** (unpaired) for depth and **paired sign-flip permutation**
for MRR — not bootstrap percentile (corrected after Codex audit, commit `ad29aa0`).

## The headline: a triple dissociation

Path memory — the fraction of the top-5 retrieved set that PDCT reshapes relative to a
naive last-turn query — is **universal**: all three corpora reshape ~54–57% of the top-5.
But what that reshaping *does to retrieval quality* is entirely corpus-dependent, ranging
from a significant gain (Govt) through neutral (FiQA) to a significant **harm** (Cloud).

| Signal | FiQA | Govt | Cloud |
|---|---|---|---|
| n (query records) | 165 | 190 | 177 |
| **Path memory active** (mean top-5 set change) | 0.569 | 0.540 | 0.562 |
| early → late depth | 0.483 → 0.628 | 0.513 → 0.556 | 0.454 → 0.634 |
| **Grows with depth** (Δ, perm p) | +0.145, **p=.013 ✅** | +0.043, p=.44 — | +0.180, **p=.002 ✅** |
| **MRR** PDCT vs last-turn (Δ, perm p) | +0.011, p=.061 ~ | +0.041, **p=.014 ✅** | −0.047, **p=.004 ❌** |
| Win / tie / loss vs last-turn | 6 / 153 / 6 | 18 / 165 / 7 | 4 / 160 / 13 |
| **Win rate** (decided queries) | 50% | **72%** | **23.5%** |
| Recall payoff when path memory fires (Δrecall@5, CI) | +0.003 [−.016,.022] — | **+0.035 [.005,.066] ✅** | **−0.037 [−.070,−.008] ❌** |
| Turn-divergence correlation (Pearson r) | 0.220 | 0.068 | 0.200 |

## Three claims, each reviewer-proof

**1. Path-dependence is real and measurable at zero cost — everywhere.**
On every corpus, swapping a last-turn query for PDCT's path-aware retrieval reshapes
more than half the top-5 (0.54–0.57). PDCT *always changes the retrieval*; the mechanism
is not corpus-specific.

**2. Depth-scaling of path memory is real but does NOT generalize to all corpora,
and critically does NOT predict benefit.**
The "path memory grows as the conversation deepens" effect is significant on FiQA
(p=.013) and Cloud (p=.002) but absent on Govt (p=.44). And the decoupling is the point:
**Cloud has the *strongest* depth effect yet the *worst* quality outcome.** So depth-scaling
is a mechanism signature, not a benefit guarantee — this kills the naive "more path memory =
better retrieval" intuition before a reviewer can.

**3. Whether path memory helps is governed by how much conversational context
disambiguates retrieval — and we can see it three ways.**
- **Govt** (procedural, multi-turn, context-dependent questions): PDCT lifts MRR
  +0.041 (p=.014), wins 72% of decided queries, and delivers a real recall gain
  (+0.035, CI excludes 0) when path memory fires. **PDCT pays off.**
- **FiQA** (finance Q&A, near-standalone turns): neutral. MRR trend +0.011 is not
  significant (p=.061), 50/50 win rate, zero recall payoff. **PDCT is quality-neutral —
  it reshapes retrieval but doesn't move the gold.**
- **Cloud** (technical docs where the last turn is often already self-contained and
  specific): PDCT *hurts* — MRR −0.047 (p=.004), wins only 23.5%, recall payoff
  −0.037. **Path memory injects stale context that displaces a good standalone query.**

## Why this is the right story to tell

A single-corpus "we win" claim invites the obvious reviewer rejoinder ("cherry-picked
corpus / metric"). The triple dissociation pre-empts it: we report a corpus where PDCT
loses, explain *why* (self-contained last turns mean path context is noise), and show the
mechanism (path memory magnitude) is decoupled from the benefit. The contribution becomes
**a diagnostic**: PDCT is a zero-cost lever whose value is predictable from corpus
structure — apply it where turns are context-dependent (Govt-like), withhold it where
turns are self-contained (Cloud-like).

## Caveats (state these in the paper)

- Decided-query counts are modest (Govt 25, Cloud 17, FiQA 12); win-rate is suggestive,
  the MRR/payoff permutation CIs are the firmer evidence.
- All corpora load with **0 missing gold** — recall@5 is not artificially capped on any.
- Depth split uses the late-turn threshold from `run_mtrag.py`; early/late n shown above.
- A natural next probe: regress per-corpus benefit on a "last-turn self-containedness"
  score to make claim 3 quantitative rather than interpretive.
