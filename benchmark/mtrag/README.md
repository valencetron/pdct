# PDCT × MTRAG cross-corpus harness

Reproduces the public-benchmark study in §7 of the PDCT paper: PDCT's conversational
cascade vs a last-turn baseline on IBM's MTRAG multi-turn RAG benchmark
(Katsis et al. 2025, arXiv:2501.03468), across the FiQA, Govt, and Cloud
passage corpora, with permutation-test significance.

## Data (not redistributed)

MTRAG is IBM's dataset, distributed under its own license at
https://github.com/IBM/mt-rag-benchmark. Fetch what the harness needs
(~60 MB zipped corpora + conversations + retrieval tasks + qrels):

```bash
python -m benchmark.mtrag.fetch_mtrag            # all corpora
python -m benchmark.mtrag.fetch_mtrag fiqa govt  # subset
```

## Run

```bash
# Leg 1: path-memory magnitude (top-5 set change vs last-turn)
# Leg 2: recall/nDCG/MRR vs qrels gold, sliced by depth
python -m benchmark.mtrag.run_mtrag --corpus govt
python -m benchmark.mtrag.analyze_leg2 --corpus govt
```

Statistics (`stats.py`): percentile bootstrap (10k iters) for CIs,
label-permutation tests for depth effects, paired sign-flip permutation
for MRR deltas. Tests: `pytest benchmark/mtrag/tests/`.

## Expected results (paper §7)

| | FiQA | Govt | Cloud |
|---|---|---|---|
| ΔMRR vs last-turn | +0.011 (p=.061) | **+0.041 (p=.014)** | −0.047 (p=.004) |
| Path memory (top-5 reshape) | 0.569 | 0.540 | 0.562 |

The triple dissociation — universal mechanism, corpus-dependent benefit —
is the point: apply path memory where turns are context-dependent,
withhold it where last turns are self-contained.
