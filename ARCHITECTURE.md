# PDCT Architecture

Written for the public repo against the bundled example corpus. Every path
below is relative to `PDCT_HOME` / this repo — no external services required
except the (optional) Anthropic API.

## The idea

Retrieval-augmented systems usually rank memory by query similarity alone.
PDCT adds **path dependence**: what the conversation has already touched
changes what gets retrieved next. A conversation that wandered through
`rate-limit-incident` and `exponential-backoff` should retrieve different
context for the query "how did we fix it?" than one that came via
`sourdough-baking`.

## Data flow

```
conversation turns
      │  (adapters: claude_code, telegram, retell, vault)
      ▼
events.jsonl  ──────────────►  ActivationEngine
  append-only concept events     heat = touch + exponential decay
      │                          + neighbor spread (blast radius)
      ▼
ConceptGraph  ◄── co-occurrence, transition, VEC_NEAR edges
      │
      ▼
retrieval cascade (dct/retrieval/)
  seed extraction → graph walk (depth, decay, score_floor levers)
  → candidate distillations → rerank → top-k rows
      │
      ▼
memory API: query_memory(seed) / read_memory(id)
```

## Components

- **`dct/events.py`, `dct/event_log.py`** — typed, append-only JSONL event
  log. Ops: read/write/traversal/turn/feedback/prune.
- **`dct/activation.py`** — heat model. `DecayConfig(half_life_seconds, radius_hop_cap, radius_falloff)`.
- **`dct/retrieval/`** — the core: `distill_index` (frontmatter-indexed corpus,
  mtime-cached), `cascade` (path-dependent graph walk), `rerank`
  (concept-match aggregation), `memory_api` (`query_memory`/`read_memory`),
  `overrides` (live tuning levers with clamps).
- **`dct/distiller.py`** — LLM batch job turning raw sessions into
  distillation notes. Requires API key.
- **`dct/judge/`** — sampled LLM scoring of retrieval quality; writes
  feedback events that reinforce graph edges.
- **`benchmark/`** — recall@k harness (`eval_v3.py`) with stratified smoke
  sampling (`--smoke N`).
- **`dct/doctor.py`** — install-time self-diagnosis; runs the functional
  stages in a temp sandbox against `examples/`.

## Tuning

Retrieval levers (`cascade_depth`, `cascade_decay`, `cascade_top_k`,
`cascade_score_floor`, `cascade_transitions_bias`) live in
`runtime/pdct-overrides.json` with typed clamps. The upstream project runs an
automated screening loop (single-lever + factorial combos, promote/revert on
composite score); the harness for that ships in `benchmark/` so you can run
your own sweeps.

## Design invariants

1. **Markdown is the storage format.** Memory stays human-readable and
   editable; the system re-indexes on mtime change.
2. **The graph is derived state.** `events.jsonl` + vault are the only truth;
   caches rebuild from them.
3. **Degrade gracefully.** No API key → retrieval still works. No embeddings
   extra → graph falls back to co-occurrence edges. No events log → index
   ranking still works.
