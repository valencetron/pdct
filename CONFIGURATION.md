# PDCT Configuration

All paths resolve through `src/dct/config.py` with this precedence:

1. Specific env var (e.g. `PDCT_EVENTS_PATH`)
2. `PDCT_HOME` (all defaults nest under it)
3. Package defaults

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PDCT_HOME` | `~/.pdct` (installer) | root for all mutable state |
| `OBSIDIAN_VAULT` / `PDCT_VAULT_ROOT` | `$PDCT_HOME/vault/distillations` | distillation corpus root(s). If the dir contains `distillations/`, that subdir is used |
| `PDCT_EVENTS_PATH` | `$PDCT_HOME/events.jsonl` | concept event log |
| `PDCT_RUNTIME_DIR` | `$PDCT_HOME/runtime` | overrides, regions, tuning state |
| `PDCT_LOGS_DIR` | `$PDCT_HOME/logs` | telemetry, ledgers |
| `DCT_DATA_DIR` | `$PDCT_HOME/data` | judge.db and other databases |
| `PDCT_OVERRIDES_PATH` | `$PDCT_RUNTIME_DIR/pdct-overrides.json` | live tuning-lever overrides |
| `PDCT_ARCHIVE_ROOT` | `$PDCT_HOME/vault/compaction-archive` | compaction archive corpus (optional) |
| `PDCT_ANCHOR_PATHS` | `$PDCT_HOME/ANCHOR.md` | `:`-separated always-on context files (optional) |
| `ANTHROPIC_API_KEY` | — | anthropic provider auth (or Claude Code OAuth login); retrieval runs without it |
| `PDCT_LLM_PROVIDER` | `anthropic` | `anthropic` or `openai-compatible` |
| `PDCT_LLM_BASE_URL` | — | OpenAI-compatible endpoint base, e.g. `http://localhost:11434/v1` |
| `PDCT_LLM_MODEL` | provider default | model name for distiller/judge |
| `PDCT_LLM_API_KEY` | — | bearer key for the OpenAI-compatible endpoint |
| `PDCT_SCHEDULER_INTERVAL` | `300` | seconds between supervisor scheduler ticks |
| `PDCT_DISABLE_ELIGIBILITY` | unset | `1` = index every note regardless of eligibility gate |
| `DCT_QUERY_COSINE_THRESHOLD` | `0.57` | embedding filter threshold (embeddings extra) |

## Tuning levers

Runtime retrieval levers live in `pdct-overrides.json` (see
`src/dct/retrieval/overrides.py` for the full key table with types, defaults,
and clamps): `cascade_depth`, `cascade_decay`, `cascade_top_k`,
`cascade_score_floor`, `cascade_transitions_bias`, and others. Each is also
env-overridable — see the `ENV` column in the overrides table.

## Distillation format

A distillation is a markdown file with YAML frontmatter:

```markdown
---
date: 2026-01-05
concepts: [vector-database-selection, embedding-storage]
tags: [distillation]
gist: One-line summary used in retrieval results
---
# Title

Body text…
```

`concepts` drive graph placement; `gist` is what the agent sees first in
retrieval results. See `examples/vault/distillations/` for working samples.
