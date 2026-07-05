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
| `PDCT_LLM_PROVIDER` | `anthropic` | `anthropic`, `openai-compatible`, or `codex-oauth` (experimental) |
| `PDCT_LLM_BASE_URL` | — | OpenAI-compatible endpoint base, e.g. `http://localhost:11434/v1` |
| `PDCT_LLM_MODEL` | provider default | model name for distiller/judge |
| `PDCT_LLM_API_KEY` | — | bearer key for the OpenAI-compatible endpoint |
| `PDCT_LLM_API_KEY_ENV` | — | name of another env var holding the key (indirection; written by `pdct configure --key-env`) |
| `PDCT_CODEX_AUTH_PATH` | `~/.codex/auth.json` | codex-oauth: Codex CLI auth file location |
| `PDCT_SCHEDULER_INTERVAL` | `300` | seconds between supervisor scheduler ticks |
| `PDCT_DISABLE_ELIGIBILITY` | unset | `1` = index every note regardless of eligibility gate |
| `DCT_QUERY_COSINE_THRESHOLD` | `0.57` | embedding filter threshold (embeddings extra) |

## LLM providers

PDCT's distiller and judge route through one provider interface with three
backends. Retrieval works with **no** provider at all (retrieval-only mode).

### `pdct configure` — the front door

Don't edit `pdct.env` by hand — run **`pdct configure`**. It detects every
backend your machine has (Anthropic key or Claude Code OAuth, Codex OAuth,
`OPENAI_API_KEY`, local Ollama on `:11434`, LM Studio on `:1234`), lets you
pick one interactively, writes `pdct.env` atomically (preserving your
comments), and finishes with a live capability probe so you end on a
verified ✅ rather than a hope.

```bash
pdct configure                      # interactive: detect → pick → probe
pdct configure --show               # resolved diagnostics (redacted)
pdct configure --show --json        # machine-readable, for agents

# non-interactive (scripts / agents):
pdct configure --provider openai-compatible \
    --base-url http://localhost:11434/v1 --model llama3.1 \
    --key-env MY_KEY_VAR            # references the var — no secret on disk
```

`--key-env NAME` writes `PDCT_LLM_API_KEY_ENV=NAME` so the key stays in your
environment; `--key VALUE` writes the literal (file is chmod 0600) but warns.
The post-write probe runs against the *just-written* config even if your
shell exports conflicting `PDCT_LLM_*` vars. Exit code reflects the probe,
so `pdct configure --provider ... && pdct daemon start` is safe to script.

### `anthropic` (default)
Claude via `ANTHROPIC_API_KEY`, or zero-key via an existing Claude Code
OAuth login (Claude Pro/Max subscription) — PDCT sends the first-party
Claude Code header shape so subscription traffic is treated normally.

### `openai-compatible`
Any `/v1/chat/completions` endpoint: OpenAI, OpenRouter, Groq, Together,
and local models via Ollama or LM Studio. Set `PDCT_LLM_BASE_URL` +
`PDCT_LLM_MODEL` (+ `PDCT_LLM_API_KEY` if the endpoint needs one).
Structured output is emulated with a JSON-schema prompt + strict parse.

### `codex-oauth` (experimental)
ChatGPT Plus/Pro subscription via the Codex CLI's OAuth login — zero API
spend. Requires an existing login: `npm install -g @openai/codex`, then run
`codex` and sign in, which writes `~/.codex/auth.json`. PDCT reads that
file, proactively refreshes tokens (writing them back atomically, `0600`,
so the Codex CLI stays in sync), retries once on 401, and speaks the
Responses API at the ChatGPT Codex backend with the first-party Codex CLI
header shape. Keep `auth.json` permissions restricted; never copy tokens
into config files. Heavy consecutive use may hit ChatGPT account-level
rate limits.

```bash
PDCT_LLM_PROVIDER=codex-oauth
# PDCT_LLM_MODEL=gpt-5.5           # backend slug (default)
# PDCT_CODEX_AUTH_PATH=~/.codex/auth.json
```

`pdct doctor` stage 6 validates whichever provider you configured with the
same functional capability gate (endpoint/auth, structured JSON, concept
quality, judge round-trip).

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
