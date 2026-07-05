# PDCT Integration Checklist

Every component of a complete PDCT installation, and the doctor check that
verifies it. Run `pdct doctor` (sandbox) or `pdct doctor --live` (your real
install) — each row below maps 1:1 to a machine-readable check ID in the
doctor's `--json` output (`stages.*[].id`).

An install is **fully integrated** when every *required* row passes.
Retrieval-only installs (no LLM configured) are valid: the `llm.*` rows
become advisory skips and distillation is disabled.

| # | Component | What it does | Doctor check ID | Required |
|---|-----------|--------------|-----------------|----------|
| 1 | Python runtime | Python ≥ 3.12 with required deps (yaml, watchdog, sklearn) | `env.python`, `env.deps` | yes |
| 2 | Optional extras | anthropic SDK, sentence-transformers embeddings | `env.optional` | no |
| 3 | PDCT_HOME | resolved install root; scaffolded by `pdct init` | `config.home` | yes |
| 4 | Vault | markdown distillation root (Obsidian vault or `$PDCT_HOME/vault`) | `config.vault` | yes (`--live`) |
| 5 | Event log | `events.jsonl` — the append-only activation record | `config.events` | no |
| 6 | Runtime dir | writable state dir (pidfiles, overrides, status) | `config.runtime` | yes (`--live`) |
| 7 | LLM credentials | any provider auth detected (OAuth / API key / endpoint) | `config.credentials` | no |
| 8 | Example corpus | bundled synthetic corpus for sandbox diagnostics | `functional.corpus` | yes |
| 9 | Index build | distillation index builds from the corpus | `functional.index` | yes |
| 10 | Event replay + heat | activation engine replays events into concept heat | `functional.replay` | yes |
| 11 | Retrieval questions | canned recall questions execute | `retrieval.questions` | yes |
| 12 | Retrieval recall | expected note surfaces in top-5 for every question | `retrieval.recall` | yes |
| 13 | Supervisor (write path) | `pdct daemon` lifecycle: watcher sees a note → event lands in events.jsonl | `daemon.supervisor` | yes |
| 14 | Live daemon | your running daemon is healthy (watcher alive, scheduler ticking) | `daemon.liveness` | no (`--live`) |
| 15 | LLM endpoint | configured provider reachable, auth valid (anthropic / openai-compatible / codex-oauth) | `llm.endpoint` | no* |
| 16 | Structured output | model returns parseable JSON matching the distillation schema | `llm.structured` | yes if provider configured |
| 17 | Concept quality | distilled concepts hit ≥2 of the expected set (minimum capability) | `llm.concepts` | yes if provider configured |
| 18 | Judge round-trip | judge returns a valid verdict object | `llm.judge` | yes if provider configured |
| 19 | Sibling: valence | advisory family-package detection — a co-installed valence harness (`~/.valence` / `VALENCE_HOME`); reads its fleet-status.json; never affects exit code; silent when absent | `env.sibling` | no (advisory) |

\* `llm.endpoint` is an advisory skip when **no** provider is configured
(retrieval-only mode — a supported state). Run **`pdct configure`** to
detect what LLM backends your machine has (Anthropic key or Claude OAuth,
OpenAI key, Codex OAuth, local Ollama / LM Studio) and set one up with a
verified capability probe; `pdct configure --show` prints the resolved
provider diagnostics. Once a provider *is* configured, rows 16–18 are hard
gates: a model that cannot produce schema-valid JSON or acceptable concepts
is reported as **below minimum capability** and distillation stays disabled.

## Minimum LLM requirements

Any provider/model passes if it can, functionally:

1. return a JSON object matching a supplied schema (natively via tool-use,
   or by following a structured-output prompt),
2. extract topically-correct concepts from a short conversation,
3. return a small JSON verdict for the judge.

Claude models (via subscription OAuth or API key), OpenAI models, and most
≥7B local instruction-tuned models pass. Verify yours with:

```
pdct doctor --json | python3 -c "import json,sys; d=json.load(sys.stdin); \
  print([c for c in d['stages']['llm'] if not c['ok']] or 'LLM capable')"
```

## Machine-readable status (for dashboards / web checkers)

Two stable JSON contracts:

- `pdct doctor --json` — every check above with `id`, `ok`, `detail`, `required`.
- `pdct daemon status --json` — supervisor liveness: `pid`, `uptime_s`,
  `scheduler.{ticks,last_rc,last_tick_ts}`, `watcher.alive`, `last_event_ts`.

Poll either to render PDCT in a fleet/status dashboard; the check IDs in
this document are the schema and will not be renamed.
