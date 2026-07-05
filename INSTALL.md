# Installing PDCT

## Requirements

- **Python 3.12+** (required)
- macOS or Linux
- ~500MB disk if you enable the embeddings extra (sentence-transformers model)
- An **LLM** — only for the distiller and judge; the retrieval core, doctor,
  and benchmark run without one. Any of: Claude Code login (subscription
  OAuth — zero API key), `ANTHROPIC_API_KEY`, or **any OpenAI-compatible
  endpoint** (OpenAI, OpenRouter, Groq, local Ollama/LM Studio) that meets
  the minimum capability gate below

## Install

```bash
./install.sh                      # core install
./install.sh --with-embeddings    # + VEC_NEAR embedding edges (recommended)
./install.sh --pdct-home /srv/pdct  # custom state directory
```

The installer:
1. verifies Python ≥ 3.12
2. creates `.venv` and installs the package (editable) with dev tools
3. scaffolds `$PDCT_HOME` (`~/.pdct` by default) with `vault/ runtime/ logs/ data/`
   and writes `pdct.env` — a template you edit and `source`
4. runs `python -m dct.doctor` against the bundled synthetic corpus

After install, the `pdct` command is on your venv PATH:

```bash
pdct init            # detect your environment, finish setup interactively
pdct configure       # detect + set up an LLM provider (verified probe)
pdct doctor --live   # validate YOUR setup end-to-end
pdct daemon start    # supervisor: vault watcher + scheduler (any POSIX OS)
pdct daemon install-service   # optional: survive reboot (launchd/systemd)
pdct recall "what did we decide about X?"   # query memory from any shell
pdct ingest transcript.json                 # manual event ingestion
```

**A green doctor means PDCT works on your machine** before you've wired any
of your own data.

## Verify

```bash
source .venv/bin/activate
python -m dct.doctor           # human-readable
python -m dct.doctor --json    # for scripts/CI
pytest -q                      # full test suite
```

## Recommended setup

- **Obsidian** (optional but recommended): point `OBSIDIAN_VAULT` at your
  vault. PDCT reads/writes plain markdown — you get a browsable, linkable
  memory system for free. Without Obsidian, any markdown directory works.
- **Embeddings extra**: enables VEC_NEAR graph edges (semantic proximity),
  which measurably improves recall on sparse corpora.
- Run `python -m dct.doctor --live` after editing `pdct.env` — it validates
  your actual vault, events log, and writability.

## LLM providers & minimum requirements

Run **`pdct configure`** — it detects what your machine has (Anthropic /
Claude OAuth, OpenAI key, Codex OAuth, local Ollama or LM Studio), writes
`pdct.env`, and ends with a live capability probe. Scriptable too:
`pdct configure --provider openai-compatible --base-url URL --model M
--key-env VAR`. Or configure `pdct.env` manually:

```bash
# Claude subscription or API key (default — auto-detected by pdct init)
PDCT_LLM_PROVIDER=anthropic

# …or any OpenAI-compatible endpoint, including local models:
PDCT_LLM_PROVIDER=openai-compatible
PDCT_LLM_BASE_URL=http://localhost:11434/v1   # Ollama example
PDCT_LLM_MODEL=llama3.1:8b
PDCT_LLM_API_KEY=                             # if the endpoint needs one
```

**Minimum capability gate** (verified functionally by `pdct doctor`, stage
`llm`): the model must (a) return schema-valid JSON for a distillation,
(b) extract topically-correct concepts, (c) return a valid judge verdict.
Models that fail are reported as *below minimum capability* — distillation
is disabled and PDCT runs in retrieval-only mode. No provider configured at
all is also fine: retrieval, doctor, daemon, and benchmark all work.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `doctor: dep:X` fails | re-run `./install.sh` inside the repo |
| `python>=3.12` fails | install via pyenv/brew/apt, re-run installer |
| retrieval recall 0/3 | file an issue with `doctor --json` output |
| `--live`: vault root missing | set `OBSIDIAN_VAULT` or `PDCT_VAULT_ROOT` in `pdct.env` |
| `llm.structured` / `llm.concepts` fail | model below minimum capability — use a stronger model |
| daemon won't start | `pdct daemon logs` shows the supervisor log tail |

See **INTEGRATION.md** for the full component checklist mapped to doctor
check IDs.
