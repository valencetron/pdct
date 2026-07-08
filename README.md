# PDCT — Path-Dependent Context Traversal

A memory and retrieval system for LLM agents that builds a **concept knowledge
graph from conversation history** and retrieves context by *traversing* it —
not just by embedding similarity.

**Paper:** [PDCT: Conversational Retrieval is Path-Dependent (working paper, v3)](https://airshiplaboratories.com/research/pdct-v3/) — this repository is the open-source implementation and benchmark harness described in §4–§10.

Most retrieval systems answer "what text looks like this query?" PDCT answers
a different question: **"given the path this conversation has taken, what does
the agent most plausibly need to remember right now?"** It does this by
maintaining:

- an **event log** of every concept touched in conversation, with timestamps
- a **concept graph** whose edges come from co-occurrence, transitions, and
  (optionally) embedding proximity
- a **heat model**: concepts warm when touched, cool with exponential decay,
  and spread activation to graph neighbors
- **distillations**: compact markdown notes (with YAML frontmatter) that are
  the retrievable memory units
- a **cascade retriever** that seeds from the query, walks the graph with
  path-dependent scoring, and reranks candidate distillations

## Quickstart

```bash
git clone https://github.com/valencetron/pdct && cd pdct
./install.sh                 # venv, deps, config scaffold, self-test
```

The installer finishes by running the **doctor** — a 4-stage self-diagnosis
(environment, configuration, functional, retrieval quality) against a bundled
synthetic corpus. If it prints `✅ PDCT healthy`, the system works on your
machine before you've configured anything.

```bash
pdct configure                # detect + set up an LLM provider (any backend)
python -m dct.doctor          # re-run anytime
python -m dct.doctor --json   # machine-readable (CI, dashboards)
python -m dct.doctor --live   # also validate YOUR vault/config
```

## Wiring it to your own data

PDCT reads distillation notes (markdown + YAML frontmatter) from a vault
directory and concept events from an `events.jsonl` log. See
[CONFIGURATION.md](CONFIGURATION.md) for every environment variable, and
[ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit.

Works well with an **Obsidian vault** (point `OBSIDIAN_VAULT` at it), but
Obsidian is not required — any directory of markdown files works.

```bash
export PDCT_HOME=~/.pdct              # all state lives here
export OBSIDIAN_VAULT=~/MyVault       # optional: use your existing vault
python -m dct.doctor --live
```

## What's in the box

| Piece | Where | What it does |
|---|---|---|
| Activation engine | `src/dct/activation.py` | heat, decay, spread |
| Event log | `src/dct/event_log.py` | append-only concept events |
| Retrieval cascade | `src/dct/retrieval/` | graph walk + rerank + memory API |
| Distiller | `src/dct/distiller.py` | conversation → distillation notes (needs Anthropic API key) |
| Judge | `src/dct/judge/` | LLM-scored retrieval quality feedback |
| Benchmark | `benchmark/` | recall@k evaluation harness |
| Doctor | `src/dct/doctor.py` | install-time self-diagnosis |
| Examples | `examples/` | synthetic corpus the doctor runs against |

## Status

Research software, under active development. The retrieval core is exercised
daily in a production agent stack; the public packaging (this repo) is young.
Issues and reproducible failure reports are very welcome — run
`python -m dct.doctor --json` and attach the output.

## License

See [LICENSE](LICENSE).
