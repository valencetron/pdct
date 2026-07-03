# Installing PDCT

## Requirements

- **Python 3.12+** (required)
- macOS or Linux
- ~500MB disk if you enable the embeddings extra (sentence-transformers model)
- An **Anthropic API key** — only for the distiller and judge; the retrieval
  core, doctor, and benchmark run without one

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

## Troubleshooting

| Symptom | Fix |
|---|---|
| `doctor: dep:X` fails | re-run `./install.sh` inside the repo |
| `python>=3.12` fails | install via pyenv/brew/apt, re-run installer |
| retrieval recall 0/3 | file an issue with `doctor --json` output |
| `--live`: vault root missing | set `OBSIDIAN_VAULT` or `PDCT_VAULT_ROOT` in `pdct.env` |
