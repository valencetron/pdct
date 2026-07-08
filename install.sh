#!/bin/bash
# PDCT installer — venv, dependencies, config scaffold, self-diagnosis.
# Usage: ./install.sh [--minimal] [--pdct-home <dir>]
#   default: full install incl. embeddings (VEC_NEAR semantic edges)
#   --minimal: skip embeddings deps (constrained boxes; ~2GB lighter)
set -euo pipefail

WITH_EMB=1
PDCT_HOME_DIR="${PDCT_HOME:-$HOME/.pdct}"
while [ $# -gt 0 ]; do
  case "$1" in
    --minimal) WITH_EMB=0 ;;
    --with-embeddings) WITH_EMB=1 ;;  # legacy no-op (now the default)
    --pdct-home) shift; PDCT_HOME_DIR="$1" ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac
  shift
done

echo "━━ PDCT install"

# 1. Python >= 3.12 — try candidates until one makes a WORKING venv
#    (some distro/homebrew pythons ship a broken ensurepip).
make_venv() {
  local py="$1"
  command -v "$py" >/dev/null || return 1
  "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)' || return 1
  rm -rf .venv
  if "$py" -m venv .venv >/dev/null 2>&1 && [ -x .venv/bin/pip ]; then
    return 0
  fi
  # ensurepip broken (common on Homebrew/Debian pythons) — bootstrap pip
  rm -rf .venv
  "$py" -m venv --without-pip .venv >/dev/null 2>&1 || return 1
  curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python - --quiet \
    >/dev/null 2>&1 && [ -x .venv/bin/pip ] && return 0
  return 1
}
PY_OK=""
USE_UV=0
if command -v uv >/dev/null 2>&1; then
  # uv bundles its own pip machinery — immune to broken system ensurepip
  rm -rf .venv
  if uv venv --python ">=3.12" .venv >/dev/null 2>&1; then
    USE_UV=1
    echo "✅ $(.venv/bin/python -V) (via uv)"
  fi
fi
if [ "$USE_UV" -eq 0 ]; then
  for cand in python3.13 python3.12 python3; do
    if make_venv "$cand"; then PY_OK="$cand"; break; fi
  done
  [ -z "$PY_OK" ] && { echo "❌ no Python 3.12+ with working venv found." \
    "Fix: install uv (https://docs.astral.sh/uv/) or repair python3 ensurepip."; exit 1; }
  echo "✅ $("$PY_OK" -V) (venv ok)"
fi
source .venv/bin/activate
if [ "$USE_UV" -eq 1 ]; then PIP="uv pip"; else PIP="pip"; pip -q install --upgrade pip; fi
if [ "$WITH_EMB" -eq 1 ]; then
  $PIP install -q -e ".[dev,embeddings]"
  echo "✅ installed dct[dev,embeddings]"
else
  $PIP install -q -e ".[dev]"
  echo "✅ installed dct[dev] — minimal (no VEC_NEAR semantic edges;"
  echo "   re-run install.sh without --minimal to add them)"
fi

# 3. Config scaffold
mkdir -p "$PDCT_HOME_DIR"/{vault/distillations,runtime,logs,data,transcripts}

# 3a. Capture source seam — the default transcript source is a real, empty dir
#     the daemon polls every tick. Drop *.json here (or point the glob
#     elsewhere) and the pipeline starts filling the vault. Ship a README so
#     the format + override are discoverable, not folklore.
TRANSCRIPTS_README="$PDCT_HOME_DIR/transcripts/README.md"
if [ ! -f "$TRANSCRIPTS_README" ]; then
  cat > "$TRANSCRIPTS_README" <<'TREOF'
# PDCT capture source

This is the **default** place PDCT looks for conversation transcripts to
ingest. The daemon's scheduler polls this directory every tick, feeds new
files into `events.jsonl`, and the distiller writes Markdown notes into
`../vault/distillations/`.

Nothing is here yet — that's normal on a fresh install. **Drop transcript
JSON files here and the pipeline starts working.** No transcripts = empty
vault (this is what `pdct doctor` flags as an advisory `capture.source`
warning; it is not an error).

## Change or add a source

Point PDCT at your own transcript directory instead (or in addition) by
setting the glob in `../pdct.env`:

    export PDCT_TRANSCRIPTS_GLOB="$HOME/my-stack/transcripts/*.json"

The env var takes precedence over this default dir. Use any glob your shell
supports. Restart the daemon after changing it:

    systemctl --user restart pdct-supervisor.service   # Linux
    # or: pdct daemon restart

## File format

Each file is one conversation, a JSON object shaped like:

    {
      "timestamp": "2026-07-08T20:00:00Z",     # ISO8601, required
      "transcript": [
        {"role": "user",  "message": "hey",  "time_in_call_secs": 0.0},
        {"role": "agent", "message": "hi",   "time_in_call_secs": 3.2}
      ],
      "metadata": {"chat_id": "123", "topic_id": "456"}   # optional
    }

- `role`: `user`, `agent` (→ stored as `assistant`), or any role string.
- `message`: the turn text (empty/whitespace turns are skipped).
- `time_in_call_secs`: ordering offset within the call (optional; falls
  back to array index).
- Filename convention: `<timestamp>_conv_<id>.json` (also accepts
  `<ts>_call_<id>.json`, or `<ts>_<anything>.json` for ad-hoc).

Malformed files are skipped with a logged warning — they won't crash the
daemon. Re-running ingest is idempotent (turns are de-duped).
TREOF
  echo "✅ capture-source README: $TRANSCRIPTS_README"
fi

ENVFILE="$PDCT_HOME_DIR/pdct.env"
if [ ! -f "$ENVFILE" ]; then
  cat > "$ENVFILE" <<ENVEOF
# PDCT configuration — source this file or export the vars.
export PDCT_HOME="$PDCT_HOME_DIR"
# Capture source: where PDCT reads transcripts to ingest. Defaults to
# \$PDCT_HOME/transcripts/*.json (a real dir; see its README.md for format).
# Uncomment to point at your own stack's transcript dir instead:
# export PDCT_TRANSCRIPTS_GLOB="\$HOME/my-stack/transcripts/*.json"
# Point at an existing Obsidian vault instead (optional):
# export OBSIDIAN_VAULT="\$HOME/Documents/MyVault"
# LLM provider (distiller/judge; retrieval works without any LLM).
# anthropic (default): Claude Code OAuth login or ANTHROPIC_API_KEY.
# openai-compatible: any /v1/chat/completions endpoint (OpenAI, Ollama,
# LM Studio, OpenRouter, Groq, local models):
# export PDCT_LLM_PROVIDER="openai-compatible"
# export PDCT_LLM_BASE_URL="http://localhost:11434/v1"
# export PDCT_LLM_MODEL="llama3.1:8b"
# export PDCT_LLM_API_KEY=""
# Anthropic API key (anthropic provider; retrieval works without):
# export ANTHROPIC_API_KEY="sk-ant-..."
ENVEOF
  echo "✅ config scaffold: $ENVFILE"
else
  echo "✅ config exists: $ENVFILE"
fi

# 3b. LLM provider auto-select — probe-first: writes a provider into
#     pdct.env only after a live capability check passes. Never fails the
#     install (retrieval works LLM-less); prints detection table on miss.
echo "━━ configuring LLM provider (auto-detect)"
if ! PDCT_HOME="$PDCT_HOME_DIR" python -m dct.cli configure --auto; then
  echo "⚠️  no LLM provider auto-configured — distillation disabled until you run:"
  echo "    pdct configure"
fi

# 4. Validate the configured home is real and writable
touch "$PDCT_HOME_DIR/.install-probe" 2>/dev/null && rm "$PDCT_HOME_DIR/.install-probe" || {
  echo "❌ PDCT_HOME not writable: $PDCT_HOME_DIR"; exit 1; }

# 5. Service drift repair — BEFORE doctor (a reinstall rebuilds .venv, so an
#    installed systemd/launchd unit points at a dead interpreter until
#    re-rendered; doctor would fail on exactly that). service-status --json
#    always exits 0 — we branch on parsed state, never exit code (set -e safe).
svc_state() {
  PDCT_HOME="$PDCT_HOME_DIR" python -m dct.cli daemon service-status --json 2>/dev/null \
    | python -c 'import json,sys; print(json.load(sys.stdin).get("state","unknown"))' 2>/dev/null \
    || echo unknown
}
# install-service, then poll up to 10s for it to come up healthy.
install_and_poll() {
  local label="$1"
  echo "━━ $label — rendering + enabling service unit"
  if PDCT_HOME="$PDCT_HOME_DIR" python -m dct.cli daemon install-service; then
    for _i in $(seq 1 20); do
      [ "$(svc_state)" = "healthy" ] && break
      sleep 0.5
    done
    echo "✅ service active (state: $(svc_state))"
  else
    echo "⚠️  service install failed — run 'pdct daemon install-service' manually"
  fi
}

SVC_STATE=$(svc_state)
case "$SVC_STATE" in
  # Fresh box (not-installed) and disabled-but-present both need a real install —
  # NOT just drift repair. This was the gap: a first-time install fell through
  # here and the daemon never persisted.
  not-installed|installed-disabled)
    install_and_poll "installing persistent service" ;;
  stale-interpreter|missing-interpreter|broken-interpreter|stale-env|installed-inactive)
    install_and_poll "OS service drift detected ($SVC_STATE)" ;;
  not-owned)
    echo "⚠️  an installed PDCT service belongs to a different install — not touching it" ;;
  manager-unavailable)
    # Headless box (VPS, no login session): the user systemd bus needs lingering.
    # Enable it ourselves, then actually try to install — don't just print a hint.
    echo "━━ service manager unreachable (headless box) — enabling user lingering"
    if loginctl enable-linger "$USER" 2>/dev/null; then
      export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
      NEW_STATE=$(svc_state)
      if [ "$NEW_STATE" = "manager-unavailable" ]; then
        echo "⚠️  still unreachable after enable-linger — may need a re-login;"
        echo "    then run: pdct daemon install-service"
      else
        install_and_poll "lingering enabled ($NEW_STATE)"
      fi
    else
      echo "⚠️  could not enable lingering (need sudo?) — run manually:"
      echo "    sudo loginctl enable-linger $USER && pdct daemon install-service"
    fi ;;
esac

# 6. Self-diagnosis (bundled example corpus — no personal setup needed)
echo "━━ running doctor"
PDCT_HOME="$PDCT_HOME_DIR" python -m dct.doctor || {
  echo "❌ doctor failed — see failures above"; exit 1; }

echo
echo "━━ PDCT installed. Next steps:"
echo "   source .venv/bin/activate && source $ENVFILE"
echo "   pdct init                       # detect your env, finish setup"
echo "   pdct doctor --live              # check YOUR setup (vault, events)"
echo "   pdct daemon start               # keep the write path alive"
echo "   pdct recall \"a question\"       # query memory from the shell"
echo "   see INSTALL.md, CONFIGURATION.md, INTEGRATION.md"
