#!/bin/bash
# check_sanitized.sh — hard gate: no personal data, secrets, or forbidden
# files in this tree. Run from repo root. Exit 0 = clean.
set -uo pipefail

FAIL=0

echo "── forbidden basenames/globs"
FOUND=$(find . -type f \( -name "*.db" -o -name "*.sqlite*" -o -name "*.pkl" \
  -o -name "*.pdf" -o -name "events.jsonl" -o -name "positions.json*" \
  -o -name ".env*" -o -name "*.pem" -o -name "*.key" -o -name "id_rsa*" \
  -o -name "id_ed25519*" -o -name "credentials*.json" -o -name "stack.json" \
  \) ! -path "./.git/*" ! -path "./.venv/*" ! -path "*__pycache__*" ! -path "./examples/events.jsonl" | head -20)
if [ -n "$FOUND" ]; then echo "❌ forbidden files:"; echo "$FOUND"; FAIL=1;
else echo "✅ none"; fi

echo "── personal identifiers"
# Note: [g]odbole-style bracket trick keeps this script from matching itself.
# 'valence' alone is the PUBLIC sibling product (github.com/valencetron/valence)
# — only private identity forms are forbidden (Build 105 family package).
PATTERNS='neilg|[g]odbole|[s]hehla|[v]alence-[e]lectron|/Users/[a-z]|Documents/OBSIDIAN'
# 'airship' removed 2026-07-07: repo now intentionally links the Airship Laboratories paper
HITS=$(grep -rInE "$PATTERNS" . --include="*.py" --include="*.md" \
  --include="*.sh" --include="*.json" --include="*.jsonl" --include="*.yaml" \
  --include="*.toml" --include="*.txt" \
  --exclude="check_sanitized.sh" --exclude-dir=".git" --exclude-dir=".venv" --exclude-dir="__pycache__" --exclude-dir=".pytest_cache" 2>/dev/null | head -20)
if [ -n "$HITS" ]; then echo "❌ personal tokens:"; echo "$HITS"; FAIL=1;
else echo "✅ none"; fi

echo "── secrets / tokens / high-entropy"
SEC=$(grep -rInE 'sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{20,}|bot[0-9]{8,}:[A-Za-z0-9_-]{30,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|eyJ[A-Za-z0-9_-]{40,}' \
  . --exclude="check_sanitized.sh" --exclude-dir=".git" --exclude-dir=".venv" --exclude-dir="__pycache__" --exclude-dir=".pytest_cache" 2>/dev/null | head -10)
if [ -n "$SEC" ]; then echo "❌ secret-shaped strings:"; echo "$SEC"; FAIL=1;
else echo "✅ none"; fi

echo "── emails / phone-shaped"
PII=$(grep -rInE '[a-zA-Z0-9._%+-]+@(gmail|yahoo|hotmail|icloud|outlook)\.[a-z]+|\+1[0-9]{10}' \
  . --exclude="check_sanitized.sh" --exclude-dir=".git" --exclude-dir=".venv" --exclude-dir="__pycache__" --exclude-dir=".pytest_cache" 2>/dev/null | head -10)
if [ -n "$PII" ]; then echo "❌ PII-shaped strings:"; echo "$PII"; FAIL=1;
else echo "✅ none"; fi

echo "── telegram chat ids"
TG=$(grep -rInE '\-100[0-9]{8,}' . --exclude="check_sanitized.sh" \
  --exclude-dir=".git" --exclude-dir=".venv" --exclude-dir="__pycache__" --exclude-dir=".pytest_cache" 2>/dev/null | head -10)
if [ -n "$TG" ]; then echo "❌ telegram chat ids:"; echo "$TG"; FAIL=1;
else echo "✅ none"; fi

echo "── binary files (outside .git)"
BIN=$(find . -type f ! -path "./.git/*" ! -path "./.venv/*" \
  ! -path "*__pycache__*" ! -path "*.pytest_cache*" ! -name "*.pyc" \
  -size +0c -exec sh -c \
  'file -b --mime "$1" | grep -q "charset=binary" && echo "$1"' _ {} \; 2>/dev/null | \
  head -10)
if [ -n "$BIN" ]; then echo "❌ binary files:"; echo "$BIN"; FAIL=1;
else echo "✅ none"; fi

if [ "$FAIL" -eq 0 ]; then echo; echo "✅ SANITIZATION GATE PASSED"; exit 0
else echo; echo "❌ SANITIZATION GATE FAILED"; exit 1; fi
