#!/bin/bash
# Claude Code UserPromptSubmit hook entry point.
# Forwards stdin JSON to the Python hook script, which prints
# the PDCT context block on stdout for Claude Code to inject.
#
# Never exits non-zero — failure must not block the user's prompt.
exec ~/example-stack/pdct/venv/bin/python \
  ~/example-stack/pdct/hooks/claude_code_pdct_hook.py
