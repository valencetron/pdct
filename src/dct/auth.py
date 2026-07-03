"""OAuth token loader for the DCT distiller.

Claude Max stacks store the live access token in macOS Keychain (refreshed
by the Claude Code CLI). Static API keys live in stack.json or env. This
module resolves whichever is available, in order of freshness:

    ~/.claude/.credentials.json  →  Keychain  →  stack.json  →  ANTHROPIC_API_KEY

Ported from tools/retell-endpoint/llm.py; batch-process adapted (no in-process
cache — we read once per run).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path


log = logging.getLogger(__name__)


_CLI_PATH = os.path.expanduser("~/.local/bin/claude")
_KEYCHAIN_SERVICE = "Claude Code-credentials"
_OAUTH_PREFIX = "sk-ant-oat"


class TokenLoadError(RuntimeError):
    """Raised when no credentials can be located anywhere."""


def is_oauth_token(tok: str) -> bool:
    """Whether ``tok`` is a Claude Max OAuth access token (vs. a static API key)."""
    return bool(tok) and tok.startswith(_OAUTH_PREFIX)


def _try_credentials_json() -> str | None:
    path = Path(os.path.expanduser("~/.claude/.credentials.json"))
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        tok = data.get("claudeAiOauth", {}).get("accessToken", "")
        return tok or None
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        log.warning("credentials.json unreadable: %s", exc)
        return None


def _try_keychain() -> str | None:
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", _KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        data = json.loads(r.stdout.strip())
        tok = data.get("claudeAiOauth", {}).get("accessToken", "")
        return tok or None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, AttributeError) as exc:
        log.warning("keychain unreadable: %s", exc)
        return None


def _try_stack_json() -> str | None:
    path = Path(os.path.expanduser("~/example-stack/config/stack.json"))
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        key = data.get("anthropic", {}).get("api_key", "")
        return key or None
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        log.warning("stack.json unreadable: %s", exc)
        return None


def _try_env() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or None


def load_oauth_token() -> str:
    """Return a token string. Raises ``TokenLoadError`` if all 4 sources empty."""
    for fetch, label in (
        (_try_credentials_json, "credentials.json"),
        (_try_keychain,          "keychain"),
        (_try_stack_json,        "stack.json"),
        (_try_env,               "env"),
    ):
        tok = fetch()
        if tok:
            log.info("auth: loaded token from %s", label)
            return tok
    raise TokenLoadError(
        "No Anthropic credentials found. Checked: "
        "~/.claude/.credentials.json, macOS Keychain (Claude Code-credentials), "
        "~/example-stack/config/stack.json, ANTHROPIC_API_KEY env."
    )


def refresh_oauth_via_cli() -> bool:
    """Force the Claude Code CLI to refresh its token on disk.

    Returns True if the CLI ran cleanly, False otherwise. Callers should
    re-invoke ``load_oauth_token`` afterwards to pick up the new token.
    """
    if not os.path.exists(_CLI_PATH):
        log.warning("refresh: claude CLI not at %s", _CLI_PATH)
        return False
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin:/bin"),
        "USER": os.environ.get("USER", ""),
    }
    try:
        r = subprocess.run(
            [_CLI_PATH, "-p", "--max-turns", "1", "--model", "haiku"],
            input="hi", capture_output=True, text=True, timeout=30, env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("refresh: cli invocation failed: %s", exc)
        return False
    if r.returncode != 0:
        log.warning("refresh: cli rc=%s stderr=%s",
                    r.returncode, (r.stderr or "")[:200])
        return False
    log.info("refresh: token refreshed via claude CLI")
    return True
