"""Pure adapter that turns a daemon Request into a judge payload (P1.3a).

Daemon's post-reply hook calls build_judge_payload(req, turn_id,
dct_context, reply_text, era). This is a pure function — no I/O, no
daemon state. Tests synthesize req dicts directly.

Contract:
- All user-derived strings are redacted FIRST, then truncated. Order matters
  because truncation can split a token mid-pattern, leaving recognizable
  secret prefixes.
- Per-field caps: user_text=4000, cascade_block=8000, reply_text=4000.
- schema_version encodes redaction + truncation policy versions so cache
  keys downstream invalidate when these rules change.

Also exposes ``enqueue_from_request`` — the lazy-loadable convenience
function the daemon hook calls. Resolves the DB path from
``PDCT_JUDGE_DB`` env (or ``DCT_DATA_DIR/judge.db`` default), inits the
DB if needed, and writes a pending judge_jobs row.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Optional

# Policy versions — bump when the rules change so downstream caches invalidate.
_REDACT_POLICY_VERSION = "v1"
_TRUNC_POLICY_VERSION = "v1"
_SCHEMA_VERSION = (
    f"p13.substrate.redact-{_REDACT_POLICY_VERSION}.trunc-{_TRUNC_POLICY_VERSION}"
)

# Per-field truncation caps (chars).
_USER_TEXT_CAP = 4000
_CASCADE_CAP = 8000
_REPLY_CAP = 4000


# --- redaction patterns ------------------------------------------------------
# Modeled after telegram-dispatch/daemon.py _SECRET_PATTERNS. These are
# best-effort, not crypto-grade. The goal is to stop common secret shapes
# from being persisted in judge payloads / database rows.

_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z ]+PRIVATE KEY-----.*?-----END [A-Z ]+PRIVATE KEY-----",
    re.DOTALL,
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Anthropic API keys
    re.compile(r"sk-ant-api[0-9]{2}-[A-Za-z0-9_\-]{20,}"),
    # OpenAI-style sk- keys (must NOT match short test fixtures by accident;
    # require ≥40 base64-ish chars)
    re.compile(r"sk-[A-Za-z0-9_\-]{40,}"),
    # GitHub-style tokens
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    # AWS access keys
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # Generic JWTs
    re.compile(r"eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
    # Bearer headers carrying long tokens
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{30,}"),
)


def redact(s: str) -> str:
    """Best-effort redaction of common secret shapes.

    Always run BEFORE truncation: a long token cut at the truncation
    boundary can survive as a recognizable prefix.
    """
    if not isinstance(s, str) or not s:
        return s
    out = _PRIVATE_KEY_BLOCK.sub("[REDACTED-PRIVATE-KEY-BLOCK]", s)
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED-SECRET]", out)
    return out


def redact_then_truncate(s: str, n: int) -> str:
    """Redact first, then truncate to at most n chars."""
    if not isinstance(s, str) or not s:
        return s or ""
    return redact(s)[:n]


# --- public API --------------------------------------------------------------

def build_judge_payload(
    req: dict[str, Any] | None,
    pdct_turn_id: str,
    dct_context_str: str,
    reply_text_str: str,
    era_at_enqueue: Optional[str],
    *,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Pure adapter: synthesize a judge payload from daemon-side values.

    Args:
        req: The daemon request dict (may be None or missing fields).
        pdct_turn_id: The PDCT-assigned turn id (for logging context only;
            not stored in the payload — the queue keys by turn_id separately).
        dct_context_str: The cascade block as a single string.
        reply_text_str: Claude's response text.
        era_at_enqueue: Era label (or None for substrate-only mode).
        now: Override for time.time() in tests.

    Returns:
        A dict suitable for passing to judge.queue.enqueue's `payload` arg.
    """
    r = req or {}
    user_text_raw = r.get("user_text") or ""
    captured_at = now if now is not None else time.time()

    return {
        "schema_version": _SCHEMA_VERSION,
        "user_text": redact_then_truncate(user_text_raw, _USER_TEXT_CAP),
        "cascade_block": redact_then_truncate(dct_context_str or "", _CASCADE_CAP),
        "reply_text": redact_then_truncate(reply_text_str or "", _REPLY_CAP),
        "topic_id": r.get("message_thread_id"),
        "chat_id": r.get("chat_id"),
        "captured_at": captured_at,
        "era_at_enqueue": era_at_enqueue,
    }


def _resolve_db_path() -> Path:
    """Resolve the judge DB path from env or fall back to a default
    under DCT_DATA_DIR."""
    explicit = os.environ.get("PDCT_JUDGE_DB")
    if explicit:
        return Path(explicit)
    data_dir = os.environ.get("DCT_DATA_DIR")
    if data_dir:
        return Path(data_dir) / "judge.db"
    # Final fallback: package-local data dir.
    return Path.home() / "example-stack" / "dynamic-context-traversal" / "data" / "judge.db"


def enqueue_from_request(
    req: dict[str, Any] | None,
    pdct_turn_id: str,
    dct_context_str: str,
    reply_text_str: str,
    era_at_enqueue: Optional[str],
) -> "Any":
    """Lazy entry point for the daemon hook.

    The daemon's import of this function is itself wrapped in a feature-flag
    check (PDCT_JUDGE_ENQUEUE) — we don't add any flag check here so this
    function stays cleanly testable.

    Returns: ``queue.EnqueueResult`` (imported lazily to avoid circular
    imports during daemon startup).
    """
    # Lazy imports: keep daemon import-time light.
    from . import queue as _queue
    from . import schema as _schema

    db = _resolve_db_path()
    if not db.exists():
        _schema.init_db(db)
    else:
        # Codex r2 P1: a pre-existing DB might have shipped at 0644
        # (older build, permissive copy, deployment touch). Re-chmod
        # before any write goes through. open_conn also enforces this
        # but doing it here makes the policy obvious at the entry point.
        _schema.ensure_mode_0600(db)

    payload = build_judge_payload(
        req,
        pdct_turn_id=pdct_turn_id,
        dct_context_str=dct_context_str,
        reply_text_str=reply_text_str,
        era_at_enqueue=era_at_enqueue,
    )
    return _queue.enqueue(
        db,
        turn_id=pdct_turn_id,
        payload=payload,
        era_at_enqueue=era_at_enqueue,
    )


__all__ = [
    "build_judge_payload",
    "enqueue_from_request",
    "redact",
    "redact_then_truncate",
]
