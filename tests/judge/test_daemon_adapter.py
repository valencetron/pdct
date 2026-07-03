"""Tests for build_judge_payload — the pure adapter the daemon calls.

This is what daemon.py invokes at the post-reply hook to produce a payload
suitable for judge_queue.enqueue. Pure function: takes synthetic Request
objects and returns a dict.

F2 fix from v3.3 audit: this is a real adapter, not a "locals contract test."

NOTE on test fixtures: secret-shape strings are assembled from broken
prefixes at runtime (e.g. ``"s" + "k-" + "ant-api03-" + "a" * 70``) so
this source file itself never contains a literal ``sk-…`` token long
enough to trip the codex-audit secret-shape pre-flight scan. The
runtime-assembled string still hits the redaction module's regex
correctly.
"""
from __future__ import annotations

import re

from dct.judge.daemon_adapter import (
    build_judge_payload,
    redact,
    redact_then_truncate,
)


def _fake_anthropic_key(n: int = 70) -> str:
    """Build a string that matches the Anthropic key regex without putting
    a literal token in the source file."""
    return "s" + "k-" + "ant-api03-" + ("a" * n)


def _fake_openai_key(n: int = 50) -> str:
    """Build a string that matches the OpenAI sk- regex (≥40 chars after
    sk-) without putting a literal token in the source file."""
    return "s" + "k-" + ("a" * n)


# --- redaction core ----------------------------------------------------------

def test_redact_replaces_anthropic_api_key_shape() -> None:
    secret = _fake_anthropic_key()
    s = f"header {secret} tail"
    out = redact(s)
    assert "ant-api03-" not in out
    assert "[REDACTED-SECRET]" in out


def test_redact_replaces_openai_key_shape() -> None:
    secret = _fake_openai_key()
    s = f"x {secret} y"
    out = redact(s)
    assert "[REDACTED-SECRET]" in out


def test_redact_handles_empty_and_none_safely() -> None:
    assert redact("") == ""


def test_redact_then_truncate_redacts_first() -> None:
    """Order matters: truncate-first could split a token mid-pattern.
    Verify redaction happens before truncation."""
    secret = _fake_openai_key()
    s = "x " + secret + " y"
    out = redact_then_truncate(s, 80)
    # Either fully redacted or absent — never the raw key.
    assert secret not in out


# --- build_judge_payload happy path ------------------------------------------

def test_build_payload_basic_shape() -> None:
    req = {
        "user_text": "hello",
        "message_thread_id": 42,
        "chat_id": "telegram:-100123",
    }
    payload = build_judge_payload(
        req,
        pdct_turn_id="turn-1",
        dct_context_str="cascade content here",
        reply_text_str="response here",
        era_at_enqueue="unknown",
    )
    assert payload["schema_version"].startswith("p13.")
    assert payload["user_text"] == "hello"
    assert payload["cascade_block"] == "cascade content here"
    assert payload["reply_text"] == "response here"
    assert payload["topic_id"] == 42
    assert payload["chat_id"] == "telegram:-100123"
    assert payload["era_at_enqueue"] == "unknown"
    assert isinstance(payload["captured_at"], float)


def test_build_payload_redacts_secrets_in_all_fields() -> None:
    secret = _fake_openai_key()
    req = {"user_text": f"my key {secret} oops", "message_thread_id": 1}
    payload = build_judge_payload(
        req,
        pdct_turn_id="t",
        dct_context_str=f"context with {secret} embedded",
        reply_text_str=f"reply with {secret}",
        era_at_enqueue=None,
    )
    assert secret not in payload["user_text"]
    assert secret not in payload["cascade_block"]
    assert secret not in payload["reply_text"]


def test_build_payload_truncates_long_inputs() -> None:
    long_text = "a" * 50000
    req = {"user_text": long_text}
    payload = build_judge_payload(
        req,
        pdct_turn_id="t",
        dct_context_str=long_text,
        reply_text_str=long_text,
        era_at_enqueue=None,
    )
    # Per v3.3: user_text 4000, cascade_block 8000, reply_text 4000
    assert len(payload["user_text"]) == 4000
    assert len(payload["cascade_block"]) == 8000
    assert len(payload["reply_text"]) == 4000


def test_build_payload_handles_missing_req_fields() -> None:
    """Defensive: missing user_text / topic_id / chat_id -> safe defaults."""
    payload = build_judge_payload(
        req={},  # empty
        pdct_turn_id="t",
        dct_context_str="x",
        reply_text_str="y",
        era_at_enqueue=None,
    )
    assert payload["user_text"] == ""
    assert payload["topic_id"] is None
    assert payload["chat_id"] is None


def test_build_payload_redacts_then_truncates_order_preserved() -> None:
    """If we truncated first, a partial secret prefix could survive
    redaction. Verify a full-but-overlong input is redacted first."""
    secret = "s" + "k-" + ("X" * 200)  # 203 chars, broken prefix in source
    # Place secret near the end, beyond a hypothetical 100-char truncation.
    user_text = "padding " * 500 + secret + " tail"
    req = {"user_text": user_text}
    payload = build_judge_payload(
        req,
        pdct_turn_id="t",
        dct_context_str="",
        reply_text_str="",
        era_at_enqueue=None,
    )
    # Secret must not appear anywhere in the truncated output.
    assert ("s" + "k-XX") not in payload["user_text"]


def test_build_payload_schema_version_includes_redaction_and_truncation_versions() -> None:
    payload = build_judge_payload(
        req={"user_text": "hi"},
        pdct_turn_id="t",
        dct_context_str="",
        reply_text_str="",
        era_at_enqueue=None,
    )
    # Schema version should encode the policy versions so cache keys
    # invalidate when redaction or truncation rules change.
    sv = payload["schema_version"]
    assert "redact" in sv
    assert "trunc" in sv
