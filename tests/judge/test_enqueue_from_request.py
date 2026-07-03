"""Tests for the lazy-loadable enqueue_from_request entry point.

This is what daemon.py calls inside its hook (under feature flag).
The function:
  - resolves the DB path from env (or default)
  - composes a payload via build_judge_payload
  - calls queue.enqueue
  - returns the EnqueueResult

The daemon-side timeout / failure-swallow contract is tested separately
(in tools/telegram-dispatch/tests once the hook lands).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from dct.judge import queue, schema
from dct.judge.daemon_adapter import enqueue_from_request


def test_enqueue_from_request_writes_pending_row(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "judge.db"
    schema.init_db(db)
    monkeypatch.setenv("PDCT_JUDGE_DB", str(db))

    req = {"user_text": "hello", "message_thread_id": 5, "chat_id": "c"}
    result = enqueue_from_request(
        req=req,
        pdct_turn_id="turn-xyz",
        dct_context_str="ctx",
        reply_text_str="reply",
        era_at_enqueue="unknown",
    )
    assert result == queue.EnqueueResult.OK

    conn = schema.open_conn(db)
    row = conn.execute(
        "SELECT turn_id, status, payload_json, era_at_enqueue "
        "FROM judge_jobs WHERE turn_id='turn-xyz'"
    ).fetchone()
    assert row["status"] == "pending"
    assert row["era_at_enqueue"] == "unknown"


def test_enqueue_from_request_uses_default_db_when_env_unset(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("PDCT_JUDGE_DB", raising=False)
    monkeypatch.setenv("DCT_DATA_DIR", str(tmp_path))

    result = enqueue_from_request(
        req={"user_text": "hi"},
        pdct_turn_id="t-1",
        dct_context_str="",
        reply_text_str="",
        era_at_enqueue=None,
    )
    assert result == queue.EnqueueResult.OK

    # Default DB at $DCT_DATA_DIR/judge.db should now exist.
    expected_db = tmp_path / "judge.db"
    assert expected_db.exists()


def test_enqueue_from_request_creates_db_if_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """If the DB doesn't exist yet, enqueue_from_request should init it
    on first call rather than raising."""
    db = tmp_path / "subdir" / "judge.db"
    monkeypatch.setenv("PDCT_JUDGE_DB", str(db))
    assert not db.exists()

    result = enqueue_from_request(
        req={"user_text": "hi"},
        pdct_turn_id="t-1",
        dct_context_str="",
        reply_text_str="",
        era_at_enqueue=None,
    )
    assert result == queue.EnqueueResult.OK
    assert db.exists()
