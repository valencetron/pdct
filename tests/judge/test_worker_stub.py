"""Tests for the stub-mode judge worker (P1.3a).

The worker drains the queue, calls a codex_invoker callable to get a
result, and commits via queue.commit_result. P1.3a never calls real codex —
the invoker passed in CI is a stub that returns deterministic JSON.

This file tests the worker's drain loop, error handling, and
recover-then-sweep startup sequence. All without ever shelling out.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from dct.judge import queue, schema, worker


def _setup_db(tmp_path: Path) -> Path:
    db = tmp_path / "judge.db"
    schema.init_db(db)
    return db


def _payload(text: str = "hi") -> dict:
    return {
        "schema_version": "p13.test",
        "user_text": text,
        "cascade_block": "",
        "reply_text": "",
        "captured_at": time.time(),
    }


def _ok_invoker(prompt: str) -> worker.JudgeInvocationResult:
    """Stub invoker: always succeeds with score=4."""
    return worker.JudgeInvocationResult(
        status="ok",
        score=4,
        rationale="stub-ok",
        era_assessment="fits",
        task_assessment="fits",
        latency_ms=1,
        fail_reason=None,
        judge_model_exact="stub",
    )


def _fail_invoker(prompt: str) -> worker.JudgeInvocationResult:
    """Stub invoker: always fails with schema_violation."""
    return worker.JudgeInvocationResult(
        status="schema_violation",
        score=None,
        rationale=None,
        era_assessment=None,
        task_assessment=None,
        latency_ms=5,
        fail_reason="schema_violation",
        judge_model_exact="stub",
    )


def _raises_invoker(prompt: str) -> worker.JudgeInvocationResult:
    raise RuntimeError("invoker boom")


# --- drain loop --------------------------------------------------------------

def test_drain_processes_pending_jobs(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload(), "x")
    queue.enqueue(db, "t-2", _payload(), "x")

    n = worker.drain_once(db, invoker=_ok_invoker, max_jobs=10)
    assert n == 2

    conn = schema.open_conn(db)
    completed = conn.execute(
        "SELECT COUNT(*) AS n FROM judge_jobs WHERE status='completed'"
    ).fetchone()
    assert completed["n"] == 2

    rows = conn.execute(
        "SELECT score, rationale, judge_model_exact FROM judge_results"
    ).fetchall()
    assert all(r["score"] == 4 for r in rows)
    assert all(r["rationale"] == "stub-ok" for r in rows)
    assert all(r["judge_model_exact"] == "stub" for r in rows)


def test_drain_respects_max_jobs(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    for i in range(5):
        queue.enqueue(db, f"t-{i}", _payload(), "x")
    n = worker.drain_once(db, invoker=_ok_invoker, max_jobs=2)
    assert n == 2

    conn = schema.open_conn(db)
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM judge_jobs WHERE status='pending'"
    ).fetchone()
    assert pending["n"] == 3


def test_drain_handles_invoker_failure(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload(), "x")
    n = worker.drain_once(db, invoker=_fail_invoker, max_jobs=10)
    assert n == 1

    conn = schema.open_conn(db)
    row = conn.execute(
        "SELECT status, fail_reason FROM judge_jobs WHERE turn_id='t-1'"
    ).fetchone()
    assert row["status"] == "failed"
    assert row["fail_reason"] == "schema_violation"


def test_drain_handles_invoker_exception_as_unexpected_error(tmp_path: Path) -> None:
    """Invoker raising should land the job in 'failed' with fail_reason set,
    not crash the drain loop."""
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload(), "x")
    queue.enqueue(db, "t-2", _payload(), "x")
    # Even though invoker explodes, the drain reports both jobs touched.
    n = worker.drain_once(db, invoker=_raises_invoker, max_jobs=10)
    assert n == 2

    conn = schema.open_conn(db)
    rows = conn.execute(
        "SELECT turn_id, status, fail_reason FROM judge_jobs"
    ).fetchall()
    for row in rows:
        assert row["status"] == "failed"
        assert row["fail_reason"] == "unexpected_error"


def test_drain_returns_zero_on_empty_queue(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    assert worker.drain_once(db, invoker=_ok_invoker, max_jobs=10) == 0


# --- run_once: full cycle (recover + sweep + drain) -------------------------

def test_run_once_recovers_stuck_then_sweeps_then_drains(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)

    # Inject a stuck claim from a previous run.
    conn = schema.open_conn(db)
    conn.execute(
        "INSERT INTO judge_jobs(turn_id, enqueued_at, claimed_at, status, "
        "payload_json, rubric_version, prompt_template_version, judge_model_requested) "
        "VALUES ('t-stuck', ?, ?, 'claimed', '{}', 'r', 'p', 'm')",
        (time.time() - 1000, time.time() - 700),
    )
    # Inject a stale pending row that should be TTL'd.
    conn.execute(
        "INSERT INTO judge_jobs(turn_id, enqueued_at, status, "
        "payload_json, rubric_version, prompt_template_version, judge_model_requested) "
        "VALUES ('t-old', ?, 'pending', '{}', 'r', 'p', 'm')",
        (time.time() - 86400 - 60,),
    )
    # And a fresh job.
    queue.enqueue(db, "t-new", _payload(), "x")

    summary = worker.run_once(db, invoker=_ok_invoker, max_jobs=10)
    assert summary.recovered >= 1
    assert summary.swept >= 1
    # Recovered stuck row + fresh row drain in this cycle:
    assert summary.drained >= 2

    rows = conn.execute(
        "SELECT turn_id, status FROM judge_jobs ORDER BY turn_id"
    ).fetchall()
    by_id = {r["turn_id"]: r["status"] for r in rows}
    assert by_id["t-old"] == "skipped"
    assert by_id["t-new"] == "completed"
    # Recovered job was claimed and completed in same run.
    assert by_id["t-stuck"] == "completed"
