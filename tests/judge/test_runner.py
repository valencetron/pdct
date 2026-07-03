"""Tests for the judge runner (P1.3b).

The runner drains the judge queue with the real invoker (mocked here),
then appends era_judge_update rows to utility.jsonl.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from dct.judge import queue, schema, worker
from dct.judge import runner


def _setup(tmp_path: Path):
    db = tmp_path / "judge.db"
    schema.init_db(db)
    util = tmp_path / "utility.jsonl"
    return db, util


def _enqueue_turn(db, turn_id: str = "turn-001"):
    queue.enqueue(db, turn_id=turn_id, payload={
        "schema_version": "test",
        "user_text": "hello",
        "cascade_block": "some context",
        "reply_text": "a reply",
        "captured_at": time.time(),
    }, era_at_enqueue="test-era")


def _ok_invoker(prompt: str) -> worker.JudgeInvocationResult:
    return worker.JudgeInvocationResult(
        status="ok", score=4, rationale="good", era_assessment="helpful",
        task_assessment=None, latency_ms=50, fail_reason=None,
        judge_model_exact="claude-haiku-4-5",
    )


def test_runner_drains_and_writes_utility(tmp_path):
    """Runner drains one job and writes an era_judge_update row to utility.jsonl."""
    db, util = _setup(tmp_path)
    _enqueue_turn(db, "turn-001")

    util.write_text(json.dumps({
        "kind": "turn", "turn_id": "turn-001",
        "self_rating": "partial", "era_judge": None,
    }) + "\n")

    summary = runner.run_once(db_path=db, util_path=util, invoker=_ok_invoker)

    assert summary.drained == 1

    rows = [json.loads(l) for l in util.read_text().splitlines() if l.strip()]
    era_rows = [r for r in rows if r.get("kind") == "era_judge_update"]
    assert len(era_rows) == 1
    assert era_rows[0]["turn_id"] == "turn-001"
    assert era_rows[0]["era_judge"] == 4
    assert era_rows[0]["era_assessment"] == "helpful"


def test_runner_no_jobs(tmp_path):
    """Runner with empty queue returns summary with drained=0."""
    db, util = _setup(tmp_path)
    summary = runner.run_once(db_path=db, util_path=util, invoker=_ok_invoker)
    assert summary.drained == 0


def test_runner_failed_job_writes_null(tmp_path):
    """A failed job writes era_judge_update with era_judge=None."""
    db, util = _setup(tmp_path)
    _enqueue_turn(db, "turn-fail")
    util.write_text(json.dumps({"kind": "turn", "turn_id": "turn-fail", "era_judge": None}) + "\n")

    def _fail_invoker(prompt: str) -> worker.JudgeInvocationResult:
        return worker.JudgeInvocationResult(
            status="parse_error", score=None, rationale=None,
            era_assessment=None, task_assessment=None, latency_ms=10,
            fail_reason="json_decode", judge_model_exact="claude-haiku-4-5",
        )

    runner.run_once(db_path=db, util_path=util, invoker=_fail_invoker)

    rows = [json.loads(l) for l in util.read_text().splitlines() if l.strip()]
    era_rows = [r for r in rows if r.get("kind") == "era_judge_update"]
    assert len(era_rows) == 1
    assert era_rows[0]["era_judge"] is None
    assert era_rows[0]["fail_reason"] == "json_decode"
