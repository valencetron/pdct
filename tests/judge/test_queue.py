"""Tests for the judge queue: enqueue, claim, commit, TTL, recovery.

Substrate-only. No era logic, no codex. Pure SQLite mechanics.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from dct.judge import queue, schema


# --- helpers -----------------------------------------------------------------

def _setup_db(tmp_path: Path) -> Path:
    db = tmp_path / "judge.db"
    schema.init_db(db)
    return db


def _payload(text: str = "hello") -> dict:
    return {
        "schema_version": "p13.v3.2",
        "user_text": text,
        "cascade_block": "",
        "reply_text": "",
        "topic_id": None,
        "chat_id": None,
        "captured_at": time.time(),
    }


# --- enqueue: basic + cap + duplicate ---------------------------------------

def test_enqueue_inserts_pending_job(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    result = queue.enqueue(
        db,
        turn_id="t-1",
        payload=_payload(),
        era_at_enqueue="unknown",
    )
    assert result == queue.EnqueueResult.OK

    conn = schema.open_conn(db)
    row = conn.execute(
        "SELECT turn_id, status, attempt_count FROM judge_jobs WHERE turn_id='t-1'"
    ).fetchone()
    assert row["status"] == "pending"
    assert row["attempt_count"] == 0


def test_enqueue_duplicate_returns_duplicate(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    queue.enqueue(db, turn_id="t-1", payload=_payload(), era_at_enqueue="x")
    result = queue.enqueue(db, turn_id="t-1", payload=_payload(), era_at_enqueue="x")
    assert result == queue.EnqueueResult.DUPLICATE


def test_enqueue_duplicate_does_not_increment_daily_counter(tmp_path: Path) -> None:
    """F4 fix: insert-first, then bump counter only if rowcount>0."""
    db = _setup_db(tmp_path)
    queue.enqueue(db, turn_id="t-1", payload=_payload(), era_at_enqueue="x")
    queue.enqueue(db, turn_id="t-1", payload=_payload(), era_at_enqueue="x")  # dup

    conn = schema.open_conn(db)
    rows = conn.execute(
        "SELECT day, enqueued_count FROM judge_daily_counters"
    ).fetchall()
    # Exactly one row for today, count = 1 (not 2).
    assert len(rows) == 1
    assert rows[0]["enqueued_count"] == 1


def test_enqueue_at_cap_returns_at_cap(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    # Force daily cap to 2 for this test
    conn = schema.open_conn(db)
    conn.execute("UPDATE judge_daily_counters SET daily_cap=2 WHERE day=?",
                 (queue._pacific_today_str(),))
    # If the row doesn't exist yet, seed it
    if conn.total_changes == 0:
        conn.execute(
            "INSERT INTO judge_daily_counters(day, daily_cap) VALUES (?, 2)",
            (queue._pacific_today_str(),),
        )

    assert queue.enqueue(db, "t-1", _payload(), "x") == queue.EnqueueResult.OK
    assert queue.enqueue(db, "t-2", _payload(), "x") == queue.EnqueueResult.OK
    assert queue.enqueue(db, "t-3", _payload(), "x") == queue.EnqueueResult.AT_CAP

    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM judge_jobs"
    ).fetchone()
    assert rows["n"] == 2  # third one rejected, not inserted


# --- claim: atomic, single-claim, none-pending case --------------------------

def test_claim_returns_none_when_no_pending(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    assert queue.claim_one(db) is None


def test_claim_returns_pending_job_and_marks_claimed(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload("hi"), "x")

    job = queue.claim_one(db)
    assert job is not None
    assert job.turn_id == "t-1"
    assert job.payload["user_text"] == "hi"

    conn = schema.open_conn(db)
    row = conn.execute(
        "SELECT status, claimed_at, attempt_count FROM judge_jobs WHERE turn_id='t-1'"
    ).fetchone()
    assert row["status"] == "claimed"
    assert row["claimed_at"] is not None
    assert row["attempt_count"] == 1


def test_claim_only_one_worker_wins_under_concurrency(tmp_path: Path) -> None:
    """F3 atomicity: if two workers race to claim, only one gets the job."""
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload(), "x")

    results: list = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(queue.claim_one(db))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    winners = [r for r in results if r is not None]
    assert len(winners) == 1, f"expected exactly 1 claim, got {len(winners)}"
    assert winners[0].turn_id == "t-1"


def test_claim_skips_jobs_older_than_24h(tmp_path: Path) -> None:
    """Stale pending rows should not be claimed by the normal path
    (TTL sweep handles them)."""
    db = _setup_db(tmp_path)
    conn = schema.open_conn(db)
    # Insert a stale pending row directly
    conn.execute(
        "INSERT INTO judge_jobs(turn_id, enqueued_at, status, payload_json, "
        "rubric_version, prompt_template_version, judge_model_requested) "
        "VALUES (?, ?, 'pending', '{}', 'r', 'p', 'm')",
        ("t-old", time.time() - 86400 - 60),
    )
    assert queue.claim_one(db) is None


# --- TTL sweep ---------------------------------------------------------------

def test_ttl_sweep_marks_old_pending_skipped(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    conn = schema.open_conn(db)
    conn.execute(
        "INSERT INTO judge_jobs(turn_id, enqueued_at, status, payload_json, "
        "rubric_version, prompt_template_version, judge_model_requested) "
        "VALUES (?, ?, 'pending', '{}', 'r', 'p', 'm')",
        ("t-old", time.time() - 86400 - 60),
    )

    n = queue.sweep_ttl(db)
    assert n == 1

    row = conn.execute(
        "SELECT status, fail_reason FROM judge_jobs WHERE turn_id='t-old'"
    ).fetchone()
    assert row["status"] == "skipped"
    assert row["fail_reason"] == "ttl_expired"

    # And a corresponding judge_results row should exist (so reports
    # see the skipped turn in the denominator).
    res = conn.execute(
        "SELECT score, fail_reason FROM judge_results WHERE turn_id='t-old'"
    ).fetchone()
    assert res is not None
    assert res["score"] is None
    assert res["fail_reason"] == "ttl_expired"


def test_ttl_sweep_leaves_recent_pending_alone(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-recent", _payload(), "x")
    n = queue.sweep_ttl(db)
    assert n == 0


# --- crash recovery ----------------------------------------------------------

def test_recover_stuck_claims_resets_to_pending(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    conn = schema.open_conn(db)
    conn.execute(
        "INSERT INTO judge_jobs(turn_id, enqueued_at, claimed_at, status, "
        "payload_json, rubric_version, prompt_template_version, judge_model_requested) "
        "VALUES ('t-stuck', ?, ?, 'claimed', '{}', 'r', 'p', 'm')",
        (time.time() - 1000, time.time() - 700),
    )

    n = queue.recover_stuck_claims(db, stuck_after_s=600)
    assert n == 1
    row = conn.execute(
        "SELECT status FROM judge_jobs WHERE turn_id='t-stuck'"
    ).fetchone()
    assert row["status"] == "pending"


def test_recover_does_not_touch_recent_claims(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    conn = schema.open_conn(db)
    conn.execute(
        "INSERT INTO judge_jobs(turn_id, enqueued_at, claimed_at, status, "
        "payload_json, rubric_version, prompt_template_version, judge_model_requested) "
        "VALUES ('t-fresh', ?, ?, 'claimed', '{}', 'r', 'p', 'm')",
        (time.time() - 60, time.time() - 30),
    )

    n = queue.recover_stuck_claims(db, stuck_after_s=600)
    assert n == 0


# --- commit_result UPSERT semantics (F4) -------------------------------------

def test_commit_success_writes_result_row(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload(), "x")
    job = queue.claim_one(db)
    assert job is not None

    queue.commit_result(
        db,
        turn_id=job.turn_id,
        score=4,
        rationale="fits the era",
        fail_reason=None,
        latency_ms=42,
        judge_model_exact="stub",
    )

    conn = schema.open_conn(db)
    row = conn.execute(
        "SELECT status, completed_at FROM judge_jobs WHERE turn_id='t-1'"
    ).fetchone()
    assert row["status"] == "completed"
    assert row["completed_at"] is not None

    res = conn.execute(
        "SELECT score, rationale, latency_ms, judge_model_exact, fail_reason "
        "FROM judge_results WHERE turn_id='t-1'"
    ).fetchone()
    assert res["score"] == 4
    assert res["rationale"] == "fits the era"
    assert res["latency_ms"] == 42
    assert res["judge_model_exact"] == "stub"
    assert res["fail_reason"] is None


def test_commit_failure_writes_fail_reason(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload(), "x")
    job = queue.claim_one(db)
    queue.commit_result(
        db,
        turn_id=job.turn_id,
        score=None,
        rationale=None,
        fail_reason="schema_violation",
    )
    conn = schema.open_conn(db)
    row = conn.execute(
        "SELECT status, fail_reason FROM judge_jobs WHERE turn_id='t-1'"
    ).fetchone()
    assert row["status"] == "failed"
    assert row["fail_reason"] == "schema_violation"


def test_retry_after_failure_overwrites_result(tmp_path: Path) -> None:
    """F4 UPSERT: a failed row can be replaced by a successful retry."""
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload(), "x")
    queue.claim_one(db)
    queue.commit_result(db, turn_id="t-1", score=None, rationale=None,
                        fail_reason="schema_violation")

    # Reset to pending and retry
    conn = schema.open_conn(db)
    conn.execute(
        "UPDATE judge_jobs SET status='pending', claimed_at=NULL, "
        "completed_at=NULL, fail_reason=NULL WHERE turn_id='t-1'"
    )
    queue.claim_one(db)
    queue.commit_result(db, turn_id="t-1", score=5, rationale="great",
                        fail_reason=None)

    res = conn.execute(
        "SELECT score, rationale, fail_reason FROM judge_results WHERE turn_id='t-1'"
    ).fetchone()
    assert res["score"] == 5
    assert res["rationale"] == "great"
    assert res["fail_reason"] is None


def test_terminal_success_is_not_overwritten_by_retry(tmp_path: Path) -> None:
    """Once a row is successfully scored, it is terminal."""
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload(), "x")
    queue.claim_one(db)
    queue.commit_result(db, turn_id="t-1", score=4, rationale="ok",
                        fail_reason=None)

    # Try to overwrite with a different success — should be no-op.
    queue.commit_result(db, turn_id="t-1", score=2, rationale="changed?",
                        fail_reason=None)

    conn = schema.open_conn(db)
    res = conn.execute(
        "SELECT score, rationale FROM judge_results WHERE turn_id='t-1'"
    ).fetchone()
    assert res["score"] == 4
    assert res["rationale"] == "ok"


# --- codex r1 P3: terminal-success protection extends to job row ------------

def test_terminal_success_followed_by_failure_does_not_diverge(
    tmp_path: Path,
) -> None:
    """Codex r1 P1 #2 regression: a successful results row + a later
    failure call must not produce judge_jobs.status='failed' while
    judge_results.score still holds the success."""
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload(), "x")
    queue.claim_one(db)
    queue.commit_result(db, turn_id="t-1", score=4, rationale="ok",
                        fail_reason=None)

    # Reset and re-run; this time invoker fails.
    conn = schema.open_conn(db)
    conn.execute("UPDATE judge_jobs SET status='pending', claimed_at=NULL, "
                 "completed_at=NULL, fail_reason=NULL WHERE turn_id='t-1'")
    queue.claim_one(db)
    queue.commit_result(db, turn_id="t-1", score=None, rationale=None,
                        fail_reason="schema_violation")

    job = conn.execute(
        "SELECT status, fail_reason FROM judge_jobs WHERE turn_id='t-1'"
    ).fetchone()
    res = conn.execute(
        "SELECT score, fail_reason FROM judge_results WHERE turn_id='t-1'"
    ).fetchone()
    # The results row is still the original success.
    assert res["score"] == 4
    assert res["fail_reason"] is None
    # The job row is also still 'completed' (terminal-protected).
    assert job["status"] == "completed"
    assert job["fail_reason"] is None


# --- codex r1 P1: reject (None, None) -- can't be both no-score-and-no-reason

def test_commit_rejects_score_and_fail_reason_both_none(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload(), "x")
    queue.claim_one(db)
    with pytest.raises(ValueError, match="score or fail_reason"):
        queue.commit_result(
            db, turn_id="t-1", score=None, rationale=None, fail_reason=None,
        )


# --- codex r1 P2: result-row metadata version updates on retry --------------

def test_retry_updates_result_metadata_versions(tmp_path: Path) -> None:
    """Codex r1 P2: when a failed row is overwritten by a successful retry,
    rubric_version / prompt_template_version / sample_policy_version
    must reflect the retry, not the original failure."""
    db = _setup_db(tmp_path)
    queue.enqueue(db, "t-1", _payload(), "x")
    queue.claim_one(db)
    queue.commit_result(
        db, turn_id="t-1", score=None, rationale=None,
        fail_reason="schema_violation",
        rubric_version="r-old", prompt_template_version="p-old",
        sample_policy_version="s-old",
    )
    conn = schema.open_conn(db)
    conn.execute(
        "UPDATE judge_jobs SET status='pending', claimed_at=NULL, "
        "completed_at=NULL, fail_reason=NULL WHERE turn_id='t-1'"
    )
    queue.claim_one(db)
    queue.commit_result(
        db, turn_id="t-1", score=5, rationale="great", fail_reason=None,
        rubric_version="r-new", prompt_template_version="p-new",
        sample_policy_version="s-new",
    )

    res = conn.execute(
        "SELECT rubric_version, prompt_template_version, sample_policy_version "
        "FROM judge_results WHERE turn_id='t-1'"
    ).fetchone()
    assert res["rubric_version"] == "r-new"
    assert res["prompt_template_version"] == "p-new"
    assert res["sample_policy_version"] == "s-new"


# --- codex r1 P1: db file mode 0600 -----------------------------------------

def test_init_db_enforces_mode_0600(tmp_path: Path) -> None:
    """The DB file (and any WAL/SHM sidecars) must end up at 0600."""
    import stat as _stat
    db = tmp_path / "judge.db"
    schema.init_db(db)
    # Touch the WAL/SHM peers by opening + writing once.
    conn = schema.open_conn(db)
    conn.execute("INSERT INTO schema_meta(key, value) VALUES ('probe', '1') "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value")
    conn.close()
    schema.init_db(db)  # re-run to chmod sidecars

    for suffix in ("", "-wal", "-shm"):
        sidecar = Path(str(db) + suffix)
        if sidecar.exists():
            mode = _stat.S_IMODE(sidecar.stat().st_mode)
            assert mode == 0o600, (
                f"{sidecar.name} has mode {oct(mode)}; expected 0o600"
            )


# --- codex r1 P1: sweep_ttl honest-denominator ------------------------------

def test_sweep_ttl_skips_job_when_results_already_terminal_success(
    tmp_path: Path,
) -> None:
    """If a results row already holds a successful score, sweep_ttl
    must NOT mark the corresponding job 'skipped' (codex r1 P1 #3)."""
    db = _setup_db(tmp_path)
    conn = schema.open_conn(db)
    # Stale pending job
    conn.execute(
        "INSERT INTO judge_jobs(turn_id, enqueued_at, status, payload_json, "
        "rubric_version, prompt_template_version, judge_model_requested) "
        "VALUES (?, ?, 'pending', '{}', 'r', 'p', 'm')",
        ("t-old", time.time() - 86400 - 60),
    )
    # ...with a pre-existing successful result row.
    conn.execute(
        "INSERT INTO judge_results(turn_id, scored_at, score, rationale, "
        "rubric_version, prompt_template_version, sample_policy_version) "
        "VALUES (?, ?, 4, 'ok', 'r', 'p', 'p13a')",
        ("t-old", time.time() - 100),
    )

    n = queue.sweep_ttl(db)
    assert n == 0, "sweep_ttl should NOT count this job (results already terminal)"

    job = conn.execute(
        "SELECT status FROM judge_jobs WHERE turn_id='t-old'"
    ).fetchone()
    # Job stays pending so the inconsistency surfaces.
    assert job["status"] == "pending"


# --- codex r2 P1: sweep_ttl with EXISTING failed result row ----------------

def test_sweep_ttl_overwrites_prior_failed_result_with_ttl_expired(
    tmp_path: Path,
) -> None:
    """If a stale pending job already has a non-success failed results
    row (e.g. fail_reason='schema_violation'), sweep_ttl must overwrite
    it with 'ttl_expired' AND mark the job 'skipped' — not leave the
    job/result fail_reason in disagreement (codex r2 P1 #1)."""
    db = _setup_db(tmp_path)
    conn = schema.open_conn(db)
    # Stale pending job
    conn.execute(
        "INSERT INTO judge_jobs(turn_id, enqueued_at, status, payload_json, "
        "rubric_version, prompt_template_version, judge_model_requested) "
        "VALUES (?, ?, 'pending', '{}', 'r', 'p', 'm')",
        ("t-old-failed", time.time() - 86400 - 60),
    )
    # Pre-existing FAILED results row
    conn.execute(
        "INSERT INTO judge_results(turn_id, scored_at, score, fail_reason, "
        "rubric_version, prompt_template_version, sample_policy_version) "
        "VALUES (?, ?, NULL, 'schema_violation', 'r', 'p', 'p13a')",
        ("t-old-failed", time.time() - 100),
    )

    n = queue.sweep_ttl(db)
    assert n == 1

    job = conn.execute(
        "SELECT status, fail_reason FROM judge_jobs WHERE turn_id='t-old-failed'"
    ).fetchone()
    res = conn.execute(
        "SELECT fail_reason FROM judge_results WHERE turn_id='t-old-failed'"
    ).fetchone()
    # Both must agree on ttl_expired.
    assert job["status"] == "skipped"
    assert job["fail_reason"] == "ttl_expired"
    assert res["fail_reason"] == "ttl_expired"


# --- codex r2 P3: cutoff boundary partition --------------------------------

def test_pending_job_at_exact_cutoff_is_swept_not_lost(tmp_path: Path) -> None:
    """A pending job whose enqueued_at == ts - ttl_s should be sweepable.

    claim_one uses enqueued_at > cutoff (excludes equal); sweep_ttl uses
    enqueued_at <= cutoff (includes equal). Together they partition the
    pending set — the codex r2 P3 catch was the prior version's `<` left
    a 1-µs gap where a job was neither claimable nor sweepable.
    """
    db = _setup_db(tmp_path)
    conn = schema.open_conn(db)
    fixed_now = 1000000.0
    ttl_s = 86400
    cutoff_exact = fixed_now - ttl_s
    conn.execute(
        "INSERT INTO judge_jobs(turn_id, enqueued_at, status, payload_json, "
        "rubric_version, prompt_template_version, judge_model_requested) "
        "VALUES (?, ?, 'pending', '{}', 'r', 'p', 'm')",
        ("t-edge", cutoff_exact),
    )

    # claim_one with the same now should not return this job.
    j = queue.claim_one(db, ttl_s=ttl_s, now=fixed_now)
    assert j is None or j.turn_id != "t-edge"

    # sweep_ttl with the same now SHOULD sweep it.
    n = queue.sweep_ttl(db, ttl_s=ttl_s, now=fixed_now)
    assert n == 1
    job = conn.execute(
        "SELECT status FROM judge_jobs WHERE turn_id='t-edge'"
    ).fetchone()
    assert job["status"] == "skipped"


# --- codex r2 P2: orphan-result guard --------------------------------------

def test_commit_result_refuses_to_create_orphan_result(tmp_path: Path) -> None:
    """commit_result on a turn_id with no judge_jobs row must raise
    rather than silently insert a results row pointing at nothing
    (codex r2 P2)."""
    db = _setup_db(tmp_path)
    with pytest.raises(ValueError, match="no judge_jobs row"):
        queue.commit_result(
            db, turn_id="t-orphan", score=4, rationale="ok",
            fail_reason=None,
        )

    # And the results row should NOT have been created.
    conn = schema.open_conn(db)
    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM judge_results"
    ).fetchone()
    assert rows["n"] == 0


# --- codex r2 P1: 0600 enforced even on EXISTING db ------------------------

def test_open_conn_re_enforces_0600_on_pre_existing_loose_db(
    tmp_path: Path,
) -> None:
    """A DB that was created (or copied in) at 0644 must be coerced to
    0600 when next opened (codex r2 P1: prior r1 fix only ran on first
    init, leaving migration-from-old-build paths exposed)."""
    import stat as _stat
    db = tmp_path / "judge.db"
    schema.init_db(db)
    # Simulate the file ending up at 0644 (e.g. external copy, older build).
    os.chmod(db, 0o644)
    assert _stat.S_IMODE(db.stat().st_mode) == 0o644
    # Any open path now must coerce.
    conn = schema.open_conn(db)
    conn.close()
    assert _stat.S_IMODE(db.stat().st_mode) == 0o600


def test_ensure_mode_0600_is_idempotent_and_cheap(tmp_path: Path) -> None:
    """Calling ensure_mode_0600 on an already-0600 file is a no-op
    (no errors, no permission downgrade)."""
    import stat as _stat
    db = tmp_path / "judge.db"
    schema.init_db(db)
    schema.ensure_mode_0600(db)
    schema.ensure_mode_0600(db)
    assert _stat.S_IMODE(db.stat().st_mode) == 0o600


def test_chmod_failure_is_logged_not_silently_swallowed(
    tmp_path: Path, monkeypatch, caplog,
) -> None:
    """Codex r2 P1: a chmod failure inside ensure_mode_0600 must surface
    via a WARNING log line. Prior r1 swallowed it entirely."""
    db = tmp_path / "judge.db"
    schema.init_db(db)
    # Knock the mode out of 0o600 so ensure_mode_0600 actually attempts
    # a chmod (the optimization skips chmod when already correct).
    os.chmod(db, 0o644)

    def _boom(path, mode):
        raise OSError("simulated chmod failure")

    monkeypatch.setattr("dct.judge.schema.os.chmod", _boom)

    with caplog.at_level("WARNING", logger="dct.judge.schema"):
        schema.ensure_mode_0600(db)
    assert any(
        "chmod 0600 failed" in r.message for r in caplog.records
    ), f"expected a WARNING log line about chmod failure; got {[r.message for r in caplog.records]}"
