"""Tests for the judge SQLite schema and migrations.

P1.3a substrate. No era logic, no live codex. Just schema correctness.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dct.judge import schema


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


# --- migration creates expected tables ---------------------------------------

def test_init_db_creates_all_expected_tables(tmp_path: Path) -> None:
    db = tmp_path / "judge.db"
    schema.init_db(db)

    conn = _open(db)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = [r["name"] for r in rows]

    # Substrate tables from §4 of v3.3 plan:
    for required in (
        "judge_jobs",
        "judge_results",
        "judge_cache",
        "judge_budget",
        "judge_daily_counters",
        "schema_meta",
    ):
        assert required in names, f"missing table {required!r} (have {names!r})"


def test_init_db_records_schema_version(tmp_path: Path) -> None:
    db = tmp_path / "judge.db"
    schema.init_db(db)

    conn = _open(db)
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    assert row is not None
    assert row["value"] == str(schema.CURRENT_SCHEMA_VERSION)


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    """Calling init_db twice on the same file must be safe."""
    db = tmp_path / "judge.db"
    schema.init_db(db)
    schema.init_db(db)  # must not raise

    conn = _open(db)
    # Still exactly one schema_meta row
    rows = conn.execute(
        "SELECT key, value FROM schema_meta WHERE key='schema_version'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["value"] == str(schema.CURRENT_SCHEMA_VERSION)


def test_init_db_creates_indexes(tmp_path: Path) -> None:
    db = tmp_path / "judge.db"
    schema.init_db(db)

    conn = _open(db)
    idx = [
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    ]
    assert "idx_jobs_status" in idx
    assert "idx_jobs_enqueued" in idx


# --- judge_jobs columns ------------------------------------------------------

def test_judge_jobs_has_required_columns(tmp_path: Path) -> None:
    db = tmp_path / "judge.db"
    schema.init_db(db)
    conn = _open(db)

    cols = {
        r["name"]: r
        for r in conn.execute("PRAGMA table_info(judge_jobs)").fetchall()
    }
    expected = {
        "turn_id",
        "enqueued_at",
        "claimed_at",
        "completed_at",
        "status",
        "fail_reason",
        "attempt_count",
        "payload_json",
        "era_at_enqueue",
        "rubric_version",
        "prompt_template_version",
        "judge_model_requested",
    }
    missing = expected - set(cols)
    assert not missing, f"judge_jobs missing columns: {missing}"

    # turn_id is primary key, status NOT NULL
    pks = [r["name"] for r in cols.values() if r["pk"]]
    assert pks == ["turn_id"]
    assert cols["status"]["notnull"] == 1


def test_judge_jobs_status_check_constraint(tmp_path: Path) -> None:
    """Inserting an invalid status must fail."""
    db = tmp_path / "judge.db"
    schema.init_db(db)
    conn = _open(db)

    # Valid status: should succeed
    conn.execute(
        "INSERT INTO judge_jobs(turn_id, enqueued_at, status, payload_json, "
        "rubric_version, prompt_template_version, judge_model_requested) "
        "VALUES('t1', 1.0, 'pending', '{}', 'r', 'p', 'm')"
    )

    # Invalid status: should raise IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO judge_jobs(turn_id, enqueued_at, status, payload_json, "
            "rubric_version, prompt_template_version, judge_model_requested) "
            "VALUES('t2', 1.0, 'bogus', '{}', 'r', 'p', 'm')"
        )


# --- judge_results columns ---------------------------------------------------

def test_judge_results_has_attempt_columns(tmp_path: Path) -> None:
    """attempt_id + attempt_count must be in v1 (folded, not ALTER)."""
    db = tmp_path / "judge.db"
    schema.init_db(db)
    conn = _open(db)

    cols = {
        r["name"]
        for r in conn.execute("PRAGMA table_info(judge_results)").fetchall()
    }
    for required in (
        "turn_id",
        "scored_at",
        "score",
        "rationale",
        "era_assessment",
        "task_assessment",
        "era_inferred",
        "era_declared",
        "era_method",
        "judge_model_exact",
        "rubric_version",
        "prompt_template_version",
        "sample_policy_version",
        "latency_ms",
        "cost_estimate_usd",
        "cache_hit",
        "fail_reason",
        "attempt_id",
        "attempt_count",
        "archived_at",
    ):
        assert required in cols, f"judge_results missing column {required!r}"


# --- daily counters / budget seed --------------------------------------------

def test_judge_budget_seeded_with_single_row(tmp_path: Path) -> None:
    db = tmp_path / "judge.db"
    schema.init_db(db)
    conn = _open(db)

    rows = conn.execute("SELECT id, cap_usd, spent_usd FROM judge_budget").fetchall()
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["cap_usd"] == pytest.approx(20.0)
    assert rows[0]["spent_usd"] == pytest.approx(0.0)


def test_judge_budget_id_check_constraint(tmp_path: Path) -> None:
    db = tmp_path / "judge.db"
    schema.init_db(db)
    conn = _open(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO judge_budget(id, week_start, spent_usd, cap_usd) "
            "VALUES (2, 0, 0, 0)"
        )


def test_daily_counters_have_columns(tmp_path: Path) -> None:
    db = tmp_path / "judge.db"
    schema.init_db(db)
    conn = _open(db)
    cols = {
        r["name"]
        for r in conn.execute(
            "PRAGMA table_info(judge_daily_counters)"
        ).fetchall()
    }
    for required in (
        "day",
        "enqueued_count",
        "completed_count",
        "failed_count",
        "skipped_count",
        "daily_cap",
    ):
        assert required in cols
