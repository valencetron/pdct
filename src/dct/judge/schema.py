"""Judge SQLite schema + idempotent migration.

Single migration file (v1). All ALTERs from the v3.3 plan are folded into
the initial schema. Future schema changes get numbered migration files.

Privacy: ``init_db`` enforces 0600 on the DB file plus its WAL/SHM
sidecars (codex r1+r2 P1). Callers should also call ``ensure_mode_0600``
on every open path — including paths that already existed — to defend
against permissive copies / migrations from older builds.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import stat as _stat
from pathlib import Path

log = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1

_SCHEMA_V1 = """
-- Job queue
CREATE TABLE IF NOT EXISTS judge_jobs (
  turn_id TEXT PRIMARY KEY,
  enqueued_at REAL NOT NULL,
  claimed_at REAL,
  completed_at REAL,
  status TEXT NOT NULL CHECK(status IN
    ('pending','claimed','completed','failed','skipped')),
  fail_reason TEXT,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  payload_json TEXT NOT NULL,
  era_at_enqueue TEXT,
  rubric_version TEXT NOT NULL,
  prompt_template_version TEXT NOT NULL,
  judge_model_requested TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON judge_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_enqueued ON judge_jobs(enqueued_at);

-- Schema metadata (single-row per key)
CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Results: terminal data, kept independently of judge_jobs lifecycle.
-- attempt_id + attempt_count + archived_at folded in at v1 (no ALTERs).
CREATE TABLE IF NOT EXISTS judge_results (
  turn_id TEXT PRIMARY KEY,
  scored_at REAL NOT NULL,
  score INTEGER,
  rationale TEXT,
  era_assessment TEXT,
  task_assessment TEXT,
  era_inferred TEXT,
  era_declared TEXT,
  era_method TEXT,
  judge_model_exact TEXT,
  rubric_version TEXT NOT NULL,
  prompt_template_version TEXT NOT NULL,
  sample_policy_version TEXT NOT NULL,
  latency_ms INTEGER,
  cost_estimate_usd REAL,
  cache_hit INTEGER NOT NULL DEFAULT 0,
  fail_reason TEXT,
  attempt_id INTEGER,
  attempt_count INTEGER NOT NULL DEFAULT 1,
  archived_at REAL
);

-- Cache: separate table so cache lookups don't compete with the job queue.
CREATE TABLE IF NOT EXISTS judge_cache (
  cache_key TEXT PRIMARY KEY,
  result_json TEXT NOT NULL,
  cached_at REAL NOT NULL
);

-- Weekly cost budget (single row, id=1)
CREATE TABLE IF NOT EXISTS judge_budget (
  id INTEGER PRIMARY KEY CHECK(id=1),
  week_start REAL NOT NULL,
  spent_usd REAL NOT NULL DEFAULT 0,
  cap_usd REAL NOT NULL DEFAULT 20.0
);

-- Daily counters: separate from cost (F5 fix in v3.3 plan)
CREATE TABLE IF NOT EXISTS judge_daily_counters (
  day TEXT PRIMARY KEY,
  enqueued_count INTEGER NOT NULL DEFAULT 0,
  completed_count INTEGER NOT NULL DEFAULT 0,
  failed_count INTEGER NOT NULL DEFAULT 0,
  skipped_count INTEGER NOT NULL DEFAULT 0,
  daily_cap INTEGER NOT NULL DEFAULT 200
);
"""


def init_db(path: str | os.PathLike[str]) -> None:
    """Create or upgrade the judge database at ``path``.

    Idempotent: safe to call on a fresh path or an existing one.

    Sets WAL mode for concurrency, ensures the parent directory exists,
    seeds the budget + schema_version rows, and **enforces 0600 mode on
    the database file plus its WAL/SHM sidecars** before returning
    (codex r1 P1 fix: a permissive umask would otherwise leave redacted
    user_text / cascade / reply contents world-readable).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # isolation_level=None → autocommit; we manage transactions ourselves
    # for the migration to be atomic.
    conn = sqlite3.connect(str(p), isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # DDL: executescript implicitly commits; CREATE TABLE IF NOT EXISTS
        # is idempotent so re-running is safe.
        conn.executescript(_SCHEMA_V1)

        # Seed rows in an explicit transaction.
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(CURRENT_SCHEMA_VERSION),),
            )
            conn.execute(
                "INSERT OR IGNORE INTO judge_budget(id, week_start, spent_usd, cap_usd) "
                "VALUES (1, strftime('%s','now','weekday 1'), 0, 20.0)"
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()

    ensure_mode_0600(p)


def ensure_mode_0600(path: str | os.PathLike[str]) -> None:
    """Force the DB file (plus any WAL/SHM sidecars) to mode 0600.

    Idempotent and cheap (one stat + one chmod per file). Should be
    called by every code path that opens the DB — including paths that
    skip ``init_db`` because the file already existed (codex r2 P1).
    A pre-existing file from an older build, a deployment touch, or a
    permissive copy might have arrived at 0644; this is the chokepoint
    that fixes that.

    Failures are LOGGED (not silently swallowed — codex r2 P1 catch).
    The function still does not raise on chmod failure because a
    read-only filesystem in tests would otherwise abort the migration;
    but the warning makes the failure visible.
    """
    p = Path(path)
    for suffix in ("", "-wal", "-shm"):
        sidecar = Path(str(p) + suffix)
        if not sidecar.exists():
            continue
        try:
            current = _stat.S_IMODE(sidecar.stat().st_mode)
            if current != 0o600:
                os.chmod(sidecar, 0o600)
        except OSError as e:
            log.warning(
                "judge.schema: chmod 0600 failed for %s: %s",
                sidecar, e,
            )


def open_conn(path: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open a connection with sane defaults for runtime use.

    Callers should still BEGIN IMMEDIATE explicitly when they need
    write-side atomicity.

    Re-asserts mode 0600 on every open (codex r2 P1 — the chmod gate
    must defend against pre-existing-DB and permissive-copy paths,
    not just first-init).
    """
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    # Defense in depth: every open path goes through here, so chmod
    # gating here catches DBs that bypassed init_db (e.g. external
    # provisioning or a path that already existed at adapter-side).
    ensure_mode_0600(path)
    return conn
