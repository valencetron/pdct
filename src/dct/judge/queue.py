"""Atomic SQLite queue for the PDCT-utility judge (P1.3a).

Operations:
- enqueue: insert + cap-respecting daily counter, all in one transaction
- claim_one: atomic UPDATE...WHERE status='pending' RETURNING
- commit_result: UPSERT into judge_results, terminal-row-protected
- sweep_ttl: mark long-pending jobs skipped + write terminal results row
- recover_stuck_claims: reset claims older than threshold to pending

Substrate-only: no era logic, no codex invocation.
"""
from __future__ import annotations

import enum
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    # Python 3.9+: stdlib zoneinfo handles DST transitions correctly.
    from zoneinfo import ZoneInfo
    _PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
except Exception:  # pragma: no cover — only triggers on missing tzdata
    from datetime import timezone, timedelta
    _PACIFIC_TZ = timezone(timedelta(hours=-8))  # fallback, PST-only

from . import schema as _schema

# Default TTL for pending jobs.
DEFAULT_TTL_S = 86400  # 24h
DEFAULT_DAILY_CAP = 200
DEFAULT_RUBRIC_VERSION = "p13.v3.2-substrate"
DEFAULT_PROMPT_VERSION = "v3_0"
DEFAULT_MODEL_REQUESTED = "stub"


class EnqueueResult(enum.Enum):
    OK = "ok"
    DUPLICATE = "duplicate"
    AT_CAP = "at_cap"


@dataclass(frozen=True)
class Job:
    turn_id: str
    payload: dict
    era_at_enqueue: Optional[str]
    enqueued_at: float
    rubric_version: str
    prompt_template_version: str
    judge_model_requested: str


# --- helpers -----------------------------------------------------------------

def _pacific_today_str(now: Optional[float] = None) -> str:
    """Today in 'YYYY-MM-DD' Pacific.

    Uses `zoneinfo.ZoneInfo("America/Los_Angeles")` so DST transitions
    bucket correctly year-round (codex r1 P2 fix: a fixed UTC-8 offset
    mis-buckets the first hour after midnight Pacific during PDT).
    """
    epoch = now if now is not None else time.time()
    pacific = datetime.fromtimestamp(epoch, tz=_PACIFIC_TZ)
    return pacific.strftime("%Y-%m-%d")


def _begin_immediate(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")


# --- enqueue -----------------------------------------------------------------

def enqueue(
    db_path: str | Path,
    turn_id: str,
    payload: dict,
    era_at_enqueue: Optional[str],
    *,
    rubric_version: str = DEFAULT_RUBRIC_VERSION,
    prompt_template_version: str = DEFAULT_PROMPT_VERSION,
    judge_model_requested: str = DEFAULT_MODEL_REQUESTED,
    now: Optional[float] = None,
) -> EnqueueResult:
    """Enqueue a job atomically, respecting daily cap.

    F4 fix: insert FIRST (with ON CONFLICT DO NOTHING), then bump daily
    counter only if rowcount>0. Prevents quota burn on duplicate turn_id.
    """
    conn = _schema.open_conn(db_path)
    try:
        ts = now if now is not None else time.time()
        today = _pacific_today_str(ts)
        _begin_immediate(conn)
        try:
            # Read current daily counter
            row = conn.execute(
                "SELECT enqueued_count, daily_cap FROM judge_daily_counters WHERE day=?",
                (today,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO judge_daily_counters(day) VALUES(?)",
                    (today,),
                )
                count, cap = 0, DEFAULT_DAILY_CAP
            else:
                count, cap = row["enqueued_count"], row["daily_cap"]

            if count >= cap:
                conn.execute("COMMIT")
                return EnqueueResult.AT_CAP

            # Insert FIRST. If duplicate, rowcount=0 and we skip the bump.
            cur = conn.execute(
                "INSERT INTO judge_jobs(turn_id, enqueued_at, status, "
                "payload_json, era_at_enqueue, rubric_version, "
                "prompt_template_version, judge_model_requested) "
                "VALUES (?, ?, 'pending', ?, ?, ?, ?, ?) "
                "ON CONFLICT(turn_id) DO NOTHING",
                (
                    turn_id,
                    ts,
                    # allow_nan=False: reject non-standard JSON (NaN, Infinity)
                    # so payloads round-trip cleanly through every consumer
                    # (codex r1 P2). If a caller hands us a NaN, we surface
                    # it as a ValueError rather than persist invalid JSON.
                    json.dumps(payload, allow_nan=False),
                    era_at_enqueue,
                    rubric_version,
                    prompt_template_version,
                    judge_model_requested,
                ),
            )
            if cur.rowcount == 0:
                conn.execute("COMMIT")
                return EnqueueResult.DUPLICATE

            conn.execute(
                "UPDATE judge_daily_counters "
                "SET enqueued_count = enqueued_count + 1 WHERE day=?",
                (today,),
            )
            conn.execute("COMMIT")
            return EnqueueResult.OK
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


# --- claim -------------------------------------------------------------------

def claim_one(
    db_path: str | Path,
    *,
    ttl_s: int = DEFAULT_TTL_S,
    now: Optional[float] = None,
) -> Optional[Job]:
    """Atomically claim the oldest pending job that is not stale.

    Returns None if no eligible pending job. Increments attempt_count.
    """
    conn = _schema.open_conn(db_path)
    try:
        ts = now if now is not None else time.time()
        cutoff = ts - ttl_s
        _begin_immediate(conn)
        try:
            # The inner SELECT + AND status='pending' re-check inside
            # UPDATE ensures atomicity even with relaxed isolation.
            row = conn.execute(
                """
                UPDATE judge_jobs
                SET status='claimed',
                    claimed_at=?,
                    attempt_count = attempt_count + 1
                WHERE turn_id = (
                    SELECT turn_id FROM judge_jobs
                    WHERE status='pending'
                      AND enqueued_at > ?
                    ORDER BY enqueued_at ASC
                    LIMIT 1
                )
                  AND status='pending'
                RETURNING turn_id, payload_json, era_at_enqueue, enqueued_at,
                          rubric_version, prompt_template_version,
                          judge_model_requested
                """,
                (ts, cutoff),
            ).fetchone()
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        if row is None:
            return None
        return Job(
            turn_id=row["turn_id"],
            payload=json.loads(row["payload_json"]),
            era_at_enqueue=row["era_at_enqueue"],
            enqueued_at=row["enqueued_at"],
            rubric_version=row["rubric_version"],
            prompt_template_version=row["prompt_template_version"],
            judge_model_requested=row["judge_model_requested"],
        )
    finally:
        conn.close()


# --- TTL sweep ---------------------------------------------------------------

def sweep_ttl(
    db_path: str | Path,
    *,
    ttl_s: int = DEFAULT_TTL_S,
    now: Optional[float] = None,
) -> int:
    """Mark pending jobs older than ttl_s as 'skipped' with fail_reason='ttl_expired'.

    Also writes a terminal results row for each so the denominator is honest.

    Truth contract (codex r2 P1 fix on top of r1):

    - **A successful results row blocks the sweep.** If a results row
      already holds ``score IS NOT NULL, fail_reason IS NULL``, the job
      is left pending — sweeping it would diverge the job from a row
      that's terminal-success. An operator should investigate.

    - **A failed results row is overwritten by ttl_expired.** Sweep is
      the terminal state for stale jobs: after 24h of no successful
      result, ``ttl_expired`` is the honest outcome regardless of which
      transient failure was previously logged. The r1 fix incorrectly
      preserved a prior failure and then still flipped the job to
      skipped, producing exactly the divergence the comment promised
      to avoid.

    - **The job-row mutation only happens after the results row is
      verified to bear ``fail_reason='ttl_expired'``** (not just any
      non-null fail_reason — codex r2 P1 catch).

    Returns the number of jobs successfully swept.
    """
    conn = _schema.open_conn(db_path)
    try:
        ts = now if now is not None else time.time()
        cutoff = ts - ttl_s
        _begin_immediate(conn)
        try:
            # Boundary alignment with claim_one (codex r2 P3): claim_one
            # treats jobs with enqueued_at > cutoff as fresh; sweep treats
            # the rest (enqueued_at <= cutoff) as stale. Together they
            # partition the pending set with no gap at exactly the cutoff.
            stale = conn.execute(
                "SELECT turn_id, rubric_version, prompt_template_version "
                "FROM judge_jobs WHERE status='pending' AND enqueued_at <= ?",
                (cutoff,),
            ).fetchall()
            n = 0
            for row in stale:
                # UPSERT: insert if no row, OR overwrite any non-success
                # row. The WHERE clause excludes only terminal-success
                # results so a prior failure (schema_violation, etc.)
                # gets correctly overwritten by ttl_expired.
                conn.execute(
                    """
                    INSERT INTO judge_results(turn_id, scored_at, score,
                        fail_reason, rubric_version, prompt_template_version,
                        sample_policy_version)
                    VALUES (?, ?, NULL, 'ttl_expired', ?, ?, 'p13a')
                    ON CONFLICT(turn_id) DO UPDATE SET
                        scored_at = excluded.scored_at,
                        fail_reason = excluded.fail_reason,
                        rubric_version = excluded.rubric_version,
                        prompt_template_version = excluded.prompt_template_version,
                        sample_policy_version = excluded.sample_policy_version
                    WHERE judge_results.score IS NULL
                    """,
                    (
                        row["turn_id"],
                        ts,
                        row["rubric_version"],
                        row["prompt_template_version"],
                    ),
                )
                # Confirm: the results row must now bear EXACTLY
                # 'ttl_expired'. r2 strengthens this from "any non-null
                # fail_reason" so a prior schema_violation that survived
                # the UPSERT (e.g. results row marked terminal-success)
                # cannot pass the gate.
                confirm = conn.execute(
                    "SELECT score, fail_reason FROM judge_results WHERE turn_id=?",
                    (row["turn_id"],),
                ).fetchone()
                if (
                    confirm is None
                    or confirm["score"] is not None
                    or confirm["fail_reason"] != "ttl_expired"
                ):
                    # Inconsistent or success-blocked: leave the job
                    # pending so the inconsistency surfaces.
                    continue
                conn.execute(
                    "UPDATE judge_jobs SET status='skipped', "
                    "fail_reason='ttl_expired', completed_at=? WHERE turn_id=?",
                    (ts, row["turn_id"]),
                )
                n += 1
            conn.execute("COMMIT")
            return n
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


# --- crash recovery ----------------------------------------------------------

def recover_stuck_claims(
    db_path: str | Path,
    *,
    stuck_after_s: int = 600,
    now: Optional[float] = None,
) -> int:
    """Reset claims older than stuck_after_s back to pending. Returns count."""
    conn = _schema.open_conn(db_path)
    try:
        ts = now if now is not None else time.time()
        cutoff = ts - stuck_after_s
        _begin_immediate(conn)
        try:
            cur = conn.execute(
                "UPDATE judge_jobs SET status='pending', claimed_at=NULL "
                "WHERE status='claimed' AND claimed_at < ?",
                (cutoff,),
            )
            conn.execute("COMMIT")
            return cur.rowcount
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


# --- commit_result -----------------------------------------------------------

def commit_result(
    db_path: str | Path,
    *,
    turn_id: str,
    score: Optional[int],
    rationale: Optional[str],
    fail_reason: Optional[str],
    era_assessment: Optional[str] = None,
    task_assessment: Optional[str] = None,
    era_inferred: Optional[str] = None,
    era_declared: Optional[str] = None,
    era_method: Optional[str] = None,
    judge_model_exact: Optional[str] = None,
    rubric_version: str = DEFAULT_RUBRIC_VERSION,
    prompt_template_version: str = DEFAULT_PROMPT_VERSION,
    sample_policy_version: str = "p13a",
    latency_ms: Optional[int] = None,
    cost_estimate_usd: Optional[float] = None,
    cache_hit: bool = False,
    attempt_id: Optional[int] = None,
    now: Optional[float] = None,
) -> None:
    """UPSERT a judge_results row and update the corresponding job status.

    Result terminality contract (codex r1 P1 fixes):

    1. ``score is None and fail_reason is None`` is rejected as a
       ValueError — that combination produces a job whose row says
       "completed" while its results row says "no answer," which would
       silently break terminal-row protection.

    2. Effective failure: a result with ``score is None`` is treated as a
       failure regardless of caller-supplied ``fail_reason``. We synthesize
       ``fail_reason='missing_score'`` if the caller didn't provide one.

    3. Job-row update is itself terminal-protected. If a prior call
       already wrote a successful results row (score IS NOT NULL,
       fail_reason IS NULL), neither table is mutated by retries. The
       job row stays ``completed``; this prevents the "results row says
       success, jobs row says failed" divergence the previous version
       allowed.

    Failed/null results rows can still be overwritten by a later
    successful retry. Version metadata (rubric, prompt, sample-policy)
    on the result row is updated alongside score/rationale to keep
    auditability honest (codex r1 P2 fix).
    """
    if score is None and fail_reason is None:
        raise ValueError(
            "commit_result requires either score or fail_reason; "
            "both None is not a valid result."
        )
    # Normalize: a missing score is a failure even if the caller didn't
    # explicitly tag one.
    effective_fail_reason = fail_reason
    if score is None and effective_fail_reason is None:
        effective_fail_reason = "missing_score"
    new_status = "completed" if effective_fail_reason is None else "failed"

    conn = _schema.open_conn(db_path)
    try:
        ts = now if now is not None else time.time()
        _begin_immediate(conn)
        try:
            # Phase 1: under the immediate lock, check whether the results
            # row is already terminal-success. If so, this whole call is a
            # no-op (terminal-row protection extends to the job row too).
            existing = conn.execute(
                "SELECT score, fail_reason FROM judge_results WHERE turn_id=?",
                (turn_id,),
            ).fetchone()
            already_terminal_success = (
                existing is not None
                and existing["score"] is not None
                and existing["fail_reason"] is None
            )
            if already_terminal_success:
                # Repair the job row in full. The results row is the
                # source of truth on terminality, so we coerce status to
                # 'completed', clear any stale fail_reason, and ensure
                # completed_at is non-null. This is the codex r2 P2 fix:
                # the prior version only touched rows where status was
                # not already 'completed', leaving stale fail_reason or
                # missing completed_at intact.
                conn.execute(
                    "UPDATE judge_jobs SET status='completed', "
                    "completed_at=COALESCE(completed_at, ?), "
                    "fail_reason=NULL "
                    "WHERE turn_id=?",
                    (ts, turn_id),
                )
                conn.execute("COMMIT")
                return

            # Phase 2: update job row. We're guaranteed the results row is
            # NOT terminal-success at this point, so the job-row update is
            # safe to apply.
            #
            # Guard against orphan results (codex r2 P2): rowcount==0
            # means the turn_id doesn't exist in judge_jobs. Refuse to
            # write a results row for a non-existent job — that would
            # silently corrupt the denominator.
            cur = conn.execute(
                "UPDATE judge_jobs SET status=?, completed_at=?, fail_reason=? "
                "WHERE turn_id=?",
                (new_status, ts, effective_fail_reason, turn_id),
            )
            if cur.rowcount == 0:
                # Orphan-result guard (codex r2 P2). Raise here; the
                # surrounding except clause will issue the ROLLBACK so
                # we don't double-rollback.
                raise ValueError(
                    f"commit_result: no judge_jobs row for turn_id={turn_id!r}; "
                    "refusing to create orphan judge_results row."
                )

            # Phase 3: UPSERT result row. The same terminal protection on
            # the results table is preserved (belt-and-suspenders).
            conn.execute(
                """
                INSERT INTO judge_results(turn_id, scored_at, score, rationale,
                    era_assessment, task_assessment, era_inferred, era_declared,
                    era_method, judge_model_exact, rubric_version,
                    prompt_template_version, sample_policy_version,
                    latency_ms, cost_estimate_usd, cache_hit, fail_reason,
                    attempt_id, attempt_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(turn_id) DO UPDATE SET
                    scored_at = excluded.scored_at,
                    score = excluded.score,
                    rationale = excluded.rationale,
                    era_assessment = excluded.era_assessment,
                    task_assessment = excluded.task_assessment,
                    era_inferred = excluded.era_inferred,
                    era_declared = excluded.era_declared,
                    era_method = excluded.era_method,
                    judge_model_exact = excluded.judge_model_exact,
                    rubric_version = excluded.rubric_version,
                    prompt_template_version = excluded.prompt_template_version,
                    sample_policy_version = excluded.sample_policy_version,
                    latency_ms = excluded.latency_ms,
                    cost_estimate_usd = excluded.cost_estimate_usd,
                    cache_hit = excluded.cache_hit,
                    fail_reason = excluded.fail_reason,
                    attempt_id = excluded.attempt_id,
                    attempt_count = judge_results.attempt_count + 1
                WHERE judge_results.score IS NULL
                   OR judge_results.fail_reason IS NOT NULL
                """,
                (
                    turn_id, ts, score, rationale,
                    era_assessment, task_assessment, era_inferred,
                    era_declared, era_method, judge_model_exact,
                    rubric_version, prompt_template_version,
                    sample_policy_version,
                    latency_ms, cost_estimate_usd, 1 if cache_hit else 0,
                    effective_fail_reason, attempt_id,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
