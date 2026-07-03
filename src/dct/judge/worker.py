"""Worker drain loop for the PDCT-utility judge (P1.3a).

Runs in a subprocess (or test process) and drains pending jobs by:
  1. Recovering stuck claims from prior crashed runs.
  2. Sweeping TTL-expired pending jobs.
  3. Draining up to max_jobs by claiming, invoking, and committing.

The codex invocation is abstracted as a callable parameter — substrate-only
P1.3a passes a stub. P1.3b will pass the real codex_client.invoke wrapper.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from . import queue

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeInvocationResult:
    """What an invoker returns for a single judge call."""
    status: Literal["ok", "timeout", "parse_error", "exit_nonzero",
                    "schema_violation", "unexpected_error"]
    score: Optional[int]
    rationale: Optional[str]
    era_assessment: Optional[str]
    task_assessment: Optional[str]
    latency_ms: Optional[int]
    fail_reason: Optional[str]
    judge_model_exact: Optional[str]


@dataclass(frozen=True)
class RunSummary:
    recovered: int
    swept: int
    drained: int


# Type alias for the invoker callable.
Invoker = Callable[[str], JudgeInvocationResult]


def _build_prompt(job: queue.Job) -> str:
    """Build the structured judge prompt from a job payload.

    Format: labelled sections for user message, cascade block, and reply.
    The invoker's system prompt tells the model how to score — this function
    only structures the content. Era label is included when present so the
    judge can contextualize the cascade block's source.
    """
    p = job.payload
    era = job.era_at_enqueue or "unknown"
    lines = [
        f"## Era: {era}",
        "",
        "## User message",
        p.get("user_text", "").strip() or "(empty)",
        "",
        "## Cascade context injected",
        p.get("cascade_block", "").strip() or "(none)",
        "",
        "## AI reply",
        p.get("reply_text", "").strip() or "(empty)",
    ]
    return "\n".join(lines)


def drain_once(
    db_path: str | Path,
    *,
    invoker: Invoker,
    max_jobs: int = 10,
) -> int:
    """Claim and process up to max_jobs pending jobs.

    Returns the number of jobs whose status transitioned (completed | failed).
    Per-job invoker exceptions are caught and committed as
    fail_reason='unexpected_error' so one bad job doesn't halt the drain.

    Result reconciliation (codex r1 P2 fix): the invoker's ``status`` is
    the source of truth. If status != "ok", we synthesize ``fail_reason``
    from status and clear ``score`` — even if the invoker also returned
    a score, we don't trust it. This eliminates the "two sources of
    truth" gap where an invoker could simultaneously claim ok+score and
    a non-ok status.
    """
    n = 0
    while n < max_jobs:
        job = queue.claim_one(db_path)
        if job is None:
            break
        n += 1
        try:
            result = invoker(_build_prompt(job))
        except Exception as e:
            log.warning("judge_invoker_exception",
                        extra={"turn_id": job.turn_id, "err": repr(e)})
            queue.commit_result(
                db_path,
                turn_id=job.turn_id,
                score=None,
                rationale=None,
                fail_reason="unexpected_error",
            )
            continue

        # status is authoritative; failed status overrides any returned score.
        if result.status == "ok":
            commit_score = result.score
            commit_fail_reason = None
            # If status says ok but score is missing, surface a synthetic
            # missing_score (queue.commit_result will reject (None, None)
            # outright; we want the benign branch where the invoker
            # contract is honest about what's wrong).
            if commit_score is None:
                commit_fail_reason = result.fail_reason or "missing_score"
        else:
            commit_score = None
            commit_fail_reason = result.fail_reason or result.status

        queue.commit_result(
            db_path,
            turn_id=job.turn_id,
            score=commit_score,
            rationale=result.rationale,
            fail_reason=commit_fail_reason,
            era_assessment=result.era_assessment,
            task_assessment=result.task_assessment,
            judge_model_exact=result.judge_model_exact,
            latency_ms=result.latency_ms,
        )
    return n


def run_once(
    db_path: str | Path,
    *,
    invoker: Invoker,
    max_jobs: int = 10,
    stuck_after_s: int = 600,
) -> RunSummary:
    """Full cycle: recover stuck claims → TTL sweep → drain.

    Designed for `launchd --once` cadence (every ~5min in production).
    """
    recovered = queue.recover_stuck_claims(db_path, stuck_after_s=stuck_after_s)
    swept = queue.sweep_ttl(db_path)
    drained = drain_once(db_path, invoker=invoker, max_jobs=max_jobs)
    return RunSummary(recovered=recovered, swept=swept, drained=drained)


__all__ = [
    "JudgeInvocationResult",
    "RunSummary",
    "Invoker",
    "drain_once",
    "run_once",
]
