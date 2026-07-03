"""Judge runner — drains queue and writes era_judge_update + composite_update rows (P1.3b).

Entry point for the judge scheduler (launchd, every 5 min). Loads the real
Haiku invoker, drains pending judge jobs, and appends:
  - era_judge_update  → utility.jsonl (era_judge score + metadata)
  - composite_update  → utility.jsonl (recomputed composite with era_judge leg)

Design: append-only (never mutates existing rows). pdct_report resolves
per-turn composite by taking the last row per turn_id where kind is
'turn' or 'composite_update'.

Codex P1 fixes (2026-05-20):
  - Write utility.jsonl BEFORE committing queue (prevents silent metric loss
    when util write fails after terminal commit).
  - Mirror worker.drain_once ok+score=None → missing_score logic so
    commit_result() never receives (None, None) which raises by contract.
  - era_judge_update rows include thread_id/topic_id for topic-filtered reports.
  - Call composite_updater.append_composite_update() after successful judge.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from dct.judge import queue, worker
from dct.judge.worker import JudgeInvocationResult, Invoker, RunSummary

log = logging.getLogger(__name__)

_UTIL_KIND = "era_judge_update"
_SCHEMA_VERSION = 1


def _append_jsonl(path: Path, row: dict) -> None:
    """Append one JSON row to path (creates file if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _lookup_turn_meta(util_path: Path, turn_id: str) -> dict:
    """Return thread_id/topic_id from the original 'turn' row in utility.jsonl.

    Codex P1 fix: era_judge_update rows need topic metadata so --topic filters
    in pdct_report.py don't silently drop them.

    Returns an empty dict if the file doesn't exist or no matching turn row found.
    """
    if not util_path.exists():
        return {}
    try:
        with util_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "turn" and r.get("turn_id") == turn_id:
                    return {
                        "thread_id": r.get("thread_id"),
                        "topic_id": r.get("topic_id") or r.get("thread_id"),
                        "chat_id": r.get("chat_id"),
                    }
    except OSError:
        pass
    return {}


def _write_era_judge_update(
    util_path: Path,
    turn_id: str,
    result: JudgeInvocationResult,
    turn_meta: dict,
) -> None:
    """Append an era_judge_update row to utility.jsonl for this turn."""
    row = {
        "kind": _UTIL_KIND,
        "schema_version": _SCHEMA_VERSION,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "turn_id": turn_id,
        # Topic metadata for --topic filtering in pdct_report (Codex P1 fix)
        "thread_id": turn_meta.get("thread_id"),
        "topic_id": turn_meta.get("topic_id"),
        "chat_id": turn_meta.get("chat_id"),
        "era_judge": result.score if result.status == "ok" else None,
        "era_assessment": result.era_assessment if result.status == "ok" else None,
        "era_judge_rationale": result.rationale if result.status == "ok" else None,
        "judge_model": result.judge_model_exact,
        "judge_latency_ms": result.latency_ms,
        "judge_status": result.status,
        "fail_reason": result.fail_reason,
    }
    _append_jsonl(util_path, row)


def _resolve_commit_args(result: JudgeInvocationResult) -> tuple:
    """Resolve (commit_score, commit_fail) from result.

    Mirrors worker.drain_once logic: status is authoritative.
    ok+score=None → fail_reason='missing_score' (Codex P1 fix: prevents
    commit_result raising on (None, None) by contract).
    """
    if result.status == "ok":
        if result.score is not None:
            return result.score, None
        else:
            return None, result.fail_reason or "missing_score"
    else:
        return None, result.fail_reason or result.status


def run_once(
    db_path: Path,
    util_path: Path,
    *,
    invoker: Optional[Invoker] = None,
    max_jobs: int = 10,
    stuck_after_s: int = 600,
) -> RunSummary:
    """Drain the judge queue and write era_judge + composite results to utility.jsonl.

    Args:
        db_path: Path to judge.db SQLite file.
        util_path: Path to utility.jsonl to append rows.
        invoker: JudgeInvocationResult callable. Defaults to real Haiku invoker.
        max_jobs: Maximum jobs to drain per call.
        stuck_after_s: Seconds before a claimed job is considered stuck.

    Returns:
        RunSummary with recovered/swept/drained counts.
    """
    if invoker is None:
        from dct.judge.invoker import invoke_judge
        invoker = invoke_judge

    recovered = queue.recover_stuck_claims(db_path, stuck_after_s=stuck_after_s)
    swept = queue.sweep_ttl(db_path)

    drained = 0
    while drained < max_jobs:
        job = queue.claim_one(db_path)
        if job is None:
            break
        drained += 1

        try:
            prompt = worker._build_prompt(job)
            result = invoker(prompt)
        except Exception as e:
            log.warning("[pdct.judge.runner] invoker exception turn=%s: %s", job.turn_id, e)
            result = JudgeInvocationResult(
                status="unexpected_error",
                score=None, rationale=None, era_assessment=None,
                task_assessment=None, latency_ms=None,
                fail_reason=repr(e), judge_model_exact=None,
            )

        commit_score, commit_fail = _resolve_commit_args(result)

        # Codex P1 fix: write utility BEFORE committing queue to DB.
        # If util write fails, the job remains in-flight (stuck) and
        # will be recovered by recover_stuck_claims on the next run.
        # This is preferable to silently losing a successful judgment.
        turn_meta = _lookup_turn_meta(util_path, job.turn_id)
        try:
            _write_era_judge_update(util_path, job.turn_id, result, turn_meta)
        except Exception as e:
            log.warning(
                "[pdct.judge.runner] util write failed turn=%s — skipping commit: %s",
                job.turn_id, e,
            )
            # Don't commit; job stays claimed and will be recovered.
            continue

        # Commit to queue DB (terminal — job won't be retried after this)
        queue.commit_result(
            db_path,
            turn_id=job.turn_id,
            score=commit_score,
            rationale=result.rationale,
            fail_reason=commit_fail,
            era_assessment=result.era_assessment,
            task_assessment=result.task_assessment,
            judge_model_exact=result.judge_model_exact,
            latency_ms=result.latency_ms,
        )

        # Codex P1 fix: call composite_updater to append composite_update row
        # so the era_judge leg contributes to pdct_utility_composite.
        if result.status == "ok" and result.score is not None:
            try:
                from dct.composite_updater import append_composite_update
                append_composite_update(
                    logs_dir=util_path.parent,
                    turn_id=job.turn_id,
                    era_judge_score=result.score,
                )
            except Exception as e:
                log.warning(
                    "[pdct.judge.runner] composite_update failed turn=%s: %s",
                    job.turn_id, e,
                )

    return RunSummary(recovered=recovered, swept=swept, drained=drained)


def main() -> None:
    """CLI entry point: `python -m dct.judge.runner`"""
    import os
    from dct.retrieval.measurement import get_logs_dir
    from dct.judge.daemon_adapter import _resolve_db_path
    from dct.judge import schema as _schema

    db_path_str = os.environ.get("PDCT_JUDGE_DB")
    db_path = Path(db_path_str) if db_path_str else _resolve_db_path()

    if not db_path.exists():
        _schema.init_db(db_path)

    util_path = get_logs_dir() / "utility.jsonl"

    summary = run_once(db_path=db_path, util_path=util_path)
    log.info(
        "[pdct.judge.runner] done — recovered=%d swept=%d drained=%d",
        summary.recovered, summary.swept, summary.drained,
    )
    print(f"recovered={summary.recovered} swept={summary.swept} drained={summary.drained}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


__all__ = ["run_once", "main"]
