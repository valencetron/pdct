"""DCT scheduler — periodic voice ingest + cross-source distillation.

One-shot script designed to be invoked by launchd every 5 minutes. Each
run does:

  1. Ingest any new voice transcripts (~/example-stack/tools/transcripts/*.json)
     into events.jsonl with dedupe.
  2. Distill pending sessions across all sources (telegram, claude-code,
     voice). Distilled notes land in the vault, where the vault-watcher
     picks them up and surfaces them on the Context Stream rail.

Limits each distillation run to avoid long jobs: default 20 groups per run.
Logs concise output to stderr so launchd's StandardErrorPath captures it.

CLI:
  python -m dct.scheduler
    [--limit N]  distill at most N groups this run (default 20)
    [--model M]  distiller LLM model (default haiku)
    [--quiet]    suppress per-step summary lines
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dct.event_log import EventLog


from dct import config as _cfg

EVENTS_JSONL = _cfg.events_path()
VAULT_ROOT = _cfg.vault_roots()[0].parent if _cfg.vault_roots() else _cfg.pdct_home() / "vault"
import os as _os
# Single source of truth (config.transcripts_glob): env override, else a real
# dir inside PDCT_HOME that install.sh scaffolds — never a phantom default
# path that made the scheduler silently ingest nothing.
TRANSCRIPTS_GLOB = _cfg.transcripts_glob()
CAPTURE_SOURCE = _cfg.capture_source()


def _emit(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(f"[scheduler {time.strftime('%I:%M %p')}] {msg}", file=sys.stderr, flush=True)


def _ingest_transcripts(*, quiet: bool) -> int:
    """Ingest transcripts with the configured capture source. Returns number
    of events appended (best-effort)."""
    try:
        from dct.ingest import run_ingest  # type: ignore
    except ImportError:
        run_ingest = None

    if run_ingest is not None:
        # Prefer direct call when available
        try:
            count = run_ingest(
                source=CAPTURE_SOURCE,
                input_glob=TRANSCRIPTS_GLOB,
                log_path=EVENTS_JSONL,
                dedupe=True,
            )
            _emit(f"{CAPTURE_SOURCE} ingest: +{count} events", quiet=quiet)
            return int(count or 0)
        except Exception as e:
            _emit(f"{CAPTURE_SOURCE} ingest (direct) failed: {e}", quiet=quiet)

    # Fallback — shell out to the CLI
    import subprocess
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "dct.ingest",
                "--source", CAPTURE_SOURCE,
                "--input", TRANSCRIPTS_GLOB,
                "--log", str(EVENTS_JSONL),
                "--dedupe",
            ],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            _emit(f"{CAPTURE_SOURCE} ingest CLI rc={proc.returncode} stderr={proc.stderr[-200:]}", quiet=quiet)
            return 0
        _emit(f"{CAPTURE_SOURCE} ingest: ok", quiet=quiet)
        return 0
    except subprocess.TimeoutExpired:
        _emit(f"{CAPTURE_SOURCE} ingest: timeout (120s)", quiet=quiet)
        return 0


_ingest_voice = _ingest_transcripts  # legacy alias


def _distill_pending(*, limit: int, model: str, quiet: bool) -> dict[str, int]:
    """Run distiller once; return counts by status."""
    from dct.distiller import group_events_by_session, distill_one

    log = EventLog(EVENTS_JSONL)
    events = log.read_all()
    groups = group_events_by_session(events)

    counts = {"written": 0, "skipped": 0, "empty": 0, "error": 0, "total": 0}
    processed = 0
    for g in groups:
        if limit > 0 and processed >= limit:
            break
        status = distill_one(
            group=g,
            vault_root=VAULT_ROOT,
            source_roots=None,
            model=model,
            force=False,
        )
        counts["total"] += 1
        if status == "written":
            counts["written"] += 1
        elif status == "skipped":
            counts["skipped"] += 1
        elif status == "empty":
            counts["empty"] += 1
        else:
            counts["error"] += 1
            if counts["error"] <= 3:
                _emit(f"distill error: group={g.session_id[:12]} status={status}", quiet=quiet)
        processed += 1
    return counts


def main() -> int:
    p = argparse.ArgumentParser(prog="dct.scheduler")
    p.add_argument("--limit", type=int, default=20,
                   help="max groups to distill per run (0 = unbounded)")
    p.add_argument("--model", default="haiku",
                   help="distiller model (default: haiku)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    start = time.time()
    _emit("run start", quiet=args.quiet)

    # 1. Ingest transcripts
    try:
        _ingest_transcripts(quiet=args.quiet)
    except Exception as e:
        _emit(f"{CAPTURE_SOURCE} ingest crashed: {e}", quiet=args.quiet)

    # 2. Distill pending
    try:
        counts = _distill_pending(limit=args.limit, model=args.model, quiet=args.quiet)
        _emit(
            f"distill: written={counts['written']} skipped={counts['skipped']} "
            f"empty={counts['empty']} error={counts['error']} total={counts['total']}",
            quiet=args.quiet,
        )
    except Exception as e:
        _emit(f"distill crashed: {e}", quiet=args.quiet)

    # 2.5 Tuning tick (Build 106) — no-op unless `pdct tune start` enabled it.
    try:
        from dct.tuning import telemetry as _tt
        if _tt.load_config().get("enabled"):
            from dct.tuning import engine as _te
            r = _te.run_tick()
            _emit(f"tune: {r.action} move={r.move} verdict={r.verdict} {r.reason}",
                  quiet=args.quiet)
    except Exception as e:
        _emit(f"tune tick crashed: {e}", quiet=args.quiet)

    # 3. Rebuild regions (brain-map cluster detection)
    try:
        from dct.regions import build_regions, DEFAULT_OUTPUT
        data = build_regions()
        DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        import json
        DEFAULT_OUTPUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        _emit(
            f"regions: {data['total_clusters']} clusters / {data['total_concepts']} concepts",
            quiet=args.quiet,
        )
    except Exception as e:
        _emit(f"regions crashed: {e}", quiet=args.quiet)

    elapsed = time.time() - start
    _emit(f"run done in {elapsed:.1f}s", quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
