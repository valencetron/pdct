"""Measurement helpers for PDCT prelim metrics.

Pure functions (turn_id, ablation_roll), constants (SKIP_REASONS),
and a best-effort JSONL appender. No daemon-side state.

Spec: docs/superpowers/specs/2026-04-29-pdct-prelim-metrics-spec.md (v4)
Plan: docs/superpowers/plans/2026-04-29-pdct-prelim-metrics-plan.md (v3)
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, Optional


# Where measurement/utility logs live. Function (not constant) so tests can
# monkeypatch PDCT_LOGS_DIR at call time without re-importing.
_DEFAULT_LOGS_DIR = (
    Path(__file__).resolve().parents[3] / "logs"
)  # repo_root/logs


def get_logs_dir() -> Path:
    """Return the directory where measurement.jsonl/utility.jsonl live.

    Honors $PDCT_LOGS_DIR. Read at call time, NOT at import time, so tests
    that set the env after import still work.
    """
    override = os.environ.get("PDCT_LOGS_DIR")
    if override:
        return Path(override)
    return _DEFAULT_LOGS_DIR


def turn_id_from(
    chat_id: Any,
    thread_id: Any,
    turn_index: int,
    started_at_unix_ms: int,
) -> str:
    """Build a stable turn_id.

    Format: f"{chat_id}|{thread_id}|{turn_index}|{started_at_unix_ms}".
    Pipe-delimited, four parts. Coerces all parts via str(); None becomes "None".
    Globally unique even across daemon restarts (turn_index resets but
    started_at_unix_ms always advances).
    """
    return f"{chat_id}|{thread_id}|{turn_index}|{started_at_unix_ms}"


def ablation_roll(turn_id: str, seed: Optional[str]) -> float:
    """Deterministic uniform [0,1) draw conditioned on (seed, turn_id).

    Uses sha256(seed|turn_id) → first 8 bytes as big-endian uint64 / 2**64.
    Same (seed, turn_id) → same float, forever. No process-global RNG.

    If seed is None, falls back to env $PDCT_ABLATION_SEED (read at call
    time), then to empty string. Empty seed is still deterministic per
    turn_id — useful for "no seed configured" environments where we
    still want the same turn to flip the same way across restarts.
    """
    if seed is None:
        seed = os.environ.get("PDCT_ABLATION_SEED", "")
    h = hashlib.sha256(f"{seed}|{turn_id}".encode("utf-8")).digest()
    n = struct.unpack(">Q", h[:8])[0]
    return n / (1 << 64)


# Skip-reason taxonomy. See spec §Stage 0.
SKIP_REASONS = frozenset({
    "none",          # cascade ran, returned hits, injected
    "ablation",      # rate roll skipped injection
    "disabled",      # DCT_CASCADE_DISABLED=1
    "error",         # cascade raised
    "empty_result",  # cascade ran, 0 hits
    "no_concepts",   # extractor returned nothing for user_text
    "cascade_timeout",  # Build 86: cascade exceeded the time bound (latency,
                        # not seeding) — kept distinct so rebuild timeouts
                        # don't masquerade as genuine no_concepts turns.
})


def _append_jsonl(path: Path, row: dict) -> None:
    """Append `row` as a single JSON line to `path`. Best-effort.

    Auto-creates parent directories. Single os.write of the encoded line
    + newline; on POSIX a single write of a small buffer (<4KB) to a
    regular file is effectively atomic vs. concurrent appenders within
    the same process. Daemon is single-process under launchd.

    Never raises. Hot-path callers (daemon turn boundary) cannot fail
    a turn because of metrics-write errors.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:  # noqa: BLE001 — best-effort, never block a turn
        # Could log via a module logger but that's its own dependency.
        # Caller observability: if rows stop appearing, check daemon log.
        pass
