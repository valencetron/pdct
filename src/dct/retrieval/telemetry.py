"""Append-only JSONL telemetry for memory_api calls.

One line per query_memory / read_memory invocation. Writes are best-effort —
logging never raises. Used to surface graph gaps, fallback frequency, and
per-surface tool-call rates.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

LOG_PATH = (
    __import__("dct.config", fromlist=["config"]).logs_dir() / "retrieval.jsonl"
)
_SEED_CAP = 512


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_call(
    *,
    surface: str,
    fn: str,
    seed: str,
    result_count: int,
    used_fallback: bool,
    latency_ms: int,
) -> None:
    rec = {
        "ts": _now_iso(),
        "surface": surface,
        "fn": fn,
        "seed": (seed or "")[:_SEED_CAP],
        "result_count": int(result_count),
        "used_fallback": bool(used_fallback),
        "latency_ms": int(latency_ms),
    }
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass  # best-effort — never raise
