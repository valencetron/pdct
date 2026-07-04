"""Opt-in, allowlisted local telemetry for the tuner (Build 106, Task 5).

Codex plan-audit #10: "no content strings" isn't enough — enforce a fixed
field allowlist at write time. Unknown fields are DROPPED, values are coerced
to safe scalar types, corpus sizes are bucketed, and nothing free-text
(paths, exception text, seeds, queries, IDs) is ever accepted.

Rows land in ``<runtime>/tune/telemetry.jsonl`` — a local file the user can
read (``pdct tune telemetry show``) and, if they choose, send to us. There is
NO network endpoint in v1; nothing leaves the machine.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from dct.tuning import engine

SCHEMA_VERSION = 1

# field -> validator/coercer returning the stored value or None (drop row field)
_ALLOWED_VERDICTS = {"promote", "reject"}
_ALLOWED_KINDS = {"verdict", "watchdog", "converged", "reopened"}


def _num(v):
    return round(float(v), 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _corpus_bucket(n: Any) -> Optional[str]:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        return None
    for cap, label in ((100, "<100"), (1000, "100-1k"), (10000, "1k-10k")):
        if n < cap:
            return label
    return ">=10k"


ALLOWLIST = {
    "kind": lambda v: v if v in _ALLOWED_KINDS else None,
    "move": lambda v: v if (isinstance(v, str) and len(v) <= 64
                            and all(c.isalnum() or c in "-_+x" for c in v)) else None,
    "lever_changes": None,  # dict handled specially below
    "verdict": lambda v: v if v in _ALLOWED_VERDICTS else None,
    "reason": lambda v: v if (isinstance(v, str) and len(v) <= 48
                              and all(c.isalnum() or c in "_-" for c in v)) else None,
    "tier1_baseline": _num,
    "tier1_candidate": _num,
    "tier2_baseline": _num,
    "tier2_candidate": _num,
    "corpus_bucket": _corpus_bucket,
    "converged": lambda v: v if isinstance(v, bool) else None,
}


def config_path():
    return engine.tune_dir() / "config.json"


def load_config() -> dict:
    return engine._load_json(config_path(), {"enabled": False, "telemetry": False})


def save_config(cfg: dict) -> None:
    engine._save_json(config_path(), cfg)


def telemetry_path():
    return engine.tune_dir() / "telemetry.jsonl"


def _sanitize_lever_changes(changes: Any) -> Optional[dict]:
    from dct.retrieval.overrides import LEVER_SPEC
    if not isinstance(changes, dict):
        return None
    out = {}
    for k, v in changes.items():
        if k not in LEVER_SPEC:
            continue  # unknown lever names dropped
        if isinstance(v, bool) or isinstance(v, (int, float)):
            out[k] = v
    return out or None


def record(row: dict) -> bool:
    """Append one allowlisted telemetry row. Returns False (and writes
    nothing) when telemetry is disabled. Never raises."""
    try:
        if not load_config().get("telemetry"):
            return False
        clean: dict = {"schema_version": SCHEMA_VERSION,
                       "ts_day": time.strftime("%Y-%m-%d")}
        for k, v in row.items():
            if k == "lever_changes":
                sv = _sanitize_lever_changes(v)
                if sv is not None:
                    clean["lever_changes"] = sv
                continue
            fn = ALLOWLIST.get(k)
            if fn is None:
                continue  # unknown field: dropped
            cv = fn(v)
            if cv is not None:
                clean[k] = cv
        p = telemetry_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(clean, separators=(",", ":")) + "\n")
        return True
    except Exception:  # noqa: BLE001 — telemetry must never break the tuner
        return False
