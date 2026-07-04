"""Runtime override file for the 11 tunable PDCT retrieval levers.

build_config() reads this file FRESH per retrieval (no caching), layered on top
of env+defaults, so a write takes effect on the very next turn — no daemon
restart. Values are clamped to safe bounds; out-of-range is clamped, wrong-type
is dropped (falls through to env/default). A blank/corrupt file is treated as
{} so retrieval never breaks.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import logging
import math
import os
from typing import Any, Dict, Optional

log = logging.getLogger("dct.overrides")

OVERRIDES_PATH = os.environ.get("PDCT_OVERRIDES_PATH") or os.path.join(
    os.environ.get("PDCT_RUNTIME_DIR") or os.path.join(
        os.environ.get("PDCT_HOME") or os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "runtime"),
    "pdct-overrides.json",
)

# key -> (type, default, min, max, env). min/max are None for bools.
# `env` is the ACTUAL env var build_config() reads (NOT DCT_+key.upper()).
LEVER_SPEC: Dict[str, Dict[str, Any]] = {
    "cascade_score_floor":               {"type": "float", "default": 0.10, "min": 0.0, "max": 0.5, "env": "DCT_CASCADE_SCORE_FLOOR"},
    "cascade_top_k":                     {"type": "int",   "default": 20,   "min": 1,   "max": 200, "env": "DCT_CASCADE_TOP_K"},
    "cascade_heat_enabled":              {"type": "bool",  "default": True, "min": None,"max": None, "env": "DCT_CASCADE_HEAT_ENABLED"},
    "cascade_heat_floor":                {"type": "float", "default": 0.01, "min": 0.0, "max": 0.5, "env": "DCT_CASCADE_HEAT_FLOOR"},
    "cascade_heat_half_life_s":          {"type": "float", "default": 21600.0, "min": 600.0, "max": 172800.0, "env": "DCT_CASCADE_HEAT_HALF_LIFE_S"},
    "cascade_eligibility_filter_enabled":{"type": "bool",  "default": True, "min": None,"max": None, "env": "DCT_CASCADE_ELIGIBILITY_FILTER"},
    "cascade_transitions_bias":          {"type": "float", "default": 0.5,  "min": 0.0, "max": 3.0, "env": "DCT_TRANSITIONS_BIAS"},
    "cascade_vec_near_decay":            {"type": "float", "default": 0.2,  "min": 0.0, "max": 1.0, "env": "DCT_VEC_NEAR_DECAY"},
    "cascade_decay":                     {"type": "float", "default": 0.40, "min": 0.1, "max": 0.8, "env": "DCT_CASCADE_DECAY"},
    "cascade_depth":                     {"type": "int",   "default": 2,    "min": 1,   "max": 4,   "env": "DCT_CASCADE_DEPTH"},
    "cascade_transitions_enabled":       {"type": "bool",  "default": True, "min": None,"max": None, "env": "DCT_TRANSITIONS_ENABLED"},
}


def clamp(key: str, value: Any) -> Optional[Any]:
    """Coerce+clamp a value per LEVER_SPEC. Returns None if key unknown or value
    is the wrong type (caller drops it)."""
    spec = LEVER_SPEC.get(key)
    if spec is None:
        return None
    t = spec["type"]
    if t == "bool":
        return value if isinstance(value, bool) else None
    # numeric — reject bool (bool is subclass of int), non-numbers, nan/inf
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if not math.isfinite(v):   # nan / inf -> drop
        return None
    v = max(spec["min"], min(spec["max"], v))
    return int(v) if t == "int" else v


def load_overrides(path: str = OVERRIDES_PATH) -> Dict[str, Any]:
    """Read + clamp the override file. Missing/corrupt -> {}. Unknown keys and
    wrong-type values are dropped."""
    try:
        with open(path) as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        cv = clamp(k, v)
        if cv is not None:
            out[k] = cv
    return out


def _meta_path(path: str) -> str:
    return path + ".meta.json"


def read_meta(path: str = OVERRIDES_PATH) -> Dict[str, Any]:
    try:
        with open(_meta_path(path)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


@contextlib.contextmanager
def _file_lock(path: str):
    """Exclusive lock via a sidecar .lock file so concurrent slider writes don't
    lose updates (atomic replace prevents corruption, not lost
    read-modify-write races)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = path + ".lock"
    lf = open(lock_path, "w")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
        lf.close()


def write_override(key: str, value: Any, path: str = OVERRIDES_PATH) -> Dict[str, Any]:
    """Validate+clamp+merge a single lever into the override file under an
    exclusive lock (no lost updates). Stamps sinceChangeAt in the sidecar meta
    for the sample counter. Raises ValueError on unknown key or unusable value."""
    cv = clamp(key, value)
    if cv is None:
        raise ValueError(f"invalid lever or value: {key}={value!r}")
    with _file_lock(path):
        cur = load_overrides(path)   # read INSIDE the lock
        cur[key] = cv
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cur, f, indent=2)
        os.replace(tmp, path)
        now = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
        mtmp = _meta_path(path) + ".tmp"
        with open(mtmp, "w") as f:
            json.dump({"sinceChangeAt": now, "lastKey": key}, f)
        os.replace(mtmp, _meta_path(path))
    return cur


def write_overrides_batch(
    changes: Dict[str, Any], path: str = OVERRIDES_PATH
) -> Dict[str, Any]:
    """Apply MULTIPLE lever changes as one atomic transaction (Build 106,
    Codex plan-audit #2 — combo promotion must not partially apply).

    A value of None means "delete this key" (drop back to env/default).
    All changes are validated FIRST; any invalid key/value aborts the whole
    batch (ValueError) with the file untouched. The merged result is written
    via a single tmp-file + os.replace under the flock, so a crash leaves the
    file either fully-old or fully-new — never mixed.
    """
    validated: Dict[str, Any] = {}
    for k, v in changes.items():
        if v is None:
            if k not in LEVER_SPEC:
                raise ValueError(f"unknown lever: {k}")
            validated[k] = None
            continue
        cv = clamp(k, v)
        if cv is None:
            raise ValueError(f"invalid lever or value: {k}={v!r}")
        validated[k] = cv
    with _file_lock(path):
        cur = load_overrides(path)   # read INSIDE the lock
        for k, v in validated.items():
            if v is None:
                cur.pop(k, None)
            else:
                cur[k] = v
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cur, f, indent=2)
        os.replace(tmp, path)
        now = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
        mtmp = _meta_path(path) + ".tmp"
        with open(mtmp, "w") as f:
            json.dump({"sinceChangeAt": now,
                       "lastKey": ",".join(sorted(validated))}, f)
        os.replace(mtmp, _meta_path(path))
    return cur


def delete_override(key: str, path: str = OVERRIDES_PATH) -> Dict[str, Any]:
    """Remove a SINGLE lever from the override file under the lock, atomically,
    leaving the other overrides intact. Used by the auto-tuner to revert a
    candidate that had no prior override (drop it back to env/default) without
    disturbing other promoted levers. No-op if the key isn't present."""
    with _file_lock(path):
        cur = load_overrides(path)
        if key not in cur:
            return cur
        cur.pop(key, None)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cur, f, indent=2)
        os.replace(tmp, path)
        now = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
        mtmp = _meta_path(path) + ".tmp"
        with open(mtmp, "w") as f:
            json.dump({"sinceChangeAt": now, "lastKey": key}, f)
        os.replace(mtmp, _meta_path(path))
    return cur


def reset_overrides(path: str = OVERRIDES_PATH) -> None:
    """Delete the override file (and meta) -> back to env/defaults."""
    for p in (path, _meta_path(path)):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
