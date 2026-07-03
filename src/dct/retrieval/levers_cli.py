"""JSON CLI bridge for the lever panel API (Node shells out to this).

  python -m dct.retrieval.levers_cli get
  python -m dct.retrieval.levers_cli set <key> <value>
  python -m dct.retrieval.levers_cli reset

Override path may be overridden via PDCT_OVERRIDES_PATH (for tests/the API).
The composite score is read from utility.jsonl (kind=composite_update rows) via
$PDCT_LOGS_DIR — verified to exist in the live log.
"""
from __future__ import annotations

import json
import os
import sys

from dct.retrieval import overrides as ov

SAMPLES_NEEDED = 80


def _path() -> str:
    return os.environ.get("PDCT_OVERRIDES_PATH", ov.OVERRIDES_PATH)


def _utility_path() -> str:
    """Resolve utility.jsonl via the SAME helper the engine uses (honors
    $PDCT_LOGS_DIR). Time field is `ts` (ISO8601); composite rows are
    kind=composite_update with `pdct_utility_composite`."""
    from dct.retrieval.measurement import get_logs_dir
    return str(get_logs_dir() / "utility.jsonl")


def _composite_rows():
    """Yield parsed composite_update rows from utility.jsonl (best-effort)."""
    try:
        with open(_utility_path()) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("kind") == "composite_update" and row.get("pdct_utility_composite") is not None:
                    yield row
    except OSError:
        return


def _score() -> dict:
    """REAL composite score from utility.jsonl: rolling mean over the last
    SAMPLES_NEEDED rows + CI (mean±1.96·sd/√n), latest, samples-since-change,
    and the honest per-row leg count from `composite_legs_used`."""
    import math
    meta = ov.read_meta(_path())
    since = meta.get("sinceChangeAt")
    rows = list(_composite_rows())
    if not rows:
        return {"available": False, "composite": None, "samples": 0,
                "samplesNeeded": SAMPLES_NEEDED, "sinceChangeAt": since,
                "legsUsed": [], "legsTotal": 4}
    vals = [float(r["pdct_utility_composite"]) for r in rows]
    window = vals[-SAMPLES_NEEDED:]
    mean = sum(window) / len(window)
    if len(window) > 1:
        sd = (sum((v - mean) ** 2 for v in window) / (len(window) - 1)) ** 0.5
        half = 1.96 * sd / math.sqrt(len(window))
        ci = [round(mean - half, 4), round(mean + half, 4)]
    else:
        ci = None
    samples_since = sum(1 for r in rows if (r.get("ts") or "") >= since) if since else 0
    legs = rows[-1].get("composite_legs_used") or []
    return {
        "available": True,
        "composite": round(mean, 4),
        "latest": round(vals[-1], 4),
        "ci": ci,
        "samples": samples_since,
        "samplesNeeded": SAMPLES_NEEDED,
        "sinceChangeAt": since,
        "legsUsed": legs,
        "legsTotal": 4,
    }


def _effective_values() -> dict:
    """Effective per-lever values from the engine's OWN build_config(). The CLI
    points build_config at the SAME override file via PDCT_OVERRIDES_PATH by
    temporarily aligning the module path."""
    from dct.retrieval import service
    # Make build_config read our (possibly test) override file.
    service.OVERRIDES_PATH = _path()
    ov.OVERRIDES_PATH = _path()
    cfg = service.build_config()
    return {key: getattr(cfg, key) for key in ov.LEVER_SPEC}


def _get() -> dict:
    path = _path()
    active = ov.load_overrides(path)
    eff = _effective_values()
    levers = []
    for key, spec in ov.LEVER_SPEC.items():
        if key in active:
            source = "override"
        elif os.environ.get(spec["env"]) is not None:   # ACTUAL env name
            source = "env"
        else:
            source = "default"
        levers.append({
            "key": key, "type": spec["type"],
            "value": eff[key],                 # effective value from build_config
            "default": spec["default"], "min": spec["min"], "max": spec["max"],
            "source": source,
        })
    return {"levers": levers, "score": _score()}


def main(argv):
    if not argv or argv[0] == "get":
        print(json.dumps(_get()))
        return 0
    if argv[0] == "set" and len(argv) == 3:
        key, raw = argv[1], argv[2]
        spec = ov.LEVER_SPEC.get(key)
        if spec is None:
            print(json.dumps({"ok": False, "error": f"unknown lever {key}"})); return 1
        try:
            if spec["type"] == "bool":
                low = raw.strip().lower()
                if low in ("1", "true", "yes", "on"):
                    val = True
                elif low in ("0", "false", "no", "off"):
                    val = False
                else:
                    print(json.dumps({"ok": False, "error": f"bad bool {raw!r}"})); return 1
            elif spec["type"] == "int":
                fv = float(raw)
                if fv != int(fv):   # reject non-integral: "40.9" invalid
                    print(json.dumps({"ok": False, "error": f"non-integer {raw!r}"})); return 1
                val = int(fv)
            else:
                val = float(raw)
        except ValueError:
            print(json.dumps({"ok": False, "error": f"bad value {raw!r}"})); return 1
        try:
            ov.write_override(key, val, path=_path())
        except ValueError as e:
            print(json.dumps({"ok": False, "error": str(e)})); return 1
        print(json.dumps({"ok": True})); return 0
    if argv[0] == "reset":
        ov.reset_overrides(path=_path())
        print(json.dumps({"ok": True})); return 0
    print(json.dumps({"ok": False, "error": "usage: get|set <k> <v>|reset"})); return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
