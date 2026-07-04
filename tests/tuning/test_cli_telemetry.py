"""CLI semantics + telemetry allowlist tests (Tasks 4 & 5)."""
import json

import pytest

from dct.cli import build_parser
from dct.retrieval import overrides as ov
from dct.tuning import engine, telemetry
from dct.tuning.cli import cmd_tune


@pytest.fixture(autouse=True)
def _tune_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_RUNTIME_DIR", str(tmp_path / "runtime"))
    yield tmp_path


def _run(argv):
    args = build_parser().parse_args(argv)
    return args.fn(args)


# ── CLI flag semantics ────────────────────────────────────────────────────

def test_start_sets_flag_and_seeds_queue(capsys):
    assert _run(["tune", "start"]) == 0
    assert telemetry.load_config()["enabled"] is True
    q = engine._load_json(engine.tune_dir() / "candidates.json", None)
    assert q == engine.DEFAULT_CANDIDATES


def test_stop_leaves_overrides_in_place(tmp_path, monkeypatch, capsys):
    p = str(tmp_path / "ov.json")
    monkeypatch.setattr(ov, "OVERRIDES_PATH", p)
    ov.write_override("cascade_depth", 3, path=p)
    _run(["tune", "start"])
    _run(["tune", "stop"])
    assert telemetry.load_config()["enabled"] is False
    assert ov.load_overrides(p) == {"cascade_depth": 3}  # untouched


def test_tick_refuses_when_disabled(capsys):
    assert _run(["tune", "tick"]) == 2


def test_restart_clears_convergence(capsys):
    engine._save_json(engine.tune_dir() / "state.json",
                      {"converged": True, "consecutive_rejections": 4,
                       "done": ["cascade_depth"]})
    _run(["tune", "restart"])
    st = engine._load_json(engine.tune_dir() / "state.json", {})
    assert st["converged"] is False and st["done"] == []


def test_status_renders(capsys):
    assert _run(["tune", "status"]) == 0
    out = capsys.readouterr().out
    assert "levers" in out and "cascade_decay" in out


# ── telemetry (Task 5) ────────────────────────────────────────────────────

def test_telemetry_off_by_default_writes_nothing():
    ok = telemetry.record({"kind": "verdict", "verdict": "promote"})
    assert ok is False
    assert not telemetry.telemetry_path().exists()


def test_telemetry_allowlist_drops_unknown_and_freetext():
    cfg = telemetry.load_config()
    cfg["telemetry"] = True
    telemetry.save_config(cfg)
    telemetry.record({
        "kind": "verdict", "verdict": "promote", "move": "cascade_depth",
        "reason": "tier2_improved",
        "tier1_candidate": 0.87654321,
        "corpus_bucket": 4321,
        "lever_changes": {"cascade_depth": 3, "evil_key": 9},
        "seed": "private query text",             # unknown -> dropped
        "path": "/home/example/secret",          # unknown -> dropped
        "error": "Traceback ...",                 # unknown -> dropped
    })
    rows = [json.loads(l) for l in
            telemetry.telemetry_path().read_text().splitlines()]
    row = rows[-1]
    assert row["schema_version"] == 1
    assert row["corpus_bucket"] == "1k-10k"       # bucketed, not exact
    assert row["tier1_candidate"] == 0.8765       # rounded
    assert row["lever_changes"] == {"cascade_depth": 3}
    for banned in ("seed", "path", "error"):
        assert banned not in row
    # exhaustiveness: every stored field is allowlisted or structural
    structural = {"schema_version", "ts_day", "lever_changes"}
    assert set(row) <= set(telemetry.ALLOWLIST) | structural


def test_telemetry_bad_verdict_dropped():
    cfg = telemetry.load_config()
    cfg["telemetry"] = True
    telemetry.save_config(cfg)
    telemetry.record({"kind": "verdict", "verdict": "DROP TABLE"})
    row = json.loads(telemetry.telemetry_path().read_text().splitlines()[-1])
    assert "verdict" not in row
