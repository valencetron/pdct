import json
import os
import subprocess
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _run(args, env):
    return subprocess.run(
        [sys.executable, "-m", "dct.retrieval.levers_cli", *args],
        cwd=SRC, capture_output=True, text=True, env=env,
    )


def test_cli_get_returns_lever_list(tmp_path):
    env = dict(os.environ, PDCT_OVERRIDES_PATH=str(tmp_path / "ov.json"))
    r = _run(["get"], env)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert "levers" in data and len(data["levers"]) == 11
    keys = {lv["key"] for lv in data["levers"]}
    assert "cascade_score_floor" in keys
    floor = next(lv for lv in data["levers"] if lv["key"] == "cascade_score_floor")
    assert floor["value"] == 0.10
    assert floor["default"] == 0.10
    assert floor["min"] == 0.0 and floor["max"] == 0.5
    assert floor["source"] == "default"


def test_cli_set_then_get_reflects_override(tmp_path):
    env = dict(os.environ, PDCT_OVERRIDES_PATH=str(tmp_path / "ov.json"))
    s = _run(["set", "cascade_top_k", "40"], env)
    assert s.returncode == 0, s.stderr
    g = _run(["get"], env)
    data = json.loads(g.stdout)
    topk = next(lv for lv in data["levers"] if lv["key"] == "cascade_top_k")
    assert topk["value"] == 40
    assert topk["source"] == "override"


def test_cli_get_detects_env_source_with_real_var_name(tmp_path):
    # env source must use the ACTUAL var name (DCT_TRANSITIONS_BIAS), not
    # DCT_CASCADE_TRANSITIONS_BIAS, and report the effective value.
    env = dict(os.environ, PDCT_OVERRIDES_PATH=str(tmp_path / "ov.json"),
               DCT_TRANSITIONS_BIAS="0.8")
    g = _run(["get"], env)
    data = json.loads(g.stdout)
    bias = next(lv for lv in data["levers"] if lv["key"] == "cascade_transitions_bias")
    assert bias["source"] == "env"
    assert bias["value"] == 0.8


def test_cli_unknown_lever_returns_error_json(tmp_path):
    env = dict(os.environ, PDCT_OVERRIDES_PATH=str(tmp_path / "ov.json"))
    r = _run(["set", "bogus", "1"], env)
    data = json.loads(r.stdout)
    assert data["ok"] is False and "bogus" in data["error"]


def test_cli_get_reads_real_composite_from_utility_log(tmp_path):
    # composite IS logged (kind=composite_update rows). The score block reads
    # them via $PDCT_LOGS_DIR.
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "utility.jsonl").write_text("\n".join([
        json.dumps({"kind": "composite_update", "pdct_utility_composite": 0.12,
                    "composite_legs_used": ["cosine_score"], "ts": "2026-06-08T03:00:00Z"}),
        json.dumps({"kind": "composite_update", "pdct_utility_composite": 0.16,
                    "composite_legs_used": ["cosine_score"], "ts": "2026-06-08T03:10:00Z"}),
    ]) + "\n")
    env = dict(os.environ, PDCT_OVERRIDES_PATH=str(tmp_path / "ov.json"),
               PDCT_LOGS_DIR=str(logs))
    g = _run(["get"], env)
    data = json.loads(g.stdout)
    s = data["score"]
    assert s["available"] is True
    assert s["composite"] == 0.14            # mean(0.12, 0.16)
    assert s["latest"] == 0.16
    assert s["legsUsed"] == ["cosine_score"]
