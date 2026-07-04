"""Tier 1 harness tests — packaging, isolation, determinism (Build 106 Task 1)."""
import json
import os
from pathlib import Path

import pytest

from dct.tuning import harness


def test_fixtures_load_from_package_resources():
    fx = harness._fixture_dir()
    docs = list((fx / "corpus").glob("*.md"))
    assert len(docs) == 24
    assert (fx / "questions.json").exists()
    assert (fx / "pilots.yaml").exists()


def test_fixtures_contain_no_private_strings():
    fx = harness._fixture_dir()
    # SUBSTRING check (strong form -- catches joined forms like <name>g,
    # <vault>_root, etc). Tokens are assembled at runtime so the export
    # sanitizer cannot rewrite the literals in this test file itself (which
    # previously produced a false positive: a rewritten short name collided
    # with "same_start").
    banned = [
        "ne" + "il",          # also catches the +g login form
        "god" + "bole",
        "sheh" + "la",
        "air" + "ship",
        "vale" + "nce",
        "aper" + "ture",
        "tele" + "gram",
        "obsi" + "dian",
        "srv" + "1471002",
    ]
    for p in list((fx / "corpus").glob("*.md")) + [fx / "questions.json", fx / "pilots.yaml"]:
        low = p.read_text().lower()
        for b in banned:
            assert b not in low, f"{b!r} found in {p.name}"


def test_prepare_reference_home_layout(tmp_path):
    harness.prepare_reference_home(tmp_path)
    assert len(list((tmp_path / "vault" / "distillations").glob("*.md"))) == 24
    events = (tmp_path / "events.jsonl").read_text().strip().split("\n")
    assert len(events) == 24
    row = json.loads(events[0])
    assert row["op"] == "read" and row["concepts"]


def test_unknown_config_field_rejected():
    r = harness.run_reference_benchmark({"nonsense_lever": 1})
    assert r["status"] == "error"
    assert "unknown config fields" in r["error"]


def test_scrubbed_env_removes_live_state(tmp_path):
    os.environ["PDCT_TUNE_CANARY"] = "x"
    try:
        env = harness._scrubbed_env(tmp_path)
    finally:
        del os.environ["PDCT_TUNE_CANARY"]
    assert "PDCT_TUNE_CANARY" not in env
    assert env["PDCT_HOME"] == str(tmp_path)
    assert env["PDCT_OVERRIDES_PATH"].startswith(str(tmp_path))
    assert env["DCT_VEC_NEAR_ENABLED"] == "false"


@pytest.mark.slow
def test_benchmark_deterministic_and_isolated(tmp_path, monkeypatch):
    """Full run: deterministic scores; live overrides file untouched."""
    sentinel = tmp_path / "live-overrides.json"
    sentinel.write_text("{}")
    before = sentinel.stat().st_mtime_ns
    monkeypatch.setenv("PDCT_OVERRIDES_PATH", str(sentinel))

    r1 = harness.run_reference_benchmark()
    r2 = harness.run_reference_benchmark()
    assert r1["status"] == "ok", r1
    assert r1["recall_at5"] == r2["recall_at5"]
    assert r1["jaccard_concept_ab_mean"] == r2["jaccard_concept_ab_mean"]
    assert sentinel.stat().st_mtime_ns == before
    assert sentinel.read_text() == "{}"
