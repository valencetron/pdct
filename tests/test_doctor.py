"""Doctor self-diagnosis smoke tests (bundled example corpus)."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_doctor_json_passes_on_example_corpus():
    # Point the provider at an unset openai-compatible endpoint so stage 6
    # is a deterministic advisory skip (no live LLM calls in unit tests).
    env = dict(**__import__("os").environ,
               PDCT_LLM_PROVIDER="openai-compatible")
    env.pop("PDCT_LLM_BASE_URL", None)
    r = subprocess.run(
        [sys.executable, "-m", "dct.doctor", "--json"],
        capture_output=True, text=True, cwd=REPO, timeout=600, env=env,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout[r.stdout.index("{"):])
    assert payload["ok"] is True
    assert set(payload["stages"]) == {"environment", "configuration",
                                      "functional", "retrieval",
                                      "daemon", "llm"}
    daemon = payload["stages"]["daemon"]
    assert any(c["id"] == "daemon.supervisor" and c["ok"] for c in daemon), daemon
    llm = payload["stages"]["llm"]
    assert all(not c["required"] for c in llm), llm  # advisory skip
    recall = [c for c in payload["stages"]["retrieval"]
              if c["name"] == "retrieval recall"][0]
    assert recall["ok"], recall


def test_example_corpus_has_no_personal_data():
    import re
    pat = re.compile(r"[n]eil|[g]odbole|[s]hehla|[v]alence|[a]irship|/Users/[a-z]", re.I)
    for p in (REPO / "examples").rglob("*"):
        if p.is_file() and p.suffix in {".md", ".json", ".jsonl"}:
            assert not pat.search(p.read_text()), f"personal token in {p}"


def test_export_pipeline_produces_sanitized_tree(tmp_path):
    """Run the real export into a temp dir; gate must pass and private files
    must be absent. (Codex P2: highest-risk path was untested.)"""
    script = REPO / "scripts" / "export_public.sh"
    if not script.exists():
        import pytest
        pytest.skip("export pipeline only exists in the private repo")
    dest = tmp_path / "export"
    r = subprocess.run(
        ["bash", str(script), str(dest)],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "SANITIZATION GATE PASSED" in r.stdout
    # Private artifacts must not exist in the export
    for forbidden in ("events.jsonl", "positions.json", "data/judge.db",
                      "runtime/pdct-overrides.json", "docs/superpowers",
                      "benchmark/pdct-questions-v3.json", "public-docs"):
        assert not (dest / forbidden).exists(), f"leaked: {forbidden}"
    # examples corpus events file IS allowed (synthetic)
    assert (dest / "examples" / "events.jsonl").exists()
    # docs are authored copies
    for doc in ("README.md", "INSTALL.md", "CONFIGURATION.md",
                "ARCHITECTURE.md", "install.sh", "check_sanitized.sh"):
        assert (dest / doc).exists(), f"missing public doc: {doc}"
