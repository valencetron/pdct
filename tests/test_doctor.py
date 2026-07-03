"""Doctor self-diagnosis smoke tests (bundled example corpus)."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_doctor_json_passes_on_example_corpus():
    r = subprocess.run(
        [sys.executable, "-m", "dct.doctor", "--json"],
        capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout[r.stdout.index("{"):])
    assert payload["ok"] is True
    assert set(payload["stages"]) == {"environment", "configuration",
                                      "functional", "retrieval"}
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
    dest = tmp_path / "export"
    r = subprocess.run(
        ["bash", str(REPO / "scripts" / "export_public.sh"), str(dest)],
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
