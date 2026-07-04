"""Tests for the pdct CLI, supervisor daemon, service templates, and
provider abstraction (Build 103)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _clean_env(home: Path, **extra) -> dict:
    env = dict(os.environ)
    for k in ("PDCT_HOME", "PDCT_VAULT_ROOT", "OBSIDIAN_VAULT",
              "PDCT_EVENTS_PATH", "PDCT_LLM_PROVIDER", "PDCT_LLM_BASE_URL",
              "PDCT_LLM_MODEL", "PDCT_LLM_API_KEY", "PYTHONPATH"):
        env.pop(k, None)
    env["PDCT_HOME"] = str(home)
    env["PDCT_SCHEDULER_INTERVAL"] = "3600"
    env.update(extra)
    return env


def _run_cli(args: list[str], env: dict, timeout: int = 120):
    return subprocess.run([sys.executable, "-m", "dct.cli", *args],
                          capture_output=True, text=True, cwd=REPO,
                          timeout=timeout, env=env)


# ── pdct init ────────────────────────────────────────────────────────────────

def test_init_scaffolds_home(tmp_path):
    home = tmp_path / "pdct"
    env = _clean_env(home)
    env.pop("ANTHROPIC_API_KEY", None)
    r = _run_cli(["init", "--home", str(home)], env)
    assert r.returncode == 0, r.stdout + r.stderr
    for sub in ("vault/distillations", "runtime", "logs", "data"):
        assert (home / sub).is_dir()
    assert (home / "events.jsonl").exists()
    envf = (home / "pdct.env").read_text()
    assert f"PDCT_HOME={home}" in envf
    assert "PDCT_LLM_BASE_URL" in envf  # provider block present


def test_init_idempotent_preserves_env_file(tmp_path):
    home = tmp_path / "pdct"
    env = _clean_env(home)
    assert _run_cli(["init", "--home", str(home)], env).returncode == 0
    (home / "pdct.env").write_text("PDCT_HOME=/custom\n")
    assert _run_cli(["init", "--home", str(home)], env).returncode == 0
    assert (home / "pdct.env").read_text() == "PDCT_HOME=/custom\n"


# ── supervisor lifecycle ─────────────────────────────────────────────────────

def test_daemon_lifecycle_event_lands(tmp_path):
    home = tmp_path / "pdct"
    (home / "vault" / "distillations").mkdir(parents=True)
    (home / "events.jsonl").touch()
    env = _clean_env(home)

    r = _run_cli(["daemon", "start"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    try:
        time.sleep(1.5)
        st = _run_cli(["daemon", "status", "--json"], env)
        assert st.returncode == 0, st.stdout
        payload = json.loads(st.stdout)
        assert payload["running"] is True
        assert payload["status"]["watcher"]["alive"] is True
        # status contract keys (fleet-probe consumers depend on these)
        assert {"pid", "started_ts", "uptime_s", "scheduler", "watcher",
                "events_path", "last_event_ts"} <= set(payload["status"])

        note = home / "vault" / "distillations" / "probe.md"
        note.write_text("---\ntitle: Probe\nconcepts: [alpha-beta]\n---\n\n"
                        "## Summary\nprobe\n")
        deadline = time.monotonic() + 10
        n = 0
        while time.monotonic() < deadline:
            txt = (home / "events.jsonl").read_text().strip()
            n = len(txt.splitlines()) if txt else 0
            if n:
                break
            time.sleep(0.25)
        assert n >= 1, "watcher event never landed in events.jsonl"
    finally:
        stop = _run_cli(["daemon", "stop"], env)
    assert stop.returncode == 0, stop.stdout
    st2 = _run_cli(["daemon", "status", "--json"], env)
    assert json.loads(st2.stdout)["running"] is False
    assert st2.returncode == 3  # documented not-running exit code


def test_daemon_start_twice_refuses(tmp_path):
    home = tmp_path / "pdct"
    (home / "vault" / "distillations").mkdir(parents=True)
    env = _clean_env(home)
    assert _run_cli(["daemon", "start"], env).returncode == 0
    try:
        r2 = _run_cli(["daemon", "start"], env)
        assert r2.returncode == 1
        assert "already running" in r2.stdout
    finally:
        _run_cli(["daemon", "stop"], env)


def test_daemon_stop_when_not_running(tmp_path):
    home = tmp_path / "pdct"
    home.mkdir()
    r = _run_cli(["daemon", "stop"], _clean_env(home))
    assert r.returncode == 1
    assert "not running" in r.stdout


# ── service templates ────────────────────────────────────────────────────────

def test_install_service_dry_run(tmp_path):
    home = tmp_path / "pdct"
    home.mkdir()
    r = _run_cli(["daemon", "install-service", "--dry-run"], _clean_env(home))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "would write" in r.stdout
    assert str(home) in r.stdout            # parameterized on PDCT_HOME
    assert "dct.supervisor" in r.stdout


def test_service_render_launchd_and_systemd(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    from dct import service
    monkeypatch.setattr(service.platform, "system", lambda: "Darwin")
    dest, content = service.render()
    assert dest.name.endswith(".plist")
    assert "com.pdct.supervisor" in content and str(tmp_path) in content
    monkeypatch.setattr(service.platform, "system", lambda: "Linux")
    dest, content = service.render()
    assert dest.name == "pdct-supervisor.service"
    assert "ExecStart" in content and str(tmp_path) in content
    monkeypatch.setattr(service.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError):
        service.render()


# ── pdct recall (talker surface) ────────────────────────────────────────────

def test_recall_against_example_corpus(tmp_path):
    home = tmp_path / "pdct"
    home.mkdir()
    # Copy the bundled events into tmp — the retrieval service appends
    # graph-rebuild events to the bound path, and pointing it at the
    # shipped corpus would pollute examples/events.jsonl.
    import shutil
    ev = tmp_path / "events.jsonl"
    shutil.copy(REPO / "examples" / "events.jsonl", ev)
    env = _clean_env(
        home,
        PDCT_VAULT_ROOT=str(REPO / "examples" / "vault"),
        PDCT_EVENTS_PATH=str(ev),
    )
    r = _run_cli(["recall", "which vector database was chosen?", "--json"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = json.loads(r.stdout)["rows"]
    assert rows and rows[0]["id"] == "2026-01-05-choosing-a-vector-db"


# ── pdct.env loading ─────────────────────────────────────────────────────────

def test_pdct_env_file_fills_gaps_but_never_overrides(tmp_path, monkeypatch):
    home = tmp_path / "pdct"
    home.mkdir()
    (home / "pdct.env").write_text(
        "PDCT_LLM_MODEL=from-file\nPDCT_LLM_PROVIDER=openai-compatible\n")
    monkeypatch.setenv("PDCT_HOME", str(home))
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "anthropic")  # explicit wins
    monkeypatch.delenv("PDCT_LLM_MODEL", raising=False)
    from dct.cli import _load_pdct_env
    _load_pdct_env()
    assert os.environ["PDCT_LLM_MODEL"] == "from-file"
    assert os.environ["PDCT_LLM_PROVIDER"] == "anthropic"


# ── provider abstraction ────────────────────────────────────────────────────

class _MockHandler(BaseHTTPRequestHandler):
    """openai-compatible mock: JSON-schema prompts get a valid object,
    everything else gets a judge-style verdict."""
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        sys_msg = req["messages"][0]["content"]
        if "JSON schema" in sys_msg:
            content = json.dumps({"title": "t", "summary": "s",
                                  "concepts": ["pgvector", "hnsw"],
                                  "key_quotes": []})
        else:
            content = '{"score": 8, "rationale": "relevant"}'
        body = json.dumps(
            {"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # noqa: D102
        pass


class _GarbageHandler(_MockHandler):
    def do_POST(self):
        body = json.dumps({"choices": [{"message":
                          {"content": "sure, here you go!"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def mock_provider(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("PDCT_LLM_BASE_URL",
                       f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("PDCT_LLM_MODEL", "mock-model")
    yield srv
    srv.shutdown()


def test_provider_complete_json_and_text(mock_provider):
    from dct import providers as prov
    ok, detail = prov.provider_available()
    assert ok, detail
    obj = prov.complete_json("sys", "user", {
        "type": "object", "required": ["title", "summary", "concepts"],
        "properties": {}})
    assert obj["concepts"] == ["pgvector", "hnsw"]
    assert "score" in prov.complete_text("judge", "q")


def test_provider_capability_gate_rejects_garbage(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _GarbageHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("PDCT_LLM_BASE_URL",
                       f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("PDCT_LLM_MODEL", "weak-model")
    from dct import providers as prov
    with pytest.raises(prov.ProviderError, match="minimum capability"):
        prov.complete_json("sys", "user",
                           {"type": "object", "required": ["title"],
                            "properties": {}})
    srv.shutdown()


def test_provider_unconfigured_is_advisory(monkeypatch):
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.delenv("PDCT_LLM_BASE_URL", raising=False)
    from dct import providers as prov
    ok, detail = prov.provider_available()
    assert not ok and "PDCT_LLM_BASE_URL" in detail
    from dct.doctor import _check_llm
    checks = _check_llm()
    assert len(checks) == 1
    assert checks[0].id == "llm.endpoint"
    assert not checks[0].required  # advisory skip — retrieval-only is valid


def test_doctor_llm_stage_passes_with_capable_mock(mock_provider, monkeypatch):
    # The doctor's expected-concept overlap requires topical extraction —
    # serve a distillation-shaped response with on-topic concepts.
    class _Capable(_MockHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            sys_msg = req["messages"][0]["content"]
            if "JSON schema" in sys_msg:
                content = json.dumps({
                    "title": "Vector DB", "summary": "pgvector benchmark",
                    "concepts": ["pgvector", "hnsw", "benchmarking",
                                 "vector-database"]})
            else:
                content = '{"score": 9, "rationale": "on point"}'
            body = json.dumps(
                {"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), _Capable)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("PDCT_LLM_BASE_URL",
                       f"http://127.0.0.1:{srv.server_port}/v1")
    from dct.doctor import _check_llm
    checks = {c.id: c for c in _check_llm()}
    assert checks["llm.endpoint"].ok
    assert checks["llm.structured"].ok
    assert checks["llm.concepts"].ok, checks["llm.concepts"].detail
    assert checks["llm.judge"].ok, checks["llm.judge"].detail
    srv.shutdown()


def test_first_party_headers_shape():
    from dct.providers import first_party_headers
    h = first_party_headers("sk-ant-oat-test")
    assert h["Authorization"] == "Bearer sk-ant-oat-test"
    assert h["anthropic-beta"] == "oauth-2025-04-20"
    assert h["anthropic-version"] == "2023-06-01"
    assert h["anthropic-client-platform"] == "claude_code_cli"
    ua = h["User-Agent"]
    assert ua.startswith("claude-cli/") and ua.endswith(" (external, cli)")
    ver = ua.split("/")[1].split(" ")[0]
    assert ver != "0.0.0"  # the 0.0.0 UA is a non-first-party tell
    import re
    assert re.fullmatch(r"\d+\.\d+\.\d+", ver)


# ── INTEGRATION.md <-> doctor sync ──────────────────────────────────────────

def test_integration_doc_matches_doctor_check_ids():
    from dct.doctor import CHECK_IDS
    # private repo keeps it in public-docs/; the public export puts it at root
    candidates = [REPO / "public-docs" / "INTEGRATION.md",
                  REPO / "INTEGRATION.md"]
    doc = next((p for p in candidates if p.exists()), None)
    assert doc is not None, "INTEGRATION.md missing"
    text = doc.read_text()
    missing = [cid for cid in CHECK_IDS if f"`{cid}`" not in text]
    assert not missing, f"INTEGRATION.md missing check IDs: {missing}"


# ── anthropic default WITHOUT the SDK (optional extra on public installs) ──

def test_distiller_routes_via_providers_when_sdk_missing(monkeypatch):
    """Codex P1: default provider=anthropic with no `anthropic` package must
    route through the urllib provider layer (the path doctor validates),
    not crash with ModuleNotFoundError."""
    import builtins
    import dct.llm as llm
    from dct import providers as prov

    real_import = builtins.__import__

    def _no_sdk(name, *a, **kw):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_sdk)
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "anthropic")

    captured = {}

    def _fake_complete_json(system, user, schema, *, model=None, max_tokens=2048):
        captured["model"] = model
        return {"title": "t", "summary": "s",
                "concepts": ["alpha-beta"], "key_quotes": []}

    monkeypatch.setattr(prov, "complete_json", _fake_complete_json)
    note = llm.call_distiller(
        [{"role": "user", "text": "hello"}], {"session_id": "x"}, [])
    assert note.title == "t"
    assert captured["model"]  # resolved model id was passed through


def test_judge_routes_via_providers_when_sdk_missing(monkeypatch):
    import builtins
    from dct import providers as prov
    from dct.judge import invoker

    real_import = builtins.__import__

    def _no_sdk(name, *a, **kw):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_sdk)
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(
        prov, "complete_text",
        lambda *a, **kw: '{"score": 4, "rationale": "ok", '
                         '"era_assessment": "helpful"}')
    r = invoker.invoke_judge("test prompt")
    assert r.status not in ("unexpected_error",), r.fail_reason
    assert r.score == 4
