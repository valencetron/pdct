"""Tests for pdct configure — env upsert, flag mode, snapshot probe, --show
redaction, and the provider status core."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dct import providers as prov  # noqa: E402
from dct import provider_status as ps  # noqa: E402
from dct.configure import (upsert_env, read_env_file, snapshot_overlay,
                           cmd_configure, cmd_show)  # noqa: E402


class Args:
    """argparse.Namespace stand-in with configure defaults."""

    def __init__(self, **kw):
        self.provider = None
        self.base_url = None
        self.model = None
        self.key = None
        self.key_env = None
        self.no_probe = False
        self.show = False
        self.json = False
        self.paths = False
        self.__dict__.update(kw)


# ── upsert_env ──────────────────────────────────────────────────────────────

def test_upsert_preserves_comments_and_unknown_lines(tmp_path):
    f = tmp_path / "pdct.env"
    f.write_text("# header comment\n"
                 "PDCT_HOME=/x\n"
                 "PDCT_LLM_PROVIDER=anthropic\n"
                 "# trailing note\n")
    upsert_env(f, {"PDCT_LLM_PROVIDER": "openai-compatible",
                   "PDCT_LLM_MODEL": "gpt-4o-mini"})
    text = f.read_text()
    assert "# header comment" in text
    assert "# trailing note" in text
    assert "PDCT_HOME=/x" in text
    assert "PDCT_LLM_PROVIDER=openai-compatible" in text
    assert text.count("PDCT_LLM_PROVIDER") == 1
    assert "PDCT_LLM_MODEL=gpt-4o-mini" in text


def test_upsert_uncomments_managed_keys(tmp_path):
    f = tmp_path / "pdct.env"
    f.write_text("# PDCT_LLM_MODEL=old-hint\n")
    upsert_env(f, {"PDCT_LLM_MODEL": "m1"})
    text = f.read_text()
    assert "PDCT_LLM_MODEL=m1" in text
    assert "old-hint" not in text


def test_upsert_none_removes_key(tmp_path):
    f = tmp_path / "pdct.env"
    f.write_text("PDCT_LLM_API_KEY=sekrit\n")
    upsert_env(f, {"PDCT_LLM_API_KEY": None})
    assert "sekrit" not in f.read_text()


def test_upsert_sets_0600(tmp_path):
    f = tmp_path / "pdct.env"
    upsert_env(f, {"PDCT_LLM_PROVIDER": "anthropic"})
    assert (f.stat().st_mode & 0o777) == 0o600


# ── flag mode ───────────────────────────────────────────────────────────────

def test_flag_mode_writes_expected_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    rc = cmd_configure(Args(provider="openai-compatible",
                            base_url="http://localhost:9/v1",
                            model="m", key_env="MY_KEY", no_probe=True))
    assert rc == 0
    vals = read_env_file(tmp_path / "pdct.env")
    assert vals["PDCT_LLM_PROVIDER"] == "openai-compatible"
    assert vals["PDCT_LLM_BASE_URL"] == "http://localhost:9/v1"
    assert vals["PDCT_LLM_MODEL"] == "m"
    assert vals["PDCT_LLM_API_KEY_ENV"] == "MY_KEY"
    assert "PDCT_LLM_API_KEY" not in vals


def test_flag_mode_invalid_combo_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    rc = cmd_configure(Args(provider="openai-compatible", no_probe=True))
    assert rc == 2
    assert not (tmp_path / "pdct.env").exists()


def test_bare_non_tty_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("dct.configure.detect_backends", lambda probe_local=True: [])
    rc = cmd_configure(Args())
    assert rc == 2
    out = capsys.readouterr().out
    assert "usage:" in out


# ── key-env indirection resolver ────────────────────────────────────────────

def test_resolve_api_key_indirection(monkeypatch):
    monkeypatch.delenv("PDCT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PDCT_LLM_API_KEY_ENV", "MY_SECRET_VAR")
    monkeypatch.setenv("MY_SECRET_VAR", "resolved-value")
    assert prov.resolve_api_key() == "resolved-value"


def test_resolve_api_key_literal_wins(monkeypatch):
    monkeypatch.setenv("PDCT_LLM_API_KEY", "literal")
    monkeypatch.setenv("PDCT_LLM_API_KEY_ENV", "MY_SECRET_VAR")
    monkeypatch.setenv("MY_SECRET_VAR", "indirect")
    assert prov.resolve_api_key() == "literal"


# ── snapshot overlay (shell env must not shadow just-written config) ────────

def test_snapshot_overlay_removes_unset_managed_keys(tmp_path):
    f = tmp_path / "pdct.env"
    f.write_text("PDCT_LLM_PROVIDER=openai-compatible\n"
                 "PDCT_LLM_BASE_URL=http://mock/v1\n"
                 "PDCT_LLM_MODEL=m\n")
    ov = snapshot_overlay(f)
    assert ov["PDCT_LLM_PROVIDER"] == "openai-compatible"
    assert ov["PDCT_LLM_API_KEY"] is None  # explicit removal


def test_env_overlay_applies_and_restores(monkeypatch):
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("PDCT_LLM_API_KEY", "shell-key")
    with prov.env_overlay({"PDCT_LLM_PROVIDER": "openai-compatible",
                           "PDCT_LLM_API_KEY": None}):
        assert os.environ["PDCT_LLM_PROVIDER"] == "openai-compatible"
        assert "PDCT_LLM_API_KEY" not in os.environ
    assert os.environ["PDCT_LLM_PROVIDER"] == "anthropic"
    assert os.environ["PDCT_LLM_API_KEY"] == "shell-key"


def test_probe_uses_written_config_not_shell(tmp_path, monkeypatch):
    """Codex finding #1: shell exports must not shadow the new config."""
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "anthropic")  # shell says anthropic
    seen = {}

    def fake_check(overlay=None, timeout=30.0):
        with prov.env_overlay(overlay or {}):
            seen["provider"] = prov.provider_name()
        res = prov.CapabilityResult()
        res.provider = seen["provider"]
        res.endpoint_ok = res.structured_ok = True
        res.concepts_ok = res.judge_ok = True
        return res

    monkeypatch.setattr(prov, "check_capability", fake_check)
    rc = cmd_configure(Args(provider="openai-compatible",
                            base_url="http://mock/v1", model="m",
                            key_env="K", no_probe=False))
    assert rc == 0
    assert seen["provider"] == "openai-compatible"  # probe saw the file, not the shell


def test_probe_failure_propagates_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))

    def fake_check(overlay=None, timeout=30.0):
        return prov.CapabilityResult()  # all False

    monkeypatch.setattr(prov, "check_capability", fake_check)
    rc = cmd_configure(Args(provider="openai-compatible",
                            base_url="http://mock/v1", model="m",
                            key_env="K", no_probe=False))
    assert rc == 1


# ── --show redaction ────────────────────────────────────────────────────────

def test_show_never_leaks_key(tmp_path, monkeypatch, capsys):
    # assembled at runtime so the export sanitizer can't match a secret shape
    fake = "sk-" + "veryfake" + "secretkey123456"
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    monkeypatch.setenv("PDCT_LLM_API_KEY", fake)
    monkeypatch.setattr("dct.configure.detect_backends", lambda probe_local=True: [])
    for js in (False, True):
        rc = cmd_show(Args(show=True, json=js))
        assert rc == 0
        out = capsys.readouterr().out
        assert fake not in out
        assert fake[:6] not in out  # no prefixes either
        assert "present" in out.lower()


def test_show_json_shape(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    monkeypatch.setattr("dct.configure.detect_backends", lambda probe_local=True: [])
    rc = cmd_show(Args(show=True, json=True))
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert {"provider", "model", "key", "backends"} <= set(info)


# ── provider status core ────────────────────────────────────────────────────

def test_detect_separates_signals(tmp_path, monkeypatch):
    """detected / configured / auth_valid are independent axes."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("PDCT_LLM_BASE_URL", "https://api.openai.com/v1")
    # anthropic auth chain forced empty
    from dct import auth
    monkeypatch.setattr(auth, "load_oauth_token",
                        lambda: (_ for _ in ()).throw(auth.TokenLoadError("x")))
    cands = ps.detect_backends(probe_local=False)
    byname = {c.name: c for c in cands}
    assert byname["anthropic"].detected is False
    assert byname["anthropic"].configured is False
    assert byname["openai"].configured is True   # env selects it
    assert byname["openai"].auth_valid is False  # but no key


def test_detect_configured_follows_env(monkeypatch):
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "codex-oauth")
    cands = ps.detect_backends(probe_local=False)
    assert any(c.name == "codex-oauth" and c.configured for c in cands)


def test_best_candidate_prefers_usable(monkeypatch):
    a = ps.BackendStatus(name="a", provider="anthropic")
    b = ps.BackendStatus(name="b", provider="openai-compatible", auth_valid=True)
    assert ps.best_candidate([a, b]) is b
    assert ps.best_candidate([a]) is None


# ── doctor cross-link (Codex #8: advisory tone + configure hint) ────────────

def test_doctor_hints_configure_when_no_provider(monkeypatch):
    from dct import doctor
    monkeypatch.setattr(prov, "provider_available",
                        lambda: (False, "no creds"))
    checks = doctor._check_llm()
    assert len(checks) == 1
    c = checks[0]
    assert c.ok is False and c.required is False  # advisory, not failure
    assert "pdct configure" in c.detail
    assert "retrieval-only" in c.detail


# ── end-to-end smoke against the mock server (Codex #11) ────────────────────

def test_smoke_configure_probe_doctor_against_mock(tmp_path, monkeypatch):
    from tests.mock_openai_server import start_server
    srv, port, _ = start_server()
    try:
        monkeypatch.setenv("PDCT_HOME", str(tmp_path))
        # shell claims anthropic — the probe must ignore it
        monkeypatch.setenv("PDCT_LLM_PROVIDER", "anthropic")
        rc = cmd_configure(Args(provider="openai-compatible",
                                base_url=f"http://127.0.0.1:{port}/v1",
                                model="mock-model", no_probe=False))
        assert rc == 0  # endpoint + structured both pass against the mock
        # doctor stage 6 under the same config
        overlay = snapshot_overlay(tmp_path / "pdct.env")
        with prov.env_overlay(overlay):
            from dct import doctor
            checks = doctor._check_llm()
        byid = {}
        for c in checks:
            byid.setdefault(c.id, c)
        assert byid["llm.endpoint"].ok
        assert byid["llm.structured"].ok
        assert byid["llm.judge"].ok
    finally:
        srv.shutdown()


# ── Codex diff-audit regression tests ───────────────────────────────────────

def test_upsert_handles_export_prefix(tmp_path):
    """`export KEY=v` lines must be updated in place, not duplicated."""
    f = tmp_path / "pdct.env"
    f.write_text("export PDCT_LLM_PROVIDER=anthropic\n")
    upsert_env(f, {"PDCT_LLM_PROVIDER": "openai-compatible"})
    text = f.read_text()
    assert text.count("PDCT_LLM_PROVIDER") == 1
    assert "openai-compatible" in text and "anthropic" not in text


def test_switching_provider_clears_stale_literal_key(tmp_path, monkeypatch):
    """Going keyless must scrub an old literal secret from pdct.env."""
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    stale = "old" + "-literal-" + "secret"
    (tmp_path / "pdct.env").write_text(f"PDCT_LLM_API_KEY={stale}\n")
    rc = cmd_configure(Args(provider="anthropic", no_probe=True))
    assert rc == 0
    assert stale not in (tmp_path / "pdct.env").read_text()


def test_overlay_suppresses_ambient_openai_key_for_custom_endpoint(tmp_path, monkeypatch):
    """Ambient OPENAI_API_KEY must not leak to a keyless local endpoint."""
    f = tmp_path / "pdct.env"
    f.write_text("PDCT_LLM_PROVIDER=openai-compatible\n"
                 "PDCT_LLM_BASE_URL=http://localhost:11434/v1\n"
                 "PDCT_LLM_MODEL=m\n")
    ov = snapshot_overlay(f)
    assert ov.get("OPENAI_API_KEY", "sentinel") is None  # explicit removal
    # but an explicit reference keeps it
    f.write_text("PDCT_LLM_PROVIDER=openai-compatible\n"
                 "PDCT_LLM_BASE_URL=http://localhost:11434/v1\n"
                 "PDCT_LLM_MODEL=m\n"
                 "PDCT_LLM_API_KEY_ENV=OPENAI_API_KEY\n")
    ov2 = snapshot_overlay(f)
    assert "OPENAI_API_KEY" not in ov2 or ov2["OPENAI_API_KEY"] is not None


def test_key_and_key_env_mutually_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    rc = cmd_configure(Args(provider="anthropic", key="a", key_env="B",
                            no_probe=True))
    assert rc == 2


def test_capability_gate_requires_all_four(monkeypatch):
    res = prov.CapabilityResult()
    res.endpoint_ok = res.structured_ok = True
    assert not res.ok  # concepts + judge still required
    res.concepts_ok = res.judge_ok = True
    assert res.ok
