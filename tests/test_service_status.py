"""Build 121 — service_status() drift detection: JSON contract per state.

Manager calls are mocked (no live systemd/launchd in CI); unit files are
real temp files so the parse path is exercised for real.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from dct import service


# ── helpers ──────────────────────────────────────────────────────────────────

def _mk_systemd_unit(path: Path, interpreter: str, pdct_home: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""[Unit]
Description=PDCT supervisor (vault watcher + scheduler)
After=default.target

[Service]
ExecStart={interpreter} -m dct.supervisor
Restart=on-failure
RestartSec=5
Environment="PDCT_HOME={pdct_home}"

[Install]
WantedBy=default.target
""")


def _mk_plist(path: Path, interpreter: str, pdct_home: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.pdct.supervisor</string>
    <key>ProgramArguments</key>
    <array>
        <string>{interpreter}</string>
        <string>-m</string>
        <string>dct.supervisor</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PDCT_HOME</key>
        <string>{pdct_home}</string>
    </dict>
    <key>RunAtLoad</key><true/>
</dict>
</plist>
""")


MGR_OK = {"available": True, "loaded": True, "enabled": True,
          "active": True, "main_pid": 4242}


@pytest.fixture
def linux_env(tmp_path, monkeypatch):
    """Point service at temp paths, pretend Linux, mock the manager healthy."""
    home = tmp_path / "pdct-home"
    home.mkdir()
    unit = tmp_path / "systemd" / "pdct-supervisor.service"
    monkeypatch.setenv("PDCT_HOME", str(home))
    monkeypatch.setattr(service.platform, "system", lambda: "Linux")
    monkeypatch.setattr(service, "_systemd_path", lambda: unit)
    monkeypatch.setattr(service, "_systemd_manager_state", lambda: dict(MGR_OK))
    return home, unit


# ── states ───────────────────────────────────────────────────────────────────

def test_not_installed(linux_env):
    st = service.service_status()
    assert st["state"] == "not-installed"
    assert st["unit_path"].endswith("pdct-supervisor.service")


def test_healthy(linux_env):
    home, unit = linux_env
    _mk_systemd_unit(unit, sys.executable, str(home))
    st = service.service_status()
    assert st["state"] == "healthy"
    assert st["facts"]["owned"] is True
    assert st["facts"]["manager"]["main_pid"] == 4242


def test_missing_interpreter_classic_reinstall_drift(linux_env, tmp_path):
    home, unit = linux_env
    dead = tmp_path / "old-venv" / "bin" / "python"  # never created
    _mk_systemd_unit(unit, str(dead), str(home))
    st = service.service_status()
    assert st["state"] == "missing-interpreter"
    assert st["facts"]["owned"] is True  # dead path with our home = orphan we adopt


def test_stale_interpreter_old_venv_still_exists(linux_env, tmp_path, monkeypatch):
    """The realistic drift: old venv python still on disk, same PDCT_HOME.
    Must be stale-interpreter (repairable), NOT not-owned (Codex #2)."""
    home, unit = linux_env
    old = tmp_path / "old-venv" / "bin" / "python"
    old.parent.mkdir(parents=True)
    old.write_text("#!/bin/sh\n")
    _mk_systemd_unit(unit, str(old), str(home))
    monkeypatch.setattr(service, "_interpreter_functional", lambda i: (True, "ok"))
    st = service.service_status()
    assert st["state"] == "stale-interpreter"
    assert st["facts"]["owned"] is True


def test_interpreter_alias_of_same_venv_is_healthy(linux_env, tmp_path, monkeypatch):
    """False-RED regression (observed on VPS): the unit records one symlink
    alias of the venv interpreter (…/bin/python) while a later shell resolves
    a different alias of the SAME venv (…/bin/python3.13). Same venv bin/ dir →
    must be healthy, NOT stale-interpreter."""
    home, unit = linux_env
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    py = venv_bin / "python"          # unit records this alias
    py313 = venv_bin / "python3.13"   # current shell resolves this alias
    py.write_text("#!/bin/sh\n")
    py313.write_text("#!/bin/sh\n")
    _mk_systemd_unit(unit, str(py), str(home))
    monkeypatch.setattr(service.sys, "executable", str(py313))
    monkeypatch.setattr(service, "_interpreter_functional", lambda i: (True, "ok"))
    st = service.service_status()
    assert st["state"] == "healthy", st["facts"]
    assert st["facts"]["owned"] is True


def test_stale_env_our_interpreter_foreign_home(linux_env, tmp_path):
    """Unit runs OUR venv python but with a different PDCT_HOME → stale-env
    (owned, repairable FAIL), not not-owned (Codex #1)."""
    home, unit = linux_env
    _mk_systemd_unit(unit, sys.executable, str(tmp_path / "old-home"))
    st = service.service_status()
    assert st["state"] == "stale-env"
    assert st["facts"]["owned"] is True


def test_broken_interpreter_import_dct_fails(linux_env, monkeypatch):
    home, unit = linux_env
    _mk_systemd_unit(unit, sys.executable, str(home))
    monkeypatch.setattr(service, "_interpreter_functional",
                        lambda i: (False, "import-dct-failed: boom"))
    st = service.service_status()
    assert st["state"] == "broken-interpreter"


def test_not_owned_foreign_home_and_foreign_interpreter(linux_env, tmp_path):
    """Different PDCT_HOME AND a different interpreter = someone else's
    install entirely — never touch."""
    home, unit = linux_env
    other = tmp_path / "their-python"
    other.write_text("#!/bin/sh\n")
    _mk_systemd_unit(unit, str(other), str(tmp_path / "someone-else"))
    st = service.service_status()
    assert st["state"] == "not-owned"
    assert st["facts"]["owned"] is False


def test_manager_unavailable_has_linger_remedy(linux_env, monkeypatch):
    home, unit = linux_env
    _mk_systemd_unit(unit, sys.executable, str(home))
    monkeypatch.setattr(service, "_systemd_manager_state",
                        lambda: {"available": False, "error": "Failed to connect to user scope bus"})
    st = service.service_status()
    assert st["state"] == "manager-unavailable"
    assert "enable-linger" in st["facts"]["remedy"]


def test_installed_disabled_intentional_via_marker(linux_env, monkeypatch):
    home, unit = linux_env
    _mk_systemd_unit(unit, sys.executable, str(home))
    mgr = dict(MGR_OK, enabled=False, active=False)
    monkeypatch.setattr(service, "_systemd_manager_state", lambda: mgr)
    monkeypatch.setattr(service, "read_state_marker",
                        lambda: {"installed": True, "enabled": False})
    st = service.service_status()
    assert st["state"] == "installed-disabled"
    assert st["facts"]["intentional"] is True


def test_installed_disabled_unintentional(linux_env, monkeypatch):
    home, unit = linux_env
    _mk_systemd_unit(unit, sys.executable, str(home))
    mgr = dict(MGR_OK, enabled=False, active=False)
    monkeypatch.setattr(service, "_systemd_manager_state", lambda: mgr)
    st = service.service_status()  # no marker → expected enabled
    assert st["state"] == "installed-disabled"
    assert st["facts"]["intentional"] is False


def test_installed_inactive(linux_env, monkeypatch):
    home, unit = linux_env
    _mk_systemd_unit(unit, sys.executable, str(home))
    mgr = dict(MGR_OK, active=False, main_pid=None)
    monkeypatch.setattr(service, "_systemd_manager_state", lambda: mgr)
    st = service.service_status()
    assert st["state"] == "installed-inactive"


def test_hand_edited_garbage_unit_is_unknown_never_crash(linux_env):
    home, unit = linux_env
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("[[[ totally not a unit file %%%\x00")
    st = service.service_status()
    assert st["state"] == "unknown"
    assert "unparseable" in st["facts"]["error"]


def test_launchd_plist_parse_and_healthy(tmp_path, monkeypatch):
    home = tmp_path / "pdct-home"
    home.mkdir()
    plist = tmp_path / "com.pdct.supervisor.plist"
    _mk_plist(plist, sys.executable, str(home))
    monkeypatch.setenv("PDCT_HOME", str(home))
    monkeypatch.setattr(service.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(service, "_launchd_path", lambda: plist)
    monkeypatch.setattr(service, "_launchd_manager_state", lambda: dict(MGR_OK))
    st = service.service_status()
    assert st["state"] == "healthy"
    assert st["platform"] == "Darwin"


def test_launchd_garbage_plist_is_unknown(tmp_path, monkeypatch):
    home = tmp_path / "pdct-home"
    home.mkdir()
    plist = tmp_path / "com.pdct.supervisor.plist"
    plist.write_text("not xml at all")
    monkeypatch.setenv("PDCT_HOME", str(home))
    monkeypatch.setattr(service.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(service, "_launchd_path", lambda: plist)
    st = service.service_status()
    assert st["state"] == "unknown"


# ── intent marker round-trip ─────────────────────────────────────────────────

def test_state_marker_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    service.write_state_marker(installed=True, enabled=True)
    m = service.read_state_marker()
    assert m["installed"] is True and m["enabled"] is True
    assert m["interpreter"] == sys.executable
    service.write_state_marker(installed=False, enabled=False)
    assert service.read_state_marker()["installed"] is False


def test_state_marker_absent_or_corrupt_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    assert service.read_state_marker() is None
    (tmp_path / service.SERVICE_STATE_FILE).write_text("{corrupt")
    assert service.read_state_marker() is None


# ── CLI contract: --json ALWAYS exits 0 ─────────────────────────────────────

def test_cli_service_status_json_exits_zero_on_drift(tmp_path, monkeypatch, capsys):
    """install.sh (set -e) parses state from JSON; drift must not kill it."""
    from dct import cli
    home = tmp_path / "pdct-home"
    home.mkdir()
    unit = tmp_path / "systemd" / "pdct-supervisor.service"
    dead = tmp_path / "gone" / "bin" / "python"
    _mk_systemd_unit(unit, str(dead), str(home))
    monkeypatch.setenv("PDCT_HOME", str(home))
    monkeypatch.setattr(service.platform, "system", lambda: "Linux")
    monkeypatch.setattr(service, "_systemd_path", lambda: unit)
    monkeypatch.setattr(service, "_systemd_manager_state", lambda: dict(MGR_OK))
    rc = cli.main(["daemon", "service-status", "--json"])
    assert rc == 0  # ← the contract
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "missing-interpreter"


def test_cli_service_status_human_exits_one_on_drift(tmp_path, monkeypatch, capsys):
    from dct import cli
    home = tmp_path / "pdct-home"
    home.mkdir()
    unit = tmp_path / "systemd" / "pdct-supervisor.service"
    dead = tmp_path / "gone" / "bin" / "python"
    _mk_systemd_unit(unit, str(dead), str(home))
    monkeypatch.setenv("PDCT_HOME", str(home))
    monkeypatch.setattr(service.platform, "system", lambda: "Linux")
    monkeypatch.setattr(service, "_systemd_path", lambda: unit)
    monkeypatch.setattr(service, "_systemd_manager_state", lambda: dict(MGR_OK))
    rc = cli.main(["daemon", "service-status"])
    assert rc == 1
    assert "install-service" in capsys.readouterr().out


def test_cli_human_exit_one_when_unintentionally_disabled(tmp_path, monkeypatch, capsys):
    """Unintentional disablement is drift → human CLI must exit 1 (Codex #3)."""
    from dct import cli
    home = tmp_path / "pdct-home"
    home.mkdir()
    unit = tmp_path / "systemd" / "pdct-supervisor.service"
    _mk_systemd_unit(unit, sys.executable, str(home))
    monkeypatch.setenv("PDCT_HOME", str(home))
    monkeypatch.setattr(service.platform, "system", lambda: "Linux")
    monkeypatch.setattr(service, "_systemd_path", lambda: unit)
    mgr = dict(MGR_OK, enabled=False, active=False)
    monkeypatch.setattr(service, "_systemd_manager_state", lambda: mgr)
    assert cli.main(["daemon", "service-status"]) == 1


def test_cli_human_exit_zero_when_intentionally_disabled(tmp_path, monkeypatch, capsys):
    from dct import cli
    home = tmp_path / "pdct-home"
    home.mkdir()
    unit = tmp_path / "systemd" / "pdct-supervisor.service"
    _mk_systemd_unit(unit, sys.executable, str(home))
    monkeypatch.setenv("PDCT_HOME", str(home))
    monkeypatch.setattr(service.platform, "system", lambda: "Linux")
    monkeypatch.setattr(service, "_systemd_path", lambda: unit)
    mgr = dict(MGR_OK, enabled=False, active=False)
    monkeypatch.setattr(service, "_systemd_manager_state", lambda: mgr)
    monkeypatch.setattr(service, "read_state_marker",
                        lambda: {"installed": True, "enabled": False})
    assert cli.main(["daemon", "service-status"]) == 0


def test_install_service_preserves_installed_disabled_and_does_not_restart(
        tmp_path, monkeypatch):
    """Marker {installed:True, enabled:False} → reinstall leaves it disabled
    and never calls restart (Codex #4)."""
    home = tmp_path / "pdct-home"
    home.mkdir()
    unit = tmp_path / "systemd" / "pdct-supervisor.service"
    monkeypatch.setenv("PDCT_HOME", str(home))
    monkeypatch.setattr(service.platform, "system", lambda: "Linux")
    monkeypatch.setattr(service, "_systemd_path", lambda: unit)
    monkeypatch.setattr(service, "read_state_marker",
                        lambda: {"installed": True, "enabled": False})
    calls: list[list[str]] = []

    class _R:
        returncode = 0
        stderr = ""
        stdout = ""

    monkeypatch.setattr(service.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or _R())
    ok, msg = service.install_service()
    assert ok and "disabled" in msg
    flat = [" ".join(c) for c in calls]
    assert not any("restart" in c or "enable" in c for c in flat)


def test_install_service_after_uninstall_marker_enables_and_starts(
        tmp_path, monkeypatch):
    """Marker from uninstall (installed:False) = fresh install → enable+restart."""
    home = tmp_path / "pdct-home"
    home.mkdir()
    unit = tmp_path / "systemd" / "pdct-supervisor.service"
    monkeypatch.setenv("PDCT_HOME", str(home))
    monkeypatch.setattr(service.platform, "system", lambda: "Linux")
    monkeypatch.setattr(service, "_systemd_path", lambda: unit)
    monkeypatch.setattr(service, "read_state_marker",
                        lambda: {"installed": False, "enabled": False})
    calls: list[list[str]] = []

    class _R:
        returncode = 0
        stderr = ""
        stdout = ""

    monkeypatch.setattr(service.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or _R())
    ok, _ = service.install_service()
    assert ok
    flat = [" ".join(c) for c in calls]
    assert any("enable" in c for c in flat)
    assert any("restart" in c for c in flat)
