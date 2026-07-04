"""OS service templates — `pdct daemon install-service`.

Optional upgrade path over the pure-Python supervisor: renders a launchd
plist (macOS) or systemd user unit (Linux) so the supervisor survives
reboot. Parameterized on PDCT_HOME and the exact venv python running now.

`pdct daemon start/stop` remains the portable baseline; this only adds
boot persistence.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from dct import config as _cfg

LAUNCHD_LABEL = "com.pdct.supervisor"
SYSTEMD_UNIT = "pdct-supervisor.service"


def _launchd_plist() -> str:
    home = _cfg.pdct_home()
    log = _cfg.logs_dir() / "supervisor.log"
    env_lines = ""
    for k in ("PDCT_HOME", "PDCT_VAULT_ROOT", "OBSIDIAN_VAULT",
              "PDCT_EVENTS_PATH", "PDCT_LLM_PROVIDER", "PDCT_LLM_BASE_URL",
              "PDCT_LLM_MODEL", "PDCT_LLM_API_KEY"):
        v = os.environ.get(k)
        if v:
            env_lines += (f"        <key>{k}</key>\n"
                          f"        <string>{v}</string>\n")
    if "PDCT_HOME" not in env_lines:
        env_lines = (f"        <key>PDCT_HOME</key>\n"
                     f"        <string>{home}</string>\n") + env_lines
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>-m</string>
        <string>dct.supervisor</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
{env_lines.rstrip()}
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{log}</string>
    <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""


def _systemd_unit() -> str:
    home = _cfg.pdct_home()
    env_lines = ""
    for k in ("PDCT_HOME", "PDCT_VAULT_ROOT", "OBSIDIAN_VAULT",
              "PDCT_EVENTS_PATH", "PDCT_LLM_PROVIDER", "PDCT_LLM_BASE_URL",
              "PDCT_LLM_MODEL", "PDCT_LLM_API_KEY"):
        v = os.environ.get(k)
        if v:
            env_lines += f'Environment="{k}={v}"\n'
    if "PDCT_HOME=" not in env_lines:
        env_lines = f'Environment="PDCT_HOME={home}"\n' + env_lines
    return f"""[Unit]
Description=PDCT supervisor (vault watcher + scheduler)
After=default.target

[Service]
ExecStart={sys.executable} -m dct.supervisor
Restart=on-failure
RestartSec=5
{env_lines.rstrip()}

[Install]
WantedBy=default.target
"""


def _launchd_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def _systemd_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT


def render() -> tuple[Path, str]:
    """(dest_path, file_content) for the current OS. Raises on unsupported."""
    sysname = platform.system()
    if sysname == "Darwin":
        return _launchd_path(), _launchd_plist()
    if sysname == "Linux":
        return _systemd_path(), _systemd_unit()
    raise RuntimeError(f"unsupported OS for install-service: {sysname} "
                       "(use `pdct daemon start` instead)")


def install_service(dry_run: bool = False) -> tuple[bool, str]:
    try:
        dest, content = render()
    except RuntimeError as e:
        return False, str(e)
    if dry_run:
        return True, f"── would write {dest}:\n{content}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    sysname = platform.system()
    if sysname == "Darwin":
        subprocess.run(["launchctl", "unload", str(dest)],
                       capture_output=True)  # idempotent re-install
        r = subprocess.run(["launchctl", "load", str(dest)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"wrote {dest} but launchctl load failed: {r.stderr.strip()}"
        return True, (f"installed + loaded {dest}\n"
                      f"status: launchctl list | grep {LAUNCHD_LABEL}\n"
                      f"uninstall: pdct daemon uninstall-service")
    # Linux/systemd
    r = subprocess.run(["systemctl", "--user", "daemon-reload"],
                       capture_output=True, text=True)
    r2 = subprocess.run(["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT],
                        capture_output=True, text=True)
    if r2.returncode != 0:
        return False, f"wrote {dest} but systemctl enable failed: {r2.stderr.strip()}"
    return True, (f"installed + started {dest}\n"
                  f"status: systemctl --user status {SYSTEMD_UNIT}\n"
                  f"uninstall: pdct daemon uninstall-service")


def uninstall_service(dry_run: bool = False) -> tuple[bool, str]:
    sysname = platform.system()
    if sysname == "Darwin":
        dest = _launchd_path()
        if dry_run:
            return True, f"── would unload + remove {dest}"
        if dest.exists():
            subprocess.run(["launchctl", "unload", str(dest)], capture_output=True)
            dest.unlink()
            return True, f"unloaded + removed {dest}"
        return False, f"not installed ({dest} missing)"
    if sysname == "Linux":
        dest = _systemd_path()
        if dry_run:
            return True, f"── would disable + remove {dest}"
        if dest.exists():
            subprocess.run(["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT],
                           capture_output=True)
            dest.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"],
                           capture_output=True)
            return True, f"disabled + removed {dest}"
        return False, f"not installed ({dest} missing)"
    return False, f"unsupported OS: {sysname}"
