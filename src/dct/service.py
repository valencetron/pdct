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


_SERVICE_ENV_KEYS = ("PDCT_HOME", "PDCT_VAULT_ROOT", "OBSIDIAN_VAULT",
                     "PDCT_EVENTS_PATH", "PDCT_LLM_PROVIDER",
                     "PDCT_LLM_BASE_URL", "PDCT_LLM_MODEL", "PDCT_LLM_API_KEY",
                     "ANTHROPIC_API_KEY", "PDCT_SCHEDULER_INTERVAL")


def _service_env() -> dict[str, str]:
    env = {k: v for k in _SERVICE_ENV_KEYS
           if (v := os.environ.get(k)) and "\n" not in v}
    env.setdefault("PDCT_HOME", str(_cfg.pdct_home()))
    return env


def _launchd_plist() -> str:
    """Hand-serialized plist with strict XML escaping (xml.sax.saxutils —
    pure Python, no pyexpat: broken on some Homebrew builds). Values with
    &, <, quotes, etc. can't corrupt the file."""
    from xml.sax.saxutils import escape
    log = escape(str(_cfg.logs_dir() / "supervisor.log"))
    env_lines = "".join(
        f"        <key>{escape(k)}</key>\n"
        f"        <string>{escape(v)}</string>\n"
        for k, v in _service_env().items())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{escape(LAUNCHD_LABEL)}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{escape(sys.executable)}</string>
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


def _systemd_escape(v: str) -> str:
    """Escape a value for a systemd Environment="K=V" directive."""
    return v.replace("\\", "\\\\").replace('"', '\\"')


def _systemd_unit() -> str:
    env_lines = "".join(
        f'Environment="{k}={_systemd_escape(v)}"\n'
        for k, v in _service_env().items())
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
        # bootout+bootstrap (not deprecated load/unload): a previously-loaded
        # plist with a changed ProgramArguments is NOT refreshed by re-load on
        # modern macOS. bootout is expected to fail when not loaded — ignore.
        domain = f"gui/{os.getuid()}"
        subprocess.run(["launchctl", "bootout", f"{domain}/{LAUNCHD_LABEL}"],
                       capture_output=True)
        r = subprocess.run(["launchctl", "bootstrap", domain, str(dest)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            # Fallback for older macOS without bootstrap semantics
            subprocess.run(["launchctl", "unload", str(dest)], capture_output=True)
            r = subprocess.run(["launchctl", "load", str(dest)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return False, f"wrote {dest} but launchctl load failed: {r.stderr.strip()}"
        subprocess.run(["launchctl", "kickstart", "-k",
                        f"{domain}/{LAUNCHD_LABEL}"], capture_output=True)
        write_state_marker(installed=True, enabled=True)
        return True, (f"installed + loaded {dest}\n"
                      f"status: pdct daemon service-status\n"
                      f"uninstall: pdct daemon uninstall-service")
    # Linux/systemd — write → daemon-reload → enable → RESTART (not just
    # enable --now: an already-active service with a changed ExecStart keeps
    # running the OLD interpreter until restarted — the Prism drift bug).
    env = _systemd_env()
    subprocess.run(["systemctl", "--user", "daemon-reload"],
                   capture_output=True, text=True, env=env)
    # Intent (Codex diff-audit #4): a marker from a prior state only suppresses
    # enable/start when the operator disabled the service WHILE INSTALLED
    # (installed=True, enabled=False). A marker from uninstall (installed=False)
    # means this call is a fresh install → enable + start. And when we preserve
    # disabled, we must also NOT restart — reinstalling a disabled unit must
    # not start it.
    marker = read_state_marker()
    keep_disabled = bool(marker and marker.get("installed")
                         and not marker.get("enabled", True))
    if keep_disabled:
        write_state_marker(installed=True, enabled=False)
        return True, (f"installed {dest} (left disabled — operator intent; "
                      f"enable with: systemctl --user enable --now {SYSTEMD_UNIT})")
    r2 = subprocess.run(["systemctl", "--user", "enable", SYSTEMD_UNIT],
                        capture_output=True, text=True, env=env)
    if r2.returncode != 0:
        return False, f"wrote {dest} but systemctl enable failed: {r2.stderr.strip()}"
    r3 = subprocess.run(["systemctl", "--user", "restart", SYSTEMD_UNIT],
                        capture_output=True, text=True, env=env)
    if r3.returncode != 0:
        return False, f"wrote {dest} but systemctl restart failed: {r3.stderr.strip()}"
    write_state_marker(installed=True, enabled=True)
    return True, (f"installed + started {dest}\n"
                  f"status: pdct daemon service-status\n"
                  f"uninstall: pdct daemon uninstall-service")


def uninstall_service(dry_run: bool = False) -> tuple[bool, str]:
    sysname = platform.system()
    if sysname == "Darwin":
        dest = _launchd_path()
        if dry_run:
            return True, f"── would unload + remove {dest}"
        if dest.exists():
            subprocess.run(["launchctl", "bootout",
                            f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
                           capture_output=True)
            subprocess.run(["launchctl", "unload", str(dest)], capture_output=True)
            dest.unlink()
            write_state_marker(installed=False, enabled=False)
            return True, f"unloaded + removed {dest}"
        return False, f"not installed ({dest} missing)"
    if sysname == "Linux":
        dest = _systemd_path()
        if dry_run:
            return True, f"── would disable + remove {dest}"
        if dest.exists():
            env = _systemd_env()
            subprocess.run(["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT],
                           capture_output=True, env=env)
            dest.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"],
                           capture_output=True, env=env)
            write_state_marker(installed=False, enabled=False)
            return True, f"disabled + removed {dest}"
        return False, f"not installed ({dest} missing)"
    return False, f"unsupported OS: {sysname}"


# ── service_status (Build 121: drift detection) ────────────────────────────
#
# Structured states (JSON contract — install.sh and doctor branch on these):
#   not-installed        no unit/plist file on disk
#   healthy              owned + interpreter functional + enabled/active as expected
#   stale-interpreter    unit's python exists but is not our venv's python
#   missing-interpreter  unit's python path no longer exists (classic reinstall drift)
#   broken-interpreter   unit's python exists but `import dct` fails
#   stale-env            unit's PDCT_HOME differs from current PDCT_HOME
#   installed-disabled   owned + fine, but disabled in the manager
#   installed-inactive   owned + enabled, but not actually running
#   manager-unavailable  systemctl/launchctl can't reach the user manager
#   not-owned            unit exists but its PDCT_HOME isn't ours — never touch
#   unknown              unit file exists but is unparseable (hand-edited?)

SERVICE_STATE_FILE = "service-state.json"


def _state_marker_path() -> Path:
    return _cfg.pdct_home() / SERVICE_STATE_FILE


def read_state_marker() -> dict | None:
    """Operator-intent marker written by install/uninstall-service.

    Distinguishes 'I disabled this on purpose' from 'reinstall broke it'.
    Returns None when absent or unreadable.
    """
    import json as _json
    p = _state_marker_path()
    try:
        return _json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def write_state_marker(installed: bool, enabled: bool | None = None) -> None:
    import json as _json
    import time as _time
    p = _state_marker_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps({
            "installed": installed,
            "enabled": enabled,
            "interpreter": sys.executable,
            "pdct_home": str(_cfg.pdct_home()),
            "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        }, indent=2))
    except OSError:
        pass  # marker is advisory; never fail install over it


def _parse_launchd_plist(text: str) -> dict | None:
    """Extract {interpreter, env} from our generated plist. None if unparseable.

    Regex-based, NOT plistlib: plistlib needs pyexpat, which is broken on
    some Homebrew python builds — the same reason _launchd_plist()
    hand-serializes. We only need to parse plists WE generated.
    """
    import re as _re
    from xml.sax.saxutils import unescape
    try:
        if "<plist" not in text or "ProgramArguments" not in text:
            return None
        m = _re.search(
            r"<key>ProgramArguments</key>\s*<array>\s*<string>([^<]*)</string>",
            text)
        if not m:
            return None
        interp = unescape(m.group(1))
        env: dict[str, str] = {}
        em = _re.search(
            r"<key>EnvironmentVariables</key>\s*<dict>(.*?)</dict>",
            text, _re.DOTALL)
        if em:
            for km, vm in _re.findall(
                    r"<key>([^<]*)</key>\s*<string>([^<]*)</string>",
                    em.group(1)):
                env[unescape(km)] = unescape(vm)
        return {"interpreter": interp or None, "env": env}
    except Exception:  # noqa: BLE001 — hand-edited garbage → unknown, never crash
        return None


def _parse_systemd_unit(text: str) -> dict | None:
    """Extract {interpreter, env} from our generated unit. None if unparseable."""
    import re as _re
    try:
        interp = None
        env: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("ExecStart="):
                interp = line[len("ExecStart="):].split()[0] or None
            elif line.startswith("Environment="):
                v = line[len("Environment="):].strip()
                if v.startswith('"') and v.endswith('"'):
                    v = v[1:-1]
                v = v.replace('\\"', '"').replace("\\\\", "\\")
                if "=" in v:
                    k, val = v.split("=", 1)
                    env[k] = val
        if interp is None:
            return None
        return {"interpreter": interp, "env": env}
    except Exception:  # noqa: BLE001
        return None


def _systemd_env() -> dict[str, str]:
    """Env for systemctl --user calls. Self-heals XDG_RUNTIME_DIR: non-login
    shells (install.sh over ssh, cron) lack it even with lingering enabled,
    and systemctl then fails with 'Failed to connect to user scope bus'."""
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return env


def _systemd_manager_state() -> dict:
    """Query loaded/enabled/active. state=manager-unavailable when the user
    manager itself is unreachable (no lingering / no user session)."""
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", SYSTEMD_UNIT,
             "--property=LoadState,UnitFileState,ActiveState,SubState,MainPID"],
            capture_output=True, text=True, timeout=10, env=_systemd_env())
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"available": False, "error": str(e)}
    if r.returncode != 0:
        return {"available": False, "error": (r.stderr or r.stdout).strip()[:200]}
    props = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    return {
        "available": True,
        "loaded": props.get("LoadState") == "loaded",
        "enabled": props.get("UnitFileState") in ("enabled", "enabled-runtime", "linked"),
        "active": props.get("ActiveState") == "active",
        "main_pid": int(props.get("MainPID") or 0) or None,
    }


def _launchd_manager_state() -> dict:
    try:
        r = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"available": False, "error": str(e)}
    if r.returncode != 0:
        # launchctl print fails for not-loaded services with rc!=0; the manager
        # itself is fine. Distinguish via `launchctl print gui/$UID` (domain).
        rd = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}"],
                            capture_output=True, text=True, timeout=10)
        if rd.returncode != 0:
            return {"available": False, "error": (rd.stderr or "").strip()[:200]}
        return {"available": True, "loaded": False, "enabled": False,
                "active": False, "main_pid": None}
    pid = None
    running = False
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("pid = "):
            try:
                pid = int(line.split("=", 1)[1].strip())
                running = True
            except ValueError:
                pass
        if line.startswith("state = ") and "running" in line:
            running = True
    return {"available": True, "loaded": True, "enabled": True,
            "active": running, "main_pid": pid}


def _interpreter_functional(interp: str) -> tuple[bool, str]:
    """(exists AND `import dct` works, detail)."""
    p = Path(interp)
    if not p.exists():
        return False, "missing"
    try:
        r = subprocess.run([interp, "-c", "import dct"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"probe-error: {e}"
    return (r.returncode == 0,
            "ok" if r.returncode == 0 else f"import-dct-failed: {(r.stderr or '').strip()[:120]}")


def service_status() -> dict:
    """Drift-detection truth model for the installed OS service.

    Never raises; never exits non-zero (callers branch on ['state']).
    """
    sysname = platform.system()
    result: dict = {"platform": sysname, "state": "not-installed",
                    "unit_path": None, "facts": {}}
    if sysname == "Darwin":
        dest, parse, mgr = _launchd_path(), _parse_launchd_plist, _launchd_manager_state
    elif sysname == "Linux":
        dest, parse, mgr = _systemd_path(), _parse_systemd_unit, _systemd_manager_state
    else:
        result["state"] = "unknown"
        result["facts"]["error"] = f"unsupported OS: {sysname}"
        return result

    result["unit_path"] = str(dest)
    if not dest.exists():
        return result

    try:
        text = dest.read_text()
    except OSError as e:
        result["state"] = "unknown"
        result["facts"]["error"] = f"unreadable unit: {e}"
        return result
    parsed = parse(text)
    if parsed is None or not parsed.get("interpreter"):
        result["state"] = "unknown"
        result["facts"]["error"] = "unit file unparseable (hand-edited?)"
        return result

    interp = parsed["interpreter"]
    unit_home = parsed["env"].get("PDCT_HOME")
    our_home = str(_cfg.pdct_home())
    marker = read_state_marker()
    result["facts"].update({
        "unit_interpreter": interp,
        "current_interpreter": sys.executable,
        "unit_pdct_home": unit_home,
        "current_pdct_home": our_home,
        "intent_marker": marker,
    })

    # Ownership: the unit's PDCT_HOME matches ours → we own it (two live
    # checkouts sharing one PDCT_HOME is itself a misconfiguration; the unit's
    # env is the authority). Home mismatch + OUR interpreter → our unit with a
    # stale env (repairable FAIL). Home mismatch + foreign interpreter →
    # genuinely someone else's install: never touch. (Codex diff-audit #1/#2:
    # the earlier "live foreign interpreter = ambiguous" rule made
    # stale-interpreter unreachable for the realistic old-venv-still-exists
    # case and stale-env unreachable entirely.)
    home_match = (unit_home is None) or (
        os.path.normpath(os.path.expanduser(unit_home))
        == os.path.normpath(os.path.expanduser(our_home)))
    interp_is_ours = os.path.normpath(interp) == os.path.normpath(sys.executable)
    interp_exists = Path(interp).exists()
    owned = home_match or interp_is_ours
    result["facts"]["owned"] = owned
    if not owned:
        result["state"] = "not-owned"
        return result
    if not home_match:
        result["state"] = "stale-env"
        return result

    m = mgr()
    result["facts"]["manager"] = m
    if not m.get("available"):
        result["state"] = "manager-unavailable"
        result["facts"]["remedy"] = ("loginctl enable-linger $USER  # then re-run"
                                     if sysname == "Linux" else "check launchd user domain")
        return result

    if not interp_exists:
        result["state"] = "missing-interpreter"
        return result
    func_ok, func_detail = _interpreter_functional(interp)
    result["facts"]["interpreter_check"] = func_detail
    if not func_ok:
        result["state"] = "broken-interpreter"
        return result
    if not interp_is_ours:
        result["state"] = "stale-interpreter"
        return result

    expected_enabled = True if marker is None else bool(marker.get("enabled", True))
    result["facts"]["expected_enabled"] = expected_enabled
    if not m.get("enabled"):
        result["state"] = "installed-disabled"
        result["facts"]["intentional"] = not expected_enabled
        return result
    if not m.get("active"):
        result["state"] = "installed-inactive"
        return result
    result["state"] = "healthy"
    return result
