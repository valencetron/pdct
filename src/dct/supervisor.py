"""PDCT supervisor — one pure-Python process that keeps the write path alive.

Runs the vault watcher (thread) and periodic scheduler ticks (subprocess) in
a single supervised loop. No launchd/systemd knowledge required — works on
any POSIX system. OS service templates (`pdct daemon install-service`) are
an optional upgrade for reboot persistence.

Layout (all under dct.config paths):
    runtime/supervisor.pid      pidfile
    runtime/supervisor.json     machine-readable status (fleet-probe contract)
    logs/supervisor.log         combined stdout/stderr of the daemonized run

Status JSON contract (stable — consumed by doctor stage 5 and any fleet
schematic / web checker):
    {
      "pid": 1234, "started_ts": 1710000000.0, "uptime_s": 512.3,
      "scheduler": {"interval_s": 300, "last_tick_ts": ..., "last_rc": 0,
                     "ticks": 3},
      "watcher": {"alive": true, "vault_root": "..."},
      "events_path": "...", "last_event_ts": 1710000100.0
    }
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from dct import config as _cfg

DEFAULT_SCHEDULER_INTERVAL = 300.0  # seconds


# ── paths ───────────────────────────────────────────────────────────────────

def pidfile_path() -> Path:
    return _cfg.runtime_dir() / "supervisor.pid"


def status_path() -> Path:
    return _cfg.runtime_dir() / "supervisor.json"


def log_path() -> Path:
    return _cfg.logs_dir() / "supervisor.log"


# ── liveness helpers ────────────────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else


def read_pid() -> int | None:
    p = pidfile_path()
    try:
        pid = int(p.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def _last_event_ts(events: Path) -> float | None:
    """Timestamp of the newest event in events.jsonl (tail read, cheap)."""
    try:
        with events.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.strip().splitlines()):
        try:
            ts = json.loads(line).get("ts")
            if isinstance(ts, (int, float)):
                return float(ts)
        except (json.JSONDecodeError, AttributeError):
            continue
    return None


# ── the supervised loop (foreground) ───────────────────────────────────────

class Supervisor:
    def __init__(self, *, scheduler_interval: float | None = None,
                 scheduler_limit: int = 20) -> None:
        env_iv = os.environ.get("PDCT_SCHEDULER_INTERVAL")
        self.interval = float(scheduler_interval if scheduler_interval is not None
                              else (env_iv or DEFAULT_SCHEDULER_INTERVAL))
        self.scheduler_limit = scheduler_limit
        self._stop = threading.Event()
        self._started_ts = time.time()
        self._sched = {"interval_s": self.interval, "last_tick_ts": None,
                       "last_rc": None, "ticks": 0}
        self._watcher_thread: threading.Thread | None = None
        self._vault_root: Path | None = None

    # signal handlers set the stop event; the loop exits cleanly
    def _install_signals(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: self._stop.set())

    def _start_watcher(self) -> None:
        roots = _cfg.vault_roots()
        root = next((r for r in roots if r.exists()), None)
        if root is None:
            return  # no vault yet — scheduler-only mode; status reflects it
        self._vault_root = root
        from dct.event_log import EventLog
        from dct.watch import run_watcher_until
        log = EventLog(_cfg.events_path())

        def _run() -> None:
            try:
                run_watcher_until(vault_root=root, log=log,
                                  until=self._stop.is_set)
            except Exception:  # noqa: BLE001 — watcher death shows in status
                pass

        t = threading.Thread(target=_run, name="pdct-watcher", daemon=True)
        t.start()
        self._watcher_thread = t

    def _tick_scheduler(self) -> None:
        """Run one scheduler pass. Interruptible: polls the stop event and
        terminates the child on shutdown so `pdct daemon stop` never hangs
        behind a long distillation run."""
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "dct.scheduler", "--quiet",
                 "--limit", str(self.scheduler_limit)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 600
            while proc.poll() is None:
                if self._stop.is_set() or time.monotonic() > deadline:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                self._stop.wait(timeout=0.5)
            rc = proc.returncode if proc.returncode is not None else -1
        except OSError:
            rc = -2
        self._sched["last_tick_ts"] = time.time()
        self._sched["last_rc"] = rc
        self._sched["ticks"] += 1

    def status_dict(self) -> dict:
        events = _cfg.events_path()
        return {
            "pid": os.getpid(),
            "started_ts": self._started_ts,
            "uptime_s": round(time.time() - self._started_ts, 1),
            "scheduler": dict(self._sched),
            "watcher": {
                "alive": bool(self._watcher_thread and
                              self._watcher_thread.is_alive()),
                "vault_root": str(self._vault_root) if self._vault_root else None,
            },
            "events_path": str(events),
            "last_event_ts": _last_event_ts(events),
        }

    def _write_status(self) -> None:
        p = status_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.status_dict(), indent=1))
        tmp.replace(p)

    def run(self) -> int:
        """Foreground supervised loop. Blocks until SIGTERM/SIGINT."""
        _cfg.runtime_dir().mkdir(parents=True, exist_ok=True)
        pidfile_path().write_text(f"{os.getpid()}\n")
        self._install_signals()
        self._start_watcher()
        self._write_status()
        next_tick = time.monotonic()  # first scheduler tick immediately
        try:
            while not self._stop.is_set():
                if time.monotonic() >= next_tick:
                    self._tick_scheduler()
                    next_tick = time.monotonic() + self.interval
                self._write_status()
                self._stop.wait(timeout=2.0)
        finally:
            self._write_status()
            try:
                pidfile_path().unlink()
            except OSError:
                pass
        return 0


# ── daemon control (start/stop/status from the CLI) ───────────────────────

def start_daemon() -> tuple[bool, str]:
    if (pid := read_pid()) is not None:
        return False, f"already running (pid {pid})"
    # Atomic start lock (O_CREAT|O_EXCL) — two concurrent `pdct daemon start`
    # calls must not both pass the pidfile check and spawn two supervisors.
    _cfg.runtime_dir().mkdir(parents=True, exist_ok=True)
    lock = _cfg.runtime_dir() / "supervisor.start-lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError:
        # Stale lock (crashed starter) is reclaimable after 30s.
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            age = 0.0
        if age < 30:
            return False, "another start is in progress"
        try:
            lock.unlink()
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except (OSError, FileExistsError):
            return False, "another start is in progress"
    try:
        return _start_daemon_locked()
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _start_daemon_locked() -> tuple[bool, str]:
    if (pid := read_pid()) is not None:  # re-check under the lock
        return False, f"already running (pid {pid})"
    _cfg.logs_dir().mkdir(parents=True, exist_ok=True)
    logf = open(log_path(), "a")  # noqa: SIM115 — handed to child
    env = dict(os.environ)
    proc = subprocess.Popen(
        [sys.executable, "-m", "dct.supervisor"],
        stdout=logf, stderr=logf, stdin=subprocess.DEVNULL,
        start_new_session=True, env=env,
    )
    logf.close()
    # wait briefly for the pidfile to confirm liveness
    for _ in range(50):
        if read_pid() == proc.pid:
            return True, f"started (pid {proc.pid}, log {log_path()})"
        if proc.poll() is not None:
            return False, (f"exited immediately rc={proc.returncode} — "
                           f"see {log_path()}")
        time.sleep(0.1)
    # No pidfile in 5s — kill the child before reporting failure, otherwise
    # a slow-starting orphan survives and a retry spawns a duplicate.
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    return False, f"no pidfile after 5s (child terminated) — see {log_path()}"


def stop_daemon(timeout: float = 10.0) -> tuple[bool, str]:
    pid = read_pid()
    if pid is None:
        return False, "not running"
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Reap if the daemon happens to be our own child (test harnesses) —
        # otherwise a zombie keeps os.kill(pid, 0) succeeding forever.
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        # The supervisor removes its pidfile on clean exit — that plus a
        # dead pid both count as stopped.
        if not pidfile_path().exists() or not _pid_alive(pid):
            return True, f"stopped (pid {pid})"
        time.sleep(0.1)
    return False, f"pid {pid} did not exit within {timeout}s"


def daemon_status() -> dict:
    """Merged liveness + last-written status file (fleet-probe contract)."""
    pid = read_pid()
    out: dict = {"running": pid is not None, "pid": pid}
    try:
        st = json.loads(status_path().read_text())
    except (OSError, json.JSONDecodeError):
        st = {}
    if st:
        if pid is not None and st.get("started_ts"):
            st["uptime_s"] = round(time.time() - st["started_ts"], 1)
        out["status"] = st
        out["stale"] = pid is None and bool(st)
    return out


def main() -> int:
    return Supervisor().run()


if __name__ == "__main__":
    raise SystemExit(main())
