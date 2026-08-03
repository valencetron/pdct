"""Warm-first facade over memory_api.query_memory (2026-07-27).

THE BUG THIS FIXES
------------------
There were two PDCT query paths and only one knew the resident worker existed:

  * bridge  : server.py -> memory_client -> pdct-memory-worker socket  (~0.5s)
  * daemon  : daemon.py -> memory_api.query_memory                     (~13.9s)

`memory_api.query_memory` calls `build_index()` and constructs the whole model
stack in-process on EVERY call. The daemon, the deep-health probe, and every
CLI caller therefore paid full cold cost while an always-warm worker holding
the identical models sat idle on a socket. Measured 2026-07-27:

    worker socket   p50 535ms   p90 1095ms
    cold in-proc    13,900ms  (same query, same machine, same minute)

This module is the missing client half for in-repo Python callers. It tries
the warm worker first and falls back to the original in-process call, so it is
never worse than the code it replaces.

WHY IT IS NOT INSIDE query_memory
---------------------------------
`MemoryWorker._do_query` itself calls `memory_api.query_memory`. Putting the
socket hop inside `query_memory` would make the worker call itself forever.
The recursion guard (`PDCT_IN_MEMORY_WORKER=1`, set by the worker's own
launchd plist / runner) is belt-and-braces for anyone who wires this in
deeper later.

Pure stdlib on the warm path — the socket call must not drag numpy/torch into
a caller that only wanted five rows of text.
"""
from __future__ import annotations

import json
import os
import socket
import time as _time
from pathlib import Path
from typing import Any, Optional

_SOCKET_NAME = ".memory-worker.sock"
_DEFAULT_RUNTIME = Path.home() / "example-stack" / "dynamic-context-traversal" / "runtime"

# read_memory returns whole distillation bodies; 4MB bounds a runaway server
# without truncating a legitimate reply. Mirrors memory_client.
_MAX_REPLY_BYTES = 4 * 1024 * 1024

_FAIL_THRESHOLD = 3
_COOLOFF_S = 30.0
_BREAKER: dict[str, float] = {"until_mono": 0.0, "fails": 0.0}

# Env var the worker sets on itself. If present we NEVER dial the socket —
# that would be the process calling itself.
_IN_WORKER_ENV = "PDCT_IN_MEMORY_WORKER"

DEFAULT_TIMEOUT_S = float(os.environ.get("PDCT_WARM_TIMEOUT_S", "10") or 10)


def _reset_state() -> None:
    _BREAKER["until_mono"] = 0.0
    _BREAKER["fails"] = 0.0


def default_socket_path() -> Path:
    runtime = os.environ.get("PDCT_RUNTIME_DIR")
    base = Path(runtime) if runtime else _DEFAULT_RUNTIME
    return base / _SOCKET_NAME


def in_worker() -> bool:
    """True when running inside the memory worker itself (recursion guard)."""
    return bool(os.environ.get(_IN_WORKER_ENV))


def _record_failure() -> None:
    _BREAKER["fails"] += 1
    if _BREAKER["fails"] >= _FAIL_THRESHOLD:
        _BREAKER["until_mono"] = _time.monotonic() + _COOLOFF_S


def _record_success() -> None:
    _BREAKER["fails"] = 0.0
    _BREAKER["until_mono"] = 0.0


def _recv_line(conn: socket.socket, deadline: float) -> Optional[bytes]:
    """Read to newline under an absolute deadline and a byte cap."""
    buf = bytearray()
    while True:
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            return None
        conn.settimeout(remaining)
        try:
            chunk = conn.recv(65536)
        except (socket.timeout, OSError):
            return None
        if not chunk:
            return None
        buf.extend(chunk)
        nl = buf.find(b"\n")
        if nl != -1:
            if nl > _MAX_REPLY_BYTES:
                return None
            return bytes(buf[:nl])
        if len(buf) > _MAX_REPLY_BYTES:
            return None


def request(req: dict, *, path: Optional[Path] = None,
            timeout_s: Optional[float] = None) -> Optional[dict]:
    """One request to the worker. ``None`` means 'fall back', always."""
    if in_worker():
        return None
    if _BREAKER["until_mono"] > _time.monotonic():
        return None

    budget = DEFAULT_TIMEOUT_S if timeout_s is None else float(timeout_s)
    sock_path = str(path if path is not None else default_socket_path())
    deadline = _time.monotonic() + budget
    conn: Optional[socket.socket] = None
    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            _record_failure()
            return None
        conn.settimeout(remaining)
        conn.connect(sock_path)
        payload = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            _record_failure()
            return None
        conn.settimeout(remaining)
        conn.sendall(payload)
        line = _recv_line(conn, deadline)
        if line is None:
            _record_failure()
            return None
        try:
            body = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            _record_failure()
            return None
        if not isinstance(body, dict):
            _record_failure()
            return None
        if "error" in body:
            # A worker-reported engine error is a real answer, not a transport
            # fault — do not trip the breaker over one bad query.
            return None
        _record_success()
        return body
    except OSError:
        _record_failure()
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass


def _dict_to_row(d: dict):
    """Rehydrate the worker's JSON into a real DistillationRow so warm and
    cold paths are type-identical to the caller (daemon.py reads r.gist,
    r.score, ... as attributes — a dict would break it silently)."""
    from dct.retrieval.memory_api import DistillationRow
    return DistillationRow(
        id=str(d.get("id") or ""),
        path=str(d.get("path") or ""),
        date=str(d.get("date") or ""),
        title=str(d.get("title") or ""),
        concepts=list(d.get("concepts") or []),
        gist=str(d.get("gist") or ""),
        score=float(d.get("score") or 0.0),
        source=str(d.get("source") or "graph"),
    )


def query_memory_warm(seed, *, _surface: str = "unknown",
                      roots: Optional[list] = None,
                      exclude_roots: Optional[list] = None,
                      timeout_s: Optional[float] = None) -> list:
    """Drop-in for memory_api.query_memory: warm worker first, cold fallback.

    Returns list[DistillationRow] either way. Never raises for transport
    reasons — a dead worker costs one failed connect and falls through.
    """
    seed_str = seed if isinstance(seed, str) else " | ".join(seed or [])
    req: dict[str, Any] = {"mode": "query", "seed": seed_str,
                           "surface": _surface}
    if roots:
        req["roots"] = [str(r) for r in roots]
    if exclude_roots:
        req["exclude_roots"] = [str(r) for r in exclude_roots]

    body = request(req, timeout_s=timeout_s)
    if body is not None:
        rows = body.get("rows")
        if isinstance(rows, list):
            try:
                return [_dict_to_row(r) for r in rows if isinstance(r, dict)]
            except Exception:  # noqa: BLE001 — malformed row => cold path
                pass

    from dct.retrieval.memory_api import query_memory
    kw: dict[str, Any] = {"_surface": _surface}
    if roots:
        kw["roots"] = [Path(p) for p in roots]
    if exclude_roots:
        kw["exclude_roots"] = [Path(p) for p in exclude_roots]
    return query_memory(seed, **kw)


def ping(*, timeout_s: float = 2.0) -> Optional[dict]:
    """Worker health probe: {'ok':True,'warm':bool,'pid':int} or None."""
    return request({"op": "ping"}, timeout_s=timeout_s)
