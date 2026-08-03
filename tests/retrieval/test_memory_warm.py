"""Tests for the warm-first query facade (memory_warm).

The defect being locked down: memory_api.query_memory builds the full model
stack in-process on every call (13.9s measured) while an always-warm worker
holding the same models sits on a socket (0.5s). memory_warm is the client
half. Every test here is about the CONTRACT that makes it safe to put in
front of the daemon's hot path:

  - warm hit returns real DistillationRow objects (attribute access, not dicts)
  - every failure mode falls back rather than raising
  - the worker never dials itself (infinite recursion)
  - the breaker stops a dead worker costing a connect per call
"""
from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path

import pytest

from dct.retrieval import memory_warm as mw
from dct.retrieval.memory_api import DistillationRow


ROW = {
    "id": "2026-07-27-abc", "path": "/v/x.md", "date": "2026-07-27",
    "title": "Deep Health", "concepts": ["health", "probe"],
    "gist": "a gist", "score": 0.77, "source": "graph",
}


@pytest.fixture
def sockdir():
    """AF_UNIX paths cap at ~104 chars; pytest's tmp_path blows past it."""
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="mw", dir="/tmp")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    mw._reset_state()
    monkeypatch.delenv(mw._IN_WORKER_ENV, raising=False)
    yield
    mw._reset_state()


def _serve_once(sock_path, reply: bytes, *, delay: float = 0.0):
    """Tiny one-shot AF_UNIX server; returns (thread, captured_requests)."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(1)
    captured: list = []

    def run():
        try:
            conn, _ = srv.accept()
            buf = b""
            while not buf.endswith(b"\n"):
                c = conn.recv(65536)
                if not c:
                    break
                buf += c
            captured.append(buf)
            if delay:
                import time as _t
                _t.sleep(delay)
            conn.sendall(reply)
            conn.close()
        except OSError:
            pass
        finally:
            srv.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, captured


# ── warm hit ──────────────────────────────────────────────────────────────
def test_warm_hit_returns_distillation_rows_not_dicts(sockdir, monkeypatch):
    """daemon.py does r.gist / r.score — a dict would AttributeError at
    runtime, in production, on the hot path. Type identity is the contract."""
    sock = sockdir / ".memory-worker.sock"
    _serve_once(sock, json.dumps({"rows": [ROW]}).encode() + b"\n")
    monkeypatch.setattr(mw, "default_socket_path", lambda: sock)

    def _explode(*a, **k):  # cold path must NOT run
        raise AssertionError("fell back despite a healthy worker")
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory", _explode)

    rows = mw.query_memory_warm("deep health", _surface="telegram")
    assert len(rows) == 1
    assert isinstance(rows[0], DistillationRow)
    assert rows[0].gist == "a gist"
    assert rows[0].score == 0.77
    assert rows[0].concepts == ["health", "probe"]


def test_warm_request_carries_seed_surface_and_roots(sockdir, monkeypatch):
    sock = sockdir / ".memory-worker.sock"
    _t, captured = _serve_once(sock, json.dumps({"rows": []}).encode() + b"\n")
    monkeypatch.setattr(mw, "default_socket_path", lambda: sock)
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory",
                        lambda *a, **k: [])

    mw.query_memory_warm("seedtext", _surface="claudeai-alex",
                         roots=["/a"], exclude_roots=["/b"])
    req = json.loads(captured[0].decode())
    assert req["mode"] == "query"
    assert req["seed"] == "seedtext"
    assert req["surface"] == "claudeai-alex"
    assert req["roots"] == ["/a"]
    assert req["exclude_roots"] == ["/b"]


def test_list_seed_is_joined_for_the_worker(sockdir, monkeypatch):
    sock = sockdir / ".memory-worker.sock"
    _t, captured = _serve_once(sock, json.dumps({"rows": []}).encode() + b"\n")
    monkeypatch.setattr(mw, "default_socket_path", lambda: sock)
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory",
                        lambda *a, **k: [])
    mw.query_memory_warm(["a", "b"], _surface="telegram")
    assert json.loads(captured[0].decode())["seed"] == "a | b"


# ── fallback: every failure mode degrades, none raises ────────────────────
@pytest.mark.parametrize("reply,label", [
    (b'{"error":"engine blew up"}\n', "worker-reported error"),
    (b'not json at all\n', "unparseable"),
    (b'["not","a","dict"]\n', "wrong toplevel type"),
    (b'{"rows":"notalist"}\n', "rows wrong type"),
    (b'', "closed without replying"),
])
def test_every_bad_reply_falls_back_to_cold(sockdir, monkeypatch, reply, label):
    sock = sockdir / ".memory-worker.sock"
    _serve_once(sock, reply)
    monkeypatch.setattr(mw, "default_socket_path", lambda: sock)
    sentinel = [DistillationRow(id="cold", path="", date="", title="COLD")]
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory",
                        lambda *a, **k: sentinel)
    rows = mw.query_memory_warm("x", _surface="telegram")
    assert rows[0].id == "cold", f"{label} should have fallen back"


def test_no_worker_at_all_falls_back(sockdir, monkeypatch):
    monkeypatch.setattr(mw, "default_socket_path",
                        lambda: sockdir / "nothing-here.sock")
    sentinel = [DistillationRow(id="cold", path="", date="", title="COLD")]
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory",
                        lambda *a, **k: sentinel)
    assert mw.query_memory_warm("x")[0].id == "cold"


def test_malformed_row_content_falls_back(sockdir, monkeypatch):
    """A row whose score is garbage must not half-populate the result."""
    sock = sockdir / ".memory-worker.sock"
    bad = dict(ROW, score="not-a-float")
    _serve_once(sock, json.dumps({"rows": [bad]}).encode() + b"\n")
    monkeypatch.setattr(mw, "default_socket_path", lambda: sock)
    sentinel = [DistillationRow(id="cold", path="", date="", title="COLD")]
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory",
                        lambda *a, **k: sentinel)
    assert mw.query_memory_warm("x")[0].id == "cold"


def test_timeout_falls_back_and_does_not_hang(sockdir, monkeypatch):
    sock = sockdir / ".memory-worker.sock"
    _serve_once(sock, json.dumps({"rows": [ROW]}).encode() + b"\n", delay=2.0)
    monkeypatch.setattr(mw, "default_socket_path", lambda: sock)
    sentinel = [DistillationRow(id="cold", path="", date="", title="COLD")]
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory",
                        lambda *a, **k: sentinel)
    import time as _t
    t0 = _t.monotonic()
    rows = mw.query_memory_warm("x", timeout_s=0.3)
    assert rows[0].id == "cold"
    assert _t.monotonic() - t0 < 1.5, "timeout budget was not honored"


# ── recursion guard ───────────────────────────────────────────────────────
def test_worker_never_dials_itself(sockdir, monkeypatch):
    """MemoryWorker._do_query calls query_memory. If this facade ever ends up
    inside that path, dialing the socket would be the process calling itself
    forever. The env guard must short-circuit BEFORE any connect."""
    monkeypatch.setenv(mw._IN_WORKER_ENV, "1")

    def _no_socket(*a, **k):
        raise AssertionError("worker tried to dial its own socket")
    monkeypatch.setattr(mw.socket, "socket", _no_socket)
    sentinel = [DistillationRow(id="cold", path="", date="", title="COLD")]
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory",
                        lambda *a, **k: sentinel)

    assert mw.in_worker() is True
    assert mw.query_memory_warm("x")[0].id == "cold"


# ── breaker ───────────────────────────────────────────────────────────────
def test_breaker_opens_after_threshold_and_skips_connect(sockdir, monkeypatch):
    monkeypatch.setattr(mw, "default_socket_path",
                        lambda: sockdir / "dead.sock")
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory",
                        lambda *a, **k: [])
    for _ in range(mw._FAIL_THRESHOLD):
        mw.query_memory_warm("x")
    assert mw._BREAKER["until_mono"] > 0

    def _no_socket(*a, **k):
        raise AssertionError("breaker open but still dialed")
    monkeypatch.setattr(mw.socket, "socket", _no_socket)
    mw.query_memory_warm("x")  # must not raise


def test_worker_error_does_not_trip_the_breaker(sockdir, monkeypatch):
    """One bad query is a real answer, not a transport fault — blinding the
    worker for 30s over it would turn a single failure into an outage."""
    sock = sockdir / ".memory-worker.sock"
    _serve_once(sock, b'{"error":"bad seed"}\n')
    monkeypatch.setattr(mw, "default_socket_path", lambda: sock)
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory",
                        lambda *a, **k: [])
    mw.query_memory_warm("x")
    assert mw._BREAKER["fails"] == 0
    assert mw._BREAKER["until_mono"] == 0.0


def test_success_resets_the_failure_counter(sockdir, monkeypatch):
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory",
                        lambda *a, **k: [])
    monkeypatch.setattr(mw, "default_socket_path",
                        lambda: sockdir / "dead.sock")
    mw.query_memory_warm("x")
    assert mw._BREAKER["fails"] == 1

    sock = sockdir / ".memory-worker.sock"
    _serve_once(sock, json.dumps({"rows": [ROW]}).encode() + b"\n")
    monkeypatch.setattr(mw, "default_socket_path", lambda: sock)
    mw.query_memory_warm("x")
    assert mw._BREAKER["fails"] == 0


def test_oversize_reply_is_rejected(sockdir, monkeypatch):
    sock = sockdir / ".memory-worker.sock"
    monkeypatch.setattr(mw, "_MAX_REPLY_BYTES", 64)
    _serve_once(sock, json.dumps({"rows": [ROW] * 20}).encode() + b"\n")
    monkeypatch.setattr(mw, "default_socket_path", lambda: sock)
    sentinel = [DistillationRow(id="cold", path="", date="", title="COLD")]
    monkeypatch.setattr("dct.retrieval.memory_api.query_memory",
                        lambda *a, **k: sentinel)
    assert mw.query_memory_warm("x")[0].id == "cold"


def test_ping_returns_worker_health(sockdir, monkeypatch):
    sock = sockdir / ".memory-worker.sock"
    _serve_once(sock, json.dumps({"ok": True, "warm": True, "pid": 42}).encode() + b"\n")
    monkeypatch.setattr(mw, "default_socket_path", lambda: sock)
    assert mw.ping()["pid"] == 42


def test_env_override_sets_default_timeout(monkeypatch):
    """PDCT_WARM_TIMEOUT_S is how the forced-negative replay squeezes the
    warm path — it must actually be read."""
    import importlib
    monkeypatch.setenv("PDCT_WARM_TIMEOUT_S", "1")
    reloaded = importlib.reload(mw)
    try:
        assert reloaded.DEFAULT_TIMEOUT_S == 1.0
    finally:
        monkeypatch.delenv("PDCT_WARM_TIMEOUT_S", raising=False)
        importlib.reload(mw)
