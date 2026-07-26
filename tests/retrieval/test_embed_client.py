"""embed_client tests (WS1 Phase 2, 2026-07-21).

The client sends a query to the off-process embed service over AF_UNIX and
returns a 384-vec or None (fail open). A fake server exercises every failure
path without loading the real model.
"""
import json
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from dct.retrieval import embed_client as ec
from dct.retrieval.concept_embeddings import MODEL_FINGERPRINT


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch):
    ec._reset_client_state()
    yield
    ec._reset_client_state()


@pytest.fixture
def tmp_path():
    """Short-path temp dir. macOS AF_UNIX paths cap at ~104 bytes; pytest's
    own tmp_path is far too long to bind a socket under."""
    d = Path(tempfile.mkdtemp(prefix="ec-", dir="/tmp"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _good_vec():
    v = np.zeros(384, dtype="float32")
    v[3] = 1.0  # unit norm, finite
    return v


class _FakeServer:
    """A minimal AF_UNIX server: reads one request line, sends one configured
    reply. `reply` is bytes (sent as-is), or None (never reply / hang),
    optionally split into `chunks` with `chunk_delay` between them."""

    def __init__(self, path, reply, *, chunks=1, chunk_delay=0.0, accept_delay=0.0):
        self.path = str(path)
        self.reply = reply
        self.chunks = chunks
        self.chunk_delay = chunk_delay
        self.accept_delay = accept_delay
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.path)
        self._sock.listen(8)
        self._stop = False
        self._t = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass
        try:
            Path(self.path).unlink()
        except OSError:
            pass

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            if self.accept_delay:
                time.sleep(self.accept_delay)
            conn.recv(65536)  # drain the request
            if self.reply is None:
                time.sleep(2.0)  # hang past any client timeout
                return
            if self.chunks <= 1:
                conn.sendall(self.reply)
            else:
                n = max(1, len(self.reply) // self.chunks)
                for i in range(0, len(self.reply), n):
                    conn.sendall(self.reply[i:i + n])
                    time.sleep(self.chunk_delay)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


def _reply(ok=True, vec=None, fingerprint=MODEL_FINGERPRINT, raw=None):
    if raw is not None:
        return raw
    body = {"ok": ok}
    if vec is not None:
        body["vec"] = np.asarray(vec, dtype="float32").tolist()
        body["fingerprint"] = fingerprint
    return (json.dumps(body) + "\n").encode("utf-8")


def test_default_socket_path_under_runtime():
    p = ec.default_socket_path()
    assert p.name == ".embed-service.sock"
    assert "runtime" in str(p)


def test_encode_query_returns_vec(tmp_path):
    sock = tmp_path / "s.sock"
    want = _good_vec()
    with _FakeServer(sock, _reply(vec=want)):
        got = ec.encode_query("a sufficiently long query", path=sock)
    assert got is not None
    assert got.shape == (384,)
    assert np.allclose(got, want)


def test_no_server_returns_none_fast(tmp_path):
    sock = tmp_path / "nope.sock"
    t0 = time.monotonic()
    got = ec.encode_query("a sufficiently long query", path=sock, timeout_s=0.3)
    assert got is None
    assert time.monotonic() - t0 < 1.0  # no hang


def test_wrong_dim_returns_none(tmp_path):
    sock = tmp_path / "s.sock"
    with _FakeServer(sock, _reply(vec=np.zeros(7, dtype="float32"))):
        assert ec.encode_query("a sufficiently long query", path=sock) is None


def test_non_finite_returns_none(tmp_path):
    sock = tmp_path / "s.sock"
    bad = _good_vec(); bad[0] = np.nan
    with _FakeServer(sock, _reply(vec=bad)):
        assert ec.encode_query("a sufficiently long query", path=sock) is None


def test_non_unit_norm_returns_none(tmp_path):
    sock = tmp_path / "s.sock"
    bad = np.zeros(384, dtype="float32"); bad[0] = 5.0  # norm 5, not unit
    with _FakeServer(sock, _reply(vec=bad)):
        assert ec.encode_query("a sufficiently long query", path=sock) is None


def test_fingerprint_mismatch_returns_none(tmp_path):
    sock = tmp_path / "s.sock"
    with _FakeServer(sock, _reply(vec=_good_vec(), fingerprint="other-model/999/v9")):
        assert ec.encode_query("a sufficiently long query", path=sock) is None


def test_nonnumeric_vec_returns_none_not_raise(tmp_path):
    """Codex r2 P1: a 384-element but non-numeric vec must fail open (None),
    not raise out of encode_query's OSError-only guard."""
    sock = tmp_path / "s.sock"
    body = {"ok": True, "fingerprint": MODEL_FINGERPRINT, "vec": ["x"] * 384}
    raw = (json.dumps(body) + "\n").encode("utf-8")
    with _FakeServer(sock, _reply(raw=raw)):
        assert ec.encode_query("a sufficiently long query", path=sock) is None


def test_ok_false_returns_none(tmp_path):
    sock = tmp_path / "s.sock"
    with _FakeServer(sock, _reply(ok=False)):
        assert ec.encode_query("a sufficiently long query", path=sock) is None


def test_non_json_returns_none(tmp_path):
    sock = tmp_path / "s.sock"
    with _FakeServer(sock, b"not json at all\n"):
        assert ec.encode_query("a sufficiently long query", path=sock) is None


def test_fragmented_response_reassembled(tmp_path):
    sock = tmp_path / "s.sock"
    want = _good_vec()
    payload = _reply(vec=want)
    with _FakeServer(sock, payload, chunks=5, chunk_delay=0.005):
        got = ec.encode_query("a sufficiently long query", path=sock, timeout_s=1.0)
    assert got is not None and np.allclose(got, want)


def test_hang_returns_none_within_timeout(tmp_path):
    sock = tmp_path / "s.sock"
    with _FakeServer(sock, None):  # accepts, never replies
        t0 = time.monotonic()
        got = ec.encode_query("a sufficiently long query", path=sock, timeout_s=0.3)
        dt = time.monotonic() - t0
    assert got is None
    assert dt < 1.5  # bounded by the timeout, not the 2s server hang


def test_client_backoff_skips_socket_after_repeated_failure(tmp_path):
    dead = tmp_path / "dead.sock"
    # Trip the breaker with consecutive failures against a dead path.
    for _ in range(ec._CLIENT_FAIL_THRESHOLD):
        assert ec.encode_query("a sufficiently long query", path=dead, timeout_s=0.2) is None
    # Now a LIVE server exists, but backoff should skip the socket entirely.
    live = tmp_path / "live.sock"
    with _FakeServer(live, _reply(vec=_good_vec())):
        t0 = time.monotonic()
        got = ec.encode_query("a sufficiently long query", path=live, timeout_s=0.2)
        dt = time.monotonic() - t0
        assert got is None          # skipped due to backoff
        assert dt < 0.1             # didn't even attempt a connect
        # After reset, the same live server is used successfully.
        ec._reset_client_state()
        assert ec.encode_query("a sufficiently long query", path=live) is not None
