"""embed_service tests (WS1 Phase 2, 2026-07-21).

The service loads the model once and answers encode/ping requests over an
AF_UNIX socket. These tests drive the REAL server (threaded, one request per
connection) with a FAKE model so no 400MB model loads. They pin the protocol,
robustness to garbage, concurrency, the read timeout, stale-socket cleanup, and
single-instance refusal.
"""
import hashlib
import json
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from dct.retrieval import embed_service as es
from dct.retrieval.concept_embeddings import MODEL_FINGERPRINT


@pytest.fixture
def tmp_path():
    """Short-path temp dir — macOS AF_UNIX paths cap at ~104 bytes."""
    d = Path(tempfile.mkdtemp(prefix="es-", dir="/tmp"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _FakeModel:
    """Deterministic one-hot 384-vec keyed by a stable hash of the text.
    One-hot vectors are already unit-norm and finite, so client validation
    passes without loading a real model."""

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 384
        v = np.zeros(384, dtype="float32")
        v[h] = 1.0
        return v

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        self.calls += 1
        return np.asarray([self._vec(t) for t in texts], dtype="float32")


def _serve(svc):
    t = threading.Thread(target=svc.serve_forever, daemon=True)
    t.start()
    return t


def _request(path, obj, *, timeout=2.0):
    """Open a fresh connection, send one JSON line, read one JSON reply."""
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.settimeout(timeout)
    c.connect(str(path))
    try:
        if obj is _NO_NEWLINE:
            c.sendall(b'{"op":"enc')  # partial, never terminated
            return c  # caller inspects the live connection
        raw = obj if isinstance(obj, bytes) else (json.dumps(obj) + "\n").encode()
        c.sendall(raw)
        buf = bytearray()
        while b"\n" not in buf:
            chunk = c.recv(65536)
            if not chunk:
                break
            buf.extend(chunk)
        if b"\n" not in buf:
            return None
        return json.loads(bytes(buf).split(b"\n", 1)[0])
    finally:
        if obj is not _NO_NEWLINE:
            c.close()


_NO_NEWLINE = object()


def _wait_ready(path, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(path).exists():
            try:
                c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                c.settimeout(0.2)
                c.connect(str(path))
                c.close()
                return True
            except OSError:
                pass
        time.sleep(0.01)
    return False


def test_encode_returns_vec(tmp_path):
    sock = tmp_path / "s.sock"
    model = _FakeModel()
    svc = es.EmbedService(sock, model_loader=lambda: model)
    _serve(svc)
    assert _wait_ready(sock)
    try:
        resp = _request(sock, {"op": "encode", "text": "hello world"})
        assert resp["ok"] is True
        assert resp["fingerprint"] == MODEL_FINGERPRINT
        assert len(resp["vec"]) == 384
        assert np.allclose(np.asarray(resp["vec"], dtype="float32"),
                           model._vec("hello world"))
    finally:
        svc.shutdown()


def test_malformed_json_survives(tmp_path):
    sock = tmp_path / "s.sock"
    svc = es.EmbedService(sock, model_loader=lambda: _FakeModel())
    _serve(svc)
    assert _wait_ready(sock)
    try:
        bad = _request(sock, b"not json at all\n")
        assert bad["ok"] is False
        # server still answers a subsequent valid request
        good = _request(sock, {"op": "encode", "text": "still alive"})
        assert good["ok"] is True and len(good["vec"]) == 384
    finally:
        svc.shutdown()


def test_unknown_op(tmp_path):
    sock = tmp_path / "s.sock"
    svc = es.EmbedService(sock, model_loader=lambda: _FakeModel())
    _serve(svc)
    assert _wait_ready(sock)
    try:
        resp = _request(sock, {"op": "bogus"})
        assert resp["ok"] is False
    finally:
        svc.shutdown()


def test_ping(tmp_path):
    sock = tmp_path / "s.sock"
    svc = es.EmbedService(sock, model_loader=lambda: _FakeModel())
    _serve(svc)
    assert _wait_ready(sock)
    try:
        resp = _request(sock, {"op": "ping"})
        assert resp["ok"] is True
        assert resp["fingerprint"] == MODEL_FINGERPRINT
    finally:
        svc.shutdown()


def test_concurrent_clients(tmp_path):
    sock = tmp_path / "s.sock"
    model = _FakeModel()
    svc = es.EmbedService(sock, model_loader=lambda: model)
    _serve(svc)
    assert _wait_ready(sock)
    try:
        results = {}
        errors = []

        def worker(i):
            try:
                text = f"query-{i}"
                resp = _request(sock, {"op": "encode", "text": text})
                results[i] = np.allclose(
                    np.asarray(resp["vec"], dtype="float32"), model._vec(text))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors
        assert len(results) == 8 and all(results.values())
    finally:
        svc.shutdown()


def test_stale_socket_file_unlinked(tmp_path):
    sock = tmp_path / "s.sock"
    # A leftover *plain file* at the socket path (no live server behind it).
    sock.write_bytes(b"stale")
    svc = es.EmbedService(sock, model_loader=lambda: _FakeModel())
    _serve(svc)
    assert _wait_ready(sock)  # bound despite the stale file
    try:
        resp = _request(sock, {"op": "ping"})
        assert resp["ok"] is True
    finally:
        svc.shutdown()


def test_second_instance_refused(tmp_path):
    sock = tmp_path / "s.sock"
    svc1 = es.EmbedService(sock, model_loader=lambda: _FakeModel())
    _serve(svc1)
    assert _wait_ready(sock)
    try:
        svc2 = es.EmbedService(sock, model_loader=lambda: _FakeModel())
        with pytest.raises(es.AlreadyRunning):
            svc2.serve_forever()
    finally:
        svc1.shutdown()


def test_second_instance_shutdown_preserves_healthy_socket(tmp_path):
    """Codex r2 P0: a losing second launch must NOT delete the live socket.
    Its entrypoint calls shutdown() in a finally; that shutdown must be a no-op
    on the pathname it never bound, leaving the first instance answering."""
    sock = tmp_path / "s.sock"
    svc1 = es.EmbedService(sock, model_loader=lambda: _FakeModel())
    _serve(svc1)
    assert _wait_ready(sock)
    try:
        svc2 = es.EmbedService(sock, model_loader=lambda: _FakeModel())
        with pytest.raises(es.AlreadyRunning):
            svc2.serve_forever()
        svc2.shutdown()  # the entrypoint's finally — must not unlink svc1's socket
        assert Path(sock).exists()               # socket file survived
        assert _request(sock, {"op": "ping"})["ok"] is True  # svc1 still answers
    finally:
        svc1.shutdown()


def test_partial_request_dropped_by_read_timeout(tmp_path):
    sock = tmp_path / "s.sock"
    svc = es.EmbedService(sock, model_loader=lambda: _FakeModel(),
                          read_timeout=0.3)
    _serve(svc)
    assert _wait_ready(sock)
    try:
        # Never send a newline; the server must drop us within read_timeout.
        c = _request(sock, _NO_NEWLINE)
        c.settimeout(1.5)
        t0 = time.monotonic()
        data = c.recv(4096)  # blocks until server closes the connection
        dt = time.monotonic() - t0
        c.close()
        assert data == b""       # server closed without replying
        assert dt < 1.4          # bounded by read_timeout, not hung
        # server is still healthy for well-formed clients
        good = _request(sock, {"op": "ping"})
        assert good["ok"] is True
    finally:
        svc.shutdown()


def test_recv_line_rejects_oversized_newline_terminated(monkeypatch):
    """Codex r1 P1: the byte cap must hold even when the oversized message
    IS newline-terminated. Previously the newline branch returned before the
    cap check, so a line longer than max_bytes slipped through."""
    a, b = socket.socketpair()
    try:
        b.sendall(b"x" * 100 + b"\n")
        b.close()
        out = es._recv_line(a, timeout_s=1.0, max_bytes=16)
        assert out is None  # over cap -> rejected, not returned
    finally:
        a.close()


def test_recv_line_accepts_within_cap(monkeypatch):
    a, b = socket.socketpair()
    try:
        b.sendall(b"hello\n")
        b.close()
        out = es._recv_line(a, timeout_s=1.0, max_bytes=16)
        assert out == b"hello"
    finally:
        a.close()
