"""Persistent warm embed service (WS1 Phase 2, 2026-07-21).

A dedicated, single-purpose process that loads ``BAAI/bge-small-en-v1.5`` once
and answers query-encode requests over an ``AF_UNIX`` stream socket. Because it
does nothing else, an encode stays ~40ms even when the daemon process is
saturated (GIL pressure, concurrent turns, Claude API calls) — that saturation
is the ~1.7s residual Phase 2 removes.

Protocol (newline-delimited JSON, one request/response per connection):
  request:  {"op":"encode","text":"<query>"}\\n   |  {"op":"ping"}\\n
  response: {"ok":true,"vec":[<384 floats>],"fingerprint":"..."}\\n
            {"ok":true,"model":"...","fingerprint":"..."}\\n   (ping)
            {"ok":false,"error":"..."}\\n

Discipline: model construction is done once at startup under a lock (torch/s-t
construction is not thread-safe — see vec_index); inference runs under an encode
lock. The handler bounds every read (deadline + byte cap) so a client that never
sends a newline is dropped instead of pinning a thread. Single-instance is
enforced with ``flock``; a stale socket left by a crashed process is probed and
unlinked, but a *healthy* peer refuses startup.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import socket
import socketserver
import threading
import time as _time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from dct.config import runtime_dir
from dct.retrieval.concept_embeddings import MODEL_FINGERPRINT

EMBED_DIM = 384
# Cap batch size server-side: bounds the single encode + response so a caller
# can't monopolize the service or blow memory (Codex r1 P1: encode_batch DoS).
# The daemon's client chunks well below this — it's a defensive backstop.
_MAX_BATCH_ITEMS = 64
_MAX_REQUEST_BYTES = 64 * 1024
_DEFAULT_READ_TIMEOUT = 5.0
_SOCKET_NAME = ".embed-service.sock"
_PROBE_TIMEOUT = 0.5

_log = logging.getLogger(__name__)


class AlreadyRunning(RuntimeError):
    """Another embed service holds the lock or a healthy socket."""


def default_socket_path() -> Path:
    return runtime_dir() / _SOCKET_NAME


def _default_model_loader() -> Any:
    from dct.retrieval.vec_index import _get_model

    return _get_model()


def _model_name() -> str:
    try:
        from dct.retrieval.vec_index import _MODEL_NAME

        return _MODEL_NAME
    except Exception:  # noqa: BLE001
        return "unknown"


def _recv_line(conn: socket.socket, timeout_s: float, max_bytes: int) -> Optional[bytes]:
    """Read until newline under an absolute deadline and a byte cap.

    Returns the line without the trailing newline, or ``None`` on timeout /
    close-before-newline / oversize (all of which drop the connection)."""
    deadline = _time.monotonic() + timeout_s
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
            if nl > max_bytes:  # newline-terminated but over the advertised cap
                return None
            return bytes(buf[:nl])
        if len(buf) > max_bytes:
            return None


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        svc: "EmbedService" = self.server.embed_service  # type: ignore[attr-defined]
        line = _recv_line(self.request, svc.read_timeout, _MAX_REQUEST_BYTES)
        if line is None:
            return
        resp = svc.handle_request(line)
        try:
            self.request.sendall((json.dumps(resp) + "\n").encode("utf-8"))
        except OSError:
            pass


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128  # listen backlog (default 5 refuses concurrent bursts)

    def __init__(self, path: str, handler, *, embed_service: "EmbedService"):
        self.embed_service = embed_service
        super().__init__(path, handler)


class EmbedService:
    def __init__(
        self,
        path,
        *,
        model_loader: Optional[Callable[[], Any]] = None,
        read_timeout: float = _DEFAULT_READ_TIMEOUT,
        lock_path=None,
    ) -> None:
        self.path = str(path)
        self.read_timeout = read_timeout
        self._model_loader = model_loader or _default_model_loader
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._enc_lock = threading.Lock()
        self._server: Optional[_ThreadingUnixServer] = None
        self._bound = False  # True only after WE created the socket at self.path
        self._lock_fd: Optional[int] = None
        self._lock_path = str(lock_path) if lock_path is not None \
            else str(Path(self.path).with_suffix(".lock"))

    # ---- model -------------------------------------------------------------
    def _ensure_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._model_loader()
        return self._model

    def warm(self) -> None:
        self._ensure_model()

    def _encode(self, text: str) -> np.ndarray:
        model = self._ensure_model()
        with self._enc_lock:
            out = model.encode([text], normalize_embeddings=True,
                               show_progress_bar=False)
        return np.asarray(out, dtype="float32")[0]

    def _encode_batch(self, texts: list) -> np.ndarray:
        """Encode a list of texts in one pass. Same model + normalization as
        _encode, so vectors are identical to the single-text path (score-
        identical with the in-process concept encode it replaces)."""
        model = self._ensure_model()
        with self._enc_lock:
            out = model.encode(list(texts), normalize_embeddings=True,
                               show_progress_bar=False)
        return np.asarray(out, dtype="float32")

    # ---- request dispatch --------------------------------------------------
    def handle_request(self, line: bytes) -> dict:
        try:
            req = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {"ok": False, "error": "bad_json"}
        if not isinstance(req, dict):
            return {"ok": False, "error": "bad_request"}
        op = req.get("op", "encode")
        if op == "ping":
            try:
                self._ensure_model()
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": f"model_load_failed: {e}"}
            return {"ok": True, "model": _model_name(),
                    "fingerprint": MODEL_FINGERPRINT}
        if op == "encode":
            text = req.get("text")
            if not isinstance(text, str) or not text:
                return {"ok": False, "error": "bad_text"}
            try:
                vec = self._encode(text)
            except Exception as e:  # noqa: BLE001
                _log.warning("[embed_service] encode failed: %s", e)
                return {"ok": False, "error": "encode_failed"}
            if vec.shape != (EMBED_DIM,) or not np.all(np.isfinite(vec)):
                return {"ok": False, "error": "bad_vec"}
            return {"ok": True, "vec": vec.tolist(),
                    "fingerprint": MODEL_FINGERPRINT}
        if op == "encode_batch":
            texts = req.get("texts")
            if (not isinstance(texts, list) or not texts
                    or not all(isinstance(t, str) and t for t in texts)):
                return {"ok": False, "error": "bad_texts"}
            if len(texts) > _MAX_BATCH_ITEMS:
                return {"ok": False, "error": "batch_too_large"}
            try:
                vecs = self._encode_batch(texts)
            except Exception as e:  # noqa: BLE001
                _log.warning("[embed_service] encode_batch failed: %s", e)
                return {"ok": False, "error": "encode_failed"}
            if (vecs.shape != (len(texts), EMBED_DIM)
                    or not np.all(np.isfinite(vecs))):
                return {"ok": False, "error": "bad_vec"}
            return {"ok": True, "vecs": vecs.tolist(),
                    "fingerprint": MODEL_FINGERPRINT}
        return {"ok": False, "error": "unknown_op"}

    # ---- lifecycle ---------------------------------------------------------
    def _acquire_lock(self) -> None:
        Path(self._lock_path).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            raise AlreadyRunning(
                f"another embed service holds {self._lock_path}") from e
        self._lock_fd = fd

    def _probe_existing(self) -> bool:
        """True if a *healthy* server already answers on the socket."""
        if not Path(self.path).exists():
            return False
        try:
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.settimeout(_PROBE_TIMEOUT)
            c.connect(self.path)
            c.sendall(b'{"op":"ping"}\n')
            line = _recv_line(c, _PROBE_TIMEOUT, _MAX_REQUEST_BYTES)
            c.close()
            if line is None:
                return False
            body = json.loads(line.decode("utf-8"))
            return bool(body.get("ok"))
        except OSError:
            return False
        except (ValueError, UnicodeDecodeError):
            return False

    def _bind(self) -> None:
        if self._probe_existing():
            raise AlreadyRunning(f"healthy embed service already on {self.path}")
        try:
            os.unlink(self.path)  # clear a stale/dead socket file
        except FileNotFoundError:
            pass
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._server = _ThreadingUnixServer(self.path, _Handler, embed_service=self)
        self._bound = True  # WE own this pathname now — only we may unlink it
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def serve_forever(self) -> None:
        self._acquire_lock()
        try:
            self._bind()
        except BaseException:
            self._release_lock()
            raise
        assert self._server is not None
        try:
            self._server.serve_forever()
        finally:
            self._unlink_bound_socket()  # remove OUR socket while the lock is held
            self._release_lock()

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None

    def _unlink_bound_socket(self) -> None:
        """Remove the socket file — but ONLY if this instance created it.

        A second-instance launch that lost the flock (or failed to bind) never
        set ``_bound``; it must NOT delete the healthy instance's socket. When
        serve_forever ran, its ``finally`` already unlinked under the lock, so
        this is idempotent."""
        if self._bound:
            try:
                os.unlink(self.path)
            except OSError:
                pass
            self._bound = False

    def shutdown(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._server.server_close()
            except Exception:  # noqa: BLE001
                pass
            self._server = None
        self._unlink_bound_socket()
        self._release_lock()
