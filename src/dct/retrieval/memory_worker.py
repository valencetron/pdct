"""Resident warm memory worker (2026-07-27).

The claude-mcp-bridge previously spawned `python -m dct.retrieval.memory_api`
fresh on EVERY pdct_query. Measured on-machine 2026-07-27: that cold path costs
56.9s (torch+s-t import 4.9s, bge-small load 3.6s, build_index 5.6s over 2875
rows, cross-encoder ms-marco-MiniLM-L-12-v2 ~9s, first encode ~1.1s), while the
second call in the same process is 0.67s and the third 0.55s — 85-100x. The
bridge's 30s CLI timeout killed every call, so pdct_query returned [].

This process loads the bi-encoder, the cross-encoder, and the distillation
index ONCE and answers queries over an AF_UNIX stream socket forever — exactly
the architecture the Telegram daemon already uses (`dct models warmed in 29.6s`
once at startup, then sub-2s forever).

Protocol (newline-delimited JSON, one request/response per connection) mirrors
the existing memory_api CLI request shape so the bridge change is minimal:
  request:  {"mode":"query","seed":...,"surface":...,"roots":[...],
             "exclude_roots":[...]}\\n
            {"mode":"read","id":"...","surface":"..."}\\n
            {"mode":"eligible","path":"<abs>"}\\n
            {"op":"ping"}\\n
  response: {"rows":[...]}\\n   |  {"ok":true,"warm":true,...}\\n (ping)
            {"error":"...","error_type":"..."}\\n

Discipline (mirrors embed_service.py, which is the proven pattern here):
  * Models are constructed ONCE, eagerly, at startup under the shared
    _MODEL_LOCK — rerank.py and vec_index.py both document meta-tensor
    corruption on torch 2.8 + sentence-transformers 5.1.2 when models are
    constructed CONCURRENTLY. A single resident process with serialized init
    makes concurrent construction structurally impossible.
  * Single-instance enforced with flock; a stale socket from a crashed process
    is probed and unlinked, but a HEALTHY peer refuses startup.
  * Every read is bounded (absolute deadline + byte cap) so a client that never
    sends a newline is dropped instead of pinning a thread.
  * Queries are serialized under a query lock: the DCT engine's caches
    (_concept_idx_cache, build_index memoization) are shared mutable state, and
    serialization also guarantees no concurrent model touch.
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
from typing import Any, Optional

from dct.config import runtime_dir

_MAX_REQUEST_BYTES = 256 * 1024
_DEFAULT_READ_TIMEOUT = 30.0
_SOCKET_NAME = ".memory-worker.sock"
_PROBE_TIMEOUT = 0.5

_log = logging.getLogger(__name__)


class AlreadyRunning(RuntimeError):
    """Another memory worker holds the lock or a healthy socket."""


def default_socket_path() -> Path:
    return runtime_dir() / _SOCKET_NAME


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
            if nl > max_bytes:
                return None
            return bytes(buf[:nl])
        if len(buf) > max_bytes:
            return None


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        svc: "MemoryWorker" = self.server.memory_worker  # type: ignore[attr-defined]
        line = _recv_line(self.request, svc.read_timeout, _MAX_REQUEST_BYTES)
        if line is None:
            return
        resp = svc.handle_request(line)
        try:
            self.request.sendall(
                (json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        except OSError:
            pass


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, path: str, handler, *, memory_worker: "MemoryWorker"):
        self.memory_worker = memory_worker
        super().__init__(path, handler)


class MemoryWorker:
    def __init__(
        self,
        path,
        *,
        read_timeout: float = _DEFAULT_READ_TIMEOUT,
        lock_path=None,
    ) -> None:
        self.path = str(path)
        self.read_timeout = read_timeout
        self._warm = False
        self._warm_lock = threading.Lock()
        # Queries are serialized: the engine's module-level caches
        # (_concept_idx_cache, build_index memoization, model singletons) are
        # shared mutable state. Serializing also makes concurrent model touch
        # impossible, which is the meta-tensor hardening requirement.
        self._query_lock = threading.Lock()
        self._server: Optional[_ThreadingUnixServer] = None
        self._bound = False
        self._lock_fd: Optional[int] = None
        self._lock_path = str(lock_path) if lock_path is not None \
            else str(Path(self.path).with_suffix(".lock"))

    # ---- warm --------------------------------------------------------------
    def warm(self) -> float:
        """Eagerly construct both models + build the index. Returns seconds.

        Constructions happen sequentially inside one lock — never concurrently
        (torch 2.8 + s-t 5.1.2 meta-tensor race, see rerank.py/vec_index.py).
        """
        t0 = _time.monotonic()
        with self._warm_lock:
            if self._warm:
                return 0.0
            from dct.retrieval.distill_index import build_index
            from dct.retrieval import rerank, vec_index

            # Sequential, never parallel. vec_index first (bi-encoder), then
            # rerank (cross-encoder) — both take the shared _MODEL_LOCK.
            vec_index._get_model()
            rerank._get_model()
            build_index()
            # The concept graph is a THIRD cold cost the model loads don't
            # cover: measured 2026-07-27, first query after a models-only warm
            # took 19.6s (graph load) while the engine logged only 522ms of
            # query time. Load it here so the first real caller doesn't pay it.
            from dct.retrieval.service import _load_or_build_graph
            _load_or_build_graph()
            self._warm = True
        return _time.monotonic() - t0

    def is_warm(self) -> bool:
        return self._warm

    # ---- request dispatch --------------------------------------------------
    def handle_request(self, line: bytes) -> dict:
        try:
            req = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {"error": "invalid JSON", "error_type": "JSONDecodeError"}
        if not isinstance(req, dict):
            return {"error": "bad request", "error_type": "ValueError"}

        # Health probe (distinct key from `mode` so it can't collide with the
        # memory_api CLI shape the bridge mirrors).
        if req.get("op") == "ping":
            return {"ok": True, "warm": self._warm, "pid": os.getpid()}

        mode = req.get("mode", "query")
        surface = req.get("surface") or "unknown"
        try:
            if mode == "query":
                return self._do_query(req, surface)
            if mode == "read":
                return self._do_read(req, surface)
            if mode == "eligible":
                return self._do_eligible(req)
            return {"error": f"unknown mode: {mode}", "error_type": "ValueError"}
        except KeyError as e:
            return {"error": str(e), "error_type": "KeyError"}
        except Exception as e:  # noqa: BLE001
            _log.warning("[memory_worker] %s failed: %s", mode, e)
            return {"error": str(e), "error_type": type(e).__name__}

    def _do_query(self, req: dict, surface: str) -> dict:
        from dct.retrieval.memory_api import query_memory, _row_to_dict

        kw: dict[str, Any] = {"_surface": surface}
        raw_roots = req.get("roots") or None
        raw_excl = req.get("exclude_roots") or None
        if raw_roots:
            kw["roots"] = [Path(p) for p in raw_roots]
        if raw_excl:
            kw["exclude_roots"] = [Path(p) for p in raw_excl]
        with self._query_lock:
            rows = query_memory(req.get("seed", ""), **kw)
        return {"rows": [_row_to_dict(r) for r in rows]}

    def _do_read(self, req: dict, surface: str) -> dict:
        from dct.retrieval.memory_api import read_memory

        with self._query_lock:
            result = read_memory(req.get("id", ""), _surface=surface)
        return {
            "id": result.id, "date": result.date, "title": result.title,
            "related_distillations": [
                {"id": r.id, "title": r.title, "score": r.score}
                for r in result.related_distillations
            ],
            "body": result.body,
        }

    def _do_eligible(self, req: dict) -> dict:
        from dct.retrieval.eligibility import is_eligible
        from dct.retrieval.distill_index import _ref_from_file, _split_frontmatter

        p = Path(req.get("path", ""))
        if not p.is_file():
            return {"ok": False, "reason": "no-such-file"}
        ref = _ref_from_file(p)
        raw = p.read_text(encoding="utf-8", errors="replace")
        _, body = _split_frontmatter(raw)
        ok, reason = is_eligible(ref, body)
        return {"ok": ok, "reason": reason}

    # ---- lifecycle ---------------------------------------------------------
    def _acquire_lock(self) -> None:
        Path(self._lock_path).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            raise AlreadyRunning(
                f"another memory worker holds {self._lock_path}") from e
        self._lock_fd = fd

    def _probe_existing(self) -> bool:
        """True if a *healthy* worker already answers on the socket."""
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
            raise AlreadyRunning(f"healthy memory worker already on {self.path}")
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._server = _ThreadingUnixServer(self.path, _Handler, memory_worker=self)
        self._bound = True
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
            self._unlink_bound_socket()
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
        """Remove the socket file — but ONLY if this instance created it."""
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
