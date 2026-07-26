"""Thin client for the off-process embed service (WS1 Phase 2, 2026-07-21).

The daemon's per-turn query encode is the ~1.7s residual: `model.encode` runs
inside the contended daemon process. This client hands the encode to a dedicated
always-warm `embed_service` over an AF_UNIX socket and returns a validated
384-vec, or ``None`` to let the caller fall back (fail open).

Discipline mirrors the cosine breaker: every failure path returns ``None`` fast,
a bounded absolute deadline caps the wait, and a small circuit breaker skips the
socket entirely after repeated failures so a dead/slow service can't tax the
turn. Nothing here loads the model.
"""
from __future__ import annotations

import json
import socket
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from dct.config import runtime_dir
from dct.retrieval.concept_embeddings import MODEL_FINGERPRINT

EMBED_DIM = 384
_SOCKET_NAME = ".embed-service.sock"

# Response byte cap: a 384-float JSON reply is ~6-8KB; 256KB is a wide margin
# that still bounds a runaway/garbage server.
_MAX_REPLY_BYTES = 256 * 1024
_UNIT_NORM_TOL = 1e-3

# Circuit breaker: after N consecutive failures, skip the socket for a cooloff
# so a down service costs one connect attempt, not one per turn.
_CLIENT_FAIL_THRESHOLD = 3
_CLIENT_COOLOFF_S = 30.0
_CLIENT_BREAKER = {"until_mono": 0.0, "fails": 0}


def _reset_client_state() -> None:
    _CLIENT_BREAKER["until_mono"] = 0.0
    _CLIENT_BREAKER["fails"] = 0


def default_socket_path() -> Path:
    return runtime_dir() / _SOCKET_NAME


def _record_failure() -> None:
    _CLIENT_BREAKER["fails"] += 1
    if _CLIENT_BREAKER["fails"] >= _CLIENT_FAIL_THRESHOLD:
        _CLIENT_BREAKER["until_mono"] = _time.monotonic() + _CLIENT_COOLOFF_S


def _record_success() -> None:
    _CLIENT_BREAKER["fails"] = 0
    _CLIENT_BREAKER["until_mono"] = 0.0


def _recv_line(conn: socket.socket, deadline: float) -> Optional[bytes]:
    """Read until newline, an absolute wall-clock deadline, or the byte cap.

    Reassembles fragmented replies. Returns the line without the trailing
    newline, or ``None`` on timeout / closed-without-newline / oversize.
    """
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
            return None  # closed before a full line
        buf.extend(chunk)
        nl = buf.find(b"\n")
        if nl != -1:
            if nl > _MAX_REPLY_BYTES:  # newline-terminated but over the advertised cap
                return None
            return bytes(buf[:nl])
        if len(buf) > _MAX_REPLY_BYTES:
            return None


def _validate(payload: bytes) -> Optional[np.ndarray]:
    try:
        body = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(body, dict) or not body.get("ok"):
        return None
    if body.get("fingerprint") != MODEL_FINGERPRINT:
        return None
    raw = body.get("vec")
    if not isinstance(raw, list) or len(raw) != EMBED_DIM:
        return None
    try:  # non-numeric elements (strings/objects) must fail closed, not raise
        vec = np.asarray(raw, dtype="float32")
        if not np.all(np.isfinite(vec)):
            return None
        norm = float(np.linalg.norm(vec))
    except (ValueError, TypeError):
        return None
    if abs(norm - 1.0) > _UNIT_NORM_TOL:
        return None
    return vec


def encode_query(
    text: str,
    *,
    path: Optional[Path] = None,
    timeout_s: float = 0.35,
) -> Optional[np.ndarray]:
    """Encode ``text`` via the off-process embed service.

    Returns a validated 384-vec, or ``None`` on any failure so the caller can
    fall back to in-process encode. Bounded by ``timeout_s`` (absolute deadline
    across connect + send + recv). Honors the client-side circuit breaker.
    """
    if _CLIENT_BREAKER["until_mono"] > _time.monotonic():
        return None

    sock_path = str(path if path is not None else default_socket_path())
    deadline = _time.monotonic() + timeout_s
    conn: Optional[socket.socket] = None
    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            _record_failure()
            return None
        conn.settimeout(remaining)
        conn.connect(sock_path)
        request = (json.dumps({"op": "encode", "text": text}) + "\n").encode("utf-8")
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            _record_failure()
            return None
        conn.settimeout(remaining)  # absolute deadline honored across send too
        conn.sendall(request)
        line = _recv_line(conn, deadline)
        if line is None:
            _record_failure()
            return None
        vec = _validate(line)
        if vec is None:
            _record_failure()
            return None
        _record_success()
        return vec
    except OSError:
        _record_failure()
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass


# Concept-batch encode (2026-07-22): the cosine filter's concept-text cache
# misses were the last in-process bge encode on the turn path (~1.4s under
# contention). Route them through the same always-warm service. Chunked so each
# reply stays under _MAX_REPLY_BYTES: a DENSE normalized bge vec serializes to
# ~8KB of JSON (not the ~5KB a sparse vec suggests), so 16 vecs ≈ 130KB — ~2x
# margin under the 256KB cap. (Codex r1 P1: a 32-batch of dense vecs ≈ 270KB,
# over the cap → _recv_line drops it → the whole change silently no-ops.)
_BATCH_CHUNK = 16


def _validate_batch(payload: bytes, n: int) -> Optional[np.ndarray]:
    try:
        body = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(body, dict) or not body.get("ok"):
        return None
    if body.get("fingerprint") != MODEL_FINGERPRINT:
        return None
    raw = body.get("vecs")
    if not isinstance(raw, list) or len(raw) != n:
        return None
    try:
        arr = np.asarray(raw, dtype="float32")
    except (ValueError, TypeError):
        return None
    if arr.shape != (n, EMBED_DIM) or not np.all(np.isfinite(arr)):
        return None
    norms = np.linalg.norm(arr, axis=1)
    if not np.all(np.abs(norms - 1.0) <= _UNIT_NORM_TOL):
        return None
    return arr


def _encode_batch_chunk(texts: list, sock_path: str,
                        deadline: float) -> Optional[np.ndarray]:
    conn: Optional[socket.socket] = None
    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            return None
        conn.settimeout(remaining)
        conn.connect(sock_path)
        request = (json.dumps({"op": "encode_batch", "texts": texts})
                   + "\n").encode("utf-8")
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            return None
        conn.settimeout(remaining)
        conn.sendall(request)
        line = _recv_line(conn, deadline)
        if line is None:
            return None
        return _validate_batch(line, len(texts))
    except OSError:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass


def encode_texts(
    texts,
    *,
    path: Optional[Path] = None,
    timeout_s: float = 1.0,
) -> Optional[np.ndarray]:
    """Encode a batch of texts via the off-process embed service.

    Returns a validated (n, 384) float32 array (in input order), or ``None`` on
    any failure so the caller can fall back to an in-process encode. Bounded by
    ``timeout_s`` (absolute deadline across all chunks). Honors the client-side
    circuit breaker. An empty input returns an empty (0, 384) array.
    """
    texts = list(texts)
    if not texts:
        return np.empty((0, EMBED_DIM), dtype="float32")
    if not all(isinstance(t, str) and t for t in texts):
        return None  # uphold "None on any failure" before json.dumps can raise
    if _CLIENT_BREAKER["until_mono"] > _time.monotonic():
        return None
    sock_path = str(path if path is not None else default_socket_path())
    deadline = _time.monotonic() + timeout_s
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), _BATCH_CHUNK):
        vecs = _encode_batch_chunk(texts[i:i + _BATCH_CHUNK], sock_path, deadline)
        if vecs is None:
            _record_failure()
            return None
        parts.append(vecs)
    _record_success()
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
