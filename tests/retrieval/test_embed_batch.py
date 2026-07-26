"""encode_batch op + encode_texts client + relevance off-proc adapter (2026-07-22).

Drives the REAL embed service (fake one-hot model) via the REAL client over an
AF_UNIX socket, so the batch protocol, chunking, ordering, validation, and the
off-proc concept adapter's fallback are exercised end-to-end without loading a
400MB model. Score-identity against real bge is checked separately."""
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
from dct.retrieval import embed_client as ec


@pytest.fixture
def tmp_sock():
    d = Path(tempfile.mkdtemp(prefix="eb-", dir="/tmp"))
    ec._reset_client_state()
    try:
        yield d / "s.sock"
    finally:
        ec._reset_client_state()
        shutil.rmtree(d, ignore_errors=True)


class _FakeModel:
    """Deterministic one-hot 384-vec — unit-norm + finite, so validation passes."""
    def _vec(self, text):
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 384
        v = np.zeros(384, dtype="float32")
        v[h] = 1.0
        return v

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        return np.asarray([self._vec(t) for t in texts], dtype="float32")


class _DenseModel:
    """Deterministic DENSE unit-norm 384-vec (full-precision floats) — mimics
    real bge serialization size, so a full chunk exercises the reply cap (the
    one-hot fake masked it; Codex r1 P1)."""
    def _vec(self, text):
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        v = np.random.RandomState(seed).randn(384).astype("float32")
        return v / np.linalg.norm(v)

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        return np.asarray([self._vec(t) for t in texts], dtype="float32")


def _serve(sock, model=None):
    model = model if model is not None else _FakeModel()
    svc = es.EmbedService(sock, model_loader=lambda: model)
    threading.Thread(target=svc.serve_forever, daemon=True).start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.settimeout(0.2); c.connect(str(sock)); c.close()
            return svc, model
        except OSError:
            time.sleep(0.01)
    raise RuntimeError("service did not start")


def test_encode_texts_roundtrip_matches_model(tmp_sock):
    svc, model = _serve(tmp_sock)
    try:
        texts = ["alpha", "beta concept", "gamma-token", "delta"]
        out = ec.encode_texts(texts, path=tmp_sock)
        assert out is not None and out.shape == (4, 384)
        assert np.allclose(out, model.encode(texts))   # score-identical + in order
    finally:
        svc.shutdown()


def test_encode_texts_chunks_preserve_order(tmp_sock):
    svc, model = _serve(tmp_sock)
    try:
        texts = [f"concept number {i}" for i in range(ec._BATCH_CHUNK * 2 + 5)]
        out = ec.encode_texts(texts, path=tmp_sock)
        assert out is not None and out.shape == (len(texts), 384)
        assert np.allclose(out, model.encode(texts))   # order preserved across chunks
    finally:
        svc.shutdown()


def test_encode_texts_empty_returns_empty(tmp_sock):
    out = ec.encode_texts([], path=tmp_sock)
    assert out is not None and out.shape == (0, 384)


def test_encode_texts_service_down_returns_none(tmp_sock):
    assert ec.encode_texts(["x"], path=tmp_sock) is None   # nothing serving


def test_encode_batch_handler_validates_input():
    svc = es.EmbedService(Path("/tmp/eb-unused.sock"), model_loader=lambda: _FakeModel())
    assert svc.handle_request(b'{"op":"encode_batch","texts":[]}')["ok"] is False
    assert svc.handle_request(b'{"op":"encode_batch","texts":[1,2]}')["ok"] is False
    assert svc.handle_request(b'{"op":"encode_batch","texts":"nope"}')["ok"] is False
    good = svc.handle_request(b'{"op":"encode_batch","texts":["a","b"]}')
    assert good["ok"] is True and len(good["vecs"]) == 2 and len(good["vecs"][0]) == 384


def test_offproc_adapter_service_then_fallback_then_skip(monkeypatch):
    from dct.retrieval.relevance import _OffProcConceptEncoder, _ConceptEncodeUnavailable

    # service returns vectors -> used directly
    monkeypatch.setattr(ec, "encode_texts",
                        lambda texts: np.ones((len(texts), 384), dtype="float32"))
    assert _OffProcConceptEncoder(fallback=None).encode(["a", "b"]).shape == (2, 384)

    # service down + fallback present -> fallback used
    monkeypatch.setattr(ec, "encode_texts", lambda texts: None)
    fb = _FakeModel()
    assert np.allclose(_OffProcConceptEncoder(fallback=fb).encode(["a"]), fb.encode(["a"]))

    # service down + no fallback -> raise (caller skips filter)
    with pytest.raises(_ConceptEncodeUnavailable):
        _OffProcConceptEncoder(fallback=None).encode(["a"])


def test_dense_full_chunk_fits_reply_cap(tmp_sock):
    # Codex r1 P1: dense bge vecs are ~8KB JSON each; a full chunk must round-trip.
    model = _DenseModel()
    svc, _ = _serve(tmp_sock, model)
    try:
        texts = [f"dense concept text number {i}" for i in range(ec._BATCH_CHUNK)]
        out = ec.encode_texts(texts, path=tmp_sock)
        assert out is not None and out.shape == (ec._BATCH_CHUNK, 384)
        assert np.allclose(out, model.encode(texts), atol=1e-5)
        reply = json.dumps({"ok": True, "fingerprint": "x",
                            "vecs": model.encode(texts).tolist()}).encode()
        assert len(reply) < ec._MAX_REPLY_BYTES   # a full chunk fits with margin
    finally:
        svc.shutdown()


def test_encode_batch_rejects_oversized():
    svc = es.EmbedService(Path("/tmp/eb-unused2.sock"), model_loader=lambda: _FakeModel())
    big = ["x"] * (es._MAX_BATCH_ITEMS + 1)
    r = svc.handle_request(json.dumps({"op": "encode_batch", "texts": big}).encode())
    assert r["ok"] is False and r["error"] == "batch_too_large"


def test_encode_texts_non_str_returns_none(tmp_sock):
    svc, _ = _serve(tmp_sock)
    try:
        assert ec.encode_texts(["ok", object()], path=tmp_sock) is None  # no raise
    finally:
        svc.shutdown()
