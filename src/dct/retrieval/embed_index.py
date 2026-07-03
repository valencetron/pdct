"""Embedding index over distillations for semantic retrieval.

Embeds each distillation's searchable text (title + gist + concepts + body
head) with BAAI/bge-small-en-v1.5 (384-dim) and caches vectors to disk,
keyed per-file by (path, mtime). Incremental: only new/changed files are
re-embedded on rebuild.

Public API:
    semantic_scores(query_text, index) -> dict[id, float]   # cosine 0..1

Used by memory_api._aggregate as an additional ranking channel — keyword/
concept overlap has a hard ceiling on vocabulary-mismatch queries; dense
similarity covers the gap.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from dct.retrieval.distill_index import DistillationRef

_CACHE_DIR = Path(__file__).resolve().parents[3] / "runtime"
_VEC_PATH = _CACHE_DIR / "distill-embeddings.npz"
_META_PATH = _CACHE_DIR / "distill-embeddings.meta.json"
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_BODY_HEAD_CHARS = 1200
# bge models want this prefix on the query side only.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_MODEL: Any = None
_MEM_CACHE: dict[str, Any] = {}  # {"ids": [...], "vecs": ndarray, "stamp": {...}}


def _get_model() -> Any:
    # Share the singleton with vec_index — both load BAAI/bge-small-en-v1.5.
    # Previously each module held its own copy (2x RAM, 2x ~3s load).
    global _MODEL
    if _MODEL is None:
        from dct.retrieval.vec_index import _get_model as _shared
        _MODEL = _shared()
    return _MODEL


def _doc_text(ref: DistillationRef) -> str:
    parts = [ref.title, ref.gist, " ".join(c.replace("-", " ") for c in ref.concepts)]
    try:
        raw = ref.path.read_text(encoding="utf-8", errors="ignore")
        if raw.startswith("---"):
            end = raw.find("---", 3)
            if end != -1:
                raw = raw[end + 3:]
        parts.append(raw.strip()[:_BODY_HEAD_CHARS])
    except OSError:
        pass
    return "\n".join(p for p in parts if p)


def _stamp(index: dict[str, DistillationRef]) -> dict[str, float]:
    out = {}
    for ref in index.values():
        try:
            out[ref.id] = ref.path.stat().st_mtime
        except OSError:
            out[ref.id] = 0.0
    return out


def _load_disk() -> tuple[list[str], np.ndarray, dict[str, float]]:
    if not (_VEC_PATH.exists() and _META_PATH.exists()):
        return [], np.zeros((0, 384), dtype=np.float32), {}
    try:
        meta = json.loads(_META_PATH.read_text())
        data = np.load(_VEC_PATH)
        # SIGBUS fix (2026-06-11): Accelerate cblas_sgemv crashed with
        # EXC_ARM_DA_ALIGN on views into the npz-backed buffer under
        # sustained inference. Force a fresh aligned contiguous array.
        vecs = np.ascontiguousarray(data["vecs"], dtype=np.float32)
        ids = list(meta["ids"])
        if len(ids) != vecs.shape[0] or meta.get("model") != _MODEL_NAME:
            return [], np.zeros((0, 384), dtype=np.float32), {}
        return ids, vecs, dict(meta["mtimes"])
    except Exception:
        return [], np.zeros((0, 384), dtype=np.float32), {}


def _save_disk(ids: list[str], vecs: np.ndarray, mtimes: dict[str, float]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_VEC_PATH, vecs=vecs.astype(np.float32))
    _META_PATH.write_text(json.dumps(
        {"ids": ids, "mtimes": mtimes, "model": _MODEL_NAME, "built_at": time.time()}
    ))


def _ensure_index(index: dict[str, DistillationRef]) -> tuple[list[str], np.ndarray]:
    """Return (ids, unit-normalized vectors) covering `index`, rebuilding stale rows."""
    stamp = _stamp(index)

    mem = _MEM_CACHE
    if mem.get("stamp") == stamp:
        return mem["ids"], mem["vecs"]

    ids, vecs, mtimes = _load_disk()
    pos = {i: n for n, i in enumerate(ids)}

    stale = [
        rid for rid, mt in stamp.items()
        if rid not in pos or mtimes.get(rid) != mt
    ]
    if stale:
        model = _get_model()
        texts = [_doc_text(index[rid]) for rid in stale]
        new_vecs = model.encode(texts, normalize_embeddings=True,
                                batch_size=32, show_progress_bar=False)
        new_vecs = np.asarray(new_vecs, dtype=np.float32)
        rows = {rid: new_vecs[i] for i, rid in enumerate(stale)}
        # Rebuild arrays in stamp order (also drops deleted files).
        out_ids = list(stamp.keys())
        out = np.zeros((len(out_ids), new_vecs.shape[1] if len(new_vecs) else 384),
                       dtype=np.float32)
        for n, rid in enumerate(out_ids):
            if rid in rows:
                out[n] = rows[rid]
            elif rid in pos:
                out[n] = vecs[pos[rid]]
        ids, vecs = out_ids, out
        _save_disk(ids, vecs, stamp)
    else:
        # Disk may contain deleted entries; filter to current stamp.
        keep = [n for n, i in enumerate(ids) if i in stamp]
        if len(keep) != len(ids):
            ids = [ids[n] for n in keep]
            vecs = vecs[keep]

    _MEM_CACHE.update({"ids": ids, "vecs": vecs, "stamp": stamp})
    return ids, vecs


def semantic_scores(
    query_text: str,
    index: dict[str, DistillationRef],
) -> dict[str, float]:
    """Cosine similarity of the query against every distillation. 0..1 per id.

    Returns {} on any failure (missing model, no files) — callers treat
    semantic scoring as an optional channel.
    """
    if not query_text.strip():
        return {}
    try:
        ids, vecs = _ensure_index(index)
        if not ids:
            return {}
        model = _get_model()
        q = model.encode([_QUERY_PREFIX + query_text], normalize_embeddings=True,
                         show_progress_bar=False)
        sims = vecs @ np.asarray(q[0], dtype=np.float32)
        # bge cosines live ~0.3..0.9; clamp to 0..1 without rescaling so
        # thresholds in callers stay interpretable.
        return {rid: float(max(0.0, min(1.0, s))) for rid, s in zip(ids, sims)}
    except Exception:
        return {}
