"""Content-addressed concept-embedding cache (WS1 Phase 1, 2026-07-21).

The cosine relevance filter re-encoded every concept text every turn — 3-18s
under daemon contention, which forced the filter into breaker-exile. Concept
texts (slug + snippet) are STABLE, so we cache their embeddings keyed by
sha256(text) and only encode the fresh query per turn. Disk-persisted so a
process restart doesn't cold-rebuild. 384-dim float32, L2-normalized (the
filter uses dot product = cosine).

Writer model (Codex diff r1 P0/P1): the daemon's cosine filter is READ-ONLY —
it fills its in-memory cache on misses but NEVER writes to disk (a per-turn
NPZ rewrite raced the warmer across processes and put 44MB of I/O on the turn
path). The nightly warmer is the SOLE disk writer: it calls rebuild() with the
current graph's concepts, so removed/renamed concepts are actually pruned from
disk (not merely orphaned). Persistence uses a pid-unique temp file + fsync +
atomic replace, so overlapping writers can't expose a partial .npz.

Correctness note: the key is the FULL concept text (slug + snippet[:200]).
If a concept's text changes, its hash changes → the new text is a cache MISS
and gets re-encoded — stale vectors are never served for changed content.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_log = logging.getLogger(__name__)
_DIM = 384

# Model fingerprint stamped into the store. If the embedding model changes,
# a stale store's vectors would silently score garbage (Codex r1 #7) — a
# fingerprint mismatch on load rejects the whole store so it rebuilds clean.
MODEL_FINGERPRINT = "bge-small-en-v1.5/384/v1"


def text_key(text: str) -> str:
    """Content hash of a concept text. Stable across processes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def concept_text(concept: str, snippet: str = "") -> str:
    """THE canonical concept-text builder — the single source of truth shared
    by the cosine filter and the warmer (Codex r1 #12: divergent text = cache
    identity drift). Mirrors the historical filter behavior: slug words, plus
    up to 200 chars of snippet when present.
    """
    slug_words = (concept or "").replace("-", " ")
    snip = (snippet or "").strip()
    return slug_words if not snip else f"{slug_words} {snip[:200]}"


class ConceptEmbeddingStore:
    """Thread-safe content-addressed embedding cache with atomic persistence."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._cache: dict[str, np.ndarray] = {}
        self._dirty = False
        self._gen = 0  # bumps on EVERY mutation; save() clears dirty only if
                       # unchanged since its snapshot (Codex diff r2: a same-
                       # length rebuild defeated the old len-guard)
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                keys = data["keys"]
                vecs = data["vecs"]
                fp = str(data["fingerprint"]) if "fingerprint" in data else ""
            # Reject a store built by a different embedding model (Codex r1 #7):
            # its vectors are incompatible and would score garbage.
            if fp != MODEL_FINGERPRINT:
                raise ValueError(
                    f"fingerprint mismatch: {fp!r} != {MODEL_FINGERPRINT!r}")
            if len(keys) != len(vecs):
                raise ValueError("keys/vecs length mismatch")
            if vecs.ndim != 2 or vecs.shape[1] != _DIM:
                raise ValueError(f"bad vecs shape {vecs.shape}")
            if not np.isfinite(vecs).all():
                raise ValueError("non-finite vectors")
            self._cache = {str(k): np.asarray(vecs[i], dtype="float32")
                           for i, k in enumerate(keys)}
        except Exception as e:  # corrupt/partial/stale → start empty, never crash
            _log.warning("[concept-emb] load rejected (%s) — starting empty", e)
            self._cache = {}

    def get_many(self, texts: Sequence[str], *, model: Any) -> np.ndarray:
        """Return (len(texts), 384) normalized vectors, encoding only misses.

        Row i corresponds to texts[i]. Duplicate texts within a call are
        encoded once. Encoding runs OUTSIDE the lock (it can be slow); the
        lock only guards the dict.
        """
        if not texts:
            return np.zeros((0, _DIM), dtype="float32")
        keys = [text_key(t) for t in texts]

        # Unique misses (dedup within the call so a repeated text encodes once).
        with self._lock:
            missing_keys = {k for k in keys if k not in self._cache}
        if missing_keys:
            # Map each missing key to one representative text.
            rep: dict[str, str] = {}
            for t, k in zip(texts, keys):
                if k in missing_keys and k not in rep:
                    rep[k] = t
            miss_keys_ordered = list(rep.keys())
            miss_texts = [rep[k] for k in miss_keys_ordered]
            fresh = np.asarray(
                model.encode(miss_texts, normalize_embeddings=True,
                             show_progress_bar=False),
                dtype="float32",
            )
            # Uphold the documented (N, 384) contract (Codex diff r1 P2): a
            # malformed/changed model must NOT poison the cache. Raising here
            # lets the filter's except-block fail open cleanly instead.
            if fresh.shape != (len(miss_keys_ordered), _DIM) or not np.isfinite(fresh).all():
                raise ValueError(
                    f"model.encode returned bad shape/values {fresh.shape}")
            with self._lock:
                for j, k in enumerate(miss_keys_ordered):
                    self._cache[k] = fresh[j]
                self._dirty = True
                self._gen += 1

        with self._lock:
            return np.array([self._cache[k] for k in keys], dtype="float32")

    def missing_texts(self, texts: Sequence[str]) -> list[str]:
        """Return the subset of `texts` NOT already cached — cache-only, no
        model touched. Lets the cosine filter decide whether it needs a local
        model at all (WS1 Phase 2, Codex #2): if the query is encoded off-proc
        AND every concept is cached, no local model is loaded this turn.
        Deduped, order-preserving on first occurrence.
        """
        keys = [text_key(t) for t in texts]
        seen: set[str] = set()
        out: list[str] = []
        with self._lock:
            for t, k in zip(texts, keys):
                if k not in self._cache and k not in seen:
                    seen.add(k)
                    out.append(t)
        return out

    def save(self) -> None:
        """Atomically persist if dirty. Best-effort; never raises.

        SOLE intended caller is the warmer (single writer). pid-unique temp +
        fsync + atomic replace so an overlapping writer can't expose a partial
        file (Codex diff r1 P0). The dirty flag is cleared only when the exact
        snapshot we wrote is still current, so a concurrent add isn't marked
        clean-but-unpersisted.
        """
        with self._lock:
            if not self._dirty:
                return
            keys = list(self._cache.keys())
            snapshot_gen = self._gen
            vecs = (np.array([self._cache[k] for k in keys], dtype="float32")
                    if keys else np.zeros((0, _DIM), dtype="float32"))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # pid+id-unique tmp, MUST end in .npz (np.savez appends .npz else).
            tmp = self.path.with_name(
                f"{self.path.stem}.tmp.{os.getpid()}.{id(vecs) & 0xffffff}.npz")
            np.savez(tmp, keys=np.array(keys), vecs=vecs,
                     fingerprint=np.array(MODEL_FINGERPRINT))
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
            with self._lock:
                # Clear dirty only if NO mutation landed since the snapshot
                # (Codex diff r2: a same-length rebuild swaps contents while
                # keeping length, so compare a generation counter not len).
                if self._gen == snapshot_gen:
                    self._dirty = False
        except Exception as e:
            _log.warning("[concept-emb] save failed: %s", e)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def vectors_for(self, texts: Sequence[str], *, model: Any) -> dict[str, np.ndarray]:
        """Return {text_key: vector} for exactly these texts, encoding misses.
        Reuses cached vectors; does not prune. The warmer assembles the
        authoritative current-concept set with this, then calls rebuild()."""
        self.get_many(texts, model=model)  # ensure all present in _cache
        with self._lock:
            return {text_key(t): self._cache[text_key(t)] for t in texts}

    def rebuild(self, keyed_vectors: dict[str, np.ndarray]) -> None:
        """Replace the entire cache with exactly `keyed_vectors` and persist.

        The warmer's authoritative write: removed/renamed concepts are pruned
        (not orphaned), making the age-out property real (Codex diff r1 P1).
        """
        with self._lock:
            self._cache = dict(keyed_vectors)
            self._dirty = True
            self._gen += 1
        self.save()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


_DEFAULT_STORE: "ConceptEmbeddingStore | None" = None
_DEFAULT_LOCK = threading.Lock()


def default_store() -> "ConceptEmbeddingStore":
    """Process-wide singleton at data/concept-embeddings.npz."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is not None:
        return _DEFAULT_STORE
    with _DEFAULT_LOCK:
        if _DEFAULT_STORE is None:
            root = Path(__file__).resolve().parents[3]  # repo root
            _DEFAULT_STORE = ConceptEmbeddingStore(
                root / "data" / "concept-embeddings.npz")
    return _DEFAULT_STORE
