"""VEC_NEAR embedding index — builds cosine-similarity edges between distillations.

Uses BAAI/bge-small-en-v1.5 (384-dim) via sentence-transformers.
Embeddings are computed at graph build time; invalidation is handled by
the mtime-keyed cache in service.py (new vault writes trigger a rebuild).

Public API:
    build_vec_near_edges(vault_root, threshold=0.70, top_k_pairs=500)
    -> list[tuple[str, str, int, str]]

    Each tuple: (concept_a, concept_b, weight, "vec_near")
    weight = max(1, int(round(cosine * 10))) — scaled to co-occurrence range.

Track C Claim 3: VEC_NEAR edges make the graph structurally heterogeneous,
distinguishing PDCT from HippoRAG/GraphRAG's purely co-occurrence graphs.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

_MODEL: Any = None
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_MODEL_LOCK = threading.Lock()
_WARM_THREAD: threading.Thread | None = None
_log = logging.getLogger(__name__)


def _get_model() -> Any:
    """Load the embedding model (singleton per process). BLOCKING — only
    warmers and offline/batch callers should use this; request-path code
    uses get_model_if_ready() instead.

    Construction is serialized by _MODEL_LOCK and validated with a probe
    encode before caching. Root cause 2026-07-16: torch 2.8 + s-t 5.1.2
    model construction is NOT thread-safe — concurrent constructions all
    fail with "Cannot copy out of meta tensor" (reproduced 6/6). The
    unlocked singleton let daemon warmer + request threads race, so the
    model never loaded and every PDCT cascade timed out for 24h+.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: .venv/bin/pip install sentence-transformers"
                ) from e
            # device="cpu": MPS backend deadlocks (_pthread_cond_wait on the
            # Metal GPU stream) on long multi-query runs — observed 2026-06-11.
            model = SentenceTransformer(_MODEL_NAME, device="cpu")
            # Validate BEFORE caching: a raced/partial construction fails
            # here (meta tensor) instead of poisoning the singleton.
            model.encode(["warmup"], normalize_embeddings=True,
                         show_progress_bar=False)
            _MODEL = model
        return _MODEL


def get_model_if_ready() -> Any | None:
    """Request-path accessor: return the warm model or None. NEVER
    constructs, never blocks. On a miss, kicks a background warm so the
    model becomes available for subsequent turns without any request
    paying the ~6s construction cost inside the 3s cascade budget."""
    if _MODEL is not None:
        return _MODEL
    ensure_warm_async()
    return None


_WARM_SPAWN_LOCK = threading.Lock()
_WARM_FAIL_TS = 0.0
_WARM_FAIL_COOLDOWN_S = 60.0


def ensure_warm_async() -> None:
    """Debounced background warm — at most one loader thread alive (spawn
    check is locked, Codex P2), with a cooldown after a failed warm so a
    request burst can't serially retry expensive construction."""
    global _WARM_THREAD, _WARM_FAIL_TS
    if _MODEL is not None:
        return
    import time as _time
    with _WARM_SPAWN_LOCK:
        if _MODEL is not None:
            return
        if _WARM_THREAD is not None and _WARM_THREAD.is_alive():
            return
        if _time.time() - _WARM_FAIL_TS < _WARM_FAIL_COOLDOWN_S:
            return
        def _warm():
            global _WARM_FAIL_TS
            try:
                _get_model()
                _log.info("[vec_index] background model warm complete")
            except Exception as e:  # noqa: BLE001
                _WARM_FAIL_TS = _time.time()
                _log.warning("[vec_index] background model warm failed: %s", e)
        t = threading.Thread(target=_warm, name="vec-index-warm", daemon=True)
        _WARM_THREAD = t
        t.start()


def reset_model() -> None:
    """Clear a poisoned singleton (e.g. meta-tensor encode failure) so the
    next warm constructs fresh. Safe to call from any thread."""
    global _MODEL
    with _MODEL_LOCK:
        _MODEL = None


# Incremental VEC_NEAR embedding cache: path -> ((mtime_ns, size),
# primary_concept|None, vector|None). ~1,900 files × 384 float32 ≈ 3MB.
# Guarded by a lock because rebuilds now run in background threads
# (service stale-while-revalidate).
_VEC_EDGE_CACHE: dict[str, tuple[tuple, str | None, Any]] = {}
_VEC_EDGE_CACHE_LOCK = threading.Lock()


def _parse_distillation(path: Path) -> tuple[list[str], str]:
    """Extract (concepts, body_text) from a distillation markdown file.

    Handles both inline (concepts: [a, b]) and multi-line YAML list format.
    Returns ([], "") on parse failure — caller skips files with no concepts.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return [], ""

    concepts: list[str] = []
    body = text

    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            frontmatter = text[3:end]
            body = text[end + 3:].strip()
            lines = frontmatter.splitlines()
            for j, line in enumerate(lines):
                if line.strip().startswith("concepts:"):
                    inline = line.split("concepts:", 1)[1].strip()
                    normalized = inline.replace(" ", "")
                    if normalized and normalized != "[]":
                        # Inline: concepts: [a, b, c]
                        raw = re.sub(r"[\[\]]", "", inline)
                        concepts = [c.strip() for c in raw.split(",") if c.strip()]
                    else:
                        # Multi-line: subsequent indented "  - item" lines
                        for k in range(j + 1, min(j + 50, len(lines))):
                            ml = lines[k]
                            if (ml.startswith(" ") or ml.startswith("\t")) and ml.strip().startswith("- "):
                                concepts.append(ml.strip()[2:].strip())
                            elif ml and not (ml.startswith(" ") or ml.startswith("\t")):
                                break
                    break

    return concepts, body


def build_vec_near_edges(
    vault_root: Path,
    *,
    threshold: float = 0.70,
    top_k_pairs: int = 500,
) -> list[tuple[str, str, int, str]]:
    """Build VEC_NEAR edges from distillation embeddings.

    For each pair of distillations whose cosine similarity exceeds `threshold`,
    emit a VEC_NEAR edge connecting their primary concept (first concept slug).
    Weight = max(1, int(round(cosine * 10))) to match co-occurrence scale.

    Args:
        vault_root: directory containing *.md distillation files.
        threshold: cosine similarity threshold (0.0–1.0). Default 0.70.
        top_k_pairs: max edges returned (sorted by similarity desc). Default 500.

    Returns list of (concept_a, concept_b, weight, "vec_near") tuples.
    Returns [] if < 2 distillations have concepts, or on model load failure.
    """
    try:
        import numpy as np
    except ImportError:
        return []  # numpy not available — fail open

    # Incremental (2026-07-16): per-file (mtime → primary, vector) cache.
    # Previously every rebuild re-read AND re-embedded all ~1,900 vault
    # files (tens of seconds); now only new/changed files are parsed and
    # embedded. Files with no concepts cache a tombstone so they aren't
    # re-read every pass. Deleted files drop out via the seen-set sweep.
    with _VEC_EDGE_CACHE_LOCK:
        cache = _VEC_EDGE_CACHE
        seen: set[str] = set()
        to_embed: list[tuple[str, tuple, str, str]] = []  # (path, stamp, primary, text)
        for f in vault_root.rglob("*.md"):
            try:
                fst = f.stat()
            except OSError:
                continue
            # (mtime_ns, size): float st_mtime can't distinguish writes
            # within one timestamp tick on coarse filesystems (Codex P2).
            mt = (fst.st_mtime_ns, fst.st_size)
            p = str(f)
            seen.add(p)
            hit = cache.get(p)
            if hit is not None and hit[0] == mt:
                continue  # unchanged — reuse cached primary/vector (or tombstone)
            concepts, body = _parse_distillation(f)
            if not concepts:
                cache[p] = (mt, None, None)  # tombstone: parsed, nothing to embed
                continue
            primary = concepts[0]
            # Embed text: concept slugs joined + first 512 chars of body
            embed_text = " ".join(concepts) + " " + body[:512]
            to_embed.append((p, mt, primary, embed_text))

        if to_embed:
            try:
                model = _get_model()
                new_vecs = model.encode(
                    [t for _, _, _, t in to_embed],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception:
                # fail open — VEC_NEAR is additive; serve whatever the cache
                # already holds rather than nothing.
                new_vecs = None
            if new_vecs is not None:
                for (p, mt, primary, _), v in zip(to_embed, new_vecs):
                    cache[p] = (mt, primary, np.asarray(v, dtype=np.float32))
        # Sweep deletions and materialize the live set.
        for p in list(cache.keys()):
            if p not in seen:
                del cache[p]
        live = [(prim, vec) for (_, prim, vec) in cache.values()
                if prim is not None and vec is not None]

    if len(live) < 2:
        return []
    items = [(prim, None) for prim, _ in live]
    vecs = np.stack([vec for _, vec in live]).astype(np.float32)

    # Cosine similarity matrix (vecs are L2-normalised → dot product = cosine)
    sims = vecs @ vecs.T  # shape (N, N)
    N = len(items)

    edges: list[tuple[str, str, int, str]] = []
    for i in range(N):
        for j in range(i + 1, N):
            cos = float(sims[i, j])
            if cos >= threshold:
                a = items[i][0]
                b = items[j][0]
                if a == b:
                    continue  # same concept slug (different files) — skip self-loop
                weight = max(1, int(round(cos * 10)))
                edges.append((a, b, weight, "vec_near"))

    # Sort by weight desc, cap at top_k_pairs
    edges.sort(key=lambda e: -e[2])
    return edges[:top_k_pairs]
