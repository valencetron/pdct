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

import re
from pathlib import Path
from typing import Any

_MODEL: Any = None
_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _get_model() -> Any:
    """Lazy-load the embedding model (singleton per process)."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            # device="cpu": MPS backend deadlocks (_pthread_cond_wait on the
            # Metal GPU stream) on long multi-query runs — observed 2026-06-11.
            _MODEL = SentenceTransformer(_MODEL_NAME, device="cpu")
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: .venv/bin/pip install sentence-transformers"
            ) from e
    return _MODEL


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

    # Collect distillations with at least one concept
    items: list[tuple[str, str]] = []  # (primary_concept, embed_text)
    for f in vault_root.rglob("*.md"):
        concepts, body = _parse_distillation(f)
        if not concepts:
            continue
        primary = concepts[0]
        # Embed text: concept slugs joined + first 512 chars of body
        embed_text = " ".join(concepts) + " " + body[:512]
        items.append((primary, embed_text))

    if len(items) < 2:
        return []

    try:
        model = _get_model()
        texts = [t for _, t in items]
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vecs = np.array(vecs, dtype=np.float32)
    except Exception:
        return []  # fail open — VEC_NEAR is additive, not required for correctness

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
