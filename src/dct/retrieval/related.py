"""Graph-cascade related distillations.

For a given distillation id, run a cascade from its concept set, then
aggregate hits across other distillations via shared concepts and rank.
"""
from __future__ import annotations

from dataclasses import dataclass

from dct.heat import ConceptGraph
from dct.retrieval.cascade import cascade
from dct.retrieval.distill_index import DistillationRef, build_index
from dct.retrieval.types import RetrievalConfig


@dataclass(frozen=True)
class RelatedRef:
    id: str
    title: str
    score: float


def _load_graph() -> ConceptGraph:
    # Lazy import to avoid pulling event-log on every test
    from dct.retrieval.service import _load_or_build_graph
    return _load_or_build_graph()


def _default_config() -> RetrievalConfig:
    from dct.retrieval.service import build_config
    return build_config()


def related_distillations(
    id: str,
    *,
    k: int = 3,
    index: dict[str, DistillationRef] | None = None,
    config: RetrievalConfig | None = None,
) -> list[RelatedRef]:
    idx = index if index is not None else build_index()
    self_ref = idx.get(id)
    if self_ref is None or not self_ref.concepts:
        return []

    graph = _load_graph()
    cfg = config or _default_config()
    hits = cascade(
        seed_concepts=self_ref.concepts,
        graph=graph,
        heat={},
        config=cfg,
        current_context=set(),
    )
    if not hits:
        return []

    # Map concept -> max cascade score
    concept_score = {h.concept: h.score for h in hits}

    # Aggregate per-distillation: max score across its concepts
    scored: list[tuple[float, DistillationRef]] = []
    for other_id, ref in idx.items():
        if other_id == id or not ref.concepts:
            continue
        best = max((concept_score.get(c, 0.0) for c in ref.concepts), default=0.0)
        if best > 0.0:
            scored.append((best, ref))

    scored.sort(key=lambda sr: (-sr[0], sr[1].id))
    return [RelatedRef(id=r.id, title=r.title, score=s) for s, r in scored[:k]]
