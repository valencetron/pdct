"""Cascade retrieval — 2-hop traversal with decay."""
from __future__ import annotations
import time
from typing import Any

from .types import ConceptHit, RetrievalConfig


def _build_adj(graph: Any) -> tuple[dict[str, dict[str, int]], int]:
    """Build adjacency map + max edge weight from a ConceptGraph.

    Cuts cascade from O(|seeds| × hops × |E|) to O(|frontier| × avg_deg).
    With ~33k edges and ~600 frontier nodes this is a 200-300x speedup.
    Built once per cascade() call (no cross-call cache; ~10ms for 33k edges).

    Track C (Codex r2 P0 fix): also merges VEC_NEAR edges from typed_edges
    into the adjacency map so they participate in traversal. Without this,
    VEC_NEAR edges would exist in metadata but never be walked.
    typed_edges uses max(existing_w, vec_w) to avoid overwriting stronger
    CO_OCCUR edges with weaker embedding similarities.
    """
    adj: dict[str, dict[str, int]] = {}
    max_w = 0
    # Undirected CO_OCCUR edges (always present)
    for a, b, w in graph.edges:
        adj.setdefault(a, {})[b] = w
        adj.setdefault(b, {})[a] = w
        if w > max_w:
            max_w = w
    # VEC_NEAR edges from typed_edges (Track C Claim 3).
    # Merge into adj: if edge already exists (CO_OCCUR), keep the max weight.
    for te in (getattr(graph, "typed_edges", None) or []):
        a_te, b_te, w_te, etype = te
        if etype != "vec_near":
            continue
        existing_ab = adj.get(a_te, {}).get(b_te, 0)
        existing_ba = adj.get(b_te, {}).get(a_te, 0)
        merged_w = max(w_te, existing_ab, existing_ba)
        adj.setdefault(a_te, {})[b_te] = merged_w
        adj.setdefault(b_te, {})[a_te] = merged_w
        if merged_w > max_w:
            max_w = merged_w
    return adj, max_w


def _neighbors_of(graph: Any, concept: str) -> dict[str, int]:
    """Return {neighbor_concept: edge_weight} for undirected traversal.

    Builds the adjacency map fresh each call. Used by callers outside cascade()
    that don't cache adj themselves. Inside cascade() we use the local `adj`
    directly to avoid recomputation per concept.
    """
    adj, _ = _build_adj(graph)
    return adj.get(concept, {})


def _normalize(weight: int, max_weight: int) -> float:
    return weight / max_weight if max_weight > 0 else 0.0


def cascade(
    seed_concepts: list[str],
    graph: Any,
    heat: dict[str, float],
    config: RetrievalConfig,
    current_context: set[str] | None = None,
) -> list[ConceptHit]:
    """Traverse the graph from seed concepts, returning ranked ConceptHits.

    hop 0: seed concepts (score=1.0).
    hop 1: direct neighbors, score = edge_weight / max_edge_weight.
    hop 2+: multiply by config.cascade_decay ** (hop - 1), compound with parent score.

    Budget: config.cascade_budget_ms wall time (checked between hops).
    Dedup: concepts in current_context are excluded.
    heat: currently unused at hop 0/1; reserved for future tie-breaking.
    """
    skip = set(current_context or ())
    hits: dict[str, ConceptHit] = {}
    adj, max_w = _build_adj(graph)

    # Track C — Directed transition biasing (Claim 2b).
    # transitions[(src, nb)] / max_trans gives a normalized bias multiplier.
    transitions: dict[tuple[str, str], int] = getattr(graph, "transitions", {}) or {}
    max_trans: int = max(transitions.values(), default=1)

    # Track C — Per-edge-type decay (Claim 3 — heterogeneous graph).
    # Build lookup: (a, b) → edge_type_value for quick lookup during hop.
    typed_edge_map: dict[tuple[str, str], str] = {}
    for te in (getattr(graph, "typed_edges", None) or []):
        a_te, b_te, _w_te, etype = te
        typed_edge_map[(a_te, b_te)] = etype
        typed_edge_map[(b_te, a_te)] = etype  # undirected lookup

    # Seeds are immutable: hop-0 with score 1.0. They are NOT replaced even
    # if a later hop finds a higher-scoring path back to them — the seed is
    # the user's intent, not a derived hit.
    seed_set: set[str] = set()
    frontier: dict[str, tuple[float, list[str]]] = {}
    for s in seed_concepts:
        if s in skip:
            continue
        if s in hits:
            continue  # duplicate seed list — first wins
        hits[s] = ConceptHit(
            concept=s, source_slug="seed", snippet="",
            score=1.0, hop=0, path=[s],
        )
        seed_set.add(s)
        frontier[s] = (1.0, [s])

    start = time.monotonic()
    deadline = start + (config.cascade_budget_ms / 1000.0)

    # Best-score-globally attribution (R3 fix): a non-seed concept is replaced
    # whenever a later traversal step finds a higher-scoring path to it. This
    # gives credit-assignment to the strongest trajectory rather than the
    # shallowest one. Seeds are never replaced (handled above).
    # Action gate: drop genuine action/verb neighbors (stoplist) before they
    # consume a slot — the slot-dilution fix. Single-token *frequency*
    # eligibility is deliberately NOT handled here; that stays with
    # service._filter_by_eligibility (which owns the env toggle + telemetry).
    from dct.classify import ACTION_STOPLIST
    for hop in range(1, config.cascade_depth + 1):
        if time.monotonic() >= deadline:
            break
        next_frontier: dict[str, tuple[float, list[str]]] = {}
        for src, (parent_score, parent_path) in frontier.items():
            for nb, w in adj.get(src, {}).items():
                if nb in skip or nb in seed_set:
                    continue
                # Action gate — drop verb/imperative neighbors (stoplist only,
                # so toy/low-freq concept nodes and the eligibility filter's
                # frequency logic are untouched). Seeds are exempt (above).
                if nb.strip().lower() in ACTION_STOPLIST:
                    continue
                edge_score = _normalize(w, max_w)

                # Per-type decay (Track C Claim 3): VEC_NEAR uses cascade_vec_near_decay.
                etype = typed_edge_map.get((src, nb), "co_occur")
                if (etype == "vec_near"
                        and getattr(config, "cascade_vec_near_enabled", True)
                        and hop > 1):
                    decay_rate = getattr(config, "cascade_vec_near_decay", 0.2)
                else:
                    decay_rate = config.cascade_decay

                hop_score = (
                    edge_score
                    * (decay_rate ** (hop - 1))
                    * (parent_score if hop > 1 else 1.0)
                )

                # Directed transition biasing (Track C Claim 2b).
                # Multiply score by (1 + bias * normalized_transition_count).
                if (config.cascade_transitions_enabled and transitions):
                    t_count = transitions.get((src, nb), 0)
                    if t_count > 0:
                        bias_mult = 1.0 + config.cascade_transitions_bias * (t_count / max_trans)
                        hop_score *= bias_mult

                # Don't bother extending if it can't beat the current best.
                existing_hit = hits.get(nb)
                if existing_hit is not None and hop_score <= existing_hit.score:
                    continue
                cand_path = parent_path + [nb]
                existing = next_frontier.get(nb)
                if existing is None or hop_score > existing[0]:
                    next_frontier[nb] = (hop_score, cand_path)
        # Promote next_frontier into hits, replacing weaker existing entries.
        for nb, (score, path) in next_frontier.items():
            existing_hit = hits.get(nb)
            if existing_hit is None or score > existing_hit.score:
                hits[nb] = ConceptHit(
                    concept=nb, source_slug=f"hop-{hop}", snippet="",
                    score=score, hop=hop, path=path,
                )
        frontier = next_frontier
        if not frontier:
            break

    return sorted(hits.values(), key=lambda h: h.score, reverse=True)
