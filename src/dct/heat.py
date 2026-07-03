"""Point-in-time heat snapshot + co-occurrence concept graph.

Server side of the Mission Control DCT heat visualization. Wraps
``ActivationEngine.snapshot`` with a time-bounded replay and a one-pass
co-occurrence graph builder so the browser can render nodes + edges
without parsing events.jsonl itself.

CLI:
    python -m dct.heat --log events.jsonl --mode graph
    python -m dct.heat --log events.jsonl --mode heat  --ts 1776668000 [--half-life 21600] [--hop-cap 0]
    python -m dct.heat --log events.jsonl --mode range
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dct.activation import ActivationEngine, DecayConfig, Graph
from dct.event_log import EventLog
from dct.events import EventOp


DEFAULT_HALF_LIFE_SECS = 21600.0   # 6 hours
DEFAULT_MIN_HEAT = 0.001


from enum import Enum

class EdgeType(str, Enum):
    """Edge type in the heterogeneous concept graph (Track C Claim 3).

    CO_OCCUR: undirected co-occurrence — concepts appeared in the same event.
    VEC_NEAR: embedding-similarity edge — cosine > threshold between
              distillation vectors (bge-small-en-v1.5, 384-dim).
    """
    CO_OCCUR = "co_occur"
    VEC_NEAR = "vec_near"


@dataclass(frozen=True)
class ConceptGraph:
    """Co-occurrence graph over all concepts ever observed in the log.

    ``nodes`` maps slug → total event-occurrence count (degree-in-events).
    ``edges`` is a list of ``(source, target, weight)`` with source < target
    lexicographically, where weight is the number of events containing both.
    ``transitions`` maps (from_concept, to_concept) → directed count. Populated
    from consecutive concept pairs in op=traversal events. The order reflects the
    actual cascade trajectory — transitions[(a,b)] ≠ transitions[(b,a)] in general.
    This asymmetry is the paper's Claim 2b (path-dependent retrieval).
    ``typed_edges`` tags each edge with an EdgeType value. CO_OCCUR = undirected
    co-occurrence (built here). VEC_NEAR = embedding-similarity (added by vec_index).
    """
    nodes: dict[str, int]
    edges: list[tuple[str, str, int]]
    transitions: dict[tuple[str, str], int] = field(default_factory=dict)
    typed_edges: list[tuple[str, str, int, str]] = field(default_factory=list)
    # typed_edges: (source, target, weight, edge_type_value)
    # CO_OCCUR edges: source < target lexicographically (matches edges list).
    # VEC_NEAR edges: added by build_vec_near_edges() in service.py; no
    # ordering constraint but (a,b) and (b,a) are not both emitted.

    def kind(self, slug: str) -> str:
        """Classify a node as 'concept' or 'action' on demand.

        Computed from self.nodes (slug -> occurrence count); no stored field,
        so it stays correct as the graph rebuilds. See dct.classify.
        """
        from dct.classify import classify_node
        return classify_node(slug, self.nodes.get(slug, 0))


def build_concept_graph(
    log: EventLog,
    *,
    topic_id: str | None = None,
    ignore_feedback: bool = False,
) -> ConceptGraph:
    """Build concept co-occurrence graph from observed events.

    Skips op=traversal and op=turn (derived/pre-distillation events).
    op=feedback events apply per-edge multipliers from metadata['multipliers']
    over metadata['path'] — Track B online edge-weight learning.

    Args:
        log: events log to read.
        topic_id: if not None, FEEDBACK events whose metadata.thread_id != topic_id
            are skipped. Co-occurrence (READ/WRITE) is always global.
        ignore_feedback: if True, ALL feedback events are skipped at read time.
            Used for the "no online learning" ablation baseline (R3.1).
    """
    node_counts: Counter[str] = Counter()
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    transition_counts: dict[tuple[str, str], int] = {}  # directed (Track C Claim 2b)
    for ev in log.read_all():
        if ev.op == EventOp.TURN:
            continue
        # Directed transitions from traversal events (Track C Claim 2b).
        # Consecutive concept pairs in a cascade reflect the traversal order.
        if ev.op == EventOp.TRAVERSAL:
            seq = ev.concepts
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                if a != b:
                    key = (a, b)
                    transition_counts[key] = transition_counts.get(key, 0) + 1
            continue  # traversal events do NOT feed undirected co-occurrence or node counts
        if ev.op == EventOp.FEEDBACK:
            if ignore_feedback:
                continue
            ev_topic = (ev.metadata or {}).get("thread_id")
            if topic_id is not None and ev_topic != topic_id:
                continue
            path = (ev.metadata or {}).get("path") or []
            mults = (ev.metadata or {}).get("multipliers") or []
            if len(path) < 2 or len(mults) != len(path) - 1:
                continue  # malformed, skip
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                try:
                    raw = mults[i]
                    w = max(1, int(round(float(raw))))
                except (TypeError, ValueError):
                    w = 1
                edge_counts[key] += w
            continue  # do NOT bump node_counts (FEEDBACK is meta-signal)

        uniq = sorted(set(ev.concepts))
        for slug in uniq:
            node_counts[slug] += 1
        for a, b in itertools.combinations(uniq, 2):
            edge_counts[(a, b)] += 1
    edges = sorted((a, b, n) for (a, b), n in edge_counts.items())
    typed_edges = [(a, b, n, EdgeType.CO_OCCUR.value) for a, b, n in edges]
    return ConceptGraph(
        nodes=dict(node_counts),
        edges=edges,
        transitions=transition_counts,
        typed_edges=typed_edges,
    )


def _adjacency_from_concept_graph(cg: ConceptGraph) -> Graph:
    """Build an undirected adjacency map for ActivationEngine blast radius."""
    adj: dict[str, list[str]] = {n: [] for n in cg.nodes}
    for a, b, _ in cg.edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def compute_heat_at(
    log_path: Path | str,
    *,
    ts: float,
    half_life: float = DEFAULT_HALF_LIFE_SECS,
    hop_cap: int = 0,
    min_heat: float = DEFAULT_MIN_HEAT,
) -> dict[str, float]:
    """Replay events up to ``ts`` and return per-concept heat at that instant.

    Events strictly after ``ts`` are ignored so the browser's time-scrubber
    sees a causally-consistent past. With ``hop_cap > 0`` the co-occurrence
    graph is built from the same ≤ts slice and fed to the engine for blast
    radius propagation.
    """
    log = EventLog(Path(log_path))
    config = DecayConfig(half_life_seconds=half_life, radius_hop_cap=hop_cap)
    engine = ActivationEngine(config=config)

    # Time-bounded consume — log is ts-sorted after read_all() returns.
    past_events = []
    for ev in log.read_all():
        if ev.ts > ts:
            break
        past_events.append(ev)
        engine.consume(ev)

    if hop_cap > 0:
        cg = _build_graph_from_events(past_events)
        engine.set_graph(_adjacency_from_concept_graph(cg))

    return engine.snapshot(ts, min_heat=min_heat)


def _build_graph_from_events(events) -> ConceptGraph:
    node_counts: Counter[str] = Counter()
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    for ev in events:
        # R3.10: same skip rules as build_concept_graph for the in-process path.
        # FEEDBACK events are edge-only meta-signal; they must not contribute to
        # heat-time co-occurrence (compute_heat_at uses this helper).
        if ev.op in (EventOp.TRAVERSAL, EventOp.TURN, EventOp.FEEDBACK):
            continue
        uniq = sorted(set(ev.concepts))
        for slug in uniq:
            node_counts[slug] += 1
        for a, b in itertools.combinations(uniq, 2):
            edge_counts[(a, b)] += 1
    edges = sorted((a, b, n) for (a, b), n in edge_counts.items())
    # This helper is for heat visualization only; no transitions/typed_edges needed.
    typed_edges = [(a, b, n, EdgeType.CO_OCCUR.value) for a, b, n in edges]
    return ConceptGraph(
        nodes=dict(node_counts),
        edges=edges,
        transitions={},
        typed_edges=typed_edges,
    )


def time_range(log: EventLog) -> tuple[float | None, float | None]:
    events = log.read_all()
    if not events:
        return (None, None)
    return (events[0].ts, events[-1].ts)


# ── CLI ─────────────────────────────────────────────────────────────────


def _emit_graph(log: EventLog) -> None:
    cg = build_concept_graph(log)
    out = {
        "nodes": [{"id": slug, "degree": count}
                  for slug, count in sorted(cg.nodes.items(), key=lambda kv: (-kv[1], kv[0]))],
        "edges": [{"source": a, "target": b, "weight": w} for a, b, w in cg.edges],
        "stats": {"node_count": len(cg.nodes), "edge_count": len(cg.edges)},
    }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


def _emit_heat(log_path: Path, ts: float, half_life: float, hop_cap: int,
               min_heat: float) -> None:
    heat = compute_heat_at(
        log_path, ts=ts, half_life=half_life, hop_cap=hop_cap, min_heat=min_heat,
    )
    top = sorted(heat.items(), key=lambda kv: (-kv[1], kv[0]))[:50]
    out = {
        "ts": ts,
        "heat": heat,
        "top": [[slug, h] for slug, h in top],
        "stats": {"active_concept_count": len(heat)},
    }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


def _emit_range(log: EventLog) -> None:
    tmin, tmax = time_range(log)
    out = {"ts_min": tmin, "ts_max": tmax}
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dct.heat",
        description="DCT heat snapshot + concept graph for the MC visualization.",
    )
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("graph", "heat", "range"))
    parser.add_argument("--ts", type=float, help="timestamp for --mode heat (unix seconds, float)")
    parser.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE_SECS,
                        help="heat decay half-life in seconds (default 6h)")
    parser.add_argument("--hop-cap", type=int, default=0,
                        help="blast-radius hop cap (0 disables; default 0)")
    parser.add_argument("--min-heat", type=float, default=DEFAULT_MIN_HEAT,
                        help="drop concepts with heat below this threshold (default 0.001)")
    args = parser.parse_args()

    log = EventLog(args.log)
    if args.mode == "graph":
        _emit_graph(log)
    elif args.mode == "heat":
        if args.ts is None:
            parser.error("--ts is required for --mode heat")
        _emit_heat(args.log, args.ts, args.half_life, args.hop_cap, args.min_heat)
    elif args.mode == "range":
        _emit_range(log)


if __name__ == "__main__":
    main()
