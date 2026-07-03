"""Region (brain-cluster) detection on the concept co-occurrence graph.

Runs label-propagation clustering — dep-free, deterministic, fast enough
to run every 5 minutes from the DCT scheduler. Emits a regions.json
describing each cluster with: id, name, concepts, size.

A "region" is a brain-map continent: a cluster of concepts that co-occur
strongly in events. Consumers (Context Stream rail, future heat-map
coloring) look up a concept's region to get a human-readable label
like "Telegram Dispatch" or "Memory System" instead of raw concept tokens.

Naming heuristic (v1): each cluster is named after its highest-weighted
concept within the cluster (a proxy for "densest / most central"). For
noisy/tiny clusters, the name is the concept itself; that's ugly but
honest. Future: LLM-name clusters.

CLI:
  python -m dct.regions        # rebuild runtime/regions.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from dct.event_log import EventLog
from dct.heat import ConceptGraph, build_concept_graph


from dct import config as _cfg

DEFAULT_EVENTS = _cfg.events_path()
DEFAULT_OUTPUT = _cfg.runtime_dir() / "regions.json"


def _build_adjacency(graph: ConceptGraph) -> tuple[dict[str, dict[str, float]], dict[str, float], float]:
    """adj[node][nb] = weight; degree[node]; 2m (sum of all weights, counting each edge twice)."""
    adj: dict[str, dict[str, float]] = {n: {} for n in graph.nodes}
    for a, b, w in graph.edges:
        if a in adj and b in adj:
            adj[a][b] = adj[a].get(b, 0.0) + float(w)
            adj[b][a] = adj[b].get(a, 0.0) + float(w)
    degree: dict[str, float] = {n: sum(adj[n].values()) for n in graph.nodes}
    m2 = sum(degree.values())  # = 2 * total_weight
    return adj, degree, m2


def run_louvain(
    graph: ConceptGraph,
    *,
    max_passes: int = 20,
    resolution: float = 1.0,
) -> dict[str, int]:
    """One-level Louvain local-moving algorithm on the concept graph.

    Returns {concept: cluster_id}. Does NOT run the aggregation phase — for
    a 742-node graph the single-pass local-moving phase gives balanced
    cluster sizes (typically 10-30 clusters of 10-50 concepts each).

    resolution: >1 → more, smaller communities; <1 → fewer, larger.
    """
    adj, degree, m2 = _build_adjacency(graph)
    if m2 == 0:
        return {n: i for i, n in enumerate(graph.nodes)}

    # Initial: each node in own community.
    node_cluster: dict[str, int] = {n: i for i, n in enumerate(graph.nodes)}
    # Σ_tot per community (sum of degrees of nodes in community).
    cluster_sigma_tot: dict[int, float] = {c: degree[n] for n, c in node_cluster.items()}

    for _ in range(max_passes):
        changed = False
        # Sorted node order for determinism.
        for node in sorted(graph.nodes.keys()):
            cur_c = node_cluster[node]
            k_i = degree[node]

            # k_{i,in}(C): sum of edge weights from node to each community.
            k_in: dict[int, float] = {}
            for nb, w in adj[node].items():
                c = node_cluster[nb]
                k_in[c] = k_in.get(c, 0.0) + w
            # Remove node's self-contribution from its current cluster.
            # (No self-loops in our input; k_{i,in}(cur_c) only counts edges to
            # *other* nodes in cur_c.)

            # Treat node as if removed from cur_c for the move-out sigma_tot.
            sigma_tot_current_without_i = cluster_sigma_tot[cur_c] - k_i

            best_c = cur_c
            best_gain = 0.0
            # Baseline gain from staying (node is "moved" back into cur_c).
            base = k_in.get(cur_c, 0.0) - resolution * k_i * sigma_tot_current_without_i / m2

            for c, k_ic in k_in.items():
                if c == cur_c:
                    continue
                gain = k_ic - resolution * k_i * cluster_sigma_tot.get(c, 0.0) / m2
                if gain > base + best_gain:
                    best_gain = gain - base
                    best_c = c

            if best_c != cur_c:
                cluster_sigma_tot[cur_c] = sigma_tot_current_without_i
                cluster_sigma_tot[best_c] = cluster_sigma_tot.get(best_c, 0.0) + k_i
                node_cluster[node] = best_c
                changed = True
        if not changed:
            break

    # Renumber clusters
    unique = sorted(set(node_cluster.values()))
    remap = {c: i for i, c in enumerate(unique)}
    return {n: remap[c] for n, c in node_cluster.items()}


def _is_bad_name(c: str) -> bool:
    """Reject ugly naming candidates: UUIDs, dates, versions, path prefixes."""
    if len(c) > 35:
        return True
    # Starts with a year like 2025/2026 → date-prefixed topic slug
    if len(c) >= 5 and c[:4].isdigit() and c[4] == "-":
        return True
    # UUID shape: 8-4-4-4-12 hex, total ~36 chars with 4 hyphens
    if len(c) == 36 and c.count("-") == 4:
        return True
    # All digits/hyphens (e.g., "5-0-7", "1-2-3")
    if all(ch.isdigit() or ch == "-" for ch in c):
        return True
    # Path-prefix slugs (filesystem artifacts, not concepts)
    if c.startswith(("users-", "documents-", "example-stack-", "home-")):
        return True
    # Long hash-looking suffix
    if len(c) >= 20 and all(ch.isalnum() for ch in c.replace("-", "")):
        words = c.split("-")
        if all(len(w) < 3 or w.isdigit() for w in words):
            return True
    return False


def name_cluster(concepts: list[str], graph: ConceptGraph) -> str:
    """Name a cluster with a preference for multi-word distinctive concepts.

    Filters out UUIDs, date-prefixed slugs, and overly long names, then
    ranks surviving candidates by intra-cluster weight + multi-word bonus.
    """
    if not concepts:
        return "empty"
    candidates = [c for c in concepts if not _is_bad_name(c)]
    if not candidates:
        candidates = concepts  # fall back if everyone got filtered

    in_cluster = set(concepts)
    scores: dict[str, float] = {}
    for a, b, w in graph.edges:
        if a in in_cluster and b in in_cluster:
            scores[a] = scores.get(a, 0.0) + w
            scores[b] = scores.get(b, 0.0) + w

    # Prefer multi-word (hyphenated) names when available — "consciousness-
    # research" is more informative than "alex". Fall back to single-word if
    # nothing multi-word survives.
    multi = [c for c in candidates if "-" in c]
    pool = multi if multi else candidates

    def rank_key(c: str) -> tuple:
        base = scores.get(c, float(graph.nodes.get(c, 0)))
        return (base, -len(c))

    return max(pool, key=rank_key)


def build_regions(events_path: Path = DEFAULT_EVENTS) -> dict[str, Any]:
    log = EventLog(events_path)
    graph = build_concept_graph(log)
    if not graph.nodes:
        return {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total_concepts": 0,
            "total_clusters": 0,
            "regions": [],
        }

    node_cluster = run_louvain(graph)

    clusters: dict[int, list[str]] = {}
    for node, c in node_cluster.items():
        clusters.setdefault(c, []).append(node)

    regions_out: list[dict[str, Any]] = []
    for c, concepts in sorted(clusters.items()):
        concepts_sorted = sorted(concepts)
        # Weight = sum of node occurrences (useful to pick hottest regions first)
        total_weight = sum(graph.nodes.get(n, 0) for n in concepts)
        regions_out.append({
            "id": c,
            "name": name_cluster(concepts_sorted, graph),
            "size": len(concepts_sorted),
            "weight": total_weight,
            "concepts": concepts_sorted,
        })

    # Sort regions by weight desc (hottest regions first in the output)
    regions_out.sort(key=lambda r: (-r["weight"], r["name"]))
    # Reassign display ids after sort so the output is stable to scan
    for i, r in enumerate(regions_out):
        r["id"] = i

    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_concepts": len(graph.nodes),
        "total_clusters": len(regions_out),
        "regions": regions_out,
    }


def concept_to_region_map(data: dict[str, Any]) -> dict[str, str]:
    """Flatten the regions output to {concept: region_name}."""
    out: dict[str, str] = {}
    for r in data.get("regions", []):
        name = r.get("name", "")
        for c in r.get("concepts", []):
            out[c] = name
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dct.regions")
    p.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    data = build_regions(args.events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    if not args.quiet:
        print(
            f"regions: {data['total_clusters']} clusters / {data['total_concepts']} concepts "
            f"-> {args.output}",
            file=sys.stderr,
        )
        # Show top 10 regions by weight
        for r in data["regions"][:10]:
            print(f"  [{r['id']:3}]  {r['name']:35}  size={r['size']:4}  weight={r['weight']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
