"""SIMPLIFIED MTRAG-native conversational cascade — inspired by PDCT's
activation/decay design but NOT semantically identical to the repo cascade
(one-hop spread, edge-share normalization, decay 0.6, no transition bias / heat /
eligibility / typed edges). It walks the MTRAG ConceptGraph and seeds via the
corpus TF-IDF extractor. No dependency on the live vault retrieval service."""
from __future__ import annotations
from collections import defaultdict
from benchmark.mtrag.build_graph import MtragGraph
from benchmark.mtrag import keyphrase

DECAY = 0.6          # per-turn activation decay (matches PDCT CONV_DECAY)
FLOOR = 0.05         # drop activation below this after decay
WARM_TOPK = 3        # top activated concepts injected as warm seeds
HOP1_WEIGHT = 0.5    # neighbor activation = HOP1_WEIGHT * edge_share


class MtragCascade:
    def __init__(self, g: MtragGraph):
        self.g = g
        self.adj: dict[str, dict[str, int]] = defaultdict(dict)
        for a, b, w in g.graph.edges:
            self.adj[a][b] = w
            self.adj[b][a] = w
        self.activation: dict[str, float] = {}

    def reset(self):
        self.activation = {}

    def _decay(self):
        self.activation = {c: v * DECAY for c, v in self.activation.items()
                           if v * DECAY >= FLOOR}

    def turn(self, user_text: str) -> dict:
        # 1) decay prior activation (state entering this turn)
        self._decay()
        # 2) derive seeds from the query via the corpus extractor. A seed that
        # is not a graph node still DEPOSITS activation (it just cannot spread);
        # this avoids dropping the whole turn when slug granularity (1-3gram)
        # makes the query slug differ from the passage-derived node slugs.
        seeds = keyphrase.extract_query_concepts(user_text, self.g.extractor, top_k=6)
        node_seeds = [s for s in seeds if s in self.g.graph.nodes]
        # 3) warm seeds from carried activation (path memory)
        warm = sorted(self.activation.items(), key=lambda kv: -kv[1])[:WARM_TOPK]
        # 4) deposit activation: query seeds at 1.0, warm seeds keep their value
        for s in seeds:
            self.activation[s] = max(self.activation.get(s, 0.0), 1.0)
        for c, v in warm:
            self.activation[c] = max(self.activation.get(c, 0.0), v)
        # 5) one-hop spread from query seeds that ARE graph nodes
        for s in node_seeds:
            nbrs = self.adj.get(s, {})
            tot = sum(nbrs.values()) or 1
            for nbr, w in nbrs.items():
                inc = HOP1_WEIGHT * (w / tot)
                self.activation[nbr] = self.activation.get(nbr, 0.0) + inc
        return {"activation": dict(self.activation),
                "seeds": seeds,
                "node_seeds": node_seeds,
                "warm_seeds": [c for c, _ in warm]}
