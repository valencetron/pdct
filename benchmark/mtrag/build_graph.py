"""Co-occurrence ConceptGraph over MTRAG passages + concept->passage index."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dct.heat import ConceptGraph  # noqa: E402
from benchmark.mtrag import keyphrase  # noqa: E402


@dataclass
class MtragGraph:
    graph: ConceptGraph
    extractor: keyphrase.CorpusExtractor
    concept_to_passages: dict[str, set] = field(default_factory=dict)
    passage_concepts: dict[str, list] = field(default_factory=dict)
    passage_text: dict[str, str] = field(default_factory=dict)


def build(passages: list[dict], top_k: int = 8) -> MtragGraph:
    docs = [p["text"] for p in passages]
    ex = keyphrase.CorpusExtractor(docs, top_k=top_k)
    nodes: dict[str, int] = defaultdict(int)
    eweight: dict[tuple, int] = defaultdict(int)
    c2p: dict[str, set] = defaultdict(set)
    pconc: dict[str, list] = {}
    ptext: dict[str, str] = {}
    for p, text in zip(passages, docs):
        cs = ex.extract(text)
        pconc[p["id"]] = cs
        ptext[p["id"]] = text
        for c in cs:
            nodes[c] += 1
            c2p[c].add(p["id"])
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = sorted((cs[i], cs[j]))
                if a != b:
                    eweight[(a, b)] += 1
    edges = sorted((a, b, w) for (a, b), w in eweight.items())
    g = ConceptGraph(nodes=dict(nodes), edges=edges)
    return MtragGraph(graph=g, extractor=ex, concept_to_passages=dict(c2p),
                      passage_concepts=pconc, passage_text=ptext)
