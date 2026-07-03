"""Shared graph stubs for cascade behavioral/integration tests.

Mirrors the _FakeGraph/edges shape used in test_cascade.py: a graph object
exposing `.edges` as a list of (src, dst, weight) tuples.
"""
from __future__ import annotations


class _ChainGraph:
    """Minimal stand-in matching ConceptGraph.edges shape (see test_cascade.py)."""

    def __init__(self, edges):
        self.edges = edges


def chain_graph(nodes, weight: int = 10):
    """Linear chain n0 -> n1 -> ... with high edge weights so each hop-N score
    clears the default score floor. Edge tuple shape matches test_cascade.py."""
    edges = [(nodes[i], nodes[i + 1], weight) for i in range(len(nodes) - 1)]
    return _ChainGraph(edges)
