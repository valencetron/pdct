"""Activation engine: derives per-concept heat from an event stream.

Heat is a pure function of the events consumed, the graph structure, and the
current time. Given identical inputs, the output is identical — this is what
makes the system replayable and evaluable.

Heat formula:
    base(c, now)   = 0.5 ** ((now - last_ts[c]) / half_life_seconds)
    radius(c, c')  = radius_falloff ** shortest_path_hops(c, c')
                     or 0 if hops > radius_hop_cap
    heat(c', now)  = max over all ignited concepts c of base(c, now) * radius(c, c')
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

from dct.event_log import EventLog
from dct.events import Event, EventOp

Graph = Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class DecayConfig:
    half_life_seconds: float
    radius_hop_cap: int = 0
    radius_falloff: float = 0.5

    def __post_init__(self) -> None:
        if self.half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")
        if self.radius_hop_cap < 0:
            raise ValueError("radius_hop_cap must be >= 0")
        if not (0.0 < self.radius_falloff <= 1.0):
            raise ValueError("radius_falloff must be in (0, 1]")


class ActivationEngine:
    def __init__(self, config: DecayConfig) -> None:
        self._config = config
        self._last_ts: dict[str, float] = {}
        self._graph: Graph = {}

    @classmethod
    def replay(cls, log: EventLog, config: DecayConfig) -> ActivationEngine:
        eng = cls(config=config)
        for event in log.read_all():
            eng.consume(event)
        return eng

    def set_graph(self, graph: Graph) -> None:
        self._graph = graph

    def consume(self, event: Event) -> None:
        # R2.7: FEEDBACK events are meta-signal (edge reinforcement); they
        # must NOT activate concepts in the heat snapshot. Skip them here.
        if event.op == EventOp.FEEDBACK:
            return
        for concept in event.concepts:
            prior = self._last_ts.get(concept)
            if prior is None or event.ts > prior:
                self._last_ts[concept] = event.ts

    def last_seen_ts(self, concept: str) -> float | None:
        return self._last_ts.get(concept)

    def heat(self, concept: str, now: float) -> float:
        best = 0.0
        for origin, last in self._last_ts.items():
            base = self._base_heat(last, now)
            hops = self._hops(origin, concept)
            if hops is None:
                continue
            contribution = base * (self._config.radius_falloff ** hops)
            if contribution > best:
                best = contribution
        return best

    def snapshot(self, now: float, min_heat: float = 0.01) -> dict[str, float]:
        """Return concept → heat for every concept with heat >= min_heat.

        Also includes radius-reached concepts (those not directly consumed but
        warmed by a neighbor). Ordering: descending by heat, ties alphabetical.

        Single-pass: one BFS per origin simultaneously computes base heat and
        propagates radius contributions, avoiding the redundant per-candidate
        BFS of the old two-phase approach.
        """
        heats: dict[str, float] = {}
        cap = self._config.radius_hop_cap
        falloff = self._config.radius_falloff
        for origin, last in self._last_ts.items():
            base = self._base_heat(last, now)
            if base > heats.get(origin, 0.0):
                heats[origin] = base
            if cap > 0:
                visited = {origin}
                frontier: deque[tuple[str, int]] = deque([(origin, 0)])
                while frontier:
                    node, depth = frontier.popleft()
                    if depth >= cap:
                        continue
                    for neighbor in self._graph.get(node, ()):
                        if neighbor in visited:
                            continue
                        visited.add(neighbor)
                        contribution = base * (falloff ** (depth + 1))
                        if contribution > heats.get(neighbor, 0.0):
                            heats[neighbor] = contribution
                        frontier.append((neighbor, depth + 1))
        filtered = {c: h for c, h in heats.items() if h >= min_heat}
        return dict(sorted(filtered.items(), key=lambda kv: (-kv[1], kv[0])))

    def _base_heat(self, last: float, now: float) -> float:
        if now <= last:
            return 1.0
        elapsed = now - last
        return 0.5 ** (elapsed / self._config.half_life_seconds)

    def _hops(self, origin: str, target: str) -> int | None:
        """Shortest-path hop count within the hop cap, or None if unreachable."""
        if origin == target:
            return 0
        cap = self._config.radius_hop_cap
        if cap == 0:
            return None
        visited = {origin}
        frontier: deque[tuple[str, int]] = deque([(origin, 0)])
        while frontier:
            node, depth = frontier.popleft()
            if depth >= cap:
                continue
            for neighbor in self._graph.get(node, ()):
                if neighbor in visited:
                    continue
                if neighbor == target:
                    return depth + 1
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))
        return None
