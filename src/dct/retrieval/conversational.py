"""Conversational cascade — turn-to-turn path memory over the stateless engine.

The retrieval engine (`service.run`) is stateless: each call seeds the cascade
from the current user_text alone. This wrapper gives a conversation *memory of
its own trajectory*, so that retrieval at turn N depends on the path T1→…→N —
the central claim of Path-Dependent Context Traversal.

It holds a decayed per-concept **activation vector**. After each turn the
retrieved concepts accumulate activation; before the next turn everything decays
(× conv_decay) and sub-floor entries are forgotten. The top-K activated concepts
are injected as *warm* seeds (score = activation, NOT 1.0) so the user's actual
query still dominates and the conversation only PRIMES the frontier.

Design: spec at benchmark/SPEC-conversational-cascade.md.

This wrapper is ADDITIVE and does not touch cascade() internals. It calls
service.run() with `seeds_override` (warm seeds ∪ derived seeds) and reads the
new `scored_hits` field to update activation. When conv_cascade_enabled is False
it behaves identically to a plain stateless run() call (no augmentation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import RetrievalConfig
from . import service
from .service import _derive_seeds, build_config
from ..heat import ConceptGraph  # ConceptGraph is defined in dct.heat


@dataclass
class TurnRecord:
    """Audit trail for one turn — lets the eval *see* the path, not just score it."""
    turn_idx: int
    user_text: str
    derived_seeds: list[str]
    warm_seeds: list[tuple[str, float]]   # (concept, activation) injected this turn
    retrieved_slugs: list[str]
    activation_snapshot: dict[str, float]  # AFTER update+decay (state entering next turn)


@dataclass
class ConvState:
    turn_idx: int = 0
    activation: dict[str, float] = field(default_factory=dict)
    seen_distillations: set[str] = field(default_factory=set)
    turn_log: list[TurnRecord] = field(default_factory=list)


class ConversationalCascade:
    """Stateful, multi-turn front-end to service.run().

    Usage:
        cc = ConversationalCascade(topic_id="bench")
        r1 = cc.turn("what's the voice pipeline bug?")
        r2 = cc.turn("and how did we fix it?")   # primed by r1's trajectory
        cc.reset()  # start a fresh conversation
    """

    def __init__(
        self,
        *,
        topic_id: str | None = None,
        config: RetrievalConfig | None = None,
        surface: str = "conv-bench",
    ) -> None:
        self.topic_id = topic_id
        self.surface = surface
        # Build config once; conv_* knobs read from here. Caller may inject a
        # config_override (sweep arm) — else build from env/file like prod.
        self.config = config if config is not None else build_config()
        self.state = ConvState()
        # Cache the graph handle for seed derivation (matches service internals).
        self._graph: ConceptGraph | None = None

    # ── public API ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Begin a new conversation. Clears all activation + seen state."""
        self.state = ConvState()

    def turn(self, user_text: str) -> dict[str, Any]:
        """Run one conversational turn. Returns the full service.run() dict,
        plus conv_* diagnostics under result['conv']."""
        cfg = self.config

        # Derive base seeds from the current query (same as stateless path).
        graph = self._get_graph()
        derived = _derive_seeds(user_text, graph)

        warm: list[tuple[str, float]] = []
        seeds_override: list[str] | None = None

        if cfg.conv_cascade_enabled and cfg.conv_seed_augment_enabled:
            warm = self._top_activated(cfg.conv_seed_topk, exclude=set(derived))
            # Warm seeds appended AFTER derived so derived (user intent) keep
            # earliest position; service dedups order-preserving.
            seeds_override = list(derived) + [c for c, _ in warm]

        # Heat priming (Channel B): pass activation as the heat view. service.run
        # currently hard-codes heat={} in the cascade call; until that passthrough
        # lands we expose activation in diagnostics and rely on Channel A. (TODO
        # wire heat= into run() — tracked in spec §3 Channel B.)
        result = service.run(
            user_text,
            topic_id=self.topic_id,
            surface=self.surface,
            config_override=cfg,
            seeds_override=seeds_override,
        )

        scored = result.get("scored_hits", []) or []

        # Apply seen-doc policy at AGGREGATION (cascade stays pure). Only in
        # conv mode; stateless path is untouched.
        if cfg.conv_cascade_enabled and self.state.seen_distillations:
            scored = self._apply_seen_policy(scored, cfg)

        # Update activation from this turn's hits, then decay.
        if cfg.conv_cascade_enabled:
            self._update_activation(scored)
            self._decay()

        retrieved_slugs = [h["source_slug"] for h in scored if h.get("source_slug")]
        if cfg.conv_cascade_enabled:
            self.state.seen_distillations.update(retrieved_slugs)

        # Record audit trail.
        rec = TurnRecord(
            turn_idx=self.state.turn_idx,
            user_text=user_text,
            derived_seeds=list(derived),
            warm_seeds=warm,
            retrieved_slugs=retrieved_slugs,
            activation_snapshot=dict(self.state.activation),
        )
        self.state.turn_log.append(rec)
        self.state.turn_idx += 1

        # Attach diagnostics for the eval.
        result["conv"] = {
            "turn_idx": rec.turn_idx,
            "derived_seeds": rec.derived_seeds,
            "warm_seeds": rec.warm_seeds,
            "activation": rec.activation_snapshot,
            "retrieved_slugs": retrieved_slugs,
            "seen_count": len(self.state.seen_distillations),
            "policy": cfg.conv_seen_policy,
        }
        return result

    # ── internals ────────────────────────────────────────────────────────────

    def _get_graph(self) -> ConceptGraph:
        if self._graph is None:
            self._graph = service._load_or_build_graph(
                topic_id=self.topic_id, ignore_feedback=False,
            )
        return self._graph

    def _top_activated(self, k: int, *, exclude: set[str]) -> list[tuple[str, float]]:
        items = [
            (c, w) for c, w in self.state.activation.items()
            if c not in exclude and w >= self.config.conv_floor
        ]
        items.sort(key=lambda x: (-x[1], x[0]))
        return items[:k]

    def _update_activation(self, scored: list[dict[str, Any]]) -> None:
        for h in scored:
            c = h.get("concept")
            if not c:
                continue
            self.state.activation[c] = self.state.activation.get(c, 0.0) + float(h.get("score", 0.0))

    def _decay(self) -> None:
        d = self.config.conv_decay
        floor = self.config.conv_floor
        decayed = {}
        for c, w in self.state.activation.items():
            nw = w * d
            if nw >= floor:
                decayed[c] = nw
        self.state.activation = decayed

    def _apply_seen_policy(
        self, scored: list[dict[str, Any]], cfg: RetrievalConfig,
    ) -> list[dict[str, Any]]:
        seen = self.state.seen_distillations
        if cfg.conv_seen_policy == "exclude":
            return [h for h in scored if h.get("source_slug") not in seen]
        # deprioritize (default): penalize seen docs' score, keep them eligible.
        out = []
        for h in scored:
            if h.get("source_slug") in seen:
                h = dict(h)
                h["score"] = float(h.get("score", 0.0)) * cfg.conv_seen_penalty
            out.append(h)
        out.sort(key=lambda x: -float(x.get("score", 0.0)))
        return out
