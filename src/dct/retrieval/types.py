"""Type definitions for the retrieval engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ConceptHit:
    """A single concept retrieved by the cascade.

    hop=0 means seed; hop=1 is direct neighbor; hop=2+ is decayed cascade.
    score is normalized [0, 1] where 1.0 is seed and lower values are
    attenuated by edge weight x decay**hop.

    path: trajectory from a seed to this concept along the cascade walk.
    For hop=0, path == [concept]. For hop>=1, path[-1] == concept and
    path[0] is the seed that originated the walk. Default [] for callers
    that construct ConceptHit without cascade context.
    """
    concept: str
    score: float
    source_slug: str
    snippet: str
    hop: int
    path: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreloadBundle:
    """Session-start context bundle.

    anchors: concatenated text of static anchor files (CLAUDE.md, soul.md...)
    today_summaries: distilled notes from today, all surfaces aggregated
    recent_summaries: {surface: concatenated_text} for last-N per surface
    total_tokens: estimated via ~4 chars/token
    """
    anchors: str
    today_summaries: str
    recent_summaries: dict[str, str]
    total_tokens: int


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration bundle for retrieval ops."""
    anchor_paths: list[Path]
    distill_root: Path
    surfaces: list[str]
    # Phase 2 (2026-05-28): additional roots to scan for distilled notes.
    # Primary use: vault/compaction-archive/ written by compaction_archive.py.
    # Each root is walked with rglob("*.md"), same as distill_root.
    # Missing/empty roots are silently skipped.
    archive_roots: list[Path] = field(default_factory=list)
    cascade_depth: int = 2
    cascade_decay: float = 0.4
    cascade_budget_ms: int = 800
    cascade_token_cap: int = 10_000
    cascade_score_floor: float = 0.05
    cascade_top_k: int = 80
    # Heat wiring (cooling principle).
    cascade_heat_enabled: bool = True
    cascade_heat_floor: float = 0.01
    cascade_heat_half_life_s: float = 21600.0
    cascade_heat_min_dict_size: int = 20
    # Eligibility filter — when ENABLED (default), drops non-seed cascade
    # hits that the downstream utility scorer treats as INELIGIBLE
    # (single-token / stopword-only concepts).
    #
    # Conditional contract (when this flag is True): post-heat, pre-trim
    # non-seed hits are filtered to scorable concepts. Seeds (hop=0)
    # always bypass — they encode user intent and a user-typed
    # single-token "[[Memory]]" must be honored in the prompt even
    # though match-rate will exclude it (the scorer already drops
    # INELIGIBLE concepts from the rate denominator, so the
    # seed-vs-non-seed asymmetry doesn't deflate the metric).
    #
    # When this flag is False, no filtering happens and `cascade_concepts`
    # may contain ineligible-for-scoring concepts. The flag exists so
    # we can A/B test the filter against historical baselines.
    #
    # PDCT v2 P1.1 — junk-concept blocklist
    # (Codex review #1: contract; Codex review #2: conditional wording).
    cascade_eligibility_filter_enabled: bool = True
    preload_anchor_cap: int = 5_000
    preload_today_cap: int = 5_000
    preload_surface_cap: int = 5_000
    preload_last_n: int = 10
    # Track C — Directed transitions (Claim 2b).
    # When enabled, next-hop scores are multiplied by
    # (1 + cascade_transitions_bias * normalized_transition_count).
    # cascade_transitions_bias=0.0 = no biasing (pure undirected walk).
    cascade_transitions_enabled: bool = True
    cascade_transitions_bias: float = 0.5
    # Track C — VEC_NEAR heterogeneous edges (Claim 3).
    # VEC_NEAR edges use cascade_vec_near_decay instead of cascade_decay.
    cascade_vec_near_enabled: bool = True
    cascade_vec_near_decay: float = 0.2
    # Conversational cascade (build #-) — turn-to-turn path memory.
    # All OFF by default: zero change to stateless callers until conv_cascade_enabled.
    # See benchmark/SPEC-conversational-cascade.md.
    conv_cascade_enabled: bool = False        # master switch
    conv_seed_augment_enabled: bool = True    # Channel A: warm activated concepts as seeds
    conv_heat_priming_enabled: bool = True    # Channel B: activation → heat tie-break
    conv_decay: float = 0.6                   # per-turn multiplicative decay of activation
    conv_floor: float = 0.05                  # drop activation entries below this
    conv_seed_topk: int = 3                   # how many warm concepts to inject as seeds
    conv_seed_immutable: bool = False         # warm seeds replaceable by stronger paths
    conv_seen_policy: str = "deprioritize"    # "deprioritize" | "exclude"
    conv_seen_penalty: float = 0.5            # score *= penalty for seen docs (deprioritize)
