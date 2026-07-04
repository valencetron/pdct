"""DCT retrieval service — CLI entry for cross-venv callers (e.g., Telegram daemon).

Runs the full retrieval pipeline (extract → preload → cascade → format) as
a single batch. Intended to be invoked by callers that can't import dct
directly (because they run under a different Python environment).

The concept graph is slow to build (walks all of events.jsonl). We cache a
pickled ConceptGraph at ~/example-stack/pdct/.retrieval-cache.pkl
and invalidate whenever events.jsonl's mtime exceeds the cache's mtime.
A fresh run after the first build is typically <100ms.

CLI contract:
  stdin (JSON): { "user_text": str, "current_context": [str], "now": float?,
                  "now_snapshot": dict?, "surface": str? }
  stdout (JSON): { "prompt_block": str, "seed_concepts": [str],
                   "cascade_concepts": [str], "cascade_count": int,
                   "node_kinds": {str: str}, "explicit_seed_concepts": [str],
                   "pre_heat_count": int, "post_heat_count": int,
                   "heat_floor": float,
                   "pre_trim_count": int, "score_floor": float, "top_k": int,
                   "bundle_tokens": int,
                   "relevance_rule_id": str,
                   "relevance_dropped_count": int,
                   "cascade_top_k_effective": int,
                   "cascade_score_floor_effective": float,
                   "query_cosine_dropped_count": int,
                   "query_cosine_threshold": float }
  on error: stderr gets {"error": str, "error_type": str}; exit code 1.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

from dct.rules import extract
from dct.heat import build_concept_graph, compute_heat_at, ConceptGraph
from dct.event_log import EventLog
from . import cascade, preload, format_for_telegram, RetrievalConfig
from .types import ConceptHit
from .overrides import load_overrides, OVERRIDES_PATH, clamp as _clamp_lever

_log = logging.getLogger(__name__)


from dct import config as _cfg

EVENTS_JSONL = _cfg.events_path()
CACHE_PATH = _cfg.pdct_home() / ".retrieval-cache.pkl"
DISTILL_ROOT = _cfg.vault_roots()[0]
# Phase 2 (2026-05-28): compaction archives land here; DCT now indexes them.
ARCHIVE_ROOT = _cfg.archive_root()

# R3.2: in-memory keyed cache replaces the on-disk pickle. Bump _CACHE_VERSION
# whenever the key tuple semantics change.
# v4 (2026-06-14): force fresh graph build for node_kinds classifier-aware
# scoring (Code/Concept Layer Split). kind() is a method so cached graphs
# would classify correctly anyway, but this is belt-and-suspenders.
_CACHE_VERSION = "v4"
_GRAPH_CACHE: dict[tuple, ConceptGraph] = {}

# Heat snapshot cache — same shape as _GRAPH_CACHE, keyed by:
#   (cache_version, events_path, mtime, half_life, ts_bucket=ts // 30)
# 30-second buckets so concurrent topics in the same chat share the snapshot.
_HEAT_CACHE: dict[tuple, dict[str, float]] = {}
_HEAT_CACHE_BUCKET_S = 30

_ANCHOR_CANDIDATES = tuple(_cfg.anchor_candidates())


def _existing_anchor_paths() -> list[Path]:
    return [p for p in _ANCHOR_CANDIDATES if p.exists()]


def _filter_by_eligibility(
    hits: list[ConceptHit],
    config: RetrievalConfig,
    graph=None,
) -> tuple[list[ConceptHit], int, list[str]]:
    """When enabled, filter the non-seed hits passed to this function so
    only scorable non-seeds remain; hop=0 seeds are preserved even if
    they would be unscorable.

    The downstream utility scorer (dct.retrieval.utility.score_turn_utility)
    treats concepts with <2 multi-char non-stopword tokens as INELIGIBLE
    (returns None — the concept is silently excluded from match-rate).
    Until P1.1 the same concepts were still injected into the prompt, so
    single-token nouns like "memory" / "ide" / "daemon" padded every PDCT
    block while contributing zero scorable signal.

    When `config.cascade_eligibility_filter_enabled` is False, this
    function is a passthrough — no filtering, dropped_concepts is empty.
    Seeds (hop=0) always bypass the filter regardless: they encode user
    intent and a user-typed "[[Memory]]" must be honored in the prompt
    even though the scorer will exclude it from match-rate. (Match-rate
    already excludes INELIGIBLE concepts from the denominator, so the
    seed-vs-non-seed asymmetry doesn't deflate the metric.)

    Returns (filtered_hits, pre_filter_count, dropped_concepts).
    """
    if not hits:
        return [], 0, []
    pre = len(hits)
    if not config.cascade_eligibility_filter_enabled:
        return hits, pre, []
    # Local import: utility is already a sibling and cheap, but keep import
    # lazy so test patches and ablation toggles can intervene.
    from .utility import concept_eligible_tokens, MIN_ELIGIBLE_TOKENS

    kind_fn = getattr(graph, "kind", None)
    out: list[ConceptHit] = []
    dropped: list[str] = []
    for h in hits:
        if h.hop == 0:               # seeds always bypass
            out.append(h)
            continue
        if kind_fn is not None:
            k = kind_fn(h.concept)
            if k == "action":
                dropped.append(h.concept)
                continue
            if k == "concept":
                # STOPWORDS floor must also apply here, else a frequency-
                # promoted pure-stopword node ('memory'/'code') consumes an
                # injection slot but is INELIGIBLE at scoring — breaking
                # "injection set == scorable set". Require >=1 eligible token
                # (matches the relaxed scoring threshold for concepts).
                if len(concept_eligible_tokens(h.concept)) >= 1:
                    out.append(h)
                else:
                    dropped.append(h.concept)
                continue
        # legacy fallback when no graph: old >=2-token rule
        if len(concept_eligible_tokens(h.concept)) >= MIN_ELIGIBLE_TOKENS:
            out.append(h)
        else:
            dropped.append(h.concept)
    return out, pre, dropped


def _trim_hits(
    hits: list[ConceptHit],
    config: RetrievalConfig,
) -> tuple[list[ConceptHit], int]:
    """Apply score floor + top-K cap for prompt injection.

    Seeds (hop=0) are always preserved — they're the user's intent and
    must not be dropped by either the floor or the cap. Non-seed hits are
    filtered by score, sorted desc, then clamped to (top_k - len(seeds)).

    Returns (trimmed_hits, pre_trim_count). Used by run() before format;
    not applied to memory_api / related callers (they have their own top-K).
    """
    if not hits:
        return [], 0
    pre_trim_count = len(hits)
    seeds = [h for h in hits if h.hop == 0]
    non_seeds = [
        h for h in hits
        if h.hop != 0 and h.score >= config.cascade_score_floor
    ]
    non_seeds.sort(key=lambda h: h.score, reverse=True)
    remaining = max(0, config.cascade_top_k - len(seeds))
    trimmed = seeds + non_seeds[:remaining]
    # Re-sort full result by score desc so the rendered cascade is stable.
    trimmed.sort(key=lambda h: h.score, reverse=True)
    return trimmed, pre_trim_count


def _filter_by_heat(
    hits: list[ConceptHit],
    heat: dict[str, float],
    config: RetrievalConfig,
) -> tuple[list[ConceptHit], int]:
    """Drop non-seed cascade hits whose heat is below the floor.

    Seeds (hop=0) always survive — they're the user's intent and the
    reignition path: a cold concept mentioned in user text becomes a seed,
    so it bypasses heat filtering, gets traversed, and warms up via the
    event it generates.

    Concepts not present in `heat` (decayed below the engine's internal
    min_heat=0.001 cutoff, or never observed) are treated as stone cold
    and dropped. Threshold: config.cascade_heat_floor.

    Returns (filtered_hits, pre_filter_count).
    """
    if not hits:
        return [], 0
    pre_count = len(hits)
    out: list[ConceptHit] = []
    floor = config.cascade_heat_floor
    for h in hits:
        if h.hop == 0:
            out.append(h)
            continue
        # Strict membership: absent from heat dict ⇒ stone cold (decayed below
        # the engine's internal min_heat=0.001 OR never observed). Drop even if
        # floor=0.0 — that floor means "drop stone-cold," not "no filter."
        h_heat = heat.get(h.concept)
        if h_heat is not None and h_heat >= floor:
            out.append(h)
    return out, pre_count


def _load_or_build_graph(
    events_path: Path | None = None,
    *,
    topic_id: str | None = None,
    ignore_feedback: bool = False,
) -> ConceptGraph:
    """Return ConceptGraph from in-memory cache or rebuild on staleness.

    R3.2: cache is in-memory only, keyed by
        (cache_version, events_path, mtime, topic_id, ignore_feedback)
    so different topics and ablation arms each get their own graph without
    collision. Process restart = cold rebuild.

    R3.5 fix: events_path defaults to None and resolves to module-level
    EVENTS_JSONL at call time. Late binding lets tests monkeypatch the
    module attribute and have the graph rebuild against the temp path —
    a default-arg `events_path: Path = EVENTS_JSONL` would freeze the
    production path at import time and silently ignore patches.
    """
    if events_path is None:
        events_path = EVENTS_JSONL
    try:
        mtime = events_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    # Track C Codex r2 P1 fix: cache key must include VEC_NEAR env flags and
    # vault mtime so ablation toggles and new distillations invalidate the cache.
    vec_near_flag = _env_bool("DCT_VEC_NEAR_ENABLED", True)
    vec_near_thresh = _env_float("DCT_VEC_NEAR_THRESHOLD", 0.70)
    try:
        vault_mtime = max(
            (f.stat().st_mtime for f in DISTILL_ROOT.rglob("*.md") if f.is_file()),
            default=0.0,
        )
    except OSError:
        vault_mtime = 0.0
    key = (
        _CACHE_VERSION, str(events_path), mtime, topic_id, ignore_feedback,
        vec_near_flag, vec_near_thresh, vault_mtime,
    )
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached

    log = EventLog(events_path)
    graph = build_concept_graph(
        log, topic_id=topic_id, ignore_feedback=ignore_feedback,
    )

    # Track C Claim 3: add VEC_NEAR edges if enabled.
    # DCT_VEC_NEAR_ENABLED=false disables (ablation arm).
    vec_near_enabled = _env_bool("DCT_VEC_NEAR_ENABLED", True)
    if vec_near_enabled:
        try:
            from dct.retrieval.vec_index import build_vec_near_edges
            from dataclasses import replace as _dc_replace
            vec_threshold = _env_float("DCT_VEC_NEAR_THRESHOLD", 0.70)
            vec_edges = build_vec_near_edges(DISTILL_ROOT, threshold=vec_threshold)
            if vec_edges:
                new_typed = list(graph.typed_edges) + vec_edges
                graph = _dc_replace(graph, typed_edges=new_typed)
                _log.info(
                    "[vec_near] added %d VEC_NEAR edges (threshold=%.2f)",
                    len(vec_edges), vec_threshold,
                )
        except Exception as e:
            _log.warning("[vec_near] build failed (non-fatal): %s", e)

    # Bound the cache so a long-lived process doesn't accumulate every
    # historical (mtime, topic) tuple forever.
    if len(_GRAPH_CACHE) > 32:
        _GRAPH_CACHE.clear()
    _GRAPH_CACHE[key] = graph

    # Emit a "rebuild" event so metric_graph_staleness can track graph activity.
    # This fires on every cache-miss build — which is exactly "the graph was rebuilt."
    try:
        rebuild_event = {
            "ts": time.time(),
            "source": "service",
            "op": "rebuild",
            "concepts": [],
            "metadata": {
                "nodes": len(graph.nodes) if hasattr(graph, "nodes") else 0,
                "edges": len(graph.typed_edges) if hasattr(graph, "typed_edges") else 0,
                "topic_id": topic_id,
            },
        }
        EVENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_JSONL.open("a", encoding="utf-8") as _ef:
            _ef.write(json.dumps(rebuild_event, separators=(",", ":")) + "\n")
    except Exception as _e:
        _log.warning("[graph_rebuild] failed to emit rebuild event: %s", _e)

    return graph


def _load_or_build_heat(
    events_path: Path,
    *,
    ts: float,
    half_life: float,
) -> dict[str, float]:
    """Return heat snapshot from cache or fresh compute. Cache invalidated by
    events.jsonl mtime change OR ts_bucket roll (every 30s)."""
    try:
        mtime = events_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    bucket = int(ts // _HEAT_CACHE_BUCKET_S)
    key = (_CACHE_VERSION, str(events_path), mtime, half_life, bucket)
    cached = _HEAT_CACHE.get(key)
    if cached is not None:
        return cached
    heat = compute_heat_at(events_path, ts=ts, half_life=half_life)
    if len(_HEAT_CACHE) > 32:
        _HEAT_CACHE.clear()
    _HEAT_CACHE[key] = heat
    return heat


import math

_MIN_PART_LEN = 3
_MAX_SEEDS_FROM_PROSE = 6
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

# Common-noise stopwords that are too generic to meaningfully seed retrieval.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "these", "those",
    "but", "not", "you", "your", "yours", "are", "was", "were", "been", "has",
    "have", "had", "can", "will", "would", "could", "should", "may", "might",
    "did", "does", "doing", "don", "all", "any", "some", "who", "what",
    "when", "where", "why", "how", "tell", "about", "more", "just", "like",
    "than", "only", "also", "both", "each", "every", "few", "most", "many",
    "now", "then", "there", "their", "them", "they", "thing", "well", "way",
    "get", "got", "let", "see", "saw", "say", "said", "one", "two", "three",
    "yes", "off", "out", "our", "ours",
})


def _seed_concepts_from_prose(text: str, graph: ConceptGraph) -> list[str]:
    """Match known concept slugs against tokens, weighted by rarity (TF-IDF-ish).

    Common concepts ("aperture", "projects", "users-user...") appear in
    hundreds of events and are semantically thin — they show up in
    any text as background noise. Distinctive concepts ("consciousness-
    research", "soul-md", "kastrup-mahayana-philosophy") appear rarely
    and are the real topics. Weight matches by the inverse of the
    concept's graph-wide occurrence count so rare concepts win.

    Two match paths (unchanged):
      1. Whole-slug substring match (concept appears verbatim).
      2. Part-token match: any hyphen-separated part of the slug appears
         as a standalone token.
    """
    if not text or not graph.nodes:
        return []
    text_lower = text.lower()
    tokens = {
        t for t in _TOKEN_RE.findall(text_lower)
        if len(t) >= _MIN_PART_LEN and t not in _STOPWORDS
    }
    if not tokens:
        return []

    # Rarity weighting. Log-scale so a concept with 10 occurrences is ~1/3 as
    # common as one with 1 occurrence. Small additive constant to avoid log(0).
    total_concepts = max(1, len(graph.nodes))

    def rarity(c: str) -> float:
        count = graph.nodes.get(c, 1)
        return math.log((total_concepts + 1) / (count + 1)) + 0.1

    from .utility import concept_eligible_tokens
    kind_fn = getattr(graph, "kind", None)
    scored: list[tuple[float, str]] = []
    for concept in graph.nodes:
        # Concept/action gate: never prose-seed an action node.
        if kind_fn is not None and kind_fn(concept) == "action":
            continue
        # STOPWORDS floor (same as the injection filter): a pure-stopword node
        # ('memory'/'code') must not be prose-seeded — it can't score anyway.
        if len(concept_eligible_tokens(concept)) < 1:
            continue
        c_lower = concept.lower()
        base_score: float
        # Whole-slug match must be word-boundary anchored. A raw substring
        # check seeded junk short slugs from inside words ("forward" → 'ard',
        # "think" → 'ink', "approach" → 'app') — root cause of the 2026-06
        # match-rate collapse. Short single-part slugs (<MIN len) only match
        # via the token path, which is already boundary-safe.
        # Perf: substring containment is a cheap pre-filter; the boundary
        # regex only runs on the few concepts that pass it (~O(matches),
        # not O(graph)).
        if (
            len(c_lower) >= _MIN_PART_LEN
            and c_lower in text_lower
            and re.search(
                rf"(?<![a-z0-9]){re.escape(c_lower)}(?![a-z0-9])", text_lower
            )
        ):
            base_score = 100.0
        else:
            parts = [p for p in c_lower.split("-") if len(p) >= _MIN_PART_LEN]
            if not parts:
                continue
            matching = [p for p in parts if p in tokens]
            if not matching:
                continue
            base_score = float(len(matching) * 10 - len(parts))

        # Square the rarity to push common concepts down hard.
        weighted = base_score * (rarity(concept) ** 2)
        scored.append((weighted, concept))

    scored.sort(key=lambda sc: (-sc[0], sc[1]))
    return [c for _s, c in scored[:_MAX_SEEDS_FROM_PROSE]]


def _derive_seeds(user_text: str, graph: ConceptGraph) -> list[str]:
    """Union of explicit (wikilink/hashtag) + prose-matched seed concepts."""
    explicit = extract(user_text)
    prose = _seed_concepts_from_prose(user_text, graph)
    # Ordered dedup — explicit first so they're preserved in earliest position
    seen: dict[str, None] = {}
    for s in explicit + prose:
        if s not in seen:
            seen[s] = None
    return list(seen.keys())


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def build_config() -> RetrievalConfig:
    """Build retrieval config. Heat-wiring + cascade-shape fields are
    env-var overridable so Alex can tune the noise floor / breadth without
    a code change + daemon redeploy. Read fresh per call (no caching) so a
    daemon-managed env reload picks up the next request.

    2026-05-02 — Alex bumped `cascade_score_floor` default from 0.05 → 0.10.
    Below ~0.10 was statistical noise; trims the dynamic block ~30-40%
    and improves PDCT signal-to-noise. Top-K dropped 80 → 40 in step.
    Both still env-overridable.
    """
    cfg_kwargs = dict(
        anchor_paths=_existing_anchor_paths(),
        distill_root=DISTILL_ROOT,
        archive_roots=[ARCHIVE_ROOT],
        surfaces=["voice", "claude-code", "telegram", "vault"],
        cascade_heat_enabled=_env_bool("DCT_CASCADE_HEAT_ENABLED", True),
        cascade_heat_floor=_env_float("DCT_CASCADE_HEAT_FLOOR", 0.01),
        cascade_heat_half_life_s=_env_float("DCT_CASCADE_HEAT_HALF_LIFE_S", 21600.0),
        cascade_heat_min_dict_size=_env_int("DCT_CASCADE_HEAT_MIN_DICT_SIZE", 20),
        cascade_eligibility_filter_enabled=_env_bool(
            "DCT_CASCADE_ELIGIBILITY_FILTER", True,
        ),
        cascade_score_floor=_env_float("DCT_CASCADE_SCORE_FLOOR", 0.10),
        cascade_top_k=_env_int("DCT_CASCADE_TOP_K", 20),
        # Track C — Directed transitions (Claim 2b)
        cascade_transitions_enabled=_env_bool("DCT_TRANSITIONS_ENABLED", True),
        cascade_transitions_bias=_env_float("DCT_TRANSITIONS_BIAS", 0.5),
        # Track C — VEC_NEAR heterogeneous edges (Claim 3)
        cascade_vec_near_enabled=_env_bool("DCT_VEC_NEAR_ENABLED", True),
        cascade_vec_near_decay=_env_float("DCT_VEC_NEAR_DECAY", 0.2),
        # Tier A traversal-core levers (Build #60). Env values are clamped
        # through LEVER_SPEC bounds (same path as file overrides) so a stray
        # DCT_CASCADE_DEPTH=999 can't blow up traversal — Codex diff-audit P1.
        cascade_decay=_clamp_lever("cascade_decay", _env_float("DCT_CASCADE_DECAY", 0.4)),
        cascade_depth=_clamp_lever("cascade_depth", _env_int("DCT_CASCADE_DEPTH", 2)),
    )
    # Runtime overrides (lever panel): read FRESH per call so a file write takes
    # effect next retrieval — no daemon restart. Clamped + validated in
    # load_overrides(); unknown/wrong-type keys are already dropped. Read the
    # module-level OVERRIDES_PATH so tests can monkeypatch it.
    overrides = load_overrides(OVERRIDES_PATH)
    for k, v in overrides.items():
        if k in cfg_kwargs:
            cfg_kwargs[k] = v
    return RetrievalConfig(**cfg_kwargs)


def run(
    user_text: str,
    current_context: list[str] | None = None,
    *,
    topic_id: str | None = None,
    ignore_feedback: bool = False,
    now: float | None = None,
    now_snapshot: dict | None = None,
    surface: str = "",
    config_override: "RetrievalConfig | None" = None,
    seeds_override: list[str] | None = None,
) -> dict[str, Any]:
    """Run full retrieval pipeline. Returns dict matching the CLI stdout contract.

    Track B additions:
        topic_id: scopes FEEDBACK events to a specific Telegram thread.
        ignore_feedback: if True, ALL feedback is stripped at read time
            (R3.1 ablation, end-to-end).

    Research additions (build #56):
        config_override: when provided, this RetrievalConfig is used verbatim
            instead of calling build_config(). This is the in-process injection
            point for the benchmark sandbox — it lets a sweep run each lever
            arm WITHOUT writing the live overrides file. build_config() is NOT
            consulted when an override is passed, so no env/file state leaks in.

    Returns now also includes ``cascade_paths`` — {concept: [seed,...,concept]}
    so downstream credit assignment knows the exact trajectory traversed.

    pre_heat_count: cascade size before heat filter (== raw cascade output).
    post_heat_count: cascade size after heat filter (input to trim).
    heat_floor: the heat threshold used (config.cascade_heat_floor).
    """
    ts = now if now is not None else time.time()
    config = config_override if config_override is not None else build_config()

    graph = _load_or_build_graph(
        topic_id=topic_id, ignore_feedback=ignore_feedback,
    )
    # seeds_override (conversational cascade, build #-): when the caller supplies
    # an explicit seed list (e.g. derived seeds ∪ warm activated concepts from
    # prior turns), use it verbatim instead of re-deriving from user_text alone.
    # Additive — None preserves stateless behavior exactly. Order-preserving
    # dedup so a warm seed that duplicates a derived seed doesn't double-count.
    if seeds_override is not None:
        _seen_s: dict[str, None] = {}
        for _s in seeds_override:
            if _s and _s not in _seen_s:
                _seen_s[_s] = None
        seed_concepts = list(_seen_s.keys())
    else:
        seed_concepts = _derive_seeds(user_text, graph)

    bundle = preload(config, now=ts)
    raw_hits = cascade(
        seed_concepts=seed_concepts,
        graph=graph,
        heat={},  # cascade itself still doesn't tie-break by heat — filter happens after
        config=config,
        current_context=set(current_context or []),
    )

    # ── Heat filter (fail-open + insufficient-data guard) ──
    heat_skipped_reason = "none"
    heat_dict_size = 0
    if not config.cascade_heat_enabled:
        warm_hits = raw_hits
        pre_heat_count = len(raw_hits)
        heat_skipped_reason = "disabled"
    else:
        try:
            heat_snapshot = _load_or_build_heat(
                EVENTS_JSONL,
                ts=ts,
                half_life=config.cascade_heat_half_life_s,
            )
            heat_dict_size = len(heat_snapshot)
            if heat_dict_size < config.cascade_heat_min_dict_size:
                # New / sparse session — filtering would over-cut. Skip.
                warm_hits = raw_hits
                pre_heat_count = len(raw_hits)
                heat_skipped_reason = "insufficient_data"
            else:
                warm_hits, pre_heat_count = _filter_by_heat(
                    raw_hits, heat_snapshot, config,
                )
        except Exception as e:
            # Fail open: log and keep cascade output unchanged. Heat filter
            # is a quality gate, not a correctness gate.
            _log.warning("[heat] compute failed, skipping filter: %s", e)
            warm_hits = raw_hits
            pre_heat_count = len(raw_hits)
            heat_skipped_reason = "compute_error"

    # ── Eligibility filter (P1.1 junk-concept blocklist) ──
    # Drop non-seed concepts the scorer would mark INELIGIBLE so injection
    # set == scorable set. Logs counts for /pdct dashboard surfacing.
    eligible_hits, pre_eligibility_count, eligibility_dropped = (
        _filter_by_eligibility(warm_hits, config, graph=graph)
    )
    # Snapshot the post-eligibility count BEFORE relevance filter mutates
    # eligible_hits — otherwise eligibility telemetry mixes in relevance
    # drops (Codex r2 P2 #3).
    post_eligibility_count = len(eligible_hits)

    # ── Relevance filter (v0 — time-and-surface-aware policy) ──
    relevance_rule_id = ""
    relevance_dropped_count = 0
    cascade_top_k_effective = config.cascade_top_k
    cascade_score_floor_effective = config.cascade_score_floor

    relevance_enabled = _env_bool("DCT_RELEVANCE_ENABLED", False)
    relevance_dry_run = _env_bool("DCT_RELEVANCE_DRY_RUN", False)
    if relevance_enabled:
        try:
            from dct.retrieval.relevance import (
                load_rules as _load_rules,
                resolve_policy as _resolve_policy,
                apply_policy as _apply_policy,
            )
            rules_path_env = os.environ.get(
                "DCT_RELEVANCE_RULES_PATH",
                str(Path.home() / "example-stack" / "tools" / "telegram-dispatch" / "relevance-rules.json"),
            )
            rules = _load_rules(Path(rules_path_env))
            policy = _resolve_policy(now_snapshot, surface=surface or "telegram", rules=rules)
            relevance_rule_id = policy.rule_id

            filtered, dropped, top_k_eff, floor_eff = _apply_policy(
                eligible_hits,
                policy,
                base_top_k=config.cascade_top_k,
                base_score_floor=config.cascade_score_floor,
            )
            relevance_dropped_count = dropped
            cascade_top_k_effective = top_k_eff
            cascade_score_floor_effective = floor_eff

            if not relevance_dry_run:
                from dataclasses import replace as _replace
                config = _replace(
                    config,
                    cascade_top_k=top_k_eff,
                    cascade_score_floor=floor_eff,
                )
                eligible_hits = filtered
        except Exception as e:
            _log.warning("[relevance] integration failed, skipping: %s", e)
            relevance_rule_id = ""
            relevance_dropped_count = 0

    # ── Query-adaptive cosine filter (v1) ──
    # Drops cascade hits whose embedding is semantically unrelated to the user
    # query. Runs after all rule-based filters; seeds (hop=0) always survive.
    # Threshold 0.57 calibrated from 16 real noise/useful pairs (noise max=0.543,
    # useful min=0.594). Env: DCT_QUERY_COSINE_ENABLED, DCT_QUERY_COSINE_THRESHOLD.
    query_cosine_dropped_count = 0
    cosine_enabled = _env_bool("DCT_QUERY_COSINE_ENABLED", True)
    cosine_threshold = float(os.environ.get("DCT_QUERY_COSINE_THRESHOLD", "0.57"))
    if cosine_enabled and user_text and len(user_text.strip()) >= 10:
        try:
            from dct.retrieval.relevance import query_cosine_filter as _cosine_filter
            eligible_hits, query_cosine_dropped_count = _cosine_filter(
                user_text=user_text,
                hits=eligible_hits,
                threshold=cosine_threshold,
            )
        except Exception as _e:
            _log.warning("[cosine] filter integration failed, skipping: %s", _e)

    hits, pre_trim_count = _trim_hits(eligible_hits, config)
    prompt_block = format_for_telegram(bundle, hits)
    cascade_paths = {h.concept: list(h.path) for h in hits if h.path}

    # Seed-count breakdown for telemetry (Codex round 1 #3).
    explicit_seeds = list(extract(user_text))
    explicit_seed_set = set(explicit_seeds)
    return {
        "prompt_block": prompt_block,
        "seed_concepts": seed_concepts,
        "explicit_seed_count": sum(1 for s in seed_concepts if s in explicit_seed_set),
        "prose_seed_count": sum(1 for s in seed_concepts if s not in explicit_seed_set),
        "cascade_concepts": [h.concept for h in hits],
        # Conversational cascade (build #-): per-hit (concept, score, slug, hop)
        # so a turn-to-turn caller can update its activation vector. Additive.
        "scored_hits": [
            {"concept": h.concept, "score": h.score,
             "source_slug": h.source_slug, "hop": h.hop}
            for h in hits
        ],
        # Codex r7 #4: map kinds over the FULL scoring set (hits ∪ seeds),
        # not just hits — every concept the scorer sees must have a kind.
        "node_kinds": {
            c: graph.kind(c)
            for c in ({h.concept for h in hits} | set(seed_concepts))
        },
        # Codex r7 #3: ONLY explicit wikilink/hashtag seeds (extract()), NOT
        # prose-derived seeds. A prose-leaked action must count as
        # action_cascade (regression-visible), never action_seed.
        "explicit_seed_concepts": list(explicit_seeds),
        "cascade_paths": cascade_paths,
        "cascade_count": len(hits),
        "pre_heat_count": pre_heat_count,
        "post_heat_count": len(warm_hits),
        "heat_floor": config.cascade_heat_floor,
        "heat_dict_size": heat_dict_size,
        "heat_skipped_reason": heat_skipped_reason,
        "pre_eligibility_count": pre_eligibility_count,
        "post_eligibility_count": post_eligibility_count,
        "eligibility_dropped_count": len(eligibility_dropped),
        "eligibility_dropped_sample": eligibility_dropped[:10],
        "eligibility_filter_enabled": config.cascade_eligibility_filter_enabled,
        "pre_trim_count": pre_trim_count,
        "score_floor": config.cascade_score_floor,
        "top_k": config.cascade_top_k,
        "bundle_tokens": bundle.total_tokens,
        # --- Relevance filter telemetry (v0) ---
        "relevance_rule_id": relevance_rule_id,
        "relevance_dropped_count": relevance_dropped_count,
        "cascade_top_k_effective": cascade_top_k_effective,
        "cascade_score_floor_effective": cascade_score_floor_effective,
        "query_cosine_dropped_count": query_cosine_dropped_count,
        "query_cosine_threshold": cosine_threshold,
        # --- Graph size telemetry (tuning-ledger build 85, 2026-06-25) ---
        # graph is the warm ConceptGraph bound at top of this function;
        # .nodes is a dict, .edges is a list — both O(1) len(). This closes
        # the instrumentation gap that left metric_latency_root_cause blind.
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
    }


def extract_concepts(text: str, *, use_llm: bool = False) -> list[str]:
    """Lightweight concept extraction for per-turn event tagging.

    use_llm=False (default): graph-backed heuristic match only. Fast
    (~20-50ms warm). Best when latency matters (voice pipeline).

    use_llm=True: LLM-first (Haiku, ~500ms warm). Extracts real topics
    even if they're not yet in the concept graph — those new concepts
    will be picked up on the next graph rebuild (5-min scheduler).

    Merge policy:
    - Short text (≤ 300 chars, i.e. a user message or reply): LLM first,
      then heuristic to backfill any graph concepts LLM missed.
    - Long text (> 300 chars, i.e. a distillation body): LLM-only when
      LLM succeeds. Distillation bodies mention other concept slugs as
      examples/paths, so the heuristic picks up those stray slugs and
      writes them as frontmatter concepts — contaminating retrieval.
      Falls back to heuristic only if LLM returns empty.
    """
    if not text or not text.strip():
        return []

    llm_concepts: list[str] = []
    if use_llm:
        try:
            from dct.llm import call_concept_extractor
            llm_concepts = call_concept_extractor(text)
        except Exception:
            llm_concepts = []

    graph = _load_or_build_graph()
    heuristic = _derive_seeds(text, graph)

    if not llm_concepts:
        return heuristic

    # Long text: LLM is authoritative — skip heuristic merge to avoid
    # slug-contamination from concept slugs mentioned in prose examples.
    if len(text) > 300:
        return llm_concepts[:10]

    # Short text: merge LLM + heuristic. LLM results come first (they're
    # authoritative for what the topics ARE); heuristic backfills anything
    # LLM missed that's already in the graph.
    seen: dict[str, None] = {}
    merged: list[str] = []
    for c in llm_concepts + heuristic:
        if c not in seen:
            seen[c] = None
            merged.append(c)
    return merged[:10]


def main(argv: list[str] | None = None) -> int:
    del argv  # unused
    try:
        req = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"invalid JSON on stdin: {e}", "error_type": "JSONDecodeError"}), file=sys.stderr)
        return 1

    mode = req.get("mode", "cascade")
    try:
        if mode == "extract":
            use_llm = bool(req.get("use_llm", False))
            result = {"concepts": extract_concepts(req.get("text", ""), use_llm=use_llm)}
        else:
            # R3.9: forward-compat additive kwargs. Old payloads that omit
            # topic_id / ignore_feedback get default behavior (None / False).
            result = run(
                user_text=req.get("user_text", ""),
                current_context=req.get("current_context") or [],
                topic_id=req.get("topic_id"),
                ignore_feedback=bool(req.get("ignore_feedback", False)),
                now=req.get("now"),
                now_snapshot=req.get("now_snapshot"),
                surface=req.get("surface", ""),
            )
    except Exception as e:
        print(json.dumps({"error": str(e), "error_type": type(e).__name__}), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
