"""Relevance filter v0 — time-and-surface-aware cascade trim.

Sits between cascade (heat/eligibility-filtered) and prompt format. Consults
a JSON rules file plus the caller-supplied (now_snapshot, surface) tuple to
decide whether to drop concepts, restrict to a prefix allow-list, or override
score floor / top-K for this turn.

Public surface:
    RelevancePolicy       — dataclass returned by resolve_policy
    NO_OP_POLICY          — sentinel default (no-op)
    resolve_policy(...)   — match (snapshot, surface) → policy
    apply_policy(...)     — filter cascade hits per policy
    load_rules(...)       — read + parse rules JSON, mtime+size-cached

Failure contract: every callable swallows internal errors and returns the
no-op result, logging once. NEVER raises — relevance is a quality gate, not
a correctness gate. The cascade ships with or without it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelevancePolicy:
    """The output of resolve_policy. All fields optional / empty by default."""

    denied_concept_prefixes: tuple[str, ...] = ()
    allowed_concept_prefixes: tuple[str, ...] = ()
    cascade_score_floor: Optional[float] = None
    cascade_top_k: Optional[int] = None
    posture_hint: str = ""
    rule_id: str = ""


NO_OP_POLICY = RelevancePolicy()


def _normalize_snapshot(snapshot: Optional[dict]) -> dict:
    """Normalize a caller-supplied snapshot dict to the resolver's match shape.

    Expected input keys (any may be missing):
        cell_key:        "sun.mid_morning" — split into day_of_week + time_of_day
        activity_names:  list[str] — current activities
        workday_status:  "Workday" | "Weekend"

    Returns a dict with always-present keys:
        day_of_week:           "sun" | ""
        time_of_day:           "mid_morning" | ""
        workday_status:        whatever the caller passed | ""
        activity_names_lower:  list[str] (lowercased, blanks/non-strings dropped)
        is_empty:              True iff every meaningful field is blank/empty.
                               Resolver uses this to short-circuit to NO_OP_POLICY
                               so empty snapshots never match a catch-all rule.

    Never raises. Malformed cell_key → blank dow/tod.
    """
    if not snapshot:
        return {
            "day_of_week": "",
            "time_of_day": "",
            "workday_status": "",
            "activity_names_lower": [],
            "is_empty": True,
        }

    cell_key = snapshot.get("cell_key", "") or ""
    if cell_key and "." in cell_key:
        dow, tod = cell_key.split(".", 1)
    else:
        dow, tod = "", ""

    raw_activities = snapshot.get("activity_names", []) or []
    activity_names_lower: list[str] = []
    for a in raw_activities:
        if isinstance(a, str) and a.strip():
            activity_names_lower.append(a.strip().lower())

    workday_status = snapshot.get("workday_status", "") or ""

    is_empty = (
        not dow and not tod and not workday_status and not activity_names_lower
    )

    return {
        "day_of_week": dow.lower(),
        "time_of_day": tod.lower(),
        "workday_status": workday_status,
        "activity_names_lower": activity_names_lower,
        "is_empty": is_empty,
    }


def _coerce_list(value) -> list:
    """Treat a bare string as [string]; pass lists through; everything else → []."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _match_dow(rule_value, norm: dict, surface: str) -> bool:
    values = _coerce_list(rule_value)
    return norm["day_of_week"] in {v.lower() for v in values if isinstance(v, str)}


def _match_tod(rule_value, norm: dict, surface: str) -> bool:
    values = _coerce_list(rule_value)
    return norm["time_of_day"] in {v.lower() for v in values if isinstance(v, str)}


def _match_workday(rule_value, norm: dict, surface: str) -> bool:
    if isinstance(rule_value, list):
        return norm["workday_status"] in rule_value
    return norm["workday_status"] == rule_value


def _match_activity_any_of(rule_value, norm: dict, surface: str) -> bool:
    """Substring (case-insensitive) match against any current activity name.

    Calendar event titles are messy: 'Family lunch with Sam',
    'ExampleCo session - Akshay'. Exact-equality would force every needle to
    enumerate variants. Substring matching: 'family lunch' matches any
    activity whose lowercased name CONTAINS 'family lunch'.

    Tradeoff: short needles are over-permissive. Multi-word needles are
    the sweet spot for the starter rules.
    """
    values = _coerce_list(rule_value)
    needles = [v.lower() for v in values if isinstance(v, str) and v.strip()]
    if not needles:
        return False
    for activity in norm["activity_names_lower"]:
        for needle in needles:
            if needle in activity:
                return True
    return False


def _match_surface_any_of(rule_value, norm: dict, surface: str) -> bool:
    values = _coerce_list(rule_value)
    return surface in {v for v in values if isinstance(v, str)}


_MATCH_HANDLERS = {
    "day_of_week": _match_dow,
    "time_of_day": _match_tod,
    "workday_status": _match_workday,
    "activity_any_of": _match_activity_any_of,
    "surface_any_of": _match_surface_any_of,
    # Reserved for v0.5 categorizer — present here means the resolver KNOWS
    # the key but no caller emits it yet, so any rule using it currently
    # never matches (intended; rules will be migrated when categorizer ships).
    "activity_class_any_of": lambda *_a, **_k: False,
}


def _rule_matches(rule: dict, norm: dict, *, surface: str) -> bool:
    """True iff every key in rule['match'] matches; unknown key fails the rule.

    `match: {}` (explicitly empty) → catch-all, matches everything.
    Missing `match` key entirely → malformed rule, does not match.
    """
    if "match" not in rule:
        return False
    match_block = rule.get("match") or {}
    if not match_block:
        return True
    for key, value in match_block.items():
        handler = _MATCH_HANDLERS.get(key)
        if handler is None:
            return False
        try:
            if not handler(value, norm, surface):
                return False
        except Exception as e:
            _log.warning("[relevance] match handler %s raised: %s", key, e)
            return False
    return True


# Cache: { str(path): ((mtime_ns, size), rules_list) }
# mtime alone has 1-second resolution on some filesystems; rapid edits within
# the same second can be missed. Combining mtime_ns + size catches all
# realistic edits.
_RULES_CACHE: dict[str, tuple[tuple[int, int], list[dict]]] = {}


def load_rules(path: Path) -> list[dict]:
    """Read + parse the relevance rules JSON file. Mtime+size-cached.

    Returns an empty list (no rules) on any failure: missing file, invalid
    JSON, unexpected top-level shape, or an entry that isn't a dict. Logs
    on failure; caller treats empty-rules == no-op.

    Accepted file shape:
        {
          "version": 1,
          "rules": [ { "id": str, "match": {...}, "policy": {...} }, ... ],
          "default_policy": {...}    # ignored in v0
        }
    """
    try:
        try:
            st = path.stat()
            cache_key = (int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            return []
        cached = _RULES_CACHE.get(str(path))
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            _log.warning("[relevance] failed to read %s: %s", path, e)
            return []
        if not isinstance(data, dict):
            _log.warning("[relevance] rules file root not an object: %s", path)
            return []
        raw_rules = data.get("rules", [])
        if not isinstance(raw_rules, list):
            _log.warning("[relevance] 'rules' key is not a list: %s", path)
            return []
        rules = [r for r in raw_rules if isinstance(r, dict)]
        _RULES_CACHE[str(path)] = (cache_key, rules)
        return rules
    except Exception as e:
        _log.warning("[relevance] load_rules unexpected failure: %s", e)
        return []


def resolve_policy(
    snapshot: Optional[dict],
    *,
    surface: str,
    rules: list[dict],
) -> RelevancePolicy:
    """Match (snapshot, surface) against rules, return first-match policy.

    Iterates rules top-to-bottom. The first rule whose `match` block is
    satisfied wins; subsequent rules are not evaluated. Returns NO_OP_POLICY
    if no rule matches OR no rules exist OR snapshot is empty.

    Failure contract: NEVER raises. Per-rule exceptions are caught and the
    rule is skipped (logged at warn). Top-level exceptions return NO_OP_POLICY.
    """
    try:
        if not rules:
            return NO_OP_POLICY
        norm = _normalize_snapshot(snapshot)
        if norm.get("is_empty"):
            # Strict no-op: no temporal context = no filtering, ever.
            # A catch-all `match: {}` rule does NOT fire here.
            return NO_OP_POLICY
        for rule in rules:
            try:
                if not _rule_matches(rule, norm, surface=surface):
                    continue
                policy_dict = rule.get("policy")
                if not isinstance(policy_dict, dict):
                    _log.warning(
                        "[relevance] rule %s has non-dict policy, skipping",
                        rule.get("id", "<no-id>"),
                    )
                    continue
                return _policy_from_dict(rule.get("id", ""), policy_dict)
            except Exception as e:
                _log.warning(
                    "[relevance] rule %s eval failed: %s",
                    rule.get("id", "<no-id>"), e,
                )
                continue
        return NO_OP_POLICY
    except Exception as e:
        _log.warning("[relevance] resolve_policy unexpected failure: %s", e)
        return NO_OP_POLICY


def _policy_from_dict(rule_id: str, p: dict) -> RelevancePolicy:
    """Construct a RelevancePolicy from a rule's policy block, defensively."""
    def _tup_of_str(v) -> tuple[str, ...]:
        if isinstance(v, list):
            return tuple(s for s in v if isinstance(s, str) and s)
        return ()

    floor = p.get("cascade_score_floor")
    top_k = p.get("cascade_top_k")
    return RelevancePolicy(
        denied_concept_prefixes=_tup_of_str(p.get("denied_concept_prefixes")),
        allowed_concept_prefixes=_tup_of_str(p.get("allowed_concept_prefixes")),
        cascade_score_floor=float(floor) if isinstance(floor, (int, float)) else None,
        cascade_top_k=int(top_k) if isinstance(top_k, int) else None,
        posture_hint=p.get("posture_hint", "") if isinstance(p.get("posture_hint"), str) else "",
        rule_id=rule_id if isinstance(rule_id, str) else "",
    )


from .types import ConceptHit  # cyclic-safe; both modules in dct.retrieval


def apply_policy(
    hits: list[ConceptHit],
    policy: RelevancePolicy,
    *,
    base_top_k: int,
    base_score_floor: float,
) -> Tuple[list[ConceptHit], int, int, float]:
    """Filter cascade hits according to the policy.

    Returns (filtered_hits, dropped_count, effective_top_k, effective_score_floor).

    Order:
        1. Deny-list (drop non-seeds matching any denied prefix).
        2. Allow-list (keep ONLY non-seeds matching any allowed prefix; if list
           is empty/missing, no allow-list filtering).
        3. Compute effective top_k / score_floor (overrides take precedence).

    Seeds (hop=0) ALWAYS bypass deny + allow lists — same invariant as
    _filter_by_heat and _trim_hits. User intent wins.

    `effective_top_k` is clamped UP to seed count if the override would
    drop seeds (we never lose user intent).

    Trim by score_floor + top_k cap is applied later by `_trim_hits` in
    service.run; this function only filters the deny/allow set and reports
    effective values for the caller to substitute into RetrievalConfig.
    """
    if not hits:
        return [], 0, _effective_top_k(policy, base_top_k, 0), _effective_floor(policy, base_score_floor)

    pre_count = len(hits)
    seeds = [h for h in hits if h.hop == 0]
    non_seeds = [h for h in hits if h.hop != 0]

    # --- Deny ---
    denied = policy.denied_concept_prefixes
    if denied:
        non_seeds = [
            h for h in non_seeds
            if not any(h.concept.startswith(p) for p in denied)
        ]

    # --- Allow ---
    allowed = policy.allowed_concept_prefixes
    if allowed:
        kept = [
            h for h in non_seeds
            if any(h.concept.startswith(p) for p in allowed)
        ]
        if not kept and non_seeds:
            _log.warning("[relevance] allow-list collapsed cascade (rule=%s)", policy.rule_id)
        non_seeds = kept

    filtered = seeds + non_seeds
    dropped_count = pre_count - len(filtered)

    top_k_eff = _effective_top_k(policy, base_top_k, len(seeds))
    floor_eff = _effective_floor(policy, base_score_floor)
    return filtered, dropped_count, top_k_eff, floor_eff


def _effective_top_k(policy: RelevancePolicy, base: int, seed_count: int) -> int:
    """Override wins, but never < seed_count (preserve user intent)."""
    chosen = policy.cascade_top_k if policy.cascade_top_k is not None else base
    return max(chosen, seed_count)


def _effective_floor(policy: RelevancePolicy, base: float) -> float:
    return policy.cascade_score_floor if policy.cascade_score_floor is not None else base


# ---------------------------------------------------------------------------
# Query-adaptive cosine filter (v1)
# ---------------------------------------------------------------------------

# Cosine circuit breaker state: after one slow encode, skip the filter
# (fail open) until the cool-off passes. monotonic — wall-clock immune.
_COSINE_BREAKER = {"until_mono": 0.0}
COSINE_SLOW_S = 1.5      # an encode slower than this trips the breaker
COSINE_COOLOFF_S = 1800  # skip window after a trip


def query_cosine_filter(
    user_text: str,
    hits: list[ConceptHit],
    *,
    threshold: float = 0.57,
    _model_override: Any = None,
) -> tuple[list[ConceptHit], int]:
    """Drop cascade hits whose embedding is semantically unrelated to the user query.

    Uses the same BAAI/bge-small-en-v1.5 singleton as vec_index (loaded once
    at graph build time in the persistent DCT service process — no cold-start
    penalty on subsequent turns; warm encode is ~11–120ms for 8 texts).

    Threshold calibration: 0.57 derived empirically from 16 real noise/useful
    pairs from production judge data:
      - noise pairs score 0.428–0.543 (avg 0.471)
      - useful pairs score 0.594–0.824 (avg 0.704)
      - gap of 0.051 gives clean separation at 0.57

    Rules:
    - Seeds (hop=0) are NEVER dropped — user intent is sacred.
    - Queries < 10 chars are skipped (noisy embeddings on short text).
    - Empty hits list → ([], 0).
    - Any exception → fail-open: return (hits, 0), log warning once.

    Args:
        user_text:       The raw user message for this turn.
        hits:            Cascade output (after heat/eligibility/policy filter).
        threshold:       Cosine similarity floor. Default 0.57 (calibrated).
        _model_override: For testing only. Pass "__RAISE__" to simulate failure.

    Returns:
        (filtered_hits, dropped_count)
    """
    if not hits:
        return [], 0

    # Short query bypass — embeddings for very short strings are noisy.
    if not user_text or len(user_text.strip()) < 10:
        return hits, 0

    # Circuit breaker (2026-07-17 22:50 PT): in-daemon, encode() ran 3-18s
    # under worker contention (42ms standalone) and single-handedly blew the
    # 4.5s cascade budget on 24 consecutive turns. The filter is an
    # optimization — dropping it costs a few noisy hits; keeping it cost
    # every PDCT read. After a slow encode, fail open for COSINE_COOLOFF_S
    # so at most one turn per window pays. Real fix (precomputed concept
    # vectors off the turn path) is carded: cascade-eater-fix-630aac.
    import time as _time
    if _time.monotonic() < _COSINE_BREAKER["until_mono"]:
        return hits, 0

    try:
        import numpy as _np

        # Test injection: simulate model failure.
        if _model_override == "__RAISE__":
            raise RuntimeError("simulated model failure")

        # Use the already-loaded singleton from vec_index. NEVER construct
        # here: this runs inside the 3s cascade budget and the model load is
        # ~6s (2026-07-16 root cause — every cascade timed out for 24h+).
        # Not-warm → fail open immediately; the accessor kicks a background
        # warm so a later turn gets the filter back.
        if _model_override is not None:
            model = _model_override
        else:
            from dct.retrieval.vec_index import get_model_if_ready
            model = get_model_if_ready()
            if model is None:
                _log.info("[relevance] embedding model not warm — skipping "
                          "cosine filter this turn (background warm kicked)")
                return hits, 0

        seeds = [h for h in hits if h.hop == 0]
        non_seeds = [h for h in hits if h.hop != 0]

        if not non_seeds:
            return hits, 0

        # Build concept text: slug words + snippet if available.
        concept_texts = []
        for h in non_seeds:
            slug_words = h.concept.replace("-", " ")
            text = slug_words if not h.snippet else slug_words + " " + h.snippet[:200]
            concept_texts.append(text)

        # Embed query + all concept texts in one batch call. Time it: a
        # slow encode trips the breaker so later turns fail open instead
        # of blowing the cascade budget (see breaker comment above).
        all_texts = [user_text.strip()] + concept_texts
        _t_enc = _time.monotonic()
        all_vecs = model.encode(all_texts, normalize_embeddings=True,
                                show_progress_bar=False)
        _enc_s = _time.monotonic() - _t_enc
        if _enc_s > COSINE_SLOW_S:
            _COSINE_BREAKER["until_mono"] = _time.monotonic() + COSINE_COOLOFF_S
            _log.warning(
                "[cosine] encode took %.1fs (> %.1fs) — breaker tripped, "
                "filter fails open for %dmin", _enc_s, COSINE_SLOW_S,
                COSINE_COOLOFF_S // 60)
        query_vec = all_vecs[0]
        concept_vecs = all_vecs[1:]

        # Cosine similarity (L2-normalised → dot product = cosine).
        kept = []
        dropped = 0
        for i, h in enumerate(non_seeds):
            sim = float(_np.dot(query_vec, concept_vecs[i]))
            if sim >= threshold:
                kept.append(h)
            else:
                dropped += 1

        return seeds + kept, dropped

    except Exception as exc:
        _log.warning("[relevance] query_cosine_filter failed, skipping: %s", exc)
        if "meta tensor" in str(exc):
            # Poisoned model from a raced construction — clear it so the
            # next background warm builds a clean one (2026-07-16).
            try:
                from dct.retrieval.vec_index import reset_model
                reset_model()
                _log.warning("[relevance] cleared poisoned embedding model")
            except Exception:  # noqa: BLE001
                pass
        return hits, 0
