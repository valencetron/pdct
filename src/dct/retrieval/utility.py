"""Surface-reuse classifier for PDCT prelim metrics.

NOT a "utility" classifier (concept reuse is a weaker proxy than utility).
Renamed in spec v4 to make this distinction explicit.

Rule ("at-least-half"):
  1. Tokenize concept on [-_/\\s]+, lowercase.
  2. Drop tokens shorter than MIN_TOKEN_LEN (3).
  3. Drop tokens in STOPWORDS.
  4. If <2 eligible tokens remain → INELIGIBLE (returns None — concept not scored).
  5. Else: count word-boundary regex matches in reply_text;
     hits / len(eligible) >= 0.5 → True.

Spec: docs/superpowers/specs/2026-04-29-pdct-prelim-metrics-spec.md (v4)
"""
from __future__ import annotations

import re
from typing import Any, Optional

# Stopwords: short English fillers + project-domain noise.
# Iteratively tunable via the metrics CLI's "never-matched concepts" output.
STOPWORDS = frozenset({
    # generic English
    "the", "and", "for", "with", "from", "this", "that", "will",
    "have", "been", "are", "was", "were", "but", "not", "use",
    "make", "get", "got", "run", "see", "one", "two", "new", "old",
    "yes", "no", "can", "may", "let", "also", "all", "any", "out",
    "into", "over", "very", "more", "less", "much", "some", "now",

    # project-domain stop terms (high-frequency labels in the graph
    # that match almost anything Claude says)
    "card", "cards", "today", "user", "assistant", "ai", "tool", "tools",
    "start", "started", "starts", "starting",  # 'start' family
    "test", "tests", "code", "file", "files", "memory",  # over-frequent
    "ide", "tk", "tg",  # ambiguous abbreviations
})

MIN_TOKEN_LEN = 3
MIN_ELIGIBLE_TOKENS = 2  # concepts that reduce to <2 are dropped from scoring

_TOKEN_SPLIT = re.compile(r"[-_/\s]+")


def concept_eligible_tokens(concept: str) -> list[str]:
    """Tokenize, lowercase, filter stopwords + min-length. May return []."""
    if not concept:
        return []
    raw = _TOKEN_SPLIT.split(concept.lower())
    return [t for t in raw if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS]


def concept_matched(
    concept: str,
    reply_text: str,
    node_kinds: Optional[dict[str, str]] = None,
) -> Optional[bool]:
    """Return True/False if concept is eligible, or None if ineligible.

    "At-least-half" — hits / len(tokens) >= 0.5. Word-boundary regex match,
    case-insensitive (reply already lowercased). TOKEN-level matching is
    always preserved.

    Eligibility threshold:
      - node_kinds None (legacy): require >=2 eligible tokens.
      - kind == "action": ineligible (None).
      - kind == "concept": require >=1 eligible token (single-token domain
        nouns score; STOPWORDS floor still excludes pure-stopword concepts
        like 'code'/'memory', which reduce to 0 eligible tokens).
    """
    tokens = concept_eligible_tokens(concept)
    min_tokens = MIN_ELIGIBLE_TOKENS  # legacy default
    if node_kinds is not None:
        kind = node_kinds.get(concept)
        if kind == "action":
            return None
        if kind == "concept":
            min_tokens = 1
    if len(tokens) < min_tokens:
        return None
    reply_lower = reply_text.lower()
    hits = sum(
        1 for t in tokens
        if re.search(rf"\b{re.escape(t)}\b", reply_lower)
    )
    return (hits / len(tokens)) >= 0.5


def _hop_for_concept(concept: str, paths: dict[str, list[str]]) -> Optional[int]:
    """Look up cascade hop for a concept. Hop = path length - 1.
    Returns None if no path info (ablation arm)."""
    if not paths:
        return None
    p = paths.get(concept)
    if not p:
        return None
    return max(0, len(p) - 1)


def score_turn_utility(
    reply_text: str,
    injected_concepts: list[str],
    cascade_paths: dict[str, list[str]],
    node_kinds: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Score how many injected concepts surfaced in the reply.

    Args:
      reply_text: assistant reply text.
      injected_concepts: concepts injected into the prompt (cascade hits or
                         shadow extract for ablation arm).
      cascade_paths: {concept: [seed,...,concept]} for hop attribution.
                     Empty/None for ablation arm — by_hop becomes None.

    Returns dict with keys:
      concepts_total      - len(injected_concepts)
      concepts_eligible   - count of concepts with >=2 eligible tokens
      concepts_matched    - count of eligible concepts that match the reply
      matched_concepts    - list of concept strings that matched
      by_hop              - {hop: {"eligible": int, "matched": int}}
                            or None if no path info supplied
      match_rate          - matched / eligible, or None if eligible==0
    """
    by_hop: Optional[dict[int, dict[str, int]]] = (
        {} if cascade_paths else None
    )
    eligible_count = 0
    matched_count = 0
    matched: list[str] = []

    for c in injected_concepts:
        result = concept_matched(c, reply_text, node_kinds=node_kinds)
        if result is None:
            continue  # ineligible, not scored
        eligible_count += 1
        if result:
            matched_count += 1
            matched.append(c)
        if by_hop is not None:
            hop = _hop_for_concept(c, cascade_paths)
            if hop is None:
                continue
            bucket = by_hop.setdefault(hop, {"eligible": 0, "matched": 0})
            bucket["eligible"] += 1
            if result:
                bucket["matched"] += 1

    match_rate: Optional[float] = (
        matched_count / eligible_count if eligible_count > 0 else None
    )

    return {
        "concepts_total": len(injected_concepts),
        "concepts_eligible": eligible_count,
        "concepts_matched": matched_count,
        "matched_concepts": matched,
        "by_hop": by_hop,
        "match_rate": match_rate,
    }
