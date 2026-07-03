"""Tests for dct.retrieval.utility — surface-reuse rate classifier.

Spec: §Stage 2 of pdct-prelim-metrics-spec v4.

The rule is "at-least-half":
  - Tokenize concept on [-_/\\s]+, lowercase.
  - Drop tokens shorter than MIN_TOKEN_LEN (3).
  - Drop tokens in STOPWORDS.
  - If <2 eligible tokens remain → concept is INELIGIBLE (returns None).
  - Else: word-boundary regex match each token in reply_text;
    matched / len(eligible) >= 0.5 → True.
"""
from __future__ import annotations

import pytest

from dct.retrieval import utility


# ──────────────────────────────────────────────────────────────────────
# concept_eligible_tokens
# ──────────────────────────────────────────────────────────────────────

def test_eligible_tokens_drops_too_short():
    assert utility.concept_eligible_tokens("a-bc-defgh") == ["defgh"]


def test_eligible_tokens_drops_stopwords():
    # 'card', 'today' are stopwords → only 'create' remains
    assert utility.concept_eligible_tokens("mc-card-create-today") == ["create"]


def test_eligible_tokens_keeps_compound():
    assert utility.concept_eligible_tokens("phase5-card-control") == ["phase5", "control"]


def test_eligible_tokens_lowercases():
    assert utility.concept_eligible_tokens("PHASE5-Card-CONTROL") == ["phase5", "control"]


def test_eligible_tokens_splits_on_underscore_slash_space():
    assert utility.concept_eligible_tokens("foo_bar/baz qux") == ["foo", "bar", "baz", "qux"]


def test_eligible_tokens_handles_empty():
    assert utility.concept_eligible_tokens("") == []


# ──────────────────────────────────────────────────────────────────────
# concept_matched
# ──────────────────────────────────────────────────────────────────────

def test_concept_matched_ineligible_returns_none():
    """Single-eligible-token concept → None (ineligible for scoring)."""
    assert utility.concept_matched("mc-card-create", "we created a card") is None


def test_concept_matched_two_token_match():
    # phase5-card-control → ["phase5","control"], reply has both → 2/2 = 1.0
    assert utility.concept_matched(
        "phase5-card-control", "phase5 control work"
    ) is True


def test_concept_matched_two_token_half_match_passes():
    # at-least-half rule: 1/2 = 0.5 → True
    assert utility.concept_matched(
        "phase5-card-control", "we shipped phase5"
    ) is True


def test_concept_matched_two_token_zero_match():
    assert utility.concept_matched(
        "phase5-card-control", "good morning everyone"
    ) is False


def test_concept_matched_three_token_one_third_fails():
    # foo-bar-baz → 3 tokens; 1 hit / 3 = 0.33 < 0.5 → False
    assert utility.concept_matched("foo-bar-baz", "only foo here") is False


def test_concept_matched_three_token_two_thirds_passes():
    # 2/3 = 0.667 ≥ 0.5 → True
    assert utility.concept_matched("foo-bar-baz", "foo and bar") is True


def test_concept_matched_word_boundary_no_match_within_word():
    """\\bphase5\\b does NOT match in 'phase50' (numeric extension); \\bcontrol\\b
    does NOT match in 'controlling' (alpha extension). So 0/2 → False."""
    assert utility.concept_matched(
        "phase5-control", "phase50 controlling"
    ) is False


def test_concept_matched_word_boundary_partial_word_match():
    """One token matches as separate word, other matches as stem extension only.
    Stem extension does NOT count (boundary), so only 1/2 → True (at-least-half)."""
    # 'walk-fast': 'walk' inside 'walking' fails boundary; 'fast' matches.
    assert utility.concept_matched(
        "walk-fast", "walking fast today"
    ) is True


def test_concept_matched_punctuation_safe():
    """Reply containing concept token followed by punctuation still matches."""
    assert utility.concept_matched(
        "phase5-control", "phase5, control: shipped."
    ) is True


def test_concept_matched_case_insensitive():
    assert utility.concept_matched(
        "phase5-control", "PHASE5 Control today"
    ) is True


# ──────────────────────────────────────────────────────────────────────
# score_turn_utility
# ──────────────────────────────────────────────────────────────────────

def test_score_turn_aggregates_correctly():
    concepts = ["phase5-card-control", "cascade-graph", "mc-card-create"]
    cascade_paths = {
        "phase5-card-control": ["seed", "phase5-card-control"],
        "cascade-graph": ["seed", "cascade-graph"],
        "mc-card-create": ["seed", "mc-card-create"],
    }
    # Reply matches phase5+control (matched), and cascade+graph (matched);
    # mc-card-create is ineligible (single eligible token after filter).
    reply = "we shipped phase5 control + cascade graph work"
    out = utility.score_turn_utility(reply, concepts, cascade_paths)
    assert out["concepts_total"] == 3
    assert out["concepts_eligible"] == 2  # mc-card-create dropped
    assert out["concepts_matched"] == 2
    assert out["match_rate"] == 1.0
    assert set(out["matched_concepts"]) == {"phase5-card-control", "cascade-graph"}


def test_score_turn_no_eligible_concepts():
    """All concepts ineligible → match_rate is None (not 0.0; avoids divide-by-zero
    skewing aggregate stats)."""
    concepts = ["mc-card", "mc-today"]
    out = utility.score_turn_utility("anything", concepts, {})
    assert out["concepts_eligible"] == 0
    assert out["match_rate"] is None


def test_score_turn_by_hop_bucketing():
    concepts = ["phase5-control", "graph-cascade-walk"]
    paths = {
        "phase5-control":      ["seed", "phase5-control"],          # hop 1
        "graph-cascade-walk":  ["seed", "mid", "graph-cascade-walk"],  # hop 2
    }
    reply = "phase5 control discussion"  # phase5-control matches; graph-cascade-walk doesn't
    out = utility.score_turn_utility(reply, concepts, paths)
    assert out["by_hop"] == {
        1: {"eligible": 1, "matched": 1},
        2: {"eligible": 1, "matched": 0},
    }


def test_score_turn_no_hop_info_returns_none_by_hop():
    """Ablation case: shadow concepts arrive with empty cascade_paths.
    by_hop should be None (CLI handles this)."""
    out = utility.score_turn_utility("text", ["phase5-control"], {})
    assert out["by_hop"] is None


def test_score_turn_empty_concepts():
    out = utility.score_turn_utility("text", [], {})
    assert out["concepts_total"] == 0
    assert out["concepts_eligible"] == 0
    assert out["match_rate"] is None
