"""Tests for dct.retrieval.correction_signal.

Spec: §Stage 3B.
"""
from __future__ import annotations

import pytest

from dct.retrieval import correction_signal as cs


# ──────────────────────────────────────────────────────────────────────
# correction patterns — should match
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_pattern", [
    ("no, that's wrong", "leading-no"),
    ("Nope, try again", "leading-no"),
    ("wrong direction", "leading-no"),
    ("incorrect, redo it", "leading-no"),
    ("hmm that's wrong", "thats-wrong"),
    ("you're wrong about that", "youre-wrong"),
    ("got that wrong", "youre-wrong"),
    ("it didn't work", "doesnt-work"),
    ("that doesn't work", "doesnt-work"),
    ("isn't working at all", "doesnt-work"),
    ("undo that change", "undo"),
    ("revert it", "undo"),
    ("rollback please", "undo"),
])
def test_correction_patterns_match(text, expected_pattern):
    out = cs.classify_user_followup(text, "prev-id")
    assert out is not None
    assert out["rating"] == "correction"
    assert out["matched_pattern"] == expected_pattern


# ──────────────────────────────────────────────────────────────────────
# continuation patterns — should match
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_pattern", [
    ("ok cool", "approve"),
    ("Okay let's do that", "approve"),
    ("good", "approve"),
    ("perfect", "approve"),
    ("approved", "approve"),
    ("yes do it", "approve"),
    ("ship it now", "ship-it"),
    ("thanks for that", "thanks"),
    ("thank you", "thanks"),
])
def test_continuation_patterns_match(text, expected_pattern):
    out = cs.classify_user_followup(text, "prev-id")
    assert out is not None
    assert out["rating"] == "continuation"
    assert out["matched_pattern"] == expected_pattern


# ──────────────────────────────────────────────────────────────────────
# false positives — should NOT match (codex round-2 #12)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "no problem at all",   # 'no' but not as correction
    "actually let me clarify",  # 'actually' DROPPED as a pattern in v4
    "that's a really cool idea",
    "let's try a different angle",
    "what if we...",
    "thinking out loud here",
])
def test_false_positives_are_neutral(text):
    out = cs.classify_user_followup(text, "prev-id")
    assert out is not None
    assert out["rating"] == "neutral"


def test_no_problem_is_neutral_not_correction():
    """Specifically: 'no problem' should NOT trigger leading-no."""
    out = cs.classify_user_followup("no problem", "prev-id")
    assert out["rating"] == "neutral"


# ──────────────────────────────────────────────────────────────────────
# skip conditions
# ──────────────────────────────────────────────────────────────────────

def test_no_prev_turn_id_returns_none():
    assert cs.classify_user_followup("no, that's wrong", None) is None
    assert cs.classify_user_followup("no, that's wrong", "") is None


def test_too_short_returns_none():
    assert cs.classify_user_followup("no", "prev") is None  # len < 4
    assert cs.classify_user_followup("", "prev") is None
    assert cs.classify_user_followup("ok", "prev") is None


def test_tool_result_prefix_skipped():
    assert cs.classify_user_followup("[tool_result: blah blah]", "prev") is None
    assert cs.classify_user_followup("[tool: blah]", "prev") is None
    assert cs.classify_user_followup("   [tool_result: with leading space]", "prev") is None


# ──────────────────────────────────────────────────────────────────────
# row shape
# ──────────────────────────────────────────────────────────────────────

def test_row_includes_excerpt():
    text = "no, that's wrong — the cascade should fire before we extract"
    out = cs.classify_user_followup(text, "prev")
    assert "excerpt" in out
    assert len(out["excerpt"]) <= 50
    assert out["excerpt"].startswith("no, that's wrong")


def test_row_excerpt_truncated_at_50():
    text = "no, that's wrong — " + "x" * 200
    out = cs.classify_user_followup(text, "prev")
    assert len(out["excerpt"]) == 50


def test_neutral_row_has_no_matched_pattern():
    out = cs.classify_user_followup("just a thought here", "prev")
    assert out["rating"] == "neutral"
    assert out["matched_pattern"] is None
