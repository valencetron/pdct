"""Tests for the new format_for_telegram_with_sections sibling API.

Spec: §Stage 1.
"""
from __future__ import annotations

import pytest

from dct.retrieval.inject import (
    format_for_telegram,
    format_for_telegram_with_sections,
)
from dct.retrieval.types import ConceptHit, PreloadBundle


def _bundle(anchors="A-anchor-text", today="T-today", recent=None) -> PreloadBundle:
    return PreloadBundle(
        anchors=anchors,
        today_summaries=today,
        recent_summaries=recent if recent is not None else {"telegram": "R-text"},
        total_tokens=42,
    )


def _hits():
    return [
        ConceptHit(concept="phase5-card-control", score=0.42, hop=1, source_slug="x", snippet="s"),
        ConceptHit(concept="cascade", score=0.31, hop=2, source_slug="y", snippet=""),
    ]


# ──────────────────────────────────────────────────────────────────────
# back-compat: existing format_for_telegram unchanged
# ──────────────────────────────────────────────────────────────────────

def test_old_api_still_returns_string():
    out = format_for_telegram(_bundle(), _hits())
    assert isinstance(out, str)
    assert "## Anchors" in out
    assert "## Today" in out
    assert "## Recent" in out
    assert "## Jogged" in out


# ──────────────────────────────────────────────────────────────────────
# new API — full bundle, all sections present
# ──────────────────────────────────────────────────────────────────────

def test_new_api_returns_dict_with_full_and_sections():
    out = format_for_telegram_with_sections(_bundle(), _hits())
    assert "full" in out
    assert "sections" in out
    assert isinstance(out["full"], str)
    assert isinstance(out["sections"], dict)
    assert set(out["sections"].keys()) == {"anchors", "today", "recent", "jogged"}


def test_new_api_each_section_has_payload_and_rendered():
    out = format_for_telegram_with_sections(_bundle(), _hits())
    for name, sec in out["sections"].items():
        assert set(sec.keys()) == {"payload", "rendered"}, f"section {name}"
        assert isinstance(sec["payload"], str)
        assert isinstance(sec["rendered"], str)


def test_new_api_full_is_concat_of_sections():
    out = format_for_telegram_with_sections(_bundle(), _hits())
    expected = "".join(s["rendered"] for s in out["sections"].values())
    assert out["full"] == expected


def test_new_api_full_bundle_all_sections_nonzero():
    out = format_for_telegram_with_sections(_bundle(), _hits())
    for name, sec in out["sections"].items():
        assert len(sec["payload"]) > 0, f"section {name} payload empty"
        assert len(sec["rendered"]) > 0, f"section {name} rendered empty"


def test_new_api_anchors_payload_excludes_header():
    out = format_for_telegram_with_sections(_bundle(anchors="HELLO"), _hits())
    assert out["sections"]["anchors"]["payload"] == "HELLO"
    assert "## Anchors" in out["sections"]["anchors"]["rendered"]
    assert "HELLO" in out["sections"]["anchors"]["rendered"]


# ──────────────────────────────────────────────────────────────────────
# empty sections render to ""
# ──────────────────────────────────────────────────────────────────────

def test_empty_today_zero_chars():
    out = format_for_telegram_with_sections(_bundle(today=""), _hits())
    assert out["sections"]["today"]["payload"] == ""
    assert out["sections"]["today"]["rendered"] == ""


def test_empty_recent_zero_chars():
    out = format_for_telegram_with_sections(_bundle(recent={}), _hits())
    assert out["sections"]["recent"]["payload"] == ""
    assert out["sections"]["recent"]["rendered"] == ""


def test_empty_jogged_zero_chars():
    out = format_for_telegram_with_sections(_bundle(), [])
    assert out["sections"]["jogged"]["payload"] == ""
    assert out["sections"]["jogged"]["rendered"] == ""


def test_empty_anchors_zero_chars():
    out = format_for_telegram_with_sections(_bundle(anchors=""), _hits())
    assert out["sections"]["anchors"]["payload"] == ""
    assert out["sections"]["anchors"]["rendered"] == ""


def test_completely_empty_bundle_full_only_jogged():
    out = format_for_telegram_with_sections(
        PreloadBundle(anchors="", today_summaries="", recent_summaries={}, total_tokens=0),
        _hits(),
    )
    # Only jogged section has content
    assert out["sections"]["anchors"]["rendered"] == ""
    assert out["sections"]["today"]["rendered"] == ""
    assert out["sections"]["recent"]["rendered"] == ""
    assert out["sections"]["jogged"]["rendered"] != ""
    assert out["full"] == out["sections"]["jogged"]["rendered"]


def test_ablation_shape_zero_retrieval_chars():
    """Spec contract: when ablation skips cascade, daemon passes empty hits
    AND empty preload bundle → retrieval-eligible sections all empty."""
    out = format_for_telegram_with_sections(
        PreloadBundle(anchors="ID", today_summaries="", recent_summaries={}, total_tokens=0),
        [],
    )
    retrieval_chars = sum(
        len(out["sections"][k]["rendered"]) for k in ("today", "recent", "jogged")
    )
    assert retrieval_chars == 0
    # anchors retained (identity preserved during ablation)
    assert len(out["sections"]["anchors"]["rendered"]) > 0
