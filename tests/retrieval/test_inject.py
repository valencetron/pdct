"""Inject tests — per-surface formatters."""
from __future__ import annotations

from dct.retrieval.inject import (
    format_for_telegram,
    format_for_retell,
    format_for_claude_code,
)
from dct.retrieval.types import ConceptHit, PreloadBundle


def _bundle(anchors="A-TEXT", today="TODAY", recent=None):
    return PreloadBundle(
        anchors=anchors,
        today_summaries=today,
        recent_summaries=recent or {"voice": "VOICE", "telegram": "TG"},
        total_tokens=10,
    )


def _hits():
    return [
        ConceptHit(concept="consciousness", score=1.0, source_slug="seed", snippet="", hop=0),
        ConceptHit(concept="phenomenology", score=0.8, source_slug="hop-1", snippet="", hop=1),
    ]


def test_format_for_telegram_contains_all_sections():
    out = format_for_telegram(_bundle(), _hits())
    assert "A-TEXT" in out
    assert "TODAY" in out
    assert "VOICE" in out
    assert "TG" in out
    assert "consciousness" in out
    assert "phenomenology" in out


def test_format_for_retell_contains_all_sections():
    out = format_for_retell(_bundle(), _hits())
    assert "A-TEXT" in out
    assert "TODAY" in out
    assert "consciousness" in out


def test_format_for_claude_code_is_dict():
    out = format_for_claude_code(_hits())
    assert "jogged" in out
    assert isinstance(out["jogged"], list)
    assert len(out["jogged"]) == 2
    jc = out["jogged"][0]
    assert "concept" in jc
    assert "score" in jc
    assert "hop" in jc


def test_format_for_claude_code_empty_when_no_hits():
    out = format_for_claude_code([])
    assert out == {"jogged": []}
