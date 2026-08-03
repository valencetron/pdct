"""Structure-aware anchor loading tests (2026-07-27 spec).

Covers: fence-aware parsing, preamble handling, tier classification
(marker / tag / date heuristic / ancestor inheritance), budget assembly
(fit, whole-section drops, original-order output, oversized-section
fallback), and the byte-identical under-cap regression guard.
"""
from __future__ import annotations
from pathlib import Path

import pytest

from dct.retrieval.anchor_sections import (
    TRUNCATION_MARKER,
    assemble_anchor_block,
    parse_sections,
)


# -- parser -------------------------------------------------------------------

def test_parse_preamble_and_headings():
    text = "preamble line\n\n# One\nbody 1\n## Two\nbody 2\n"
    secs = parse_sections(text)
    assert [s.level for s in secs] == [0, 1, 2]
    assert secs[0].heading == "" and "preamble line" in secs[0].text
    assert secs[1].heading == "# One"
    assert secs[2].heading == "## Two"


def test_parse_no_preamble_when_file_starts_with_heading():
    secs = parse_sections("# Top\nbody\n")
    assert len(secs) == 1
    assert secs[0].level == 1


def test_parse_ignores_headings_inside_fences():
    text = "# Real\n```\n# not a heading\n```\nafter\n"
    secs = parse_sections(text)
    assert len(secs) == 1
    assert "# not a heading" in secs[0].text


def test_parse_tilde_fences():
    text = "# Real\n~~~\n# nope\n~~~\n# Second\nx\n"
    secs = parse_sections(text)
    assert [s.heading for s in secs] == ["# Real", "# Second"]


def test_parse_reassembles_verbatim():
    text = "pre\n# A\nbody a\n\n## B\n```\n# fenced\n```\nbody b\n"
    secs = parse_sections(text)
    assert "\n".join(s.text for s in secs) == text


# -- tier classification ------------------------------------------------------

def test_tier_inviolable_marker():
    secs = parse_sections("## Inviolable Rule: colorblindness\nred/green.\n")
    assert secs[0].tier == "inviolable"


def test_tier_explicit_tag_overrides():
    secs = parse_sections("## Old stories [tier:journal]\nx\n")
    assert secs[0].tier == "journal"


def test_tier_date_suffix_is_journal():
    secs = parse_sections("## Fable benchmark notes (2026-07-14)\nx\n")
    assert secs[0].tier == "journal"


def test_tier_default_core():
    secs = parse_sections("## Voice and tone\nx\n")
    assert secs[0].tier == "core"


def test_tier_child_inherits_ancestor():
    text = "## Inviolable Rules\nx\n### Email conduct\ny\n"
    secs = parse_sections(text)
    assert secs[0].tier == "inviolable"
    assert secs[1].tier == "inviolable"  # inherited from ## ancestor


def test_tier_child_own_marker_wins_over_ancestor():
    text = "## Inviolable Rules\nx\n### Random musings (2026-01-01)\ny\n"
    secs = parse_sections(text)
    assert secs[1].tier == "journal"


def test_tier_sibling_does_not_inherit():
    text = "## Inviolable Rules\nx\n## Normal section\ny\n"
    secs = parse_sections(text)
    assert secs[1].tier == "core"


# -- assembly -----------------------------------------------------------------

def _mk(text: str):
    return parse_sections(text)


def test_assembly_under_cap_is_byte_identical():
    text = "pre\n# A\nbody\n## Journal (2026-01-01)\nold\n"
    out, dropped = assemble_anchor_block(_mk(text), cap_tokens=10_000)
    assert out == text
    assert dropped == []


def test_assembly_drops_journal_before_core():
    text = (
        "## Inviolable Rule: one\n" + "i" * 800 + "\n"
        "## Core stuff\n" + "c" * 800 + "\n"
        "## Journal (2026-01-01)\n" + "j" * 800 + "\n"
    )
    out, dropped = assemble_anchor_block(_mk(text), cap_tokens=500)
    assert [s.tier for s in dropped] == ["journal"]
    assert "iii" in out and "ccc" in out and "jjj" not in out


def test_assembly_drops_core_before_inviolable():
    text = (
        "## Core stuff\n" + "c" * 400 + "\n"
        "## Inviolable Rule: one\n" + "i" * 400 + "\n"
    )
    out, dropped = assemble_anchor_block(_mk(text), cap_tokens=180)
    assert [s.tier for s in dropped] == ["core"]
    assert "iii" in out and "ccc" not in out


def test_assembly_output_preserves_original_order():
    text = (
        "## Journal early (2026-01-01)\nj1\n"
        "## Core A\nca\n"
        "## Inviolable Rule\nir\n"
        "## Core B\ncb\n"
    )
    out, dropped = assemble_anchor_block(_mk(text), cap_tokens=10_000)
    assert dropped == []
    # under cap: original order untouched
    assert out.index("Journal early") < out.index("Core A") < \
        out.index("Inviolable") < out.index("Core B")


def test_assembly_kept_sections_stay_in_document_order_when_dropping():
    text = (
        "## Core A\n" + "a" * 100 + "\n"
        "## Journal (2026-01-01)\n" + "j" * 4000 + "\n"
        "## Core B\n" + "b" * 100 + "\n"
    )
    out, dropped = assemble_anchor_block(_mk(text), cap_tokens=200)
    assert [s.tier for s in dropped] == ["journal"]
    assert out.index("Core A") < out.index("Core B")


def test_assembly_truncation_notice_present_and_loud():
    text = "## Core\nkeep\n## Journal (2026-01-01)\n" + "j" * 4000 + "\n"
    out, dropped = assemble_anchor_block(_mk(text), cap_tokens=100)
    assert len(dropped) == 1
    assert "section(s) dropped" in out
    assert "journal" in out.rsplit("---", 1)[-1]


def test_assembly_oversized_single_section_sentence_fallback():
    body = ("This is a sentence. " * 500).strip()
    text = "## Core giant\n" + body + "\n"
    cap = 300
    out, dropped = assemble_anchor_block(_mk(text), cap_tokens=cap)
    # Section alone exceeds the cap: truncated at a boundary, not dropped.
    assert "## Core giant" in out
    assert TRUNCATION_MARKER in out
    assert len(out) // 4 <= cap + 8  # small slack for the marker line


def test_assembly_never_cuts_mid_word_on_whole_section_drop():
    text = "## Core\nalpha bravo charlie\n## Journal (2026-01-01)\n" + "x" * 2000 + "\n"
    out, _ = assemble_anchor_block(_mk(text), cap_tokens=120)
    kept_body = out.split("\n---\n")[0]
    assert kept_body.endswith("alpha bravo charlie\n") or \
        kept_body.rstrip().endswith("charlie")


# -- golden: live SOUL.md (skipped where absent, e.g. pdct-public CI) ---------

_SOUL = Path.home() / "example-stack" / "SOUL.md"


@pytest.mark.skipif(not _SOUL.exists(), reason="live SOUL.md not present")
def test_golden_soul_inviolable_and_core_survive_generous_cap():
    """Regression guard for the 2026-07-27 incident.

    Measured 2026-07-27: SOUL.md has ~43 inviolable sections (~12.3k tokens),
    ~1.5k core, ~12k journal. At any cap that covers inviolable+core (15k
    here), drops must be journal-only — the legacy head-chop dropped 26
    inviolable sections at the old 5k cap and would drop them at ANY cap
    below file size.
    """
    secs = parse_sections(_SOUL.read_text(), source=_SOUL)
    inviolable = [s for s in secs if s.tier == "inviolable"]
    assert inviolable, "SOUL.md has no inviolable sections? investigate"
    out, dropped = assemble_anchor_block(secs, cap_tokens=15_000)
    assert dropped, "SOUL.md now fits in 15k? update this test's premise"
    assert all(s.tier == "journal" for s in dropped)
    for s in inviolable:
        assert s.heading in out


@pytest.mark.skipif(not _SOUL.exists(), reason="live SOUL.md not present")
def test_golden_soul_5k_cap_keeps_only_highest_tier():
    """At the old 5k cap (public default), every kept token should be
    inviolable-tier — the budget is too small for anything else."""
    secs = parse_sections(_SOUL.read_text(), source=_SOUL)
    _, dropped = assemble_anchor_block(secs, cap_tokens=5_000)
    dropped_ids = {s.order for s in dropped}
    kept = [s for s in secs if s.order not in dropped_ids]
    assert kept, "nothing kept at 5k cap? investigate"
    assert all(s.tier == "inviolable" for s in kept)
