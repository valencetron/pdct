"""Tests for the retrieval eligibility filter."""
from __future__ import annotations

from dataclasses import dataclass, field

from dct.retrieval.eligibility import is_eligible


@dataclass
class FakeRef:
    title: str = "A Real Title"
    concepts: list = field(default_factory=lambda: ["alpha", "beta"])


_PROSE = (
    "This session investigated the PDCT retrieval pipeline and found two bugs. "
    "The first was a substring matching error that seeded junk concepts into the "
    "graph. The second was a missing registry write during the build phase. "
    "Both were fixed and verified against the live filesystem before shipping. "
    "Alex approved the eligibility filter as the next work item for the system. "
    "The team agreed that the same gate should apply to both live and test paths. "
    "This keeps the benchmark honest while improving the real conversation quality. "
)


def test_eligible_real_distillation():
    ok, reason = is_eligible(FakeRef(), _PROSE)
    assert ok is True
    assert reason == ""


def test_thin_body_excluded():
    ok, reason = is_eligible(FakeRef(), "too short")
    assert ok is False
    assert reason == "thin"


def test_no_concepts_excluded():
    ok, reason = is_eligible(FakeRef(concepts=[]), _PROSE)
    assert ok is False
    assert reason == "no-concepts"


def test_transcript_dump_excluded():
    body = "\n".join(
        f"[tool:Bash cmd=sed -n '{i},{i+5}p' daemon.py] user: [tool_result: [rc=0] "
        f"def fn_{i}(): return {i}] assistant: next"
        for i in range(20)
    )
    ok, reason = is_eligible(FakeRef(title="1003690648082_20196"), body)
    assert ok is False
    assert reason == "transcript-dump"


def test_pruned_recap_excluded():
    body = "## Conversation recap (pruned)\n" + _PROSE
    ok, reason = is_eligible(FakeRef(), body)
    assert ok is False
    assert reason == "pruned-recap"


def test_bare_id_title_no_prose_excluded():
    # Long enough body and has concepts, but title is a bare topic key and the
    # body is keyword soup with no real sentences.
    body = "telegram dispatch daemon trace formatting token usage " * 20
    ok, reason = is_eligible(FakeRef(title="1003690648082_None"), body)
    assert ok is False
    assert reason == "bare-id-title"


def test_bare_id_title_with_prose_kept():
    # Bare-ish title but the body is genuine prose => keep it.
    ok, reason = is_eligible(FakeRef(title="100369064808219971"), _PROSE)
    assert ok is True
    assert reason == ""


def test_traces_with_real_prose_kept():
    # A summary that quotes a couple of tool lines but is mostly prose stays in.
    body = _PROSE + "\nThe command `[tool:Bash]` returned [rc=0] as expected.\n" + _PROSE
    ok, reason = is_eligible(FakeRef(), body)
    assert ok is True
    assert reason == ""
