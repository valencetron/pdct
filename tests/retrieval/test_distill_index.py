from pathlib import Path
import pytest

from dct.retrieval.distill_index import build_index, find_by_id, DistillationRef


def _write(path: Path, frontmatter: dict, body: str = "") -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = "---\n" + yaml.safe_dump(frontmatter, sort_keys=False).strip() + "\n---\n"
    path.write_text(fm + body)


def test_build_index_walks_both_roots(tmp_path: Path) -> None:
    daemon_root = tmp_path / "distillations"
    dct_root = tmp_path / "dct-distillations"
    _write(daemon_root / "Topic A" / "2026-04-20-1530-something.md",
           {"title": "Something", "topic_key": "1:2"})
    _write(dct_root / "claude-code" / "abcd-1234.md",
           {"title": "Other", "concepts": ["foo", "bar"], "gist": "A short gist"})

    idx = build_index(roots=[daemon_root, dct_root], include_ineligible=True)

    assert "2026-04-20-1530-something" in idx
    assert "abcd-1234" in idx
    assert idx["abcd-1234"].title == "Other"
    assert idx["abcd-1234"].concepts == ["foo", "bar"]
    assert idx["abcd-1234"].gist == "A short gist"


def test_build_index_handles_missing_frontmatter(tmp_path: Path) -> None:
    root = tmp_path / "distillations"
    p = root / "loose.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("just a body, no frontmatter\n")
    idx = build_index(roots=[root], include_ineligible=True)
    assert "loose" in idx
    assert idx["loose"].concepts == []
    assert idx["loose"].title == "loose"


def test_find_by_id_returns_none_when_absent(tmp_path: Path) -> None:
    idx = build_index(roots=[tmp_path / "missing"])
    assert find_by_id("nope", idx) is None


def test_date_extracted_from_frontmatter_or_filename(tmp_path: Path) -> None:
    root = tmp_path / "distillations"
    _write(root / "2026-04-21 thing.md", {"title": "Thing"})
    _write(root / "alt.md", {"title": "Alt", "date": "2026-04-22"})
    idx = build_index(roots=[root], include_ineligible=True)
    assert idx["2026-04-21 thing"].date == "2026-04-21"
    assert idx["alt"].date == "2026-04-22"


def test_eligibility_filter_default_on_and_bypass(tmp_path: Path) -> None:
    """build_index() filters by default, populates reason_counts, and the
    include_ineligible escape hatch bypasses only the gate (Codex P2 wiring)."""
    root = tmp_path / "distillations"
    prose = (
        "This session reworked the retrieval eligibility gate end to end. "
        "The team verified that live and test paths share one corpus now. "
        "Alex approved the change after reviewing the honest benchmark numbers. "
        "We confirmed no regression in the existing retrieval test suite today. "
        "The filter excludes raw transcript dumps and thin empty distillations. "
        "Codex reviewed the diff and raised two priority one findings to address. "
        "Both findings were fixed and the full retrieval suite was rerun cleanly. "
    )
    # Eligible: has concepts + real prose body.
    _write(root / "good.md", {"title": "Good", "concepts": ["alpha", "beta"]}, prose)
    # Ineligible: thin body.
    _write(root / "thin.md", {"title": "Thin", "concepts": ["x"]}, "tiny")
    # Ineligible: no concepts.
    _write(root / "noconcept.md", {"title": "NoConcept"}, prose)

    reasons: dict = {}
    idx = build_index(roots=[root], reason_counts=reasons)
    assert "good" in idx
    assert "thin" not in idx
    assert "noconcept" not in idx
    assert reasons.get("thin") == 1
    assert reasons.get("no-concepts") == 1

    full = build_index(roots=[root], include_ineligible=True)
    assert {"good", "thin", "noconcept"} <= set(full)
