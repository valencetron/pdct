"""Tests for Phase 2: preload.py archive_roots integration.

Verifies that _load_all_distilled() correctly walks vault/compaction-archive/
via config.archive_roots, that archive-format files parse the same as
distillation-format files, and that preload() injects archive gists into
today_summaries / recent_summaries.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dct.retrieval.preload import (
    _load_all_distilled,
    _reset_note_cache,
    preload,
)
from dct.retrieval.types import RetrievalConfig


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_archive_file(
    archive_root: Path,
    *,
    topic: str = "1003690648082_20196",
    compacted_at: str,
    gist: str = "One-line gist summary.",
    chunks: int = 2,
) -> Path:
    """Write a compaction_archive.py-compatible .md file under archive_root/<topic>/."""
    topic_dir = archive_root / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    slug = compacted_at.replace(":", "").replace("-", "").replace("T", "-")[:17]
    p = topic_dir / f"{slug}.md"

    chunk_text = "\n\n".join(
        f"## [CHUNK: chunk-{i}]\ncontinues_from: null\nscope: topic-{i}\n\nChunk {i} body text."
        for i in range(chunks)
    )
    # Build content without textwrap.dedent to avoid indentation issues when
    # this helper is called from an indented test function.
    lines = [
        "---",
        f"topic: '{topic}'",
        "topic_label: General",
        f"compacted_at: '{compacted_at}'",
        "source: telegram",
        "participants:",
        "- alex",
        "- orion",
        "turn_count: 20",
        "tags:",
        "- compaction",
        "- test",
        "prose_recap_words: 120",
        f'gist: "{gist}"',
        "concepts:",
        "- compaction",
        "- test",
        "---",
        "",
        "## Prose Recap",
        "",
        "## Summary",
        f"This is the prose recap body for {topic} at {compacted_at}.",
        "",
        "## Key Decisions",
        "Decision A.",
        "",
        chunk_text,
    ]
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# T1 — archive_roots: files are picked up by _load_all_distilled
# ---------------------------------------------------------------------------

def test_load_all_distilled_picks_up_archive_root(tmp_path):
    """Files in archive_roots are loaded alongside distill_root files."""
    _reset_note_cache()

    distill_root = tmp_path / "distill"
    distill_root.mkdir()
    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    _write_archive_file(archive_root, compacted_at="2026-05-20T10:00:00Z", gist="Archive gist A.")

    cfg = RetrievalConfig(
        anchor_paths=[],
        distill_root=distill_root,
        surfaces=[],
        archive_roots=[archive_root],
    )

    notes = _load_all_distilled(cfg)
    assert len(notes) == 1
    assert notes[0].gist == "Archive gist A."
    assert notes[0].distilled_at == "2026-05-20T10:00:00Z"
    assert notes[0].channel == "telegram"


def test_load_all_distilled_merges_distill_and_archive(tmp_path):
    """Notes from both distill_root and archive_root are merged and sorted newest-first."""
    _reset_note_cache()

    distill_root = tmp_path / "distill"
    distill_root.mkdir()
    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    # Write a distillation-format note
    slug_dir = distill_root / "voice-abc"
    slug_dir.mkdir()
    (slug_dir / "voice-abc.md").write_text(
        "---\n"
        "source: voice\n"
        "session_id: abc\n"
        "compacted_at: 2026-05-19T08:00:00Z\n"
        "gist: Distillation gist.\n"
        "concepts: []\n"
        "---\n\n## Summary\nOlder distillation.\n"
    )

    # Write a newer archive-format note
    _write_archive_file(archive_root, compacted_at="2026-05-20T10:00:00Z", gist="Newer archive gist.")

    cfg = RetrievalConfig(
        anchor_paths=[],
        distill_root=distill_root,
        surfaces=[],
        archive_roots=[archive_root],
    )

    notes = _load_all_distilled(cfg)
    assert len(notes) == 2
    # Sorted newest-first
    assert notes[0].gist == "Newer archive gist."
    assert notes[1].gist == "Distillation gist."


def test_load_all_distilled_missing_archive_root_is_skipped(tmp_path):
    """A non-existent archive_root does not crash — it's simply skipped."""
    _reset_note_cache()

    distill_root = tmp_path / "distill"
    distill_root.mkdir()

    cfg = RetrievalConfig(
        anchor_paths=[],
        distill_root=distill_root,
        surfaces=[],
        archive_roots=[tmp_path / "nonexistent-archive"],
    )

    notes = _load_all_distilled(cfg)
    assert notes == []


def test_load_all_distilled_cache_invalidated_by_new_archive_file(tmp_path):
    """A new archive file is picked up after the scan TTL expires.

    Contract change (2026-07-16 latency campaign): the old design promised
    immediate visibility by keying the cache on max-mtime — which meant
    every write forced a full re-read+re-parse of every note (observed
    1.9-14.4s inside the cascade). The new incremental cache serves the
    last list with zero I/O inside a 15s TTL and re-parses ONLY changed
    files on expiry. ≤15s staleness for "today/recent" context is the
    accepted trade."""
    from dct.retrieval.preload import _NOTE_LIST
    _reset_note_cache()

    distill_root = tmp_path / "distill"
    distill_root.mkdir()
    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    cfg = RetrievalConfig(
        anchor_paths=[],
        distill_root=distill_root,
        surfaces=[],
        archive_roots=[archive_root],
    )

    # First load — empty
    notes = _load_all_distilled(cfg)
    assert len(notes) == 0

    # Write a new archive file.
    _write_archive_file(archive_root, compacted_at="2026-05-21T10:00:00Z", gist="Post-write gist.")

    # Within the TTL the cached (stale) list is served with NO scan at all
    # (scanner monkeypatched to explode — Codex P2: assert the contract,
    # not just the return value).
    import dct.retrieval.preload  # the module, not the shadowing function
    import sys
    pl = sys.modules["dct.retrieval.preload"]
    real_scan = pl._scan_notes
    def _boom(cfg_):
        raise AssertionError("scan must not run inside the TTL")
    pl._scan_notes = _boom
    try:
        assert _load_all_distilled(cfg) == []
    finally:
        pl._scan_notes = real_scan

    # After TTL expiry the scan sees the new file (only it gets parsed).
    _NOTE_LIST["checked_mono"] = float("-inf")
    notes2 = _load_all_distilled(cfg)
    assert len(notes2) == 1
    assert notes2[0].gist == "Post-write gist."


# ---------------------------------------------------------------------------
# T2 — preload() injects archive gists into today_summaries / recent
# ---------------------------------------------------------------------------

def test_preload_injects_archive_gist_into_today(tmp_path, anchor_dir):
    """preload() includes today's archive files in today_summaries."""
    _reset_note_cache()

    distill_root = tmp_path / "distill"
    distill_root.mkdir()
    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    today_ts = datetime(2026, 5, 20, 10, 0, 0, tzinfo=timezone.utc)
    _write_archive_file(
        archive_root,
        compacted_at=_iso(today_ts),
        gist="Today's compaction gist.",
    )

    cfg = RetrievalConfig(
        anchor_paths=[anchor_dir / "CLAUDE.md"],
        distill_root=distill_root,
        surfaces=[],
        archive_roots=[archive_root],
    )

    bundle = preload(cfg, now=today_ts.timestamp() + 3600)
    assert "Today's compaction gist." in bundle.today_summaries


def test_preload_injects_archive_gist_into_recent(tmp_path, anchor_dir):
    """preload() includes older archive files in recent_summaries."""
    _reset_note_cache()

    distill_root = tmp_path / "distill"
    distill_root.mkdir()
    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    yesterday_ts = datetime(2026, 5, 19, 10, 0, 0, tzinfo=timezone.utc)
    _write_archive_file(
        archive_root,
        compacted_at=_iso(yesterday_ts),
        gist="Yesterday's compaction gist.",
    )

    today_ts = datetime(2026, 5, 20, 10, 0, 0, tzinfo=timezone.utc)

    cfg = RetrievalConfig(
        anchor_paths=[anchor_dir / "CLAUDE.md"],
        distill_root=distill_root,
        surfaces=[],
        archive_roots=[archive_root],
    )

    bundle = preload(cfg, now=today_ts.timestamp())
    assert "Yesterday's compaction gist." in bundle.recent_summaries.get("recent", "")


# ---------------------------------------------------------------------------
# T3 — live smoke: real archive files on disk are picked up
#
# This test is marked @pytest.mark.local — it depends on the developer's
# vault state and should only run locally, not in CI. Run explicitly with:
#   pytest -m local tests/retrieval/test_preload_archive_roots.py
# ---------------------------------------------------------------------------

@pytest.mark.local
def test_live_archive_root_smoke(tmp_path):
    """Real vault/compaction-archive/ files are loaded correctly.

    Skipped if ARCHIVE_ROOT doesn't exist (CI without a vault) or is empty.
    Uses an isolated distill_root (tmp_path) so any failures are unambiguously
    from archive parsing, not from distillation state.
    """
    from dct.retrieval.service import ARCHIVE_ROOT
    from dct.retrieval.preload import _parse_distilled

    if not ARCHIVE_ROOT.is_dir():
        pytest.skip(f"ARCHIVE_ROOT not on disk: {ARCHIVE_ROOT}")

    archive_files = list(ARCHIVE_ROOT.rglob("*.md"))
    if not archive_files:
        pytest.skip(f"ARCHIVE_ROOT is empty: {ARCHIVE_ROOT}")

    _reset_note_cache()

    # Use an empty tmp distill_root so all notes come exclusively from the archive.
    cfg = RetrievalConfig(
        anchor_paths=[],
        distill_root=tmp_path / "empty-distill",  # does not exist → treated as empty
        surfaces=[],
        archive_roots=[ARCHIVE_ROOT],
    )
    (tmp_path / "empty-distill").mkdir()

    notes = _load_all_distilled(cfg)

    # Every parseable archive file should produce a note.
    parseable = [f for f in archive_files if _parse_distilled(f) is not None]
    assert len(notes) == len(parseable), (
        f"Expected {len(parseable)} notes from {len(archive_files)} archive files, "
        f"got {len(notes)}. Silently dropped: "
        f"{set(f.stem for f in archive_files) - set(n.session_id for n in notes)}"
    )

    # Every archive note has a non-empty gist (real archives always have gist).
    notes_with_empty_gist = [n for n in notes if not n.gist]
    assert not notes_with_empty_gist, (
        f"Archive notes with empty gist (should not happen): "
        f"{[n.session_id for n in notes_with_empty_gist]}"
    )
