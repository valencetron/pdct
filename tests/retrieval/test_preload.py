"""Preload tests."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dct.retrieval.preload import (
    _load_anchors,
    _estimate_tokens,
    _load_all_distilled,
    DistilledNote,
    _split_today_and_recent,
    preload,
)
from dct.retrieval.types import RetrievalConfig, PreloadBundle


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# -- token estimator & anchors -------------------------------------------------

def test_estimate_tokens_char_quarter():
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("abcd") == 1
    assert _estimate_tokens("a" * 40) == 10


def test_load_anchors_concatenates(config: RetrievalConfig):
    text, tokens = _load_anchors(config)
    assert "Static anchor A" in text
    assert "Static anchor B" in text
    assert tokens > 0


def test_load_anchors_respects_token_cap(config: RetrievalConfig, anchor_dir: Path):
    (anchor_dir / "CLAUDE.md").write_text("X" * 40_000)
    capped = RetrievalConfig(
        anchor_paths=config.anchor_paths,
        distill_root=config.distill_root,
        surfaces=config.surfaces,
        preload_anchor_cap=100,
    )
    text, tokens = _load_anchors(capped)
    assert tokens <= 100


# -- distilled loader ---------------------------------------------------------
# FIX (2026-05-27): _load_distilled(channel=) replaced with _load_all_distilled().
# Distillations live at distill_root/<slug>/<slug>.md — no surface subdirs.

def test_load_all_distilled_parses_frontmatter(config, distill_root, write_distilled_fn):
    write_distilled_fn(
        distill_root,
        channel="voice",
        session_id="abc",
        concepts=["consciousness", "memory"],
        summary="Talked about consciousness and memory.",
        distilled_at="2026-04-22T14:30:00Z",
    )
    notes = _load_all_distilled(config)
    assert len(notes) == 1
    n = notes[0]
    assert n.session_id == "abc"
    assert n.channel == "voice"
    assert n.concepts == ["consciousness", "memory"]
    assert n.distilled_at == "2026-04-22T14:30:00Z"
    assert "consciousness and memory" in n.gist


def test_load_all_distilled_empty_root_returns_empty(config, distill_root):
    # No files written — should return empty list
    notes = _load_all_distilled(config)
    assert notes == []


def test_load_all_distilled_missing_root_returns_empty(config, tmp_path):
    cfg = RetrievalConfig(
        anchor_paths=config.anchor_paths,
        distill_root=tmp_path / "nonexistent",
        surfaces=config.surfaces,
    )
    assert _load_all_distilled(cfg) == []


# -- today/recent split --------------------------------------------------------

def test_split_today_and_recent(config, distill_root, write_distilled_fn):
    today = datetime(2026, 4, 22, 10, 0, 0, tzinfo=timezone.utc)
    yesterday = datetime(2026, 4, 21, 10, 0, 0, tzinfo=timezone.utc)
    two_days_ago = datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc)

    write_distilled_fn(distill_root, "voice", "v_today", concepts=["a"], summary="today voice", distilled_at=_iso(today))
    write_distilled_fn(distill_root, "voice", "v_yesterday", concepts=["b"], summary="yesterday voice", distilled_at=_iso(yesterday))
    write_distilled_fn(distill_root, "voice", "v_two", concepts=["c"], summary="two days voice", distilled_at=_iso(two_days_ago))
    write_distilled_fn(distill_root, "telegram", "t_today", concepts=["d"], summary="today tg", distilled_at=_iso(today))

    now = today.timestamp() + 3600
    today_notes, recent_by_surface = _split_today_and_recent(config, now=now, last_n=2)

    today_ids = {n.session_id for n in today_notes}
    assert "v_today" in today_ids
    assert "t_today" in today_ids

    # FIX: recent is now a flat "recent" bucket, not per-surface.
    recent_ids = {n.session_id for n in recent_by_surface["recent"]}
    assert "v_yesterday" in recent_ids
    assert "v_two" in recent_ids
    # No surface-split keys anymore
    assert "voice" not in recent_by_surface
    assert "claude-code" not in recent_by_surface


def test_split_respects_last_n(config, distill_root, write_distilled_fn):
    base = datetime(2026, 4, 22, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        day = datetime(2026, 4, 21 - i, 10, 0, 0, tzinfo=timezone.utc)
        write_distilled_fn(distill_root, "voice", f"v{i}", concepts=["x"], summary="s", distilled_at=_iso(day))

    today_notes, recent = _split_today_and_recent(config, now=base.timestamp(), last_n=2)
    assert len(recent["recent"]) == 2
    ids = [n.session_id for n in recent["recent"]]
    assert ids == ["v0", "v1"]


# -- public preload ------------------------------------------------------------

def test_preload_assembles_bundle(config, distill_root, write_distilled_fn):
    today = datetime(2026, 4, 22, 10, 0, 0, tzinfo=timezone.utc)
    yesterday = datetime(2026, 4, 21, 10, 0, 0, tzinfo=timezone.utc)
    write_distilled_fn(distill_root, "voice", "v_today", concepts=["a"], summary="today voice talk", distilled_at=_iso(today))
    write_distilled_fn(distill_root, "telegram", "t_yest", concepts=["b"], summary="yesterday chat", distilled_at=_iso(yesterday))

    bundle = preload(config, now=today.timestamp() + 60)
    assert isinstance(bundle, PreloadBundle)
    assert "Static anchor A" in bundle.anchors
    assert "today voice talk" in bundle.today_summaries
    # FIX: recent is now keyed "recent", not per surface
    assert "yesterday chat" in bundle.recent_summaries["recent"]
    assert bundle.total_tokens > 0


def test_preload_respects_today_cap(config, distill_root, write_distilled_fn):
    today = datetime(2026, 4, 22, 10, 0, 0, tzinfo=timezone.utc)
    write_distilled_fn(
        distill_root, "voice", "huge",
        concepts=["x"],
        summary="X" * 40_000,
        distilled_at=_iso(today),
    )
    capped = RetrievalConfig(
        anchor_paths=config.anchor_paths,
        distill_root=config.distill_root,
        surfaces=config.surfaces,
        preload_today_cap=100,
    )
    bundle = preload(capped, now=today.timestamp() + 60)
    assert _estimate_tokens(bundle.today_summaries) <= 100
