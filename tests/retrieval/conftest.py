"""Shared fixtures for retrieval tests."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from dct.events import Event, EventSource, EventOp
from dct.retrieval.types import RetrievalConfig


@pytest.fixture
def anchor_dir(tmp_path: Path) -> Path:
    d = tmp_path / "anchors"
    d.mkdir()
    (d / "CLAUDE.md").write_text("# Claude Instructions\nStatic anchor A.\n")
    (d / "soul.md").write_text("# Soul\nStatic anchor B.\n")
    return d


@pytest.fixture
def distill_root(tmp_path: Path) -> Path:
    root = tmp_path / "distill"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def config(anchor_dir: Path, distill_root: Path) -> RetrievalConfig:
    return RetrievalConfig(
        anchor_paths=[anchor_dir / "CLAUDE.md", anchor_dir / "soul.md"],
        distill_root=distill_root,
        surfaces=["voice", "claude-code", "telegram", "vault"],
    )


def write_distilled(
    distill_root: Path,
    channel: str,
    session_id: str,
    *,
    concepts: list[str],
    summary: str,
    distilled_at: str,
) -> Path:
    """Helper to write a distilled note in the slug-per-subdir layout.

    FIX (2026-05-27): Real distillations live at distill_root/<slug>/<slug>.md
    with `compacted_at` and `source` frontmatter fields. The old layout
    (surface-named subdirs, `distilled_at`, `source_channel`) never existed
    on disk and caused _load_distilled to return empty for every query.
    """
    slug_dir = distill_root / f"{channel}-{session_id}"
    slug_dir.mkdir(parents=True, exist_ok=True)
    p = slug_dir / f"{channel}-{session_id}.md"
    frontmatter = (
        "---\n"
        f"title: {session_id}\n"
        f"source: {channel}\n"
        f"session_id: {session_id}\n"
        f"compacted_at: {distilled_at}\n"
        f"concepts: {json.dumps(concepts)}\n"
        f"gist: {summary}\n"
        "---\n"
    )
    p.write_text(frontmatter + f"\n## Summary\n{summary}\n")
    return p


@pytest.fixture
def write_distilled_fn():
    """Fixture returning the helper (keeps the API together)."""
    return write_distilled


def evt(ts: float, source: EventSource, op: EventOp, concepts: list[str]) -> Event:
    return Event(ts=ts, source=source, op=op, concepts=concepts)


@pytest.fixture
def sample_events() -> list[Event]:
    """Four events across sources; all at ts=1000..1030."""
    return [
        evt(1000.0, EventSource.TELEGRAM, EventOp.READ, ["consciousness", "phenomenology"]),
        evt(1010.0, EventSource.VOICE, EventOp.READ, ["consciousness", "memory"]),
        evt(1020.0, EventSource.CLAUDE_CODE, EventOp.WRITE, ["dct", "phase7"]),
        evt(1030.0, EventSource.VAULT, EventOp.WRITE, ["dct", "phenomenology"]),
    ]


@pytest.fixture
def events_log_path(tmp_path: Path, sample_events) -> Path:
    """Write sample_events to a jsonl and return the path."""
    p = tmp_path / "events.jsonl"
    with p.open("w") as f:
        for ev in sample_events:
            f.write(json.dumps(ev.to_dict()) + "\n")
    return p
