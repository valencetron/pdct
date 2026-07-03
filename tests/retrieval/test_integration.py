"""End-to-end smoke test — cascade + preload + inject for one surface."""
from __future__ import annotations
from datetime import datetime, timezone

from dct.retrieval import cascade, preload, format_for_telegram


class _FakeGraph:
    def __init__(self, edges):
        self.edges = edges


def test_e2e_retrieval_for_telegram_turn(config, distill_root, write_distilled_fn):
    today = datetime(2026, 4, 22, 10, 0, 0, tzinfo=timezone.utc)
    write_distilled_fn(
        distill_root, "voice", "v1",
        concepts=["consciousness"],
        summary="Discussed consciousness.",
        distilled_at=today.isoformat().replace("+00:00", "Z"),
    )

    g = _FakeGraph([
        ("consciousness", "phenomenology", 4),
        ("consciousness", "memory", 2),
    ])

    bundle = preload(config, now=today.timestamp() + 3600)
    hits = cascade(
        seed_concepts=["consciousness"],
        graph=g,
        heat={},
        config=config,
    )
    prompt = format_for_telegram(bundle, hits)

    assert "Static anchor A" in prompt
    assert "Discussed consciousness" in prompt
    assert "consciousness" in prompt
    assert "phenomenology" in prompt
    assert "## Anchors" in prompt
    assert "## Today" in prompt
    assert "## Jogged (cascade)" in prompt
