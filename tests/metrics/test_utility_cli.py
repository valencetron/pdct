"""Tests for `python -m dct.metrics utility`."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from dct.metrics import utility as utility_cmd


def _utility_row(
    eligible=4, matched=2,
    by_hop=None,
    skip_reason="none", days_ago=0,
    matched_concepts=None,
    turn_id="t1",
):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    if by_hop is None and skip_reason != "ablation":
        by_hop = {"1": {"eligible": eligible, "matched": matched}}
    return {
        "kind": "turn",
        "schema_version": 6,  # post node_kinds era (CLI gates pre-6 by default)
        "ts": ts,
        "turn_id": turn_id,
        "concepts_total": eligible + 1,
        "concepts_eligible": eligible,
        "concepts_matched": matched,
        "matched_concepts": matched_concepts or [],
        "by_hop": by_hop,
        "match_rate": (matched / eligible) if eligible else None,
        "pdct_skipped_reason": skip_reason,
        "shadow_or_actual": "actual" if skip_reason == "none" else "shadow",
    }


def _write(tmp_path, rows):
    p = tmp_path / "utility.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_cli_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    rc = utility_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0
    assert "n=0" in out


def test_cli_reports_arms(tmp_path, monkeypatch, capsys):
    """Mix of pdct-on (skip_reason=none) and ablation rows. CLI shows both."""
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    rows = [
        _utility_row(eligible=4, matched=2, skip_reason="none"),
        _utility_row(eligible=4, matched=3, skip_reason="none"),
        _utility_row(eligible=2, matched=1, skip_reason="ablation", by_hop=None),
    ]
    _write(tmp_path, rows)
    rc = utility_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0
    assert "PDCT-on" in out
    assert "Ablation" in out


def test_cli_reports_wilson_ci(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    rows = [
        _utility_row(eligible=10, matched=5, skip_reason="none")
        for _ in range(20)
    ]
    _write(tmp_path, rows)
    rc = utility_cmd.run(days=7)
    out = capsys.readouterr().out
    assert "[" in out and "]" in out  # CI brackets
    assert "0.5" in out or "0.50" in out  # rate ~0.5


def test_cli_omits_hop_split_for_ablation(tmp_path, monkeypatch, capsys):
    """Ablation rows have by_hop=None — must not crash when CLI tries to aggregate."""
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    rows = [
        _utility_row(eligible=4, matched=2, skip_reason="ablation", by_hop=None),
        _utility_row(eligible=4, matched=2, skip_reason="ablation", by_hop=None),
    ]
    _write(tmp_path, rows)
    rc = utility_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0
    # When PDCT-on group is empty, hop section should say n/a or skip cleanly
    assert "Ablation" in out


def test_cli_top_never_matched(tmp_path, monkeypatch, capsys):
    """Concepts that appear in many rows but never as 'matched' surface in
    a never-matched table. Stage-2 spec section "Top 10 never-matched concepts"."""
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))

    # We need to reconstruct which concepts were eligible but unmatched.
    # The utility row stores `matched_concepts` only — to track never-matched,
    # we need an injected_concepts field too, OR we approximate via aggregate.
    # For the CLI, we'll log injected_concepts on the row in Stage 2 wiring.
    rows = [
        {
            "kind": "turn",
            "schema_version": 6,  # post node_kinds era (CLI gates pre-6)
            "ts": datetime.now(timezone.utc).isoformat(),
            "turn_id": f"t{i}",
            "pdct_skipped_reason": "none",
            "concepts_total": 3,
            "concepts_eligible": 2,
            "concepts_matched": 1,
            "matched_concepts": ["phase5-control"],
            "injected_concepts": ["phase5-control", "ide-stuff", "card-cleanup"],
            "by_hop": {"1": {"eligible": 2, "matched": 1}},
            "match_rate": 0.5,
        }
        for i in range(10)
    ]
    _write(tmp_path, rows)
    rc = utility_cmd.run(days=7)
    out = capsys.readouterr().out
    # 'ide-stuff' should appear as never-matched
    assert "ide-stuff" in out or "never" in out.lower()


def test_cli_corrupt_line_tolerated(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    p = tmp_path / "utility.jsonl"
    p.write_text(
        json.dumps(_utility_row()) + "\n"
        + "garbage\n"
        + json.dumps(_utility_row()) + "\n"
    )
    rc = utility_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0
    assert "n=" in out
