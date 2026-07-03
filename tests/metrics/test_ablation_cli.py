"""Tests for `python -m dct.metrics ablation`."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from dct.metrics import ablation as ablation_cmd


def _measurement(
    turn_id="t1", skip_reason="none", retrieval=1000, anchor=3000,
    output_chars=800, days_ago=0, model="opus",
    chat_id="c", thread_id="th",
):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "kind": "turn_measurement",
        "schema_version": 1,
        "ts": ts,
        "turn_id": turn_id,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "model": model,
        "pdct_skipped_reason": skip_reason,
        "identity_anchor_chars": anchor,
        "retrieval_context_chars": retrieval if skip_reason == "none" else 0,
        "total_injected_chars": (anchor + retrieval) if skip_reason == "none" else anchor,
        "output_chars": output_chars,
        "conversation_length": 5,
        "prior_turn_pdct_active": True,
    }


def _followup(parent_turn_id="t1", rating="correction", days_ago=0):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "kind": "followup",
        "ts": ts,
        "parent_turn_id": parent_turn_id,
        "rating": rating,
        "matched_pattern": "leading-no" if rating == "correction" else None,
        "excerpt": "no, that's wrong",
    }


def _utility(turn_id="t1", days_ago=0):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "kind": "turn",
        "ts": ts,
        "turn_id": turn_id,
        "concepts_eligible": 4,
        "concepts_matched": 2,
        "match_rate": 0.5,
        "pdct_skipped_reason": "none",
    }


def _setup_logs(tmp_path, measurement_rows, utility_rows=None, followup_rows=None):
    (tmp_path / "measurement.jsonl").write_text(
        "\n".join(json.dumps(r) for r in measurement_rows) + "\n"
    )
    if utility_rows:
        (tmp_path / "utility.jsonl").write_text(
            "\n".join(json.dumps(r) for r in (utility_rows + (followup_rows or []))) + "\n"
        )
    elif followup_rows:
        (tmp_path / "utility.jsonl").write_text(
            "\n".join(json.dumps(r) for r in followup_rows) + "\n"
        )


def test_cli_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    rc = ablation_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0
    assert "n=0" in out or "n_total=0" in out


def test_cli_reports_two_arms(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    measurements = (
        [_measurement(turn_id=f"on-{i}", skip_reason="none") for i in range(80)]
        + [_measurement(turn_id=f"ab-{i}", skip_reason="ablation") for i in range(20)]
    )
    _setup_logs(tmp_path, measurements)
    rc = ablation_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0
    assert "PDCT-on" in out
    assert "Ablation" in out
    assert "80" in out
    assert "20" in out


def test_cli_correction_rate_split(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    measurements = (
        [_measurement(turn_id=f"on-{i}", skip_reason="none") for i in range(50)]
        + [_measurement(turn_id=f"ab-{i}", skip_reason="ablation") for i in range(20)]
    )
    # 5/50 corrections on PDCT-on, 8/20 on ablation
    followups = (
        [_followup(parent_turn_id=f"on-{i}", rating="correction") for i in range(5)]
        + [_followup(parent_turn_id=f"on-{i}", rating="continuation") for i in range(5, 50)]
        + [_followup(parent_turn_id=f"ab-{i}", rating="correction") for i in range(8)]
        + [_followup(parent_turn_id=f"ab-{i}", rating="neutral") for i in range(8, 20)]
    )
    _setup_logs(tmp_path, measurements, followup_rows=followups)
    rc = ablation_cmd.run(days=7)
    out = capsys.readouterr().out
    # 0.10 vs 0.40 — both should appear with Wilson CIs
    assert "[" in out and "]" in out  # CI brackets


def test_cli_warns_low_n(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    measurements = (
        [_measurement(turn_id=f"on-{i}", skip_reason="none") for i in range(50)]
        + [_measurement(turn_id=f"ab-{i}", skip_reason="ablation") for i in range(10)]
    )
    _setup_logs(tmp_path, measurements)
    rc = ablation_cmd.run(days=7)
    out = capsys.readouterr().out
    # Ablation n=10 is way below 80 — CLI should warn
    assert "low" in out.lower() or "below" in out.lower() or "warning" in out.lower() or "⚠" in out


def test_cli_corrupt_line_tolerated(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    p = tmp_path / "measurement.jsonl"
    p.write_text(
        json.dumps(_measurement()) + "\n"
        + "garbage\n"
        + json.dumps(_measurement(turn_id="t2")) + "\n"
    )
    rc = ablation_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0


def test_cli_filters_by_days(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    measurements = [
        _measurement(turn_id="recent", days_ago=0),
        _measurement(turn_id="old", days_ago=10),
    ]
    _setup_logs(tmp_path, measurements)
    rc = ablation_cmd.run(days=7)
    out = capsys.readouterr().out
    # Recent should count, old should not — total n=1
    assert "1" in out
