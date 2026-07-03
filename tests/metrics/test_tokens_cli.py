"""Tests for `python -m dct.metrics tokens`.

Reads logs/measurement.jsonl and prints aggregates. Tolerates corrupt
last lines.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from dct.metrics import tokens as tokens_cmd


def _measurement_row(
    chars=4000,
    anchor=3000,
    retrieval=1000,
    cascade_latency=50,
    output_chars=800,
    days_ago=0,
    skip_reason="none",
):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "schema_version": 1,
        "kind": "turn_measurement",
        "ts": ts,
        "turn_id": f"c|t|{int(time.time()*1e6)}|x",
        "model": "opus",
        "pdct_skipped_reason": skip_reason,
        "identity_anchor_chars": anchor,
        "retrieval_context_chars": retrieval,
        "total_injected_chars": chars,
        "total_injected_tokens_est": chars // 4,
        "prompt_total_chars": chars + 8000,
        "prompt_total_tokens_est": (chars + 8000) // 4,
        "output_chars": output_chars,
        "output_tokens_est": output_chars // 4,
        "cascade_latency_ms": cascade_latency,
        "utility_latency_ms": 1,
    }


def _write_log(tmp_path, rows):
    p = tmp_path / "measurement.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_cli_handles_empty_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    rc = tokens_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0
    assert "n=0" in out


def test_cli_handles_missing_log(tmp_path, monkeypatch, capsys):
    """No log file at all — must not crash."""
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path / "does-not-exist"))
    rc = tokens_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0
    assert "n=0" in out


def test_cli_reports_aggregate_means(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    rows = [
        _measurement_row(retrieval=1000, anchor=3000, chars=4000),
        _measurement_row(retrieval=2000, anchor=3000, chars=5000),
        _measurement_row(retrieval=3000, anchor=3000, chars=6000),
    ]
    _write_log(tmp_path, rows)
    rc = tokens_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0
    assert "n=3" in out
    # mean retrieval = 2000
    assert "2000" in out
    # mean total = 5000
    assert "5000" in out


def test_cli_filters_by_days(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    rows = [
        _measurement_row(days_ago=0),
        _measurement_row(days_ago=10),  # outside 7-day window
    ]
    _write_log(tmp_path, rows)
    rc = tokens_cmd.run(days=7)
    out = capsys.readouterr().out
    assert "n=1" in out


def test_cli_corrupt_last_line_tolerated(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    p = tmp_path / "measurement.jsonl"
    p.write_text(
        json.dumps(_measurement_row()) + "\n"
        + json.dumps(_measurement_row()) + "\n"
        + '{"schema_version":1,"kind":"turn_measure'  # truncated, no newline
    )
    rc = tokens_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0
    assert "n=2" in out


def test_cli_corrupt_middle_line_skipped(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    p = tmp_path / "measurement.jsonl"
    p.write_text(
        json.dumps(_measurement_row()) + "\n"
        + "this is garbage\n"
        + json.dumps(_measurement_row()) + "\n"
    )
    rc = tokens_cmd.run(days=7)
    out = capsys.readouterr().out
    assert rc == 0
    assert "n=2" in out


def test_cli_p50_p95(tmp_path, monkeypatch, capsys):
    """With 100 evenly-spaced rows, p50 and p95 columns should be present."""
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    rows = [_measurement_row(retrieval=i*10, chars=3000+i*10) for i in range(1, 101)]
    _write_log(tmp_path, rows)
    rc = tokens_cmd.run(days=7)
    out = capsys.readouterr().out
    assert "p50" in out
    assert "p95" in out


def test_cli_skipreason_breakdown_in_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    rows = [
        _measurement_row(skip_reason="none"),
        _measurement_row(skip_reason="none"),
        _measurement_row(skip_reason="ablation"),
        _measurement_row(skip_reason="empty_result"),
    ]
    _write_log(tmp_path, rows)
    rc = tokens_cmd.run(days=7)
    out = capsys.readouterr().out
    # token panel should at least mention one skip reason
    assert "none" in out.lower() or "ablation" in out.lower()
