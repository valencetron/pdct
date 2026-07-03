"""Tests for dct.retrieval.measurement — pure helpers for PDCT prelim metrics.

Stage 0 of the prelim-metrics build. See:
  docs/superpowers/plans/2026-04-29-pdct-prelim-metrics-plan.md
"""
from __future__ import annotations

import os
import statistics

import pytest

from dct.retrieval import measurement


# ──────────────────────────────────────────────────────────────────────
# turn_id_from
# ──────────────────────────────────────────────────────────────────────

def test_turn_id_format_is_pipe_delimited_4_parts():
    tid = measurement.turn_id_from("-100", "0", 42, 1777529000123)
    assert tid == "-100|0|42|1777529000123"
    assert tid.count("|") == 3


def test_turn_id_unique_under_restart_simulation():
    """Same (chat,thread,turn_index) but different started_at_ms → distinct ids."""
    a = measurement.turn_id_from("c", "t", 0, 1_000_000_000_000)
    b = measurement.turn_id_from("c", "t", 0, 1_000_000_000_001)
    assert a != b


def test_turn_id_handles_int_chat_id():
    """chat_id arrives from JSON as int sometimes; must coerce cleanly."""
    tid = measurement.turn_id_from(0, 0, 0, 1)
    assert tid == "0|0|0|1"


def test_turn_id_handles_none_thread():
    """topic_id can be None for top-level chats."""
    tid = measurement.turn_id_from("c", None, 0, 1)
    assert "|None|" in tid or "||" in tid  # either form acceptable, non-crash matters


# ──────────────────────────────────────────────────────────────────────
# ablation_roll
# ──────────────────────────────────────────────────────────────────────

def test_ablation_roll_in_unit_interval():
    r = measurement.ablation_roll("anything", "seed-x")
    assert 0.0 <= r < 1.0


def test_ablation_roll_deterministic_same_inputs():
    a = measurement.ablation_roll("turn-1", "seed-x")
    b = measurement.ablation_roll("turn-1", "seed-x")
    assert a == b


def test_ablation_roll_different_for_different_turns():
    a = measurement.ablation_roll("turn-1", "seed-x")
    b = measurement.ablation_roll("turn-2", "seed-x")
    assert a != b


def test_ablation_roll_different_for_different_seeds():
    a = measurement.ablation_roll("turn-1", "seed-a")
    b = measurement.ablation_roll("turn-1", "seed-b")
    assert a != b


def test_ablation_roll_no_seed_uses_env(monkeypatch):
    monkeypatch.setenv("PDCT_ABLATION_SEED", "from-env")
    a = measurement.ablation_roll("turn-1", None)
    monkeypatch.setenv("PDCT_ABLATION_SEED", "from-env")
    b = measurement.ablation_roll("turn-1", None)
    assert a == b


def test_ablation_roll_no_seed_no_env_still_deterministic(monkeypatch):
    """Even with no seed and no env, same turn_id is reproducible
    (uses empty-string seed)."""
    monkeypatch.delenv("PDCT_ABLATION_SEED", raising=False)
    a = measurement.ablation_roll("turn-1", None)
    b = measurement.ablation_roll("turn-1", None)
    assert a == b


def test_ablation_roll_uniformity_within_3_sigma():
    """10k draws at fixed seed should split ~50/50 around 0.5.
    Allow 3σ on a binomial with p=0.5, n=10000 → σ=50, so within ~150."""
    seed = "uniform-test"
    n = 10000
    below_half = sum(
        1 for i in range(n)
        if measurement.ablation_roll(f"t{i}", seed) < 0.5
    )
    expected = n // 2
    sigma = (n * 0.25) ** 0.5
    assert abs(below_half - expected) < 3 * sigma, (
        f"got {below_half}, expected {expected} ± {3*sigma}"
    )


def test_ablation_roll_at_rate_zero_never_skips():
    """If we treat (roll < rate) as the skip condition, rate=0 → never skip."""
    rate = 0.0
    skips = sum(
        1 for i in range(1000)
        if measurement.ablation_roll(f"t{i}", "s") < rate
    )
    assert skips == 0


def test_ablation_roll_at_rate_one_always_skips():
    """rate=1 → always skip."""
    rate = 1.0
    skips = sum(
        1 for i in range(1000)
        if measurement.ablation_roll(f"t{i}", "s") < rate
    )
    assert skips == 1000


# ──────────────────────────────────────────────────────────────────────
# SKIP_REASONS constant set
# ──────────────────────────────────────────────────────────────────────

def test_skip_reasons_exact_set():
    expected = {"none", "ablation", "disabled", "error", "empty_result",
                "no_concepts", "cascade_timeout"}
    assert measurement.SKIP_REASONS == expected


def test_skip_reasons_immutable():
    """Caller code must not be able to add to it."""
    with pytest.raises((AttributeError, TypeError)):
        measurement.SKIP_REASONS.add("hacked")  # type: ignore


# ──────────────────────────────────────────────────────────────────────
# get_logs_dir
# ──────────────────────────────────────────────────────────────────────

def test_get_logs_dir_default(monkeypatch):
    monkeypatch.delenv("PDCT_LOGS_DIR", raising=False)
    p = measurement.get_logs_dir()
    assert str(p).endswith("/logs")


def test_get_logs_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path))
    assert measurement.get_logs_dir() == tmp_path


def test_get_logs_dir_function_not_constant(monkeypatch, tmp_path):
    """Re-reads env at call time (so test monkeypatch works)."""
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path / "a"))
    a = measurement.get_logs_dir()
    monkeypatch.setenv("PDCT_LOGS_DIR", str(tmp_path / "b"))
    b = measurement.get_logs_dir()
    assert a != b


# ──────────────────────────────────────────────────────────────────────
# _append_jsonl
# ──────────────────────────────────────────────────────────────────────

def test_append_jsonl_creates_file(tmp_path):
    p = tmp_path / "logs" / "test.jsonl"
    measurement._append_jsonl(p, {"k": 1})
    assert p.exists()
    content = p.read_text().strip()
    assert content == '{"k": 1}'


def test_append_jsonl_appends_newline_per_row(tmp_path):
    p = tmp_path / "test.jsonl"
    measurement._append_jsonl(p, {"a": 1})
    measurement._append_jsonl(p, {"b": 2})
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert '"a": 1' in lines[0]
    assert '"b": 2' in lines[1]


def test_append_jsonl_handles_unicode(tmp_path):
    p = tmp_path / "test.jsonl"
    measurement._append_jsonl(p, {"text": "▸ Trace"})
    assert "▸" in p.read_text()


def test_append_jsonl_never_raises_on_unwritable(tmp_path):
    """Best-effort. Daemon hot path; must not break a turn."""
    p = tmp_path / "nonexistent" / "deeply" / "nested" / "test.jsonl"
    # Should auto-create parents
    measurement._append_jsonl(p, {"k": 1})
    assert p.exists()
