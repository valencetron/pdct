"""Tests for ablation flow logic — the precedence of disabled → ablation → cascade.

These tests exercise the helper functions used by daemon.run_agent (the
real wiring is exercised by Stage 4B e2e). They live here because the
helpers are pure / live in dct.retrieval.measurement.
"""
from __future__ import annotations

import pytest

from dct.retrieval import measurement


def test_rate_zero_never_skips():
    skips = sum(
        1 for i in range(1000)
        if measurement.ablation_roll(f"t{i}", "fixed") < 0.0
    )
    assert skips == 0


def test_rate_one_always_skips():
    skips = sum(
        1 for i in range(1000)
        if measurement.ablation_roll(f"t{i}", "fixed") < 1.0
    )
    assert skips == 1000


def test_rate_quarter_uniform_within_3_sigma():
    n = 1000
    rate = 0.25
    skips = sum(
        1 for i in range(n)
        if measurement.ablation_roll(f"t{i}", "fixed-q") < rate
    )
    expected = int(n * rate)
    sigma = (n * rate * (1 - rate)) ** 0.5
    assert abs(skips - expected) < 3 * sigma, (
        f"got {skips}, expected {expected} ± {3*sigma:.1f}"
    )


def test_rate_split_reproducible_with_same_seed():
    """Same seed + same turn_id sequence → identical skip vector."""
    seed = "repro-seed"
    rate = 0.3
    a = [measurement.ablation_roll(f"turn-{i}", seed) < rate for i in range(200)]
    b = [measurement.ablation_roll(f"turn-{i}", seed) < rate for i in range(200)]
    assert a == b


def test_rate_split_changes_with_different_seed():
    rate = 0.5
    a = [measurement.ablation_roll(f"turn-{i}", "seed-A") < rate for i in range(200)]
    b = [measurement.ablation_roll(f"turn-{i}", "seed-B") < rate for i in range(200)]
    assert a != b


def test_skip_reasons_includes_all_required():
    """Sanity check the SKIP_REASONS set against spec table."""
    required = {"none", "ablation", "disabled", "error", "empty_result", "no_concepts"}
    assert required <= measurement.SKIP_REASONS
