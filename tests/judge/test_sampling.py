"""Tests for deterministic-hash turn sampling.

P1.3a uses a single fixed sampling rate (25%, configurable via env).
The decision is deterministic on turn_id so we get reproducible behavior
across runs and the same turn never flips between sampled/not-sampled.
"""
from __future__ import annotations

from dct.judge.sampling import should_sample_turn


def test_same_turn_id_same_decision() -> None:
    for tid in ("t-1", "abc", "00000000", "longer-turn-id-here"):
        assert should_sample_turn(tid, rate=0.25) == should_sample_turn(tid, rate=0.25)


def test_rate_zero_never_samples() -> None:
    for tid in ("a", "b", "c", "d", "e", "f", "g"):
        assert should_sample_turn(tid, rate=0.0) is False


def test_rate_one_always_samples() -> None:
    for tid in ("a", "b", "c", "d", "e", "f", "g"):
        assert should_sample_turn(tid, rate=1.0) is True


def test_rate_25_pct_lands_in_band_over_population() -> None:
    """Over many synthetic ids, ~25% should sample. Not exact (it's
    deterministic, not random), but should be in a reasonable band."""
    sampled = sum(
        1 for i in range(2000)
        if should_sample_turn(f"turn-{i}", rate=0.25)
    )
    pct = sampled / 2000
    assert 0.20 <= pct <= 0.30, f"expected ~25%, got {pct:.3f}"


def test_invalid_rate_clamps() -> None:
    """Defensive: out-of-range rate clamps to [0, 1] rather than crash."""
    assert should_sample_turn("t", rate=-0.5) is False
    assert should_sample_turn("t", rate=1.5) is True


def test_empty_turn_id_returns_false() -> None:
    assert should_sample_turn("", rate=1.0) is False
    assert should_sample_turn(None, rate=1.0) is False  # type: ignore[arg-type]
