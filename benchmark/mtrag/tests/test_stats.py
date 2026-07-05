from benchmark.mtrag import stats


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [0.0, 0.0, 1.0, 1.0, 1.0]  # mean 0.6
    m, lo, hi = stats.bootstrap_ci(vals, n_boot=2000, seed=0)
    assert abs(m - 0.6) < 1e-9
    assert lo <= m <= hi
    # deterministic given seed
    assert (m, lo, hi) == stats.bootstrap_ci(vals, n_boot=2000, seed=0)


def test_bootstrap_ci_empty():
    assert stats.bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_paired_delta_detects_clear_separation():
    a = [1.0] * 30
    b = [0.0] * 30
    d = stats.paired_bootstrap_delta(a, b, n_boot=2000, seed=0)
    assert d["delta"] == 1.0
    assert d["significant"] is True
    assert d["lo"] > 0


def test_paired_delta_null_when_identical():
    a = [0.3, 0.7, 0.1, 0.9, 0.5] * 6
    d = stats.paired_bootstrap_delta(a, a, n_boot=2000, seed=0)
    assert d["delta"] == 0.0
    assert d["significant"] is False


def test_paired_delta_requires_equal_length():
    import pytest
    with pytest.raises(ValueError):
        stats.paired_bootstrap_delta([1, 2, 3], [1, 2], seed=0)


def test_paired_p_is_permutation_not_bootstrap():
    # all-positive diffs: a valid sign-flip permutation p must be > 0
    # (the old bootstrap-centered p returned exactly 0 here — invalid).
    a = [0.6, 0.7, 0.65, 0.55, 0.62] * 6
    b = [0.5, 0.6, 0.55, 0.45, 0.52] * 6
    d = stats.paired_bootstrap_delta(a, b, n_boot=2000, seed=0)
    assert d["delta"] > 0
    assert 0.0 < d["p"] <= 1.0
    assert d["significant"] is True  # CI still excludes 0


def test_unpaired_p_label_permutation_identical_groups():
    a = [0.5, 0.5, 0.5, 0.5]
    b = [0.5, 0.5, 0.5, 0.5]
    d = stats.unpaired_bootstrap_delta(a, b, n_boot=1000, seed=0)
    assert d["delta"] == 0.0
    assert d["p"] == 1.0  # every permutation ties the observed 0


def test_unpaired_delta_unequal_groups_clear_separation():
    a = [1.0] * 40
    b = [0.0] * 25  # different size on purpose
    d = stats.unpaired_bootstrap_delta(a, b, n_boot=2000, seed=0)
    assert d["delta"] == 1.0
    assert d["significant"] is True
    assert d["n_a"] == 40 and d["n_b"] == 25


def test_unpaired_delta_overlapping_is_insignificant():
    a = [0.5, 0.6, 0.4, 0.55, 0.45] * 8
    b = [0.52, 0.48, 0.5, 0.51, 0.49] * 8
    d = stats.unpaired_bootstrap_delta(a, b, n_boot=2000, seed=0)
    assert d["significant"] is False
