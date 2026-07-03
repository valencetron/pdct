import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_v3", Path(__file__).resolve().parents[2] / "benchmark" / "eval_v3.py")
eval_v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_v3)
ceiling_adjusted = eval_v3.ceiling_adjusted
honest_axes = eval_v3.honest_axes
redact_exc = eval_v3.redact_exc


def test_redact_exc_drops_raw_text():
    # Codex diff r4: raw exception text can embed paths/URLs/creds; only the
    # class name may surface in durable artifacts/logs.
    e = ConnectionError("https://secret.internal/api?token=abc123 ~/x")
    out = redact_exc(e)
    assert out == "ConnectionError"
    assert "token" not in out and "secret" not in out and "/Users" not in out


def test_full_support_is_identity():
    assert ceiling_adjusted(0.8, 1.0) == 0.8


def test_partial_support_lifts_score():
    # gold doc only supports 80% of keywords; model got 0.6 of total ->
    # recovered 0.6/0.8 = 0.75 of the achievable mass.
    assert abs(ceiling_adjusted(0.6, 0.8) - 0.75) < 1e-9


def test_caps_at_one():
    assert ceiling_adjusted(0.8, 0.8) == 1.0
    assert ceiling_adjusted(0.9, 0.8) == 1.0


def test_missing_support_returns_raw():
    assert ceiling_adjusted(0.6, None) == 0.6
    assert ceiling_adjusted(0.6, 0.0) == 0.6


# --- aggregation: the exclusion logic is the whole point of the build ---


def test_honest_axes_exclusions():
    rows = [
        # positive, retrieved hit, gold supports 80% -> ceiling lifts 0.6->0.75
        {"id": "a", "score": 0.6, "retrieval_hit5": True, "is_positive": True},
        # positive, retrieved MISS -> excluded from gen-when-retrieved & ceiling
        {"id": "b", "score": 0.9, "retrieval_hit5": False, "is_positive": True},
        # positive, no gold -> excluded from recall, but counts in GEN raw
        {"id": "c", "score": 1.0, "retrieval_hit5": None, "is_positive": True},
        # NEGATIVE: grade_negative emits a 0/1 score, but is_positive=False so
        # it must be excluded from GEN raw (Codex diff #1).
        {"id": "e", "score": 1.0, "retrieval_hit5": None, "is_positive": False},
    ]
    qmap = {
        "a": {"gold_keyword_support": 0.8},
        "b": {"gold_keyword_support": 1.0},
        "c": {},
        "e": {},
    }
    ax = honest_axes(rows, qmap)
    # recall@5: only a (True) and b (False) are applicable -> 1/2
    assert abs(ax["retrieval_recall_at5"] - 0.5) < 1e-9
    assert ax["retrieval_n"] == 2
    # GEN raw: a,b,c are positive (e excluded despite score=1.0) -> (0.6+0.9+1.0)/3
    assert abs(ax["gen_raw"] - (2.5 / 3)) < 1e-9
    assert ax["gen_n"] == 3
    # GEN when retrieved@5: only a -> 0.6
    assert abs(ax["gen_when_retrieved5"] - 0.6) < 1e-9
    assert ax["gen_retrieved_n"] == 1
    # ceiling-adjusted on a only: 0.6/0.8 = 0.75
    assert abs(ax["gen_when_retrieved5_ceiling_adj"] - 0.75) < 1e-9
    # unsupported: only a (0.8 < 0.999); b is 1.0
    assert ax["benchmark_unsupported"] == 1


def test_zero_support_counts_as_unsupported():
    # Codex diff #4: support==0.0 is the worst possible cap, NOT a missing
    # annotation. The `or 1.0` bug would have hidden it.
    rows = [{"id": "z", "score": 0.0, "retrieval_hit5": True, "is_positive": True}]
    qmap = {"z": {"gold_keyword_support": 0.0}}
    ax = honest_axes(rows, qmap)
    assert ax["benchmark_unsupported"] == 1
    # missing annotation (None) must NOT count as unsupported
    ax2 = honest_axes(rows, {"z": {}})
    assert ax2["benchmark_unsupported"] == 0


def test_support_boundary_0999_counts():
    # Codex diff r2 #2: 0.999 is < 1.0 so it IS capped; 1.0 is not.
    rows = [{"id": "z", "score": 0.5, "retrieval_hit5": True, "is_positive": True}]
    assert honest_axes(rows, {"z": {"gold_keyword_support": 0.999}})["benchmark_unsupported"] == 1
    assert honest_axes(rows, {"z": {"gold_keyword_support": 1.0}})["benchmark_unsupported"] == 0


def test_is_positive_backward_compat_falls_back_to_category():
    # Codex diff r2 #1: legacy rows without is_positive use category.
    rows = [
        {"id": "a", "score": 0.6, "retrieval_hit5": True, "category": "factual-recall"},
        {"id": "n", "score": 1.0, "retrieval_hit5": None, "category": "negative"},
    ]
    ax = honest_axes(rows, {"a": {}, "n": {}})
    assert ax["gen_n"] == 1  # only the factual row, negative excluded
    assert abs(ax["gen_raw"] - 0.6) < 1e-9


def test_probe_errors_counted():
    rows = [
        {"id": "a", "score": 0.0, "retrieval_hit5": False, "is_positive": True,
         "retrieval_probe_error": "ConnectionError"},
        {"id": "b", "score": 0.9, "retrieval_hit5": True, "is_positive": True},
    ]
    ax = honest_axes(rows, {"a": {}, "b": {}})
    assert ax["retrieval_probe_errors"] == 1
    assert ax["retrieval_probe_errors_answerable"] == 1


def test_probe_error_counted_by_presence_even_if_empty():
    # Codex diff r3 #1: an exception with an empty message must still count.
    rows = [
        {"id": "a", "score": 0.0, "retrieval_hit5": False, "is_positive": True,
         "retrieval_probe_error": ""},  # empty but PRESENT
        # no-gold probe failure: counted in total but NOT answerable (r3 #2)
        {"id": "n", "score": 1.0, "retrieval_hit5": None, "is_positive": False,
         "retrieval_probe_error": "TimeoutError"},
    ]
    ax = honest_axes(rows, {"a": {}, "n": {}})
    assert ax["retrieval_probe_errors"] == 2
    assert ax["retrieval_probe_errors_answerable"] == 1


def test_unknown_category_excluded_from_positives():
    # Codex diff r3 #3: legacy row with missing/unknown category is NOT
    # silently treated as positive.
    rows = [
        {"id": "a", "score": 0.7, "retrieval_hit5": True},  # no category
        {"id": "b", "score": 0.5, "retrieval_hit5": True, "category": "factual-recall"},
    ]
    ax = honest_axes(rows, {"a": {}, "b": {}})
    assert ax["gen_n"] == 1
    assert abs(ax["gen_raw"] - 0.5) < 1e-9
