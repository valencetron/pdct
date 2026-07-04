import pytest
import json

from dct.retrieval import overrides as ov


def test_lever_spec_has_all_keys():
    keys = set(ov.LEVER_SPEC.keys())
    assert keys == {
        "cascade_score_floor", "cascade_top_k", "cascade_heat_enabled",
        "cascade_heat_floor", "cascade_heat_half_life_s",
        "cascade_eligibility_filter_enabled", "cascade_transitions_bias",
        "cascade_vec_near_decay",
        "cascade_decay", "cascade_depth", "cascade_transitions_enabled",
    }


def test_lever_spec_includes_traversal_core_levers():
    from dct.retrieval.overrides import LEVER_SPEC
    assert "cascade_decay" in LEVER_SPEC
    assert "cascade_depth" in LEVER_SPEC
    d = LEVER_SPEC["cascade_decay"]
    assert d["type"] == "float" and d["default"] == 0.4
    assert d["min"] == 0.1 and d["max"] == 0.8
    assert d["env"] == "DCT_CASCADE_DECAY"
    p = LEVER_SPEC["cascade_depth"]
    assert p["type"] == "int" and p["default"] == 2
    assert p["min"] == 1 and p["max"] == 4
    assert p["env"] == "DCT_CASCADE_DEPTH"


def test_lever_spec_includes_transitions_enabled():
    from dct.retrieval.overrides import LEVER_SPEC
    assert "cascade_transitions_enabled" in LEVER_SPEC
    t = LEVER_SPEC["cascade_transitions_enabled"]
    assert t["type"] == "bool" and t["default"] is True
    assert t["min"] is None and t["max"] is None
    assert t["env"] == "DCT_TRANSITIONS_ENABLED"


def test_clamp_traversal_core_bounds():
    assert ov.clamp("cascade_decay", 0.05) == 0.1
    assert ov.clamp("cascade_decay", 0.95) == 0.8
    assert ov.clamp("cascade_decay", 0.5) == 0.5
    assert ov.clamp("cascade_depth", 0) == 1
    assert ov.clamp("cascade_depth", 9) == 4
    assert ov.clamp("cascade_depth", 3) == 3


def test_new_lever_bounds_match_scorer_param_bounds():
    """The 2 new numeric levers' bounds must equal autoresearch.scorer.PARAM_BOUNDS."""
    from dct.retrieval.overrides import LEVER_SPEC
    pytest.importorskip("autoresearch")
    from autoresearch.scorer import PARAM_BOUNDS
    for key in ("cascade_decay", "cascade_depth"):
        lo, hi = PARAM_BOUNDS[key]
        assert LEVER_SPEC[key]["min"] == lo, f"{key} min drifted from scorer.py"
        assert LEVER_SPEC[key]["max"] == hi, f"{key} max drifted from scorer.py"


def test_clamp_in_range_returns_value():
    assert ov.clamp("cascade_score_floor", 0.2) == 0.2


def test_clamp_above_max_clamps_down():
    assert ov.clamp("cascade_score_floor", 0.99) == 0.5


def test_clamp_below_min_clamps_up():
    assert ov.clamp("cascade_top_k", 0) == 1


def test_clamp_wrong_type_returns_none():
    assert ov.clamp("cascade_top_k", "abc") is None


def test_clamp_rejects_nan_and_inf():
    assert ov.clamp("cascade_score_floor", float("nan")) is None
    assert ov.clamp("cascade_score_floor", float("inf")) is None


def test_lever_spec_defaults_match_build_config(tmp_path, monkeypatch):
    # the spec table must match the engine's real defaults.
    from dct.retrieval import service
    p = tmp_path / "none.json"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", str(p))
    monkeypatch.setattr(service, "OVERRIDES_PATH", str(p), raising=False)
    for spec in ov.LEVER_SPEC.values():
        monkeypatch.delenv(spec["env"], raising=False)
    cfg = service.build_config()
    for key, spec in ov.LEVER_SPEC.items():
        assert getattr(cfg, key) == spec["default"], f"{key} default mismatch"


def test_clamp_bool_passthrough():
    assert ov.clamp("cascade_heat_enabled", False) is False


def test_clamp_int_coerces_float():
    assert ov.clamp("cascade_top_k", 30.0) == 30
    assert isinstance(ov.clamp("cascade_top_k", 30.0), int)


def test_load_overrides_missing_file_returns_empty(tmp_path):
    p = tmp_path / "nope.json"
    assert ov.load_overrides(str(p)) == {}


def test_load_overrides_corrupt_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json")
    assert ov.load_overrides(str(p)) == {}


def test_load_overrides_clamps_and_drops_unknown(tmp_path):
    p = tmp_path / "ov.json"
    p.write_text(json.dumps({
        "cascade_score_floor": 0.99,      # clamp to 0.5
        "cascade_top_k": 30,              # ok
        "bogus_key": 5,                   # dropped
        "cascade_heat_enabled": "yes",   # wrong type -> dropped
    }))
    out = ov.load_overrides(str(p))
    assert out == {"cascade_score_floor": 0.5, "cascade_top_k": 30}


def test_write_override_merges_and_clamps(tmp_path):
    p = tmp_path / "ov.json"
    ov.write_override("cascade_score_floor", 0.99, path=str(p))   # clamp -> 0.5
    ov.write_override("cascade_top_k", 40, path=str(p))            # merge
    out = ov.load_overrides(str(p))
    assert out == {"cascade_score_floor": 0.5, "cascade_top_k": 40}


def test_write_override_rejects_unknown_key(tmp_path):
    p = tmp_path / "ov.json"
    try:
        ov.write_override("bogus", 1, path=str(p))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_write_override_stamps_meta(tmp_path):
    p = tmp_path / "ov.json"
    ov.write_override("cascade_top_k", 40, path=str(p))
    meta = ov.read_meta(path=str(p))
    assert "sinceChangeAt" in meta and meta["sinceChangeAt"]


def test_reset_overrides_removes_file(tmp_path):
    p = tmp_path / "ov.json"
    ov.write_override("cascade_top_k", 40, path=str(p))
    ov.reset_overrides(path=str(p))
    assert ov.load_overrides(str(p)) == {}
