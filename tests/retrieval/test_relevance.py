"""Unit tests for the relevance filter (v0)."""
from __future__ import annotations

import pytest

from dct.retrieval.relevance import RelevancePolicy, NO_OP_POLICY, _normalize_snapshot


def test_relevance_policy_default_is_no_op():
    p = RelevancePolicy()
    assert p.denied_concept_prefixes == ()
    assert p.allowed_concept_prefixes == ()
    assert p.cascade_score_floor is None
    assert p.cascade_top_k is None
    assert p.posture_hint == ""
    assert p.rule_id == ""


def test_no_op_policy_singleton_matches_default():
    assert NO_OP_POLICY == RelevancePolicy()


def test_normalize_snapshot_full():
    snap = {
        "cell_key": "sun.mid_morning",
        "activity_names": ["Family time", "Family lunch"],
        "workday_status": "Weekend",
    }
    norm = _normalize_snapshot(snap)
    assert norm["day_of_week"] == "sun"
    assert norm["time_of_day"] == "mid_morning"
    assert norm["workday_status"] == "Weekend"
    assert norm["activity_names_lower"] == ["family time", "family lunch"]
    assert norm["is_empty"] is False


def test_normalize_snapshot_missing_cell_key_still_not_empty():
    norm = _normalize_snapshot({"activity_names": ["foo"], "workday_status": "Workday"})
    assert norm["day_of_week"] == ""
    assert norm["time_of_day"] == ""
    assert norm["workday_status"] == "Workday"
    assert norm["activity_names_lower"] == ["foo"]
    assert norm["is_empty"] is False


def test_normalize_snapshot_malformed_cell_key_logs_and_falls_back():
    norm = _normalize_snapshot({"cell_key": "garbage_no_dot", "activity_names": []})
    assert norm["day_of_week"] == ""
    assert norm["time_of_day"] == ""
    assert norm["is_empty"] is True


def test_normalize_snapshot_empty_dict_marked_empty():
    norm = _normalize_snapshot({})
    assert norm == {
        "day_of_week": "",
        "time_of_day": "",
        "workday_status": "",
        "activity_names_lower": [],
        "is_empty": True,
    }


def test_normalize_snapshot_none_marked_empty():
    norm = _normalize_snapshot(None)
    assert norm["is_empty"] is True


def test_normalize_snapshot_non_string_activity_names_skipped():
    norm = _normalize_snapshot({"activity_names": ["Family time", None, 42, ""]})
    assert norm["activity_names_lower"] == ["family time"]
    assert norm["is_empty"] is False


from dct.retrieval.relevance import _rule_matches


_NORM_SUN_MORN = {
    "day_of_week": "sun",
    "time_of_day": "mid_morning",
    "workday_status": "Weekend",
    "activity_names_lower": ["family time", "family lunch"],
    "is_empty": False,
}


def test_rule_matches_single_key_dow_hit():
    rule = {"match": {"day_of_week": ["sun", "sat"]}}
    assert _rule_matches(rule, _NORM_SUN_MORN, surface="telegram") is True


def test_rule_matches_single_key_dow_miss():
    rule = {"match": {"day_of_week": ["mon"]}}
    assert _rule_matches(rule, _NORM_SUN_MORN, surface="telegram") is False


def test_rule_matches_multi_key_all_hit():
    rule = {
        "match": {
            "day_of_week": ["sun"],
            "workday_status": "Weekend",
            "activity_any_of": ["Family time"],
        },
    }
    assert _rule_matches(rule, _NORM_SUN_MORN, surface="telegram") is True


def test_rule_matches_multi_key_one_miss_fails_all():
    rule = {
        "match": {
            "day_of_week": ["sun"],
            "workday_status": "Workday",
        },
    }
    assert _rule_matches(rule, _NORM_SUN_MORN, surface="telegram") is False


def test_rule_matches_activity_any_of_case_insensitive():
    rule = {"match": {"activity_any_of": ["FAMILY LUNCH"]}}
    assert _rule_matches(rule, _NORM_SUN_MORN, surface="telegram") is True


def test_rule_matches_activity_any_of_substring():
    norm = {
        "day_of_week": "sat",
        "time_of_day": "midday",
        "workday_status": "Weekend",
        "activity_names_lower": ["family lunch with sam"],
        "is_empty": False,
    }
    rule = {"match": {"activity_any_of": ["family lunch"]}}
    assert _rule_matches(rule, norm, surface="telegram") is True


def test_rule_matches_activity_any_of_no_partial_word_collision():
    norm = {
        "day_of_week": "sat",
        "time_of_day": "midday",
        "workday_status": "Weekend",
        "activity_names_lower": ["lunch meeting"],
        "is_empty": False,
    }
    rule = {"match": {"activity_any_of": ["lunch"]}}
    assert _rule_matches(rule, norm, surface="telegram") is True


def test_rule_matches_surface_any_of_hit():
    rule = {"match": {"surface_any_of": ["voice", "telegram"]}}
    assert _rule_matches(rule, _NORM_SUN_MORN, surface="voice") is True


def test_rule_matches_surface_any_of_miss():
    rule = {"match": {"surface_any_of": ["voice"]}}
    assert _rule_matches(rule, _NORM_SUN_MORN, surface="telegram") is False


def test_rule_matches_unknown_key_does_not_match():
    rule = {"match": {"activity_class_any_of": ["personal"]}}
    assert _rule_matches(rule, _NORM_SUN_MORN, surface="telegram") is False


def test_rule_matches_empty_match_block_matches_everything():
    rule = {"match": {}}
    assert _rule_matches(rule, _NORM_SUN_MORN, surface="telegram") is True


def test_rule_matches_missing_match_block_does_not_match():
    rule = {"id": "broken", "policy": {}}
    assert _rule_matches(rule, _NORM_SUN_MORN, surface="telegram") is False


def test_rule_matches_string_value_for_listy_key_is_treated_as_one_element():
    rule = {"match": {"day_of_week": "sun"}}
    assert _rule_matches(rule, _NORM_SUN_MORN, surface="telegram") is True


import json as _json
import os as _os
from pathlib import Path as _Path

from dct.retrieval.relevance import load_rules, _RULES_CACHE


@pytest.fixture(autouse=True)
def _clear_rules_cache():
    _RULES_CACHE.clear()
    yield
    _RULES_CACHE.clear()


def test_load_rules_missing_file_returns_empty(tmp_path):
    p = tmp_path / "nope.json"
    assert load_rules(p) == []


def test_load_rules_valid_file(tmp_path):
    p = tmp_path / "rules.json"
    p.write_text(_json.dumps({
        "version": 1,
        "rules": [{"id": "r1", "match": {"day_of_week": ["sun"]}, "policy": {}}],
    }))
    rules = load_rules(p)
    assert len(rules) == 1
    assert rules[0]["id"] == "r1"


def test_load_rules_invalid_json_returns_empty(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    assert load_rules(p) == []


def test_load_rules_unexpected_shape_returns_empty(tmp_path):
    p = tmp_path / "weird.json"
    p.write_text(_json.dumps({"rules": "should be a list"}))
    assert load_rules(p) == []


def test_load_rules_caches_by_mtime_and_size(tmp_path):
    p = tmp_path / "rules.json"
    p.write_text(_json.dumps({"version": 1, "rules": [{"id": "v1", "match": {}, "policy": {}}]}))
    rules1 = load_rules(p)
    assert rules1[0]["id"] == "v1"

    mtime_ns = p.stat().st_mtime_ns
    p.write_text(_json.dumps({"version": 1, "rules": [{"id": "v2", "match": {}, "policy": {}}]}))
    _os.utime(p, ns=(mtime_ns, mtime_ns))
    rules2 = load_rules(p)
    assert rules2[0]["id"] == "v1", "should serve cached when mtime+size unchanged"

    _os.utime(p, None)
    rules3 = load_rules(p)
    assert rules3[0]["id"] == "v2", "should reload after mtime bump"


def test_load_rules_size_change_invalidates_cache(tmp_path):
    p = tmp_path / "rules.json"
    p.write_text(_json.dumps({"version": 1, "rules": [{"id": "v1", "match": {}, "policy": {}}]}))
    load_rules(p)

    p.write_text(_json.dumps({
        "version": 1,
        "rules": [
            {"id": "v1", "match": {}, "policy": {}},
            {"id": "v2-extra", "match": {}, "policy": {"posture_hint": "extra-content-here"}},
        ],
    }))
    rules2 = load_rules(p)
    assert len(rules2) == 2


def test_load_rules_skips_non_dict_rule_entries(tmp_path):
    p = tmp_path / "mixed.json"
    p.write_text(_json.dumps({
        "version": 1,
        "rules": [
            {"id": "good", "match": {}, "policy": {}},
            "garbage",
            42,
            None,
        ],
    }))
    rules = load_rules(p)
    assert len(rules) == 1
    assert rules[0]["id"] == "good"


from dct.retrieval.relevance import resolve_policy


_RULES_THREE = [
    {
        "id": "weekend-personal",
        "match": {"day_of_week": ["sat", "sun"], "activity_any_of": ["Family time"]},
        "policy": {
            "denied_concept_prefixes": ["exampleco-labs-buildout"],
            "cascade_score_floor": 0.20,
            "cascade_top_k": 20,
            "posture_hint": "weekend-personal",
        },
    },
    {
        "id": "deep-work",
        "match": {"workday_status": "Workday", "activity_any_of": ["Akshay setup"]},
        "policy": {
            "cascade_score_floor": 0.05,
            "cascade_top_k": 60,
            "posture_hint": "deep-work-engaged",
        },
    },
    {
        "id": "fallback",
        "match": {},
        "policy": {"posture_hint": "default"},
    },
]


def test_resolve_policy_first_match_wins():
    snap = {
        "cell_key": "sun.morning",
        "activity_names": ["Family time"],
        "workday_status": "Weekend",
    }
    p = resolve_policy(snap, surface="telegram", rules=_RULES_THREE)
    assert p.rule_id == "weekend-personal"
    assert p.denied_concept_prefixes == ("exampleco-labs-buildout",)
    assert p.cascade_score_floor == 0.20
    assert p.cascade_top_k == 20
    assert p.posture_hint == "weekend-personal"


def test_resolve_policy_falls_through_to_catchall():
    snap = {
        "cell_key": "tue.midday",
        "activity_names": ["Lunch"],
        "workday_status": "Workday",
    }
    p = resolve_policy(snap, surface="telegram", rules=_RULES_THREE)
    assert p.rule_id == "fallback"
    assert p.posture_hint == "default"


def test_resolve_policy_no_rules_returns_no_op():
    snap = {"cell_key": "sun.morning", "activity_names": ["Family time"]}
    p = resolve_policy(snap, surface="telegram", rules=[])
    assert p == NO_OP_POLICY


def test_resolve_policy_empty_snapshot_returns_no_op_strict():
    p = resolve_policy({}, surface="telegram", rules=_RULES_THREE)
    assert p == NO_OP_POLICY


def test_resolve_policy_none_snapshot_returns_no_op_strict():
    p = resolve_policy(None, surface="telegram", rules=_RULES_THREE)
    assert p == NO_OP_POLICY


def test_resolve_policy_partial_snapshot_can_match_catchall():
    snap = {"workday_status": "Workday"}
    p = resolve_policy(snap, surface="telegram", rules=_RULES_THREE)
    assert p.rule_id == "fallback"


def test_resolve_policy_skips_rule_with_invalid_policy_block():
    rules = [
        {"id": "broken", "match": {}, "policy": "not a dict"},
        {"id": "good", "match": {}, "policy": {"posture_hint": "ok"}},
    ]
    p = resolve_policy({"workday_status": "Workday"}, surface="telegram", rules=rules)
    assert p.rule_id == "good"


def test_resolve_policy_handler_exception_skips_rule():
    bad_rule = {"id": "explosive", "match": {"day_of_week": 12345}, "policy": {}}
    good_rule = {"id": "good", "match": {}, "policy": {"posture_hint": "fallback"}}
    p = resolve_policy({"cell_key": "sun.morning"}, surface="telegram",
                       rules=[bad_rule, good_rule])
    assert p.rule_id == "good"


from dct.retrieval.relevance import apply_policy
from dct.retrieval.types import ConceptHit


def _hit(concept: str, hop: int = 1, score: float = 0.5) -> ConceptHit:
    return ConceptHit(
        concept=concept,
        score=score,
        source_slug="",
        snippet="",
        hop=hop,
        path=[],
    )


def test_apply_policy_no_op_passes_through():
    hits = [_hit("trading"), _hit("buildout"), _hit("family")]
    filtered, dropped, top_k_eff, floor_eff = apply_policy(
        hits, NO_OP_POLICY, base_top_k=40, base_score_floor=0.10,
    )
    assert len(filtered) == 3
    assert dropped == 0
    assert top_k_eff == 40
    assert floor_eff == 0.10


def test_apply_policy_deny_list_drops_matching_prefix():
    hits = [
        _hit("exampleco-labs-buildout-power", hop=1),
        _hit("exampleco-labs-buildout-permits", hop=2),
        _hit("family-time", hop=1),
    ]
    pol = RelevancePolicy(denied_concept_prefixes=("exampleco-labs-buildout",))
    filtered, dropped, _, _ = apply_policy(hits, pol, base_top_k=40, base_score_floor=0.10)
    assert {h.concept for h in filtered} == {"family-time"}
    assert dropped == 2


def test_apply_policy_seed_immune_to_deny_list():
    hits = [
        _hit("exampleco-labs-buildout-power", hop=0),
        _hit("exampleco-labs-buildout-permits", hop=2),
    ]
    pol = RelevancePolicy(denied_concept_prefixes=("exampleco-labs-buildout",))
    filtered, dropped, _, _ = apply_policy(hits, pol, base_top_k=40, base_score_floor=0.10)
    assert len(filtered) == 1
    assert filtered[0].concept == "exampleco-labs-buildout-power"
    assert dropped == 1


def test_apply_policy_allow_list_keeps_only_matching():
    hits = [_hit("buildout-x"), _hit("trading-y"), _hit("personal-z")]
    pol = RelevancePolicy(allowed_concept_prefixes=("personal",))
    filtered, dropped, _, _ = apply_policy(hits, pol, base_top_k=40, base_score_floor=0.10)
    assert {h.concept for h in filtered} == {"personal-z"}
    assert dropped == 2


def test_apply_policy_allow_list_keeps_seeds_regardless():
    hits = [_hit("buildout-x", hop=0), _hit("personal-y", hop=1)]
    pol = RelevancePolicy(allowed_concept_prefixes=("personal",))
    filtered, _, _, _ = apply_policy(hits, pol, base_top_k=40, base_score_floor=0.10)
    assert {h.concept for h in filtered} == {"buildout-x", "personal-y"}


def test_apply_policy_allow_list_collapse_safety_keeps_top_seeds():
    hits = [_hit("foo", hop=1), _hit("bar", hop=2)]
    pol = RelevancePolicy(allowed_concept_prefixes=("nonexistent-prefix",))
    filtered, dropped, _, _ = apply_policy(hits, pol, base_top_k=40, base_score_floor=0.10)
    assert filtered == []
    assert dropped == 2


def test_apply_policy_score_floor_override_applied():
    pol = RelevancePolicy(cascade_score_floor=0.5)
    _, _, _, floor_eff = apply_policy([], pol, base_top_k=40, base_score_floor=0.10)
    assert floor_eff == 0.5


def test_apply_policy_top_k_override_applied():
    pol = RelevancePolicy(cascade_top_k=20)
    _, _, top_k_eff, _ = apply_policy([], pol, base_top_k=40, base_score_floor=0.10)
    assert top_k_eff == 20


def test_apply_policy_top_k_override_clamped_to_seed_count():
    seeds = [_hit(f"s{i}", hop=0) for i in range(5)]
    pol = RelevancePolicy(cascade_top_k=2)
    _, _, top_k_eff, _ = apply_policy(seeds, pol, base_top_k=40, base_score_floor=0.10)
    assert top_k_eff == 5


def test_apply_policy_deny_then_allow_both_applied():
    hits = [
        _hit("personal-good"),
        _hit("personal-bad"),
        _hit("work-x"),
    ]
    pol = RelevancePolicy(
        denied_concept_prefixes=("personal-bad",),
        allowed_concept_prefixes=("personal",),
    )
    filtered, dropped, _, _ = apply_policy(hits, pol, base_top_k=40, base_score_floor=0.10)
    assert {h.concept for h in filtered} == {"personal-good"}
    assert dropped == 2
