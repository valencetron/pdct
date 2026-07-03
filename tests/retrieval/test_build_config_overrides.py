import json

from dct.retrieval import service
from dct.retrieval import overrides as ov


def test_build_config_applies_override_without_restart(tmp_path, monkeypatch):
    p = tmp_path / "pdct-overrides.json"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", str(p))
    monkeypatch.setattr(service, "OVERRIDES_PATH", str(p), raising=False)

    # no file -> default
    cfg1 = service.build_config()
    assert cfg1.cascade_score_floor == 0.10

    # write override -> next build_config reflects it (no process restart)
    p.write_text(json.dumps({"cascade_score_floor": 0.25, "cascade_top_k": 40}))
    cfg2 = service.build_config()
    assert cfg2.cascade_score_floor == 0.25
    assert cfg2.cascade_top_k == 40


def test_build_config_clamps_dangerous_override(tmp_path, monkeypatch):
    p = tmp_path / "pdct-overrides.json"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", str(p))
    monkeypatch.setattr(service, "OVERRIDES_PATH", str(p), raising=False)
    p.write_text(json.dumps({"cascade_score_floor": 0.99, "cascade_top_k": 0}))
    cfg = service.build_config()
    assert cfg.cascade_score_floor == 0.5   # clamped
    assert cfg.cascade_top_k == 1           # clamped


def test_build_config_corrupt_override_uses_defaults(tmp_path, monkeypatch):
    p = tmp_path / "pdct-overrides.json"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", str(p))
    monkeypatch.setattr(service, "OVERRIDES_PATH", str(p), raising=False)
    p.write_text("{ broken")
    cfg = service.build_config()  # must not raise
    assert cfg.cascade_score_floor == 0.10


def test_build_config_applies_traversal_core_overrides(tmp_path, monkeypatch):
    # Codex R6 #4: clear ambient DCT_* env so defaults aren't masked by a value
    # exported in the shell/daemon.
    for spec in ov.LEVER_SPEC.values():
        if spec.get("env"):
            monkeypatch.delenv(spec["env"], raising=False)
    p = tmp_path / "pdct-overrides.json"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", str(p))
    monkeypatch.setattr(service, "OVERRIDES_PATH", str(p), raising=False)

    # defaults hold when no file
    cfg0 = service.build_config()
    assert cfg0.cascade_decay == 0.3
    assert cfg0.cascade_depth == 2

    # written override reaches the live config (no restart)
    p.write_text(json.dumps({"cascade_decay": 0.6, "cascade_depth": 3}))
    cfg1 = service.build_config()
    assert cfg1.cascade_decay == 0.6
    assert cfg1.cascade_depth == 3


import pytest
from dct.retrieval.overrides import LEVER_SPEC


def _non_default_value(spec):
    """Pick an in-bounds value different from the default."""
    t = spec["type"]
    if t == "bool":
        return not spec["default"]
    if t == "int":
        cand = spec["default"] + 1
        if cand > spec["max"]:
            cand = spec["default"] - 1
        return cand
    cand = round((spec["min"] + spec["max"]) / 2, 4)
    if cand == spec["default"]:
        cand = round(cand + (spec["max"] - cand) / 2, 4)
    return cand


@pytest.mark.parametrize("lever", sorted(LEVER_SPEC.keys()))
def test_every_lever_reaches_live_config(lever, tmp_path, monkeypatch):
    """GUARD: a lever in LEVER_SPEC must actually change build_config()'s output.
    Catches the silent-no-op trap where a spec key has no build_config wire-up."""
    # Codex R6 #4: clear ALL ambient DCT_* env first.
    for s in LEVER_SPEC.values():
        if s.get("env"):
            monkeypatch.delenv(s["env"], raising=False)
    p = tmp_path / "pdct-overrides.json"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", str(p))
    monkeypatch.setattr(service, "OVERRIDES_PATH", str(p), raising=False)

    spec = LEVER_SPEC[lever]
    val = _non_default_value(spec)
    p.write_text(json.dumps({lever: val}))
    cfg = service.build_config()
    assert hasattr(cfg, lever), f"{lever} is not a RetrievalConfig field"
    assert getattr(cfg, lever) == val, (
        f"{lever} written as {val} but live config shows {getattr(cfg, lever)} "
        f"- silent no-op: missing build_config wire-up"
    )


@pytest.mark.parametrize("lever", sorted(LEVER_SPEC.keys()))
def test_every_lever_env_reaches_live_config(lever, tmp_path, monkeypatch):
    """GUARD: spec['env'] must be the env var build_config actually reads."""
    p = tmp_path / "pdct-overrides.json"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", str(p))
    monkeypatch.setattr(service, "OVERRIDES_PATH", str(p), raising=False)
    for s in LEVER_SPEC.values():
        if s.get("env"):
            monkeypatch.delenv(s["env"], raising=False)

    spec = LEVER_SPEC[lever]
    val = _non_default_value(spec)
    env_str = "true" if val is True else "false" if val is False else str(val)
    monkeypatch.setenv(spec["env"], env_str)
    cfg = service.build_config()
    assert getattr(cfg, lever) == val, (
        f"{lever}: env {spec['env']}={env_str} did not reach config "
        f"(got {getattr(cfg, lever)}) - wrong env mapping"
    )


# ── Build #60: integration — override file -> build_config -> cascade behavior ──
from dct.retrieval.cascade import cascade
from tests.retrieval._graph_helpers import chain_graph


def test_depth_override_flows_through_build_config_to_cascade(tmp_path, monkeypatch):
    """Writing cascade_depth to the override file changes traversal reach via the
    full override -> build_config -> cascade path."""
    for s in LEVER_SPEC.values():
        if s.get("env"):
            monkeypatch.delenv(s["env"], raising=False)
    p = tmp_path / "pdct-overrides.json"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", str(p))
    monkeypatch.setattr(service, "OVERRIDES_PATH", str(p), raising=False)
    g = chain_graph(["a", "b", "c"])  # c is hop-2

    p.write_text(json.dumps({"cascade_depth": 1, "cascade_score_floor": 0.0}))
    cfg1 = service.build_config()
    p.write_text(json.dumps({"cascade_depth": 2, "cascade_score_floor": 0.0}))
    cfg2 = service.build_config()
    reach1 = {h.concept for h in cascade(seed_concepts=["a"], graph=g, heat={}, config=cfg1)}
    reach2 = {h.concept for h in cascade(seed_concepts=["a"], graph=g, heat={}, config=cfg2)}
    assert "c" not in reach1
    assert "c" in reach2


def test_decay_override_flows_through_build_config_to_cascade(tmp_path, monkeypatch):
    """Writing cascade_decay to the override file changes hop-2 scores in a real
    cascade() call."""
    for s in LEVER_SPEC.values():
        if s.get("env"):
            monkeypatch.delenv(s["env"], raising=False)
    p = tmp_path / "pdct-overrides.json"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", str(p))
    monkeypatch.setattr(service, "OVERRIDES_PATH", str(p), raising=False)
    g = chain_graph(["a", "b", "c"])

    p.write_text(json.dumps({"cascade_decay": 0.8, "cascade_depth": 3, "cascade_score_floor": 0.0}))
    cfg_hi = service.build_config()
    p.write_text(json.dumps({"cascade_decay": 0.2, "cascade_depth": 3, "cascade_score_floor": 0.0}))
    cfg_lo = service.build_config()
    c_hi = next(h.score for h in cascade(seed_concepts=["a"], graph=g, heat={}, config=cfg_hi) if h.concept == "c")
    c_lo = next(h.score for h in cascade(seed_concepts=["a"], graph=g, heat={}, config=cfg_lo) if h.concept == "c")
    assert c_lo < c_hi


def test_transitions_enabled_override_flows_through_build_config(tmp_path, monkeypatch):
    """The bool lever reaches the live config via the override file (panel/runtime
    control path — NOT grid-swept, by design)."""
    for s in LEVER_SPEC.values():
        if s.get("env"):
            monkeypatch.delenv(s["env"], raising=False)
    p = tmp_path / "pdct-overrides.json"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", str(p))
    monkeypatch.setattr(service, "OVERRIDES_PATH", str(p), raising=False)
    p.write_text(json.dumps({"cascade_transitions_enabled": False}))
    cfg = service.build_config()
    assert cfg.cascade_transitions_enabled is False


def test_env_traversal_core_values_are_clamped(tmp_path, monkeypatch):
    """Codex diff-audit P1: env-derived cascade_depth/decay must clamp through
    LEVER_SPEC bounds, NOT pass raw. A stray DCT_CASCADE_DEPTH=999 would blow up
    range(1, depth+1) in cascade()."""
    p = tmp_path / "pdct-overrides.json"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", str(p))
    monkeypatch.setattr(service, "OVERRIDES_PATH", str(p), raising=False)
    monkeypatch.setenv("DCT_CASCADE_DEPTH", "999")
    monkeypatch.setenv("DCT_CASCADE_DECAY", "99")
    cfg = service.build_config()
    assert cfg.cascade_depth == 4      # clamped to max
    assert cfg.cascade_decay == 0.8    # clamped to max
    # below-min too
    monkeypatch.setenv("DCT_CASCADE_DEPTH", "0")
    monkeypatch.setenv("DCT_CASCADE_DECAY", "0.001")
    cfg2 = service.build_config()
    assert cfg2.cascade_depth == 1
    assert cfg2.cascade_decay == 0.1
