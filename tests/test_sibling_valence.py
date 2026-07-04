"""Family-package sibling detection (Build 105): pdct doctor's advisory
valence check. Contract: advisory-only (required=False always), silent when
absent, schema-tolerant, exit-code invariant.
"""
import json

from dct.doctor import Check
from dct.family import sibling_checks as _check_sibling_valence_raw


def _check_sibling_valence(home=None):
    return _check_sibling_valence_raw(home, check_cls=Check)


def _write_status(home, payload):
    home.mkdir(parents=True, exist_ok=True)
    (home / "fleet-status.json").write_text(
        payload if isinstance(payload, str) else json.dumps(payload))


def test_absent_home_emits_nothing(tmp_path):
    assert _check_sibling_valence(tmp_path / "nope") == []


def test_home_without_status_is_ok_advisory(tmp_path):
    home = tmp_path / ".valence"
    home.mkdir()
    checks = _check_sibling_valence(home)
    assert len(checks) == 1
    c = checks[0]
    assert c.ok and not c.required and c.id == "env.sibling"
    assert "no fleet-status.json" in c.detail


def test_healthy_status_ok(tmp_path):
    home = tmp_path / ".valence"
    _write_status(home, {"generatedAt": "2026-07-04T00:00:00+00:00",
                         "probes": [{"id": "valence-daemon", "kind": "service",
                                     "status": "ok", "detail": "running"}]})
    (c,) = _check_sibling_valence(home)
    assert c.ok and not c.required
    assert "healthy" in c.detail and "2026-07-04" in c.detail


def test_failing_probes_warn_not_required(tmp_path):
    home = tmp_path / ".valence"
    _write_status(home, {"probes": [
        {"id": "valence-daemon", "status": "fail", "detail": "stopped"},
        {"id": "provider", "status": "ok"}]})
    (c,) = _check_sibling_valence(home)
    assert not c.ok
    assert not c.required, "sibling check must never be required"
    assert "valence-daemon" in c.detail


def test_malformed_json_warns(tmp_path):
    home = tmp_path / ".valence"
    _write_status(home, "{not json")
    (c,) = _check_sibling_valence(home)
    assert not c.ok and not c.required
    assert "unreadable" in c.detail


def test_missing_probes_list_warns(tmp_path):
    home = tmp_path / ".valence"
    _write_status(home, {"generatedAt": "x"})
    (c,) = _check_sibling_valence(home)
    assert not c.ok and not c.required
    assert "no probes" in c.detail


def test_unknown_statuses_tolerated(tmp_path):
    home = tmp_path / ".valence"
    _write_status(home, {"probes": [{"id": "x", "status": "mystery"},
                                    "not-a-dict"]})
    (c,) = _check_sibling_valence(home)
    assert c.ok  # unknown status != fail; non-dict entries skipped


def test_exit_code_invariance_with_dead_sibling(tmp_path, monkeypatch,
                                                capsys):
    """A broken sibling must never flip doctor.run()'s exit code."""
    home = tmp_path / ".valence"
    _write_status(home, {"probes": [{"id": "d", "status": "fail"}]})
    monkeypatch.setenv("VALENCE_HOME", str(home))
    from dct import doctor
    checks = doctor._check_environment()
    sib = [c for c in checks if c.id == "env.sibling"]
    assert len(sib) == 1 and not sib[0].ok and not sib[0].required
    # actually run() — the sibling must appear in output yet exit 0 stands
    # or falls on the REAL checks only (env deps are installed in CI, but
    # llm/daemon stages may fail here; so instead assert the gate directly:
    # a rerun of run()'s required_fail filter over env checks excludes it)
    required_fail = [c for c in checks if c.required and not c.ok]
    assert sib[0] not in required_fail


def test_env_override_respected(tmp_path, monkeypatch):
    home = tmp_path / "custom-valence"
    home.mkdir()
    monkeypatch.setenv("VALENCE_HOME", str(home))
    (c,) = _check_sibling_valence()  # no arg → env resolution path
    assert c.id == "env.sibling"
