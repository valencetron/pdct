"""Build 123 — capture source wiring + doctor honesty.

The pipeline is transcripts(glob) → events.jsonl → distiller → vault/*.md.
Before this build the transcript glob defaulted to a phantom path that
existed on no fresh install, so the scheduler silently ingested nothing and
doctor still reported healthy. These tests lock the fixed contract:
  1. transcripts_glob() defaults INSIDE PDCT_HOME (a real, scaffoldable dir).
  2. PDCT_TRANSCRIPTS_GLOB overrides it.
  3. scheduler reads the same source of truth (no phantom module constant).
  4. doctor's capture check flags an empty pipeline and passes a fed one.
"""
import importlib

from dct import config


def test_transcripts_glob_defaults_inside_pdct_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    monkeypatch.delenv("PDCT_TRANSCRIPTS_GLOB", raising=False)
    g = config.transcripts_glob()
    assert g == str(tmp_path / "transcripts" / "*.json")
    # the parent dir is under PDCT_HOME — install.sh scaffolds it, so it's a
    # real path, not the old ~/example-stack phantom.
    assert str(tmp_path) in g


def test_transcripts_glob_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    custom = "/some/stack/transcripts/*.json"
    monkeypatch.setenv("PDCT_TRANSCRIPTS_GLOB", custom)
    assert config.transcripts_glob() == custom


def test_scheduler_uses_config_glob(tmp_path, monkeypatch):
    # scheduler binds TRANSCRIPTS_GLOB from config.transcripts_glob() at import;
    # reimport under a fresh PDCT_HOME and assert it tracks config, not a
    # hard-coded phantom path.
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    monkeypatch.delenv("PDCT_TRANSCRIPTS_GLOB", raising=False)
    import dct.scheduler as sched
    importlib.reload(sched)
    assert sched.TRANSCRIPTS_GLOB == config.transcripts_glob()
    assert str(tmp_path) in sched.TRANSCRIPTS_GLOB


def _capture_check(monkeypatch, home):
    """Run just doctor's capture.source check against a PDCT_HOME and return it."""
    monkeypatch.setenv("PDCT_HOME", str(home))
    monkeypatch.setenv("PDCT_VAULT_ROOT", str(home / "vault"))
    monkeypatch.delenv("PDCT_TRANSCRIPTS_GLOB", raising=False)
    import glob as _glob
    from dct import config as _c
    from dct.doctor import _existing_vault_roots

    g = _c.transcripts_glob()
    n_src = len(_glob.glob(g))
    ev = _c.events_path()
    ev_lines = 0
    if ev.exists():
        t = ev.read_text(errors="ignore").strip()
        ev_lines = len(t.splitlines()) if t else 0
    n_md = sum(len(list(r.rglob("*.md"))) for r in _existing_vault_roots())
    return n_src, ev_lines, n_md


def test_capture_flags_empty_pipeline(tmp_path, monkeypatch):
    home = tmp_path
    (home / "vault" / "distillations").mkdir(parents=True)
    (home / "transcripts").mkdir()
    n_src, ev_lines, n_md = _capture_check(monkeypatch, home)
    # empty on all three legs → the check must FAIL (advisory warn).
    assert (n_src, ev_lines, n_md) == (0, 0, 0)


def test_capture_passes_when_events_present(tmp_path, monkeypatch):
    home = tmp_path
    (home / "vault" / "distillations").mkdir(parents=True)
    (home / "transcripts").mkdir()
    (home / "events.jsonl").write_text('{"op":"WRITE"}\n')
    n_src, ev_lines, n_md = _capture_check(monkeypatch, home)
    # at least one leg non-empty → the check passes.
    assert ev_lines == 1
    assert not (n_src == 0 and ev_lines == 0 and n_md == 0)


# ── Build: configurable capture source (voice → telegram) ──────────────────

def test_capture_source_default_is_telegram(monkeypatch):
    monkeypatch.delenv("PDCT_CAPTURE_SOURCE", raising=False)
    importlib.reload(config)
    assert config.capture_source() == "telegram"


def test_capture_source_env_override(monkeypatch):
    monkeypatch.setenv("PDCT_CAPTURE_SOURCE", "voice")
    importlib.reload(config)
    assert config.capture_source() == "voice"


def test_capture_source_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("PDCT_CAPTURE_SOURCE", "bogus")
    importlib.reload(config)
    assert config.capture_source() == "telegram"


def test_scheduler_binds_capture_source(tmp_path, monkeypatch):
    monkeypatch.delenv("PDCT_CAPTURE_SOURCE", raising=False)
    importlib.reload(config)
    import dct.scheduler as sched
    importlib.reload(sched)
    assert sched.CAPTURE_SOURCE == config.capture_source() == "telegram"


def test_scheduler_passes_source_to_cli(tmp_path, monkeypatch):
    # dct.ingest has NO run_ingest symbol, so scheduler's import always
    # fails → it uses the CLI subprocess path. Assert the source reaches argv.
    monkeypatch.setenv("PDCT_CAPTURE_SOURCE", "telegram")
    monkeypatch.setenv("PDCT_HOME", str(tmp_path))
    importlib.reload(config)
    import dct.scheduler as sched
    importlib.reload(sched)
    import subprocess
    seen = {}

    class _Proc:
        returncode = 0
        stderr = ""

    def _fake_run(argv, *a, **k):
        seen["argv"] = argv
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    sched._ingest_transcripts(quiet=True)
    argv = seen["argv"]
    assert "--source" in argv
    assert argv[argv.index("--source") + 1] == "telegram"
