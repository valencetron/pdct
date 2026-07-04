"""PDCT self-diagnosis: `python -m dct.doctor`.

Four stages, each independently reported:

  1. environment — python version, required deps, optional extras, writability
  2. configuration — resolved paths (PDCT_HOME, vault, events, runtime, logs),
     whether a vault/corpus is present, whether an API key is available
  3. functional — build a graph + index from a corpus (the bundled examples/
     corpus by default) and run smoke retrievals
  4. retrieval quality — canned synthetic questions against the example corpus;
     verifies the expected note surfaces in top-k

Exit code 0 = all required checks pass. `--json` emits machine-readable
results (for CI or a hosted web checker). By default the functional stages
run against the bundled example corpus in a TEMP data dir — never against
the operator's live installation — unless `--live` is passed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REQUIRED_DEPS = ["yaml", "watchdog", "sklearn"]
OPTIONAL_DEPS = {
    "anthropic": "anthropic provider (pip install 'dct[anthropic]') — distiller & judge",
    "sentence_transformers": "embeddings extra (pip install 'dct[embeddings]') — VEC_NEAR edges",
    "numpy": "embeddings extra — vector math",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = _REPO_ROOT / "examples"

# Stable machine-readable check IDs — the INTEGRATION.md checklist and any
# external consumer (web checker, fleet schematic) key on these, never on
# the human-readable names. tests/test_integration_doc.py enforces the
# doc<->doctor mapping. Do not rename; add new IDs instead.
CHECK_IDS = [
    "env.python",          "env.deps",           "env.optional",
    "config.home",         "config.vault",       "config.events",
    "config.runtime",      "config.credentials",
    "functional.corpus",   "functional.index",   "functional.replay",
    "retrieval.questions", "retrieval.recall",
    "daemon.supervisor",   "daemon.liveness",
    "llm.endpoint",        "llm.structured",     "llm.concepts",
    "llm.judge",
    "env.sibling",
]


class Check:
    def __init__(self, name: str, ok: bool, detail: str = "",
                 required: bool = True, id: str = ""):
        self.name, self.ok, self.detail, self.required = name, ok, detail, required
        self.id = id

    def to_dict(self):
        return {"id": self.id, "name": self.name, "ok": self.ok,
                "detail": self.detail, "required": self.required}


def _check_environment() -> list[Check]:
    checks = []
    v = sys.version_info
    checks.append(Check("python>=3.12", v >= (3, 12),
                        f"found {v.major}.{v.minor}.{v.micro}", id="env.python"))
    for mod in REQUIRED_DEPS:
        try:
            importlib.import_module(mod)
            checks.append(Check(f"dep:{mod}", True, "importable", id="env.deps"))
        except ImportError as e:
            checks.append(Check(f"dep:{mod}", False, str(e), id="env.deps"))
    for mod, why in OPTIONAL_DEPS.items():
        try:
            importlib.import_module(mod)
            checks.append(Check(f"optional:{mod}", True, why, required=False,
                                id="env.optional"))
        except ImportError:
            checks.append(Check(f"optional:{mod}", False, f"missing — {why}",
                                required=False, id="env.optional"))
    from dct import family  # sibling harness detection (advisory)
    checks.extend(family.sibling_checks(check_cls=Check))
    return checks


def _check_configuration(live: bool) -> list[Check]:
    from dct import config as cfg
    checks = []
    home = cfg.pdct_home()
    checks.append(Check("PDCT_HOME resolved", True, str(home), id="config.home"))
    if live:
        checks.append(Check("PDCT_HOME exists", home.exists(), str(home), id="config.home"))
        roots = cfg.vault_roots()
        existing = [r for r in roots if r.exists()]
        n_md = sum(len(list(r.rglob("*.md"))) for r in existing)
        checks.append(Check("vault root present", bool(existing),
                            f"{len(existing)}/{len(roots)} roots exist, {n_md} .md files", id="config.vault"))
        ev = cfg.events_path()
        checks.append(Check("events.jsonl present", ev.exists(),
                            f"{ev} ({ev.stat().st_size} bytes)" if ev.exists() else str(ev),
                            required=False, id="config.events"))
        try:
            cfg.runtime_dir().mkdir(parents=True, exist_ok=True)
            probe = cfg.runtime_dir() / ".doctor-probe"
            probe.write_text("ok"); probe.unlink()
            checks.append(Check("runtime dir writable", True, str(cfg.runtime_dir()), id="config.runtime"))
        except OSError as e:
            checks.append(Check("runtime dir writable", False, str(e), id="config.runtime"))
    # Provider-aware (contract: "any provider auth detected") — an
    # openai-compatible install with BASE_URL+MODEL counts as credentialed.
    from dct import providers as _prov
    has_auth, detail = _prov.provider_available()
    checks.append(Check("llm credentials", has_auth,
                        detail if has_auth else
                        f"{detail} — distiller & judge disabled; retrieval "
                        "works without them", required=False,
                        id="config.credentials"))
    return checks


def _check_functional(corpus: Path) -> list[Check]:
    """Build index + graph from `corpus` inside a temp sandbox and smoke it."""
    checks = []
    if not corpus.exists():
        return [Check("example corpus present", False, f"{corpus} missing", id="functional.corpus")]
    checks.append(Check("example corpus present", True,
                        f"{len(list(corpus.rglob('*.md')))} notes", id="functional.corpus"))
    tmp = Path(tempfile.mkdtemp(prefix="pdct-doctor-"))
    old_env = {k: os.environ.get(k) for k in
               ("PDCT_HOME", "PDCT_VAULT_ROOT", "OBSIDIAN_VAULT", "PDCT_EVENTS_PATH")}
    try:
        os.environ["PDCT_HOME"] = str(tmp)
        os.environ["PDCT_VAULT_ROOT"] = str(corpus / "vault")
        events_src = corpus / "events.jsonl"
        if events_src.exists():
            shutil.copy(events_src, tmp / "events.jsonl")
        # Re-import with sandbox env (config reads env at call time; index
        # caches key on resolved roots so sandbox roots get their own entry).
        from dct.retrieval.distill_index import build_index
        t0 = time.time()
        idx = build_index(roots=[Path(os.environ["PDCT_VAULT_ROOT"]) / "distillations"])
        checks.append(Check("index builds", len(idx) > 0,
                            f"{len(idx)} distillations in {time.time()-t0:.2f}s", id="functional.index"))
        from dct.event_log import EventLog
        from dct.activation import ActivationEngine, DecayConfig
        ev = tmp / "events.jsonl"
        if ev.exists():
            log = EventLog(ev)
            eng = ActivationEngine.replay(log, config=DecayConfig(half_life_seconds=3600))
            snap = eng.snapshot(now=time.time(), min_heat=0.0)
            checks.append(Check("event replay + heat", True, f"{len(snap)} warm concepts", id="functional.replay"))
        else:
            checks.append(Check("event replay + heat", False, "no events.jsonl in corpus",
                                required=False, id="functional.replay"))
    except Exception as e:  # noqa: BLE001 — doctor reports, never raises
        checks.append(Check("functional stage", False, f"{type(e).__name__}: {e}", id="functional.index"))
    finally:
        for k, val in old_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        shutil.rmtree(tmp, ignore_errors=True)
    return checks


def _check_retrieval_quality(corpus: Path) -> list[Check]:
    """Canned synthetic questions: expected note id must appear in top-5."""
    checks = []
    qfile = corpus / "doctor-questions.json"
    if not qfile.exists():
        return [Check("canned questions present", False, str(qfile), required=False, id="retrieval.questions")]
    questions = json.loads(qfile.read_text())
    from dct.retrieval import service as _svc
    from dct.retrieval.distill_index import build_index
    from dct.retrieval.memory_api import _aggregate, _cascade_for_seed  # smoke internals
    roots = [corpus / "vault" / "distillations"]
    idx = build_index(roots=roots)
    # Bind the cascade to the example corpus — NOT the live install. The
    # service resolves EVENTS_JSONL/DISTILL_ROOT late, so patch the module
    # attrs for the duration of this stage (same trick the test suite uses).
    _old_events, _old_root = _svc.EVENTS_JSONL, _svc.DISTILL_ROOT
    # Bind a TEMP COPY of the corpus events — the service appends graph
    # "rebuild" events to whatever is bound, and binding the bundled file
    # directly pollutes the shipped example corpus on every doctor run.
    _ev_tmpdir = Path(tempfile.mkdtemp(prefix="pdct-doctor-events-"))
    _ev_copy = _ev_tmpdir / "events.jsonl"
    src_events = corpus / "events.jsonl"
    if src_events.exists():
        shutil.copy(src_events, _ev_copy)
    else:
        _ev_copy.touch()
    _svc.EVENTS_JSONL = _ev_copy
    _svc.DISTILL_ROOT = roots[0]
    try:
        passed = 0
        for q in questions:
            try:
                rows = _aggregate([_cascade_for_seed(q["question"])], idx,
                                  query_text=q["question"])
                top_ids = [r.id for r in rows[:5]]
                hit = q["expected_id"] in top_ids
            except Exception as e:  # noqa: BLE001
                checks.append(Check(f"q:{q.get('id', '?')}", False,
                                    f"{type(e).__name__}: {e}", id="retrieval.questions"))
                continue
            passed += hit
            checks.append(Check(f"q:{q['id']}", hit,
                                f"expected {q['expected_id']} in top5={top_ids}",
                                id="retrieval.questions"))
        checks.append(Check("retrieval recall", passed == len(questions),
                            f"{passed}/{len(questions)} canned questions hit top-5", id="retrieval.recall"))
    finally:
        _svc.EVENTS_JSONL, _svc.DISTILL_ROOT = _old_events, _old_root
        shutil.rmtree(_ev_tmpdir, ignore_errors=True)
    return checks


def _check_llm() -> list[Check]:
    """Stage 6 — LLM capability gate. Provider-aware and model-agnostic:
    ANY model passes if it functionally meets minimum requirements.

      llm.endpoint    configured provider reachable + auth valid
      llm.structured  returns parseable JSON matching the distillation schema
      llm.concepts    distilled concepts hit >=2 of the expected set
      llm.judge       judge round-trip returns a valid verdict

    No provider configured/credentialed = advisory skip (retrieval-only
    installs are valid). Configured-but-broken = required failure.
    """
    from dct import providers as prov
    checks = []
    usable, detail = prov.provider_available()
    if not usable:
        return [Check("llm provider", False,
                      f"{detail} — distillation disabled, retrieval-only "
                      "mode available", required=False, id="llm.endpoint")]
    # Contract: llm.endpoint = "reachable + auth valid" — actually probe it
    # so connectivity/auth failures aren't misreported as capability ones.
    reach_ok, reach_detail = prov.probe_endpoint()
    checks.append(Check(f"provider:{prov.provider_name()}", reach_ok,
                        reach_detail, id="llm.endpoint"))
    if not reach_ok:
        checks.append(Check("structured output", False,
                            "skipped — endpoint unreachable/auth invalid",
                            id="llm.structured"))
        checks.append(Check("concept extraction quality", False,
                            "skipped — endpoint unreachable/auth invalid",
                            id="llm.concepts"))
        checks.append(Check("judge round-trip", False,
                            "skipped — endpoint unreachable/auth invalid",
                            id="llm.judge"))
        return checks

    # (b) structured-output fidelity on a canned synthetic session
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "concepts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "summary", "concepts"],
    }
    synthetic = (
        "user: I benchmarked pgvector against a dedicated vector database "
        "for our retrieval feature. pgvector won on operational simplicity.\n"
        "assistant: Good call — one fewer service to run. Did you index "
        "with HNSW?\n"
        "user: Yes, HNSW with m=16. Recall at 10 was 0.94, plenty for us."
    )
    expected = {"pgvector", "vector-database", "hnsw", "benchmarking",
                "retrieval", "vector-search", "recall", "database"}
    try:
        obj = prov.complete_json(
            "Distill this conversation into a note. concepts are lowercase "
            "hyphen-separated slugs.", synthetic, schema, max_tokens=1024)
        checks.append(Check("structured output", True,
                            f"valid JSON with {sorted(obj.keys())}",
                            id="llm.structured"))
        got = {str(c).strip().lower().replace(" ", "-")
               for c in (obj.get("concepts") or [])}
        overlap = {g for g in got
                   if any(e in g or g in e for e in expected)}
        ok = len(overlap) >= 2
        checks.append(Check(
            "concept extraction quality", ok,
            f"matched {sorted(overlap)[:4]} of expected set" if ok else
            f"only matched {sorted(overlap)} — model below minimum "
            "capability; distillation disabled, retrieval-only mode available",
            id="llm.concepts"))
    except prov.ProviderError as e:
        checks.append(Check("structured output", False, str(e)[:300],
                            id="llm.structured"))
        checks.append(Check("concept extraction quality", False,
                            "skipped — structured output failed",
                            id="llm.concepts"))

    # (d) judge round-trip: plain-text JSON verdict
    try:
        text = prov.complete_text(
            "You are a relevance judge. Respond with ONLY a JSON object "
            '{"score": <0-10 integer>, "rationale": "<one sentence>"}.',
            "Question: which vector database did the team choose?\n"
            "Retrieved note: The team benchmarked pgvector and chose it "
            "for operational simplicity.", max_tokens=128)
        t = text.strip()
        if t.startswith("```"):
            t = "\n".join(t.splitlines()[1:-1] if t.splitlines()[-1].strip() == "```"
                          else t.splitlines()[1:]).strip()
        verdict = json.loads(t[t.find("{"):t.rfind("}") + 1])
        ok = isinstance(verdict.get("score"), (int, float))
        checks.append(Check("judge round-trip", ok,
                            f"verdict score={verdict.get('score')}",
                            id="llm.judge"))
    except (prov.ProviderError, json.JSONDecodeError, ValueError) as e:
        checks.append(Check("judge round-trip", False,
                            f"{type(e).__name__}: {str(e)[:200]}",
                            id="llm.judge"))
    return checks


def _check_daemon(live: bool) -> list[Check]:
    """Stage 5 — loop liveness.

    Sandbox mode (default): start a supervisor against a temp PDCT_HOME,
    touch a note, assert the event lands in events.jsonl, stop it. Proves
    the *write path* works on this machine, not just retrieval.

    Live mode (--live): additionally probe the operator's running daemon
    via the supervisor status contract (required only if installed).
    """
    checks = []
    import shutil as _sh
    import subprocess as _sp
    import textwrap
    tmp = Path(tempfile.mkdtemp(prefix="pdct-doctor-daemon-"))
    try:
        script = textwrap.dedent("""
            import os, sys, time
            from pathlib import Path
            home = Path(os.environ["PDCT_HOME"])
            (home / "vault" / "distillations").mkdir(parents=True)
            (home / "events.jsonl").touch()
            from dct import supervisor as sup
            ok, msg = sup.start_daemon()
            if not ok:
                print(f"START-FAIL {msg}"); sys.exit(2)
            try:
                time.sleep(1.5)
                note = home / "vault" / "distillations" / "probe.md"
                note.write_text("---\\ntitle: Doctor Probe\\n"
                                "concepts: [doctor-probe, loop-liveness]\\n---\\n\\n"
                                "## Summary\\nliveness probe\\n")
                deadline = time.monotonic() + 10
                n = 0
                while time.monotonic() < deadline:
                    txt = (home / "events.jsonl").read_text().strip()
                    n = len(txt.splitlines()) if txt else 0
                    if n >= 1:
                        break
                    time.sleep(0.25)
                print(f"EVENTS {n}")
                sys.exit(0 if n >= 1 else 3)
            finally:
                sup.stop_daemon()
        """)
        env = dict(os.environ)
        env.update({"PDCT_HOME": str(tmp), "PDCT_SCHEDULER_INTERVAL": "3600"})
        for k in ("PDCT_VAULT_ROOT", "OBSIDIAN_VAULT", "PDCT_EVENTS_PATH"):
            env.pop(k, None)
        r = _sp.run([sys.executable, "-c", script], env=env,
                    capture_output=True, text=True, timeout=60)
        detail = (r.stdout.strip() or r.stderr.strip()[-200:])
        checks.append(Check("supervisor lifecycle + event lands", r.returncode == 0,
                            detail, id="daemon.supervisor"))
    except Exception as e:  # noqa: BLE001
        checks.append(Check("supervisor lifecycle + event lands", False,
                            f"{type(e).__name__}: {e}", id="daemon.supervisor"))
    finally:
        _sh.rmtree(tmp, ignore_errors=True)

    if live:
        try:
            from dct import supervisor as sup
            st = sup.daemon_status()
            if st["running"]:
                inner = st.get("status", {})
                watcher_ok = bool(inner.get("watcher", {}).get("alive"))
                checks.append(Check(
                    "live daemon healthy", watcher_ok,
                    f"pid={st['pid']} uptime={inner.get('uptime_s')}s "
                    f"watcher={'alive' if watcher_ok else 'DOWN'} "
                    f"ticks={inner.get('scheduler', {}).get('ticks')}",
                    id="daemon.liveness"))
            else:
                checks.append(Check(
                    "live daemon healthy", False,
                    "not running — start with `pdct daemon start` "
                    "(advisory: retrieval works without it)",
                    required=False, id="daemon.liveness"))
        except Exception as e:  # noqa: BLE001
            checks.append(Check("live daemon healthy", False,
                                f"{type(e).__name__}: {e}", required=False,
                                id="daemon.liveness"))
    return checks


def run(json_out: bool = False, live: bool = False,
        corpus: Path | None = None) -> int:
    corpus = corpus or EXAMPLES_DIR
    stages = {
        "environment": _check_environment(),
        "configuration": _check_configuration(live),
        "functional": _check_functional(corpus),
        "retrieval": _check_retrieval_quality(corpus),
        "daemon": _check_daemon(live),
        "llm": _check_llm(),
    }
    required_fail = [c for cs in stages.values() for c in cs
                     if c.required and not c.ok]
    if json_out:
        print(json.dumps({
            "ok": not required_fail,
            "stages": {k: [c.to_dict() for c in v] for k, v in stages.items()},
        }, indent=2))
    else:
        for stage, cs in stages.items():
            print(f"\n── {stage} " + "─" * (50 - len(stage)))
            for c in cs:
                mark = "✅" if c.ok else ("❌" if c.required else "⚠️ ")
                print(f"  {mark} {c.name}: {c.detail}")
        print(f"\n{'✅ PDCT healthy' if not required_fail else '❌ FAILED: ' + ', '.join(c.name for c in required_fail)}")
    return 0 if not required_fail else 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="dct.doctor", description="PDCT self-diagnosis")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--live", action="store_true",
                    help="also check the live installation (vault, events, runtime)")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="alternate corpus dir (default: bundled examples/)")
    args = ap.parse_args()
    return run(json_out=args.json, live=args.live, corpus=args.corpus)


if __name__ == "__main__":
    raise SystemExit(main())
