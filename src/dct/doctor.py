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

REQUIRED_DEPS = ["yaml", "watchdog", "sklearn", "anthropic"]
OPTIONAL_DEPS = {
    "sentence_transformers": "embeddings extra (pip install 'dct[embeddings]') — VEC_NEAR edges",
    "numpy": "embeddings extra — vector math",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = _REPO_ROOT / "examples"


class Check:
    def __init__(self, name: str, ok: bool, detail: str = "", required: bool = True):
        self.name, self.ok, self.detail, self.required = name, ok, detail, required

    def to_dict(self):
        return {"name": self.name, "ok": self.ok, "detail": self.detail,
                "required": self.required}


def _check_environment() -> list[Check]:
    checks = []
    v = sys.version_info
    checks.append(Check("python>=3.12", v >= (3, 12), f"found {v.major}.{v.minor}.{v.micro}"))
    for mod in REQUIRED_DEPS:
        try:
            importlib.import_module(mod)
            checks.append(Check(f"dep:{mod}", True, "importable"))
        except ImportError as e:
            checks.append(Check(f"dep:{mod}", False, str(e)))
    for mod, why in OPTIONAL_DEPS.items():
        try:
            importlib.import_module(mod)
            checks.append(Check(f"optional:{mod}", True, why, required=False))
        except ImportError:
            checks.append(Check(f"optional:{mod}", False, f"missing — {why}", required=False))
    return checks


def _check_configuration(live: bool) -> list[Check]:
    from dct import config as cfg
    checks = []
    home = cfg.pdct_home()
    checks.append(Check("PDCT_HOME resolved", True, str(home)))
    if live:
        checks.append(Check("PDCT_HOME exists", home.exists(), str(home)))
        roots = cfg.vault_roots()
        existing = [r for r in roots if r.exists()]
        n_md = sum(len(list(r.rglob("*.md"))) for r in existing)
        checks.append(Check("vault root present", bool(existing),
                            f"{len(existing)}/{len(roots)} roots exist, {n_md} .md files"))
        ev = cfg.events_path()
        checks.append(Check("events.jsonl present", ev.exists(),
                            f"{ev} ({ev.stat().st_size} bytes)" if ev.exists() else str(ev),
                            required=False))
        try:
            cfg.runtime_dir().mkdir(parents=True, exist_ok=True)
            probe = cfg.runtime_dir() / ".doctor-probe"
            probe.write_text("ok"); probe.unlink()
            checks.append(Check("runtime dir writable", True, str(cfg.runtime_dir())))
        except OSError as e:
            checks.append(Check("runtime dir writable", False, str(e)))
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY")) or \
        Path("~/.claude/.credentials.json").expanduser().exists()
    checks.append(Check("anthropic credentials", has_key,
                        "found" if has_key else
                        "no ANTHROPIC_API_KEY / claude credentials — distiller & judge "
                        "disabled; retrieval works without them", required=False))
    return checks


def _check_functional(corpus: Path) -> list[Check]:
    """Build index + graph from `corpus` inside a temp sandbox and smoke it."""
    checks = []
    if not corpus.exists():
        return [Check("example corpus present", False, f"{corpus} missing")]
    checks.append(Check("example corpus present", True,
                        f"{len(list(corpus.rglob('*.md')))} notes"))
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
                            f"{len(idx)} distillations in {time.time()-t0:.2f}s"))
        from dct.event_log import EventLog
        from dct.activation import ActivationEngine, DecayConfig
        ev = tmp / "events.jsonl"
        if ev.exists():
            log = EventLog(ev)
            eng = ActivationEngine.replay(log, config=DecayConfig(half_life_seconds=3600))
            snap = eng.snapshot(now=time.time(), min_heat=0.0)
            checks.append(Check("event replay + heat", True, f"{len(snap)} warm concepts"))
        else:
            checks.append(Check("event replay + heat", False, "no events.jsonl in corpus",
                                required=False))
    except Exception as e:  # noqa: BLE001 — doctor reports, never raises
        checks.append(Check("functional stage", False, f"{type(e).__name__}: {e}"))
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
        return [Check("canned questions present", False, str(qfile), required=False)]
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
    _svc.EVENTS_JSONL = corpus / "events.jsonl"
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
                                    f"{type(e).__name__}: {e}"))
                continue
            passed += hit
            checks.append(Check(f"q:{q['id']}", hit,
                                f"expected {q['expected_id']} in top5={top_ids}"))
        checks.append(Check("retrieval recall", passed == len(questions),
                            f"{passed}/{len(questions)} canned questions hit top-5"))
    finally:
        _svc.EVENTS_JSONL, _svc.DISTILL_ROOT = _old_events, _old_root
    return checks


def run(json_out: bool = False, live: bool = False,
        corpus: Path | None = None) -> int:
    corpus = corpus or EXAMPLES_DIR
    stages = {
        "environment": _check_environment(),
        "configuration": _check_configuration(live),
        "functional": _check_functional(corpus),
        "retrieval": _check_retrieval_quality(corpus),
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
