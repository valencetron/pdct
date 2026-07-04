"""Tier 1 reference benchmark harness ("wind tunnel").

Runs the shipped synthetic fixture corpus through the real retrieval stack in a
throwaway PDCT_HOME, in a SUBPROCESS with a scrubbed environment, so:

  * it never reads the user's vault, events log, or live overrides file
    (Codex plan-audit #1, #6 — shadow evaluation must not touch live state);
  * scores are deterministic and comparable across machines (Codex #7):
    optional embeddings disabled, heat disabled, fixed config, sorted corpus;
  * any candidate lever config can be evaluated via ``config_overrides``
    without writing pdct-overrides.json.

Public API:
    run_reference_benchmark(config_overrides=None) -> dict
        {"recall_at5": float, "jaccard_concept_ab_mean": float,
         "n_questions": int, "per_question": [...], "pilots": [...]}

Fixtures ship as package data (Codex #5) and load via importlib.resources.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, Optional

# Levers a candidate config may override in the wind tunnel. Mirrors
# LEVER_SPEC in dct.retrieval.overrides (validated there at apply time; here we
# just pass them through to RetrievalConfig fields of the same name).
_CONFIG_FIELDS = {
    "cascade_depth", "cascade_decay", "cascade_score_floor", "cascade_top_k",
    "cascade_transitions_bias", "cascade_token_cap", "cascade_budget_ms",
}

_TIMEOUT_S = 600


def _fixture_dir() -> Path:
    """Locate the shipped fixtures whether installed as a wheel or from a checkout."""
    pkg = resources.files("dct.tuning") / "fixtures"
    # resources.files returns a Traversable; for wheel installs it's still a
    # real dir because we ship plain files (no zip). as_file would handle zips,
    # but setuptools installs are unpacked — keep it simple and assert.
    p = Path(str(pkg))
    if not p.is_dir():
        raise FileNotFoundError(f"tuning fixtures not found at {p}")
    return p


def prepare_reference_home(dest: Path) -> Path:
    """Build a self-contained PDCT_HOME at `dest` from the shipped fixtures.

    Layout: vault/distillations/*.md (sorted copy) + data/events.jsonl
    synthesized from fixture frontmatter concepts (one READ event per doc, in
    sorted-id order → identical co-occurrence graph on every machine).
    """
    fx = _fixture_dir()
    vault = dest / "vault" / "distillations"
    vault.mkdir(parents=True, exist_ok=True)
    import yaml  # runtime dep of dct already

    # dct.config.events_path() resolves to $PDCT_HOME/events.jsonl.
    events_path = dest / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as ev:
        for doc in sorted((fx / "corpus").glob("*.md")):
            shutil.copy2(doc, vault / doc.name)
            raw = doc.read_text(encoding="utf-8")
            fm = yaml.safe_load(raw.split("---", 2)[1])
            concepts = list(fm.get("concepts") or [])
            if not concepts:
                continue
            ev.write(json.dumps({
                "ts": 1750000000.0,  # fixed — determinism over recency realism
                "source": "vault",
                "op": "read",
                "concepts": concepts,
                "metadata": {"fixture": doc.stem},
            }, separators=(",", ":")) + "\n")
    return dest


def _scrubbed_env(home: Path) -> dict:
    """Subprocess env: pinned to the reference home, all live-state and
    nondeterminism vectors removed."""
    env = {k: v for k, v in os.environ.items() if not (
        k.startswith("PDCT_") or k.startswith("DCT_") or k == "OBSIDIAN_VAULT"
    )}
    env["PDCT_HOME"] = str(home)
    # Point the overrides file INSIDE the sandbox so the live file is never read.
    env["PDCT_OVERRIDES_PATH"] = str(home / "runtime" / "overrides.json")
    env["DCT_VEC_NEAR_ENABLED"] = "false"      # no optional embeddings
    env["PYTHONHASHSEED"] = "0"
    return env


def run_reference_benchmark(
    config_overrides: Optional[dict[str, Any]] = None,
    *,
    home: Optional[Path] = None,
    timeout_s: int = _TIMEOUT_S,
) -> dict:
    """Run the Tier 1 benchmark in an isolated subprocess. Never raises on
    benchmark failure — returns {"status": "error", "error": ...} instead so
    the tuner's never-break-retrieval contract holds."""
    unknown = set(config_overrides or {}) - _CONFIG_FIELDS
    if unknown:
        return {"status": "error", "error": f"unknown config fields: {sorted(unknown)}"}

    tmp_ctx = None
    try:
        if home is None:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="pdct-tier1-")
            home = Path(tmp_ctx.name)
        prepare_reference_home(home)
        proc = subprocess.run(
            [sys.executable, "-m", "dct.tuning.harness",
             "--home", str(home),
             "--config-json", json.dumps(config_overrides or {})],
            capture_output=True, text=True, timeout=timeout_s,
            env=_scrubbed_env(home),
        )
        if proc.returncode != 0:
            return {"status": "error",
                    "error": f"harness rc={proc.returncode}: {proc.stderr[-500:]}"}
        out = json.loads(proc.stdout)
        out["status"] = "ok"
        return out
    except Exception as e:  # noqa: BLE001 — contract: never raise
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


# ---------------------------------------------------------------------------
# Subprocess entry — runs with PDCT_HOME pinned to the sandbox, so all
# module-level paths (DISTILL_ROOT, EVENTS_JSONL, _DEFAULT_ROOTS) bind to it.
# ---------------------------------------------------------------------------

def _subprocess_main(argv: list[str]) -> int:
    import argparse
    from dataclasses import replace as dc_replace

    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--config-json", default="{}")
    args = ap.parse_args(argv)

    overrides = json.loads(args.config_json)

    import yaml

    from dct.retrieval import memory_api
    from dct.retrieval.conversational import ConversationalCascade
    from dct.retrieval.service import build_config

    cfg = build_config()
    # Deterministic mode: heat off (wall-clock dependent), conv cascade on for pilots.
    pinned = {"cascade_heat_enabled": False}
    pinned.update({k: v for k, v in overrides.items() if k in _CONFIG_FIELDS})
    cfg = dc_replace(cfg, **pinned)
    pilot_cfg = dc_replace(cfg, conv_cascade_enabled=True, conv_seed_augment_enabled=True)

    fx = _fixture_dir()

    # --- recall@5 over the question set ---
    qspec = json.loads((fx / "questions.json").read_text())
    per_q = []
    hits = 0
    for q in qspec["questions"]:
        rows = memory_api.query_memory(q["question"], _surface="tier1-bench")
        top5 = [r.id for r in rows[:5]]
        ok = any(e in top5 for e in q["expected_ids"])
        hits += int(ok)
        per_q.append({"id": q["id"], "hit": ok, "top5": top5})
    recall = hits / max(1, len(qspec["questions"]))

    # --- path-dependence pilots (Jaccard concept A vs B) ---
    def _jaccard(a: set, b: set) -> float:
        u = a | b
        return (len(a & b) / len(u)) if u else 1.0

    pspec = yaml.safe_load((fx / "pilots.yaml").read_text())
    pilots_out = []
    for pilot in pspec["pilots"]:
        finals: dict[str, set] = {}
        for arm_key, arm in pilot["arms"].items():
            cc = ConversationalCascade(topic_id=None, config=pilot_cfg,
                                       surface="tier1-pilot")
            cc.reset()
            turns = (list(pilot["same_start"]) + list(arm.get("middle") or [])
                     + list(pilot["same_end"]))
            last: set = set()
            for t in turns:
                r = cc.turn(t)
                last = set(((r.get("conv") or {}).get("activation") or {}).keys())
            finals[arm_key] = last
        j = _jaccard(finals.get("A", set()), finals.get("B", set()))
        pilots_out.append({"id": pilot["id"], "jaccard_concept_AB": round(j, 4)})

    jvals = [p["jaccard_concept_AB"] for p in pilots_out]
    print(json.dumps({
        "recall_at5": round(recall, 4),
        "n_questions": len(per_q),
        "per_question": per_q,
        "pilots": pilots_out,
        "jaccard_concept_ab_mean": round(sum(jvals) / len(jvals), 4) if jvals else None,
        "config_overrides": overrides,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_subprocess_main(sys.argv[1:]))
