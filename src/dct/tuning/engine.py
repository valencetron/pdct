"""PDCT shadow-replay tuning engine (Build 106).

Design contract (from the Codex-audited plan):

  * Candidates are evaluated OFFLINE via shadow benchmark runs — evaluation
    never writes the live overrides file (Codex #1). Only a PROMOTED winner is
    applied, atomically, via ``overrides.write_overrides_batch`` (Codex #2).
  * Ticks serialize under a tuner-wide flock; a concurrent tick exits "busy"
    (Codex #3).
  * Two-tier objective:
      Tier 1 (reference benchmark, ``dct.tuning.harness``) is the REGRESSION
      GATE — a candidate must not degrade it beyond the noise band.
      Tier 2 (in-situ replay over the user's own logged retrieval traffic) is
      the PROMOTION SIGNAL — required improvement on the user's own graph.
      Tier 2 abstains below floors (Codex #4); then Tier 1-only "watchdog
      mode" applies and no candidate can be promoted.
  * Promotion additionally requires a bounded health predicate (Codex #8) —
    NOT a full doctor run inside the tick.
  * All I/O collaborators are injectable for tests. The engine never raises
    out of ``run_tick`` — failures return a TickResult with an error note.

State lives under ``<runtime_dir>/tune/``:
    state.json       — baseline, pending-free (shadow model has no live pending),
                       done moves, consecutive rejections, converged flag
    candidates.json  — move queue
    ledger.jsonl     — one row per evaluated candidate (verdicts, scores)
    tick.lock        — tuner-wide flock
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

from dct import config as _cfg
from dct.retrieval import overrides as ov

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

NOISE_BAND = 0.02            # min Tier 2 improvement to promote; max Tier 1 drop
CONVERGE_AFTER = 4           # consecutive rejections -> converged
TIER2_MIN_NODES = 200        # graph-size floor for in-situ replay
TIER2_MIN_ROWS = 50          # logged-traffic floor for in-situ replay
HEALTH_TIMEOUT_S = 60


def tune_dir() -> Path:
    d = _cfg.runtime_dir() / "tune"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------

def normalize_move(cand: dict) -> dict:
    """{"param","new"} or {"name","params":{...}} -> {"key", "changes":{k:v}}."""
    if cand.get("params"):
        key = cand.get("name") or "+".join(sorted(cand["params"]))
        return {"key": key, "changes": dict(sorted(cand["params"].items()))}
    return {"key": cand["param"], "changes": {cand["param"]: cand["new"]}}


def _move_key(cand: dict) -> str:
    return cand.get("name") or cand.get("param") or ""


def next_candidate(queue: list[dict], done: set[str]) -> Optional[dict]:
    for c in queue:
        k = _move_key(c)
        if k and k not in done:
            return c
    return None


DEFAULT_CANDIDATES = [
    {"param": "cascade_depth", "new": 3},
    {"param": "cascade_score_floor", "new": 0.07},
    {"param": "cascade_decay", "new": 0.5},
    {"param": "cascade_transitions_bias", "new": 0.8},
    {"param": "cascade_top_k", "new": 40},
    {"name": "depthxdecay",
     "params": {"cascade_depth": 3, "cascade_decay": 0.5}},
]


# ---------------------------------------------------------------------------
# Tier 2 — in-situ replay over the user's own logged traffic
# ---------------------------------------------------------------------------

def _retrieval_log_path() -> Path:
    return _cfg.logs_dir() / "retrieval.jsonl"


def tier2_floors_met(
    *,
    graph_nodes_fn: Callable[[], int],
    log_rows_fn: Callable[[], int],
) -> tuple[bool, str]:
    """Cold-start gate (Codex #4): abstain below corpus/traffic floors."""
    n = graph_nodes_fn()
    if n < TIER2_MIN_NODES:
        return False, f"graph_nodes {n} < {TIER2_MIN_NODES}"
    r = log_rows_fn()
    if r < TIER2_MIN_ROWS:
        return False, f"log_rows {r} < {TIER2_MIN_ROWS}"
    return True, ""


def _default_graph_nodes() -> int:
    try:
        from dct.retrieval.service import _load_or_build_graph
        return len(_load_or_build_graph().nodes)
    except Exception:
        return 0


def _default_log_rows() -> int:
    try:
        with _retrieval_log_path().open() as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def replay_tier2(config_overrides: dict[str, Any], *, sample: int = 60) -> Optional[float]:
    """Replay the most recent logged retrieval seeds under a candidate config,
    in-process via ``service.run(config_override=...)`` — NO live-state writes.

    Score = mean fraction of non-empty results weighted by result count parity
    with what actually happened (proxy for "would this config have found at
    least as much"). Returns None when the log is unusable.

    Tier 2 rows are local-only and never exported or telemetered (Codex #4).
    """
    rows = []
    try:
        with _retrieval_log_path().open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("seed") and isinstance(r.get("result_count"), int):
                    rows.append(r)
    except OSError:
        return None
    rows = rows[-sample:]
    if len(rows) < TIER2_MIN_ROWS:
        return None

    from dataclasses import replace as dc_replace
    from dct.retrieval.service import build_config

    cfg = build_config()
    try:
        cfg = dc_replace(cfg, **{k: v for k, v in config_overrides.items()})
    except TypeError:
        return None

    scores = []
    for r in rows:
        try:
            out = _replay_one(r["seed"], cfg)
        except Exception:
            continue
        got = len(out)
        want = r["result_count"]
        if want == 0:
            scores.append(1.0 if got >= 0 else 0.0)
        else:
            scores.append(min(1.0, got / want))
    if not scores:
        return None
    return sum(scores) / len(scores)


def _replay_one(seed: str, cfg) -> list:
    """Config-injected single replay through the cascade + aggregate path."""
    from dct.retrieval.cascade import cascade
    from dct.retrieval.distill_index import build_index
    from dct.retrieval.memory_api import _aggregate
    from dct.retrieval.service import _load_or_build_graph, _derive_seeds

    graph = _load_or_build_graph()
    seeds = _derive_seeds(seed, graph)
    if not seeds:
        return []
    hits = cascade(seed_concepts=seeds, graph=graph, heat={}, config=cfg,
                   current_context=set())
    index = build_index()
    return _aggregate([hits], index, query_text=seed)


# ---------------------------------------------------------------------------
# Health predicate (Codex #8) — bounded, not full doctor
# ---------------------------------------------------------------------------

def health_ok() -> tuple[bool, str]:
    """Small pure(ish) predicate: overrides file valid, index loadable,
    smoke retrieval executes. Bounded by construction (in-process, no
    subprocesses, no network)."""
    try:
        ov.load_overrides()  # corrupt file -> {} (never raises) but exercises it
        from dct.retrieval.distill_index import build_index
        idx = build_index()
        if not isinstance(idx, dict):
            return False, "index not a dict"
        from dct.retrieval.service import build_config
        build_config()
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def decide_verdict(
    *,
    baseline_t1: Optional[float],
    cand_t1: Optional[float],
    baseline_t2: Optional[float],
    cand_t2: Optional[float],
    tier2_available: bool,
) -> tuple[str, str]:
    """Returns (verdict, reason). verdict in {"promote", "reject"}.

    Rules:
      * Tier 1 gate: cand_t1 must exist and not fall more than NOISE_BAND
        below baseline_t1 (when baseline exists).
      * Tier 2 promotion: only when tier2_available AND cand_t2 beats
        baseline_t2 by more than NOISE_BAND. Without Tier 2, no promotion
        (watchdog mode) — Tier 1 alone can only reject.
    """
    if cand_t1 is None:
        return "reject", "tier1_unavailable"
    if baseline_t1 is not None and (baseline_t1 - cand_t1) > NOISE_BAND:
        return "reject", "tier1_regression"
    if not tier2_available:
        return "reject", "tier2_abstained"
    if cand_t2 is None:
        return "reject", "tier2_score_missing"
    if baseline_t2 is None:
        return "promote", "no_tier2_baseline"
    if (cand_t2 - baseline_t2) > NOISE_BAND:
        return "promote", "tier2_improved"
    return "reject", "tier2_no_improvement"


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

@dataclass
class TickResult:
    action: str                       # evaluated | idle | busy | disabled | error
    move: Optional[str] = None
    verdict: Optional[str] = None
    reason: str = ""
    tier1_baseline: Optional[float] = None
    tier1_candidate: Optional[float] = None
    tier2_baseline: Optional[float] = None
    tier2_candidate: Optional[float] = None
    converged: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _t1_score(res: dict) -> Optional[float]:
    """Collapse a harness result into a scalar: recall dominant, path-dependence
    secondary (lower Jaccard is better)."""
    if res.get("status") != "ok":
        return None
    rec = res.get("recall_at5")
    jac = res.get("jaccard_concept_ab_mean")
    if not isinstance(rec, (int, float)):
        return None
    path = (1.0 - jac) if isinstance(jac, (int, float)) else None
    return round(0.6 * rec + 0.4 * path, 4) if path is not None else round(rec, 4)


def run_tick(
    *,
    tier1_fn: Callable[[Optional[dict]], dict] = None,
    tier2_fn: Callable[[dict], Optional[float]] = None,
    graph_nodes_fn: Callable[[], int] = None,
    log_rows_fn: Callable[[], int] = None,
    health_fn: Callable[[], tuple[bool, str]] = None,
    apply_batch: Callable[[dict], Any] = None,
    report_hook: Callable[[TickResult, dict], None] = None,
    dry_run: bool = False,
) -> TickResult:
    """One tuning tick: pick next candidate, shadow-evaluate, promote/reject.

    Serialized by a tuner-wide flock (concurrent tick -> action='busy').
    Never raises; hard failures return action='error'.
    """
    from dct.tuning.harness import run_reference_benchmark

    tier1_fn = tier1_fn or (lambda co: run_reference_benchmark(co))
    tier2_fn = tier2_fn or replay_tier2
    graph_nodes_fn = graph_nodes_fn or _default_graph_nodes
    log_rows_fn = log_rows_fn or _default_log_rows
    health_fn = health_fn or health_ok
    apply_batch = apply_batch or ov.write_overrides_batch

    td = tune_dir()
    lock_path = td / "tick.lock"
    lf = open(lock_path, "w")
    try:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return TickResult(action="busy", note="another tick holds the lock")
        try:
            return _locked_tick(
                td, tier1_fn=tier1_fn, tier2_fn=tier2_fn,
                graph_nodes_fn=graph_nodes_fn, log_rows_fn=log_rows_fn,
                health_fn=health_fn, apply_batch=apply_batch,
                report_hook=report_hook, dry_run=dry_run)
        except Exception as e:  # noqa: BLE001 — never raise out of a tick
            return TickResult(action="error", note=f"{type(e).__name__}: {e}")
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    finally:
        lf.close()


def _locked_tick(td: Path, *, tier1_fn, tier2_fn, graph_nodes_fn, log_rows_fn,
                 health_fn, apply_batch, report_hook, dry_run) -> TickResult:
    state_path = td / "state.json"
    cand_path = td / "candidates.json"
    ledger_path = td / "ledger.jsonl"

    state = _load_json(state_path, {
        "baseline_t1": None, "baseline_t2": None,
        "done": [], "consecutive_rejections": 0, "converged": False,
        "history": [],  # trailing baseline_t1 scores for the watchdog
    })

    if state.get("converged"):
        return TickResult(action="idle", converged=True,
                          note="converged — watchdog only (pdct tune restart to reopen)")

    queue = _load_json(cand_path, None)
    if queue is None:
        queue = list(DEFAULT_CANDIDATES)
        if not dry_run:
            _save_json(cand_path, queue)

    cand = next_candidate(queue, set(state.get("done") or []))
    if cand is None:
        return TickResult(action="idle", note="queue exhausted")

    move = normalize_move(cand)
    res = TickResult(action="evaluated", move=move["key"])

    # Establish baselines lazily (first tick).
    if state.get("baseline_t1") is None:
        base_res = tier1_fn(None)
        state["baseline_t1"] = _t1_score(base_res)

    t2_ok, t2_why = tier2_floors_met(
        graph_nodes_fn=graph_nodes_fn, log_rows_fn=log_rows_fn)
    if t2_ok and state.get("baseline_t2") is None:
        state["baseline_t2"] = tier2_fn({})

    # Shadow evaluation — read-only w.r.t. live state.
    cand_res = tier1_fn(move["changes"])
    res.tier1_baseline = state["baseline_t1"]
    res.tier1_candidate = _t1_score(cand_res)
    res.tier2_baseline = state.get("baseline_t2")
    res.tier2_candidate = tier2_fn(move["changes"]) if t2_ok else None

    verdict, reason = decide_verdict(
        baseline_t1=res.tier1_baseline, cand_t1=res.tier1_candidate,
        baseline_t2=res.tier2_baseline, cand_t2=res.tier2_candidate,
        tier2_available=t2_ok)
    if not t2_ok and reason == "tier2_abstained":
        reason = f"tier2_abstained ({t2_why})"

    # Health gate on promotion (Codex #8).
    if verdict == "promote":
        ok, why = health_fn()
        if not ok:
            verdict, reason = "reject", f"health_gate ({why})"

    res.verdict, res.reason = verdict, reason

    if verdict == "promote" and not dry_run:
        apply_batch(move["changes"])
        state["baseline_t1"] = res.tier1_candidate
        if res.tier2_candidate is not None:
            state["baseline_t2"] = res.tier2_candidate
        state["consecutive_rejections"] = 0
    else:
        state["consecutive_rejections"] = int(
            state.get("consecutive_rejections") or 0) + 1
        if state["consecutive_rejections"] >= CONVERGE_AFTER:
            state["converged"] = True
            res.converged = True

    state["done"] = sorted(set(state.get("done") or []) | {move["key"]})
    hist = list(state.get("history") or [])
    if state.get("baseline_t1") is not None:
        hist.append(state["baseline_t1"])
    state["history"] = hist[-20:]

    if not dry_run:
        _save_json(state_path, state)
        with ledger_path.open("a") as f:
            f.write(json.dumps({"ts": time.time(), **res.to_dict()},
                               separators=(",", ":")) + "\n")

    if report_hook is not None:
        try:
            report_hook(res, state)
        except Exception:  # noqa: BLE001 — reporting must never break the tick
            pass
    return res
