"""`pdct tune` subcommand implementations (Build 106, Task 4).

Semantics (Codex plan-audit #11):
    start    — set enabled flag + seed candidate queue (idempotent)
    stop     — clear enabled flag, freeze state; promoted overrides are LEFT
               IN PLACE (they won; stopping the tuner doesn't undo science)
    restart  — reset convergence + rejection streak + reseed queue, re-enable
    tick     — run one tick now (respects the tuner lock; works without daemon)
    status   — levers vs shipped defaults, move history, convergence state
    telemetry— on|off|show (show prints the exact payload rows)
"""
from __future__ import annotations

import argparse
import json

from dct.retrieval import overrides as ov
from dct.tuning import engine, telemetry, watchdog


def cmd_tune(args: argparse.Namespace) -> int:
    action = args.action

    if action == "start":
        cfg = telemetry.load_config()
        cfg["enabled"] = True
        telemetry.save_config(cfg)
        td = engine.tune_dir()
        if not (td / "candidates.json").exists():
            engine._save_json(td / "candidates.json",
                              list(engine.DEFAULT_CANDIDATES))
        print("tuning enabled — the daemon will run ticks on its schedule; "
              "run `pdct tune tick` to run one now")
        return 0

    if action == "stop":
        cfg = telemetry.load_config()
        cfg["enabled"] = False
        telemetry.save_config(cfg)
        print("tuning disabled — state frozen; promoted overrides left in place")
        return 0

    if action == "restart":
        td = engine.tune_dir()
        state = engine._load_json(td / "state.json", {})
        state["converged"] = False
        state["consecutive_rejections"] = 0
        state["drift_streak"] = 0
        state["done"] = []
        engine._save_json(td / "state.json", state)
        engine._save_json(td / "candidates.json",
                          list(engine.DEFAULT_CANDIDATES))
        cfg = telemetry.load_config()
        cfg["enabled"] = True
        telemetry.save_config(cfg)
        print("tuning restarted — convergence cleared, queue reseeded")
        return 0

    if action == "tick":
        if not telemetry.load_config().get("enabled"):
            print("tuning is disabled — `pdct tune start` first")
            return 2
        r = engine.run_tick()
        _telemeter_tick(r)
        print(json.dumps(r.to_dict(), indent=1))
        return 0 if r.action != "error" else 1

    if action == "watchdog":
        r = watchdog.run_watchdog()
        print(json.dumps(r, indent=1))
        return 0 if r.get("action") != "error" else 1

    if action == "status":
        return _status(json_out=getattr(args, "json", False))

    if action == "telemetry":
        return _telemetry(args)

    print(f"unknown action: {action}")
    return 2


def _telemeter_tick(r) -> None:
    telemetry.record({
        "kind": "verdict", "move": r.move, "verdict": r.verdict,
        "reason": (r.reason or "").split(" ")[0],
        "tier1_baseline": r.tier1_baseline, "tier1_candidate": r.tier1_candidate,
        "tier2_baseline": r.tier2_baseline, "tier2_candidate": r.tier2_candidate,
        "converged": r.converged,
        "corpus_bucket": engine._default_graph_nodes(),
    })


def _status(*, json_out: bool) -> int:
    td = engine.tune_dir()
    state = engine._load_json(td / "state.json", {})
    cfg = telemetry.load_config()
    live = ov.load_overrides()
    levers = {
        k: {"default": spec["default"], "live": live.get(k, spec["default"]),
            "overridden": k in live}
        for k, spec in ov.LEVER_SPEC.items()
    }
    history = []
    try:
        with (td / "ledger.jsonl").open() as f:
            for line in f:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass

    out = {
        "enabled": bool(cfg.get("enabled")),
        "telemetry": bool(cfg.get("telemetry")),
        "converged": bool(state.get("converged")),
        "consecutive_rejections": state.get("consecutive_rejections", 0),
        "baseline_t1": state.get("baseline_t1"),
        "baseline_t2": state.get("baseline_t2"),
        "moves_done": state.get("done") or [],
        "levers": levers,
        "recent_history": history[-10:],
    }
    if json_out:
        print(json.dumps(out, indent=1))
        return 0

    print(f"tuning: {'ENABLED' if out['enabled'] else 'disabled'}"
          f"{'  ·  CONVERGED' if out['converged'] else ''}")
    print(f"baseline  tier1={out['baseline_t1']}  tier2={out['baseline_t2']}"
          f"  rejections-streak={out['consecutive_rejections']}")
    print("\nlevers (live vs shipped default):")
    for k, v in levers.items():
        mark = " *" if v["overridden"] else ""
        print(f"  {k:38s} {v['live']!s:>10}  (default {v['default']}){mark}")
    if out["moves_done"]:
        print(f"\nmoves evaluated: {', '.join(out['moves_done'])}")
    verdicts = [h for h in history if h.get("verdict")]
    if verdicts:
        print("\nrecent verdicts:")
        for h in verdicts[-5:]:
            print(f"  {h.get('move'):24s} {h.get('verdict'):8s} {h.get('reason','')}")
    return 0


def _telemetry(args: argparse.Namespace) -> int:
    sub = getattr(args, "telemetry_action", None) or "show"
    cfg = telemetry.load_config()
    if sub == "on":
        cfg["telemetry"] = True
        telemetry.save_config(cfg)
        print("telemetry ON — rows are LOCAL ONLY (see `pdct tune telemetry show`); "
              "nothing is transmitted anywhere")
        return 0
    if sub == "off":
        cfg["telemetry"] = False
        telemetry.save_config(cfg)
        print("telemetry off")
        return 0
    # show
    p = telemetry.telemetry_path()
    if not p.exists():
        print("(no telemetry rows)")
        return 0
    print(p.read_text().rstrip())
    return 0


def register(sub) -> None:
    p = sub.add_parser("tune", help="self-tuning loop (shadow-replay autotuner)")
    p.add_argument("action", choices=["start", "stop", "restart", "tick",
                                      "watchdog", "status", "telemetry"])
    p.add_argument("telemetry_action", nargs="?",
                   choices=["on", "off", "show"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_tune)
