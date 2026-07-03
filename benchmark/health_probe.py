"""PDCT health probe — the thermostat.

Runs the canary question set (benchmark/canary-v1.json: questions whose
target docs meet the distillation contract, plus negatives) in two passes:

1. Retrieval pass (fast, no LLM): recall@1 / recall@5 against pinned
   target distillation ids.
2. Optional end-to-end pass (--full): reuses eval_v3's tool loop +
   judge for answer-quality scoring.

Compares results to OPERATING_RANGE and emits a verdict:
  HEALTHY    — all metrics in range
  DEGRADED   — one metric out of range
  UNHEALTHY  — two or more out of range

Output: benchmark/.v3-work/health-<ts>.json + human summary to stdout.
Exit code: 0 healthy, 1 degraded, 2 unhealthy (cron/launchd friendly).

Usage:
    PYTHONPATH=src .venv/bin/python benchmark/health_probe.py [--full]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CANARY = ROOT / "benchmark" / "canary-v1.json"
WORK = ROOT / "benchmark" / ".v3-work"

# The thermostat. 68-72 degrees.
OPERATING_RANGE = {
    "recall_at_5": 0.85,   # >= : retrieval finds the right doc
    "recall_at_1": 0.65,   # >= : reranker puts it first
    "latency_p50_s": 8.0,  # <= : per-query retrieval latency (warm)
    # --full only:
    "answer_mean": 0.60,   # >= : end-to-end answer quality
    "negative_mean": 0.80,  # >= : no fabrication on trick questions
}


def retrieval_pass(questions: list[dict]) -> dict:
    from dct.retrieval.memory_api import query_memory

    pos = [q for q in questions if q["category"] != "negative"]
    hit1 = hit5 = 0
    lats: list[float] = []
    misses: list[dict] = []
    for q in pos:
        t0 = time.monotonic()
        rows = query_memory(q["question"], _surface="health-probe")
        lats.append(time.monotonic() - t0)
        ids = [r.id for r in rows]
        target = q["source_distillation_id"]
        if ids and ids[0] == target:
            hit1 += 1
        if target in ids:
            hit5 += 1
        else:
            misses.append({
                "question": q["question"][:90],
                "target": target,
                "got": ids[:3],
            })
    lats.sort()
    n = len(pos)
    return {
        "n": n,
        "recall_at_1": round(hit1 / n, 3),
        "recall_at_5": round(hit5 / n, 3),
        "latency_p50_s": round(lats[len(lats) // 2], 2) if lats else 0.0,
        "misses": misses,
    }


def full_pass(questions: list[dict]) -> dict:
    """End-to-end answer quality via eval_v3's tool loop + judge."""
    import eval_v3 as ev  # noqa: benchmark dir on path when run from there

    client = ev._client_factory()
    scores: dict[str, list[float]] = {}
    for q in questions:
        block = ""
        try:
            r = ev.service.run(q["question"])
            block = r.get("prompt_block", "") or ""
        except Exception:
            pass
        reply, _ = ev.run_tool_loop(client, q["question"], block)
        if q["category"] == "negative":
            g = ev.grade_negative(reply)
        else:
            g = ev.grade_positive(reply, q)
        scores.setdefault(q["category"], []).append(g["score"])
    flat = [s for v in scores.values() for s in v]
    neg = scores.get("negative", [])
    return {
        "answer_mean": round(sum(flat) / len(flat), 3) if flat else 0.0,
        "negative_mean": round(sum(neg) / len(neg), 3) if neg else None,
        "by_category": {
            k: round(sum(v) / len(v), 3) for k, v in sorted(scores.items())
        },
    }


def verdict(metrics: dict) -> tuple[str, list[str]]:
    breaches: list[str] = []
    for key, bound in OPERATING_RANGE.items():
        val = metrics.get(key)
        if val is None:
            continue
        ok = val <= bound if key.startswith("latency") else val >= bound
        if not ok:
            op = "<=" if key.startswith("latency") else ">="
            breaches.append(f"{key}={val} (want {op} {bound})")
    if not breaches:
        return "HEALTHY", breaches
    return ("DEGRADED" if len(breaches) == 1 else "UNHEALTHY"), breaches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run end-to-end answer-quality pass (LLM)")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "benchmark"))
    questions = json.load(open(CANARY))["questions"]

    metrics: dict = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    metrics.update(retrieval_pass(questions))
    if args.full:
        metrics.update(full_pass(questions))

    v, breaches = verdict(metrics)
    metrics["verdict"] = v
    metrics["breaches"] = breaches

    WORK.mkdir(exist_ok=True)
    out = WORK / f"health-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(metrics, indent=1))

    print(f"=== PDCT HEALTH: {v} ===")
    for k in ("recall_at_1", "recall_at_5", "latency_p50_s",
              "answer_mean", "negative_mean"):
        if metrics.get(k) is not None:
            print(f"  {k}: {metrics[k]}")
    for b in breaches:
        print(f"  BREACH: {b}")
    if metrics.get("misses"):
        print(f"  misses: {len(metrics['misses'])} (see {out.name})")
    print(f"-> {out}")

    sys.exit(0 if v == "HEALTHY" else 1 if v == "DEGRADED" else 2)


if __name__ == "__main__":
    main()
