#!/usr/bin/env python3
"""PDCT prelim metrics report.

Reads the three jsonl logs and prints a compact health/utility/cost report.

Usage:
    python3 pdct_report.py [--since DAYS] [--topic THREAD_ID] [--json] [--judge]

Logs read:
  dynamic-context-traversal/logs/measurement.jsonl   (per-turn cost/latency)
  dynamic-context-traversal/logs/utility.jsonl       (per-turn concept utility + followup classifications)
  dynamic-context-traversal/logs/retrieval.jsonl     (query_memory/read_memory telemetry)
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import os as _os
LOGS_DIR = Path(_os.environ.get("PDCT_LOGS_DIR", "")) or Path(__file__).resolve().parent.parent / "logs"


def _parse_ts(row: dict) -> datetime | None:
    ts = row.get("ts")
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _load(path: Path, since: datetime | None, topic: str | None) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since is not None:
            ts = _parse_ts(r)
            if ts is None or ts < since:
                continue
        if topic is not None:
            tid = str(r.get("thread_id") or r.get("topic_id") or "")
            if tid != topic:
                continue
        rows.append(r)
    return rows


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _fmt_ms(v: float) -> str:
    return f"{v:.0f}ms" if v < 1000 else f"{v/1000:.2f}s"


def _fmt_n(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}k"
    return f"{v:.0f}"


# ── Track B: feedback events (events.jsonl) ──────────────────────────────

EVENTS_JSONL = Path(__file__).resolve().parent.parent / "events.jsonl"


def feedback_section(
    events_path: Path | None = None,
    since: datetime | None = None,
    topic: str | None = None,
) -> dict:
    """Read events.jsonl and return Track B feedback summary.

    Returns:
        total_feedback_events: int — number of op=feedback events in window
        trajectory_length_dist: dict[int, int] — {hops: count}
    """
    path = events_path or EVENTS_JSONL
    if not path.exists():
        return {"total_feedback_events": 0, "trajectory_length_dist": {}}

    total = 0
    traj_dist: dict[int, int] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("op") != "feedback":
            continue
        # Time filter
        if since is not None:
            ts = _parse_ts(row)
            if ts is None or ts < since:
                continue
        # Topic filter (feedback events carry thread_id in metadata)
        if topic is not None:
            meta = row.get("metadata") or {}
            tid = str(meta.get("thread_id") or "")
            if tid != str(topic):
                continue

        total += 1
        meta = row.get("metadata") or {}
        path_list = meta.get("path") or []
        hops = max(0, len(path_list) - 1)
        traj_dist[hops] = traj_dist.get(hops, 0) + 1

    return {
        "total_feedback_events": total,
        "trajectory_length_dist": traj_dist,
    }


def report(since_days: float | None, topic: str | None, as_json: bool) -> dict:
    since: datetime | None = None
    if since_days is not None:
        since = datetime.now(tz=timezone.utc) - timedelta(days=since_days)

    meas = _load(LOGS_DIR / "measurement.jsonl", since, topic)
    util_all = _load(LOGS_DIR / "utility.jsonl", since, topic)
    retr = _load(LOGS_DIR / "retrieval.jsonl", since, None)  # topic not in retrieval rows

    util = [r for r in util_all if r.get("kind") == "turn"]
    follow = [r for r in util_all if r.get("kind") == "followup"]

    # Era-gate for MATCH-QUALITY aggregates only (match_rate / concepts_total /
    # concepts_matched). The node_kinds classifier-aware scoring (Code/Concept
    # Layer Split, 2026-06-14) changed these semantics at schema_version=6, so
    # pre-6 rows are not comparable. cosine/self_rating/composite have their
    # OWN schema lineage and keep using the full `util` list.
    UTIL_MATCH_SCHEMA_MIN = 6
    util_match = [
        r for r in util
        if isinstance(r.get("schema_version"), int)
        and r.get("schema_version") >= UTIL_MATCH_SCHEMA_MIN
    ]

    # ---- cost / latency ----
    inj_tokens = [r.get("total_injected_tokens_est", 0) for r in meas]
    pdct_chars = [r.get("retrieval_context_chars", 0) for r in meas]
    out_tokens = [r.get("output_tokens_est", 0) for r in meas]
    in_tokens = [r.get("input_tokens", 0) for r in meas]
    cached = [r.get("cached_tokens", 0) for r in meas]
    cascade_lat = [r.get("cascade_latency_ms", 0) for r in meas if r.get("cascade_latency_ms")]
    util_lat = [r.get("utility_latency_ms", 0) for r in meas if r.get("utility_latency_ms")]

    # ---- ablation ----
    ablation_skipped = [r for r in meas if r.get("pdct_skipped_reason") == "ablation"]
    no_concepts = [r for r in meas if r.get("pdct_skipped_reason") == "no_concepts"]
    pdct_active = [r for r in meas if r.get("pdct_skipped_reason") == "none"]
    rates = sorted({r.get("ablation_rate", 0.0) for r in meas})

    # ---- utility (match-quality: era-gated to schema>=6) ----
    match_rates = [
        r.get("match_rate") for r in util_match
        if r.get("concepts_total") and r.get("match_rate") is not None
    ]
    concepts_total = [r.get("concepts_total", 0) for r in util_match]
    concepts_matched = [r.get("concepts_matched", 0) for r in util_match]
    # P1.2: TF-IDF cosine — present from schema_version=2 onward; absent in v1.
    cosine_scores = [r.get("cosine_score") for r in util if r.get("cosine_score") is not None]

    # P1.4: Self-rating distribution (only count known-valid values to avoid corrupt data
    # inflating n_self_rated while disappearing from the display)
    _VALID_SELF_RATINGS = frozenset({"useful", "partial", "noise", "absent"})
    self_ratings = [
        r.get("self_rating") for r in util
        if r.get("self_rating") in _VALID_SELF_RATINGS
    ]
    rating_counts: dict[str, int] = {}
    for v in self_ratings:
        rating_counts[v] = rating_counts.get(v, 0) + 1

    # P1.3b: era_judge score distribution — from era_judge_update rows (last per turn wins)
    _era_updates: dict[str, dict] = {}
    for r in util_all:
        if r.get("kind") == "era_judge_update" and "turn_id" in r:
            _era_updates[r["turn_id"]] = r  # last update per turn wins (file order)
    _era_scores = [
        u["era_judge"] for u in _era_updates.values()
        if isinstance(u.get("era_judge"), int)
    ]
    _era_dist: dict[int, int] = {}
    for s in _era_scores:
        _era_dist[s] = _era_dist.get(s, 0) + 1

    # P1.5: composite score
    # Resolution rule: scan all util_all rows in file order; last row per turn_id wins.
    # composite_update rows (P1.3b) appear after their turn row, so file-order
    # tie-breaking naturally prefers the update without timestamp parsing.
    _composite_by_turn: dict[str, float] = {}
    _composite_legs_tally: dict[str, int] = {}
    _seen_tally_turns: set[str] = set()

    for row in util_all:
        tid = row.get("turn_id")
        if tid is None:
            continue
        kind = row.get("kind", "turn")
        if kind not in ("turn", "composite_update"):
            continue
        score = row.get("pdct_utility_composite")
        if score is not None:
            try:
                v = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            _composite_by_turn[tid] = v  # file-order: last row per turn_id wins

    # Count leg presence from winning row only — avoids double-counting
    # turns that have both a 'turn' and 'composite_update' row.
    # Strategy: scan in reverse so the last (winning) row per turn_id is seen first.
    for row in reversed(util_all):
        tid = row.get("turn_id")
        if tid not in _composite_by_turn:
            continue
        if tid in _seen_tally_turns:
            continue  # already counted the winning row for this turn
        kind = row.get("kind", "turn")
        if kind not in ("turn", "composite_update"):
            continue
        for leg in row.get("composite_legs_used", []):
            _composite_legs_tally[leg] = _composite_legs_tally.get(leg, 0) + 1
        _seen_tally_turns.add(tid)

    _composite_scores = list(_composite_by_turn.values())
    _composite_avg = (
        sum(_composite_scores) / len(_composite_scores)
        if _composite_scores else None
    )

    # ---- followup signals ----
    follow_ratings = collections.Counter(r.get("rating") for r in follow)

    # ---- retrieval ----
    retr_by_fn = collections.Counter(r.get("fn") for r in retr)
    retr_fallback = sum(1 for r in retr if r.get("used_fallback"))
    retr_lat = [r.get("latency_ms", 0) for r in retr if r.get("latency_ms")]

    # ---- topic breakdown ----
    by_topic = collections.Counter(str(r.get("thread_id") or "")  for r in meas)

    summary = {
        "window_days": since_days,
        "topic_filter": topic,
        "n_turns": len(meas),
        "n_pdct_active": len(pdct_active),
        "n_no_concepts": len(no_concepts),
        "n_ablation_skipped": len(ablation_skipped),
        "ablation_rates_seen": rates,
        "tokens": {
            "injected_avg": round(statistics.mean(inj_tokens), 0) if inj_tokens else 0,
            "injected_p50": round(_pct(inj_tokens, 50), 0),
            "injected_p95": round(_pct(inj_tokens, 95), 0),
            "pdct_chars_avg": round(statistics.mean(pdct_chars), 0) if pdct_chars else 0,
            "input_avg": round(statistics.mean(in_tokens), 0) if in_tokens else 0,
            "cached_avg": round(statistics.mean(cached), 0) if cached else 0,
            "output_avg": round(statistics.mean(out_tokens), 0) if out_tokens else 0,
            "cache_hit_pct": round(100 * sum(cached) / max(sum(in_tokens) + sum(cached), 1), 1),
        },
        "latency_ms": {
            "cascade_p50": round(_pct(cascade_lat, 50), 0),
            "cascade_p95": round(_pct(cascade_lat, 95), 0),
            "utility_p50": round(_pct(util_lat, 50), 0) if util_lat else 0,
        },
        "utility": {
            "n_scored": len(util),
            # Codex r1 P2: schema>=6 match-scored count, distinct from
            # n_scored. When 0, match aggregates render n/a (not 0.0%).
            "n_match_scored": len(util_match),
            # Codex r1 P2: match aggregates are era-gated to schema>=6, so report
            # how many turns actually fed them — distinguishes "0% match" from
            # "no comparable schema-6 data in window".
            "n_match_scored": len(util_match),
            "concepts_total_avg": round(statistics.mean(concepts_total), 1) if concepts_total else None,
            "concepts_matched_avg": round(statistics.mean(concepts_matched), 2) if concepts_matched else None,
            "match_rate_avg": round(statistics.mean(match_rates), 3) if match_rates else None,
            "n_with_cosine": len(cosine_scores),
            "cosine_avg": round(statistics.mean(cosine_scores), 3) if cosine_scores else None,
            "cosine_p50": round(_pct(cosine_scores, 50), 3) if cosine_scores else None,
            "cosine_p95": round(_pct(cosine_scores, 95), 3) if cosine_scores else None,
            "n_self_rated": len(self_ratings),
            "self_rating_dist": rating_counts,
            "n_era_judged": len(_era_scores),
            "era_judge_avg": round(sum(_era_scores) / len(_era_scores), 2) if _era_scores else None,
            "era_judge_dist": _era_dist,
        },
        "followups": {
            "n": len(follow),
            "by_rating": dict(follow_ratings),
        },
        "retrieval": {
            "n_calls": len(retr),
            "by_fn": dict(retr_by_fn),
            "fallback_pct": round(100 * retr_fallback / max(len(retr), 1), 1),
            "latency_p50": round(_pct(retr_lat, 50), 0),
            "latency_p95": round(_pct(retr_lat, 95), 0),
        },
        "by_topic": dict(by_topic.most_common()),
        # P1.5: composite
        "n_composite": len(_composite_scores),
        "composite_avg": _composite_avg,
        "composite_legs_tally": _composite_legs_tally,
    }

    return summary


def render(s: dict) -> str:
    lines = []
    win = f"last {s['window_days']:g}d" if s["window_days"] else "all-time"
    if s["topic_filter"]:
        win += f" / topic={s['topic_filter']}"
    lines.append(f"━━ PDCT prelim metrics report ({win}) ━━")
    lines.append("")

    n = s["n_turns"]
    if n == 0:
        lines.append("No turns in window. Nothing to report.")
        return "\n".join(lines)

    lines.append(f"Turns: {n}  (PDCT active: {s['n_pdct_active']} · no-concepts: {s['n_no_concepts']} · ablation-skipped: {s['n_ablation_skipped']})")
    rates = s["ablation_rates_seen"]
    if rates:
        lines.append(f"Ablation rates seen: {rates}")
    lines.append("")

    # P1.5: composite headline
    _comp_avg = s.get("composite_avg")
    _n_composite = s.get("n_composite", 0)
    lines.append("── Composite score (P1.5) ──")
    if _comp_avg is not None:
        filled = round(_comp_avg * 20)
        bar = "█" * filled + "░" * (20 - filled)
        lines.append(
            f"  pdct_utility_composite ({_n_composite}/{n} turns): "
            f"{_comp_avg:.3f}  [{bar}]"
        )
        tally = s.get("composite_legs_tally", {})
        leg_parts = []
        for leg in ("match_rate", "cosine_score", "self_rating", "era_judge"):
            count = tally.get(leg, 0)
            leg_parts.append(f"{leg}={count if count else 'n/a'}")
        lines.append("  legs: " + "  ".join(leg_parts))
    else:
        lines.append("  pdct_utility_composite: no data yet (schema_version=4 rows needed)")
    lines.append("")

    t = s["tokens"]
    lines.append("── Tokens ──")
    lines.append(f"  injected/turn: avg {_fmt_n(t['injected_avg'])} · p50 {_fmt_n(t['injected_p50'])} · p95 {_fmt_n(t['injected_p95'])}")
    lines.append(f"  PDCT block (chars): avg {_fmt_n(t['pdct_chars_avg'])}")
    lines.append(f"  input avg: {_fmt_n(t['input_avg'])}  cached avg: {_fmt_n(t['cached_avg'])}  cache-hit: {t['cache_hit_pct']}%")
    lines.append(f"  output avg: {_fmt_n(t['output_avg'])}")
    lines.append("")

    l = s["latency_ms"]
    lines.append("── Latency ──")
    lines.append(f"  cascade: p50 {_fmt_ms(l['cascade_p50'])} · p95 {_fmt_ms(l['cascade_p95'])}")
    if l["utility_p50"]:
        lines.append(f"  utility: p50 {_fmt_ms(l['utility_p50'])}")
    lines.append("")

    u = s["utility"]
    lines.append("── Utility (concept hit rate) ──")
    lines.append(f"  scored turns: {u['n_scored']}")
    _nms = u.get("n_match_scored", u["n_scored"])
    if _nms and u.get("match_rate_avg") is not None:
        lines.append(f"  match-scored turns (schema≥6): {_nms}")
        lines.append(f"  concepts/turn: {u['concepts_total_avg']} total · {u['concepts_matched_avg']} matched")
        lines.append(f"  match rate: {u['match_rate_avg']:.1%}")
    elif u["n_scored"]:
        lines.append("  match rate: n/a (no comparable schema≥6 turns in window)")
    if u.get("n_with_cosine"):
        lines.append(
            f"  TF-IDF cosine ({u['n_with_cosine']} turns): "
            f"avg {u['cosine_avg']:.3f} · p50 {u['cosine_p50']:.3f} · p95 {u['cosine_p95']:.3f}"
        )
    # P1.4: self-rating distribution
    if u.get("n_self_rated"):
        dist = u.get("self_rating_dist", {})
        total = u["n_self_rated"]
        parts = []
        for label in ("useful", "partial", "noise", "absent"):
            cnt = dist.get(label, 0)
            if cnt:
                parts.append(f"{label}={cnt} ({100*cnt//total}%)")
        lines.append(f"  self-rating ({total} turns): " + " · ".join(parts))
    else:
        lines.append("  self-rating: n/a")
    # P1.3b: era_judge score distribution
    n_era = u.get("n_era_judged", 0)
    if n_era:
        era_avg = u.get("era_judge_avg")
        era_dist = u.get("era_judge_dist", {})
        dist_str = "  ".join(f"{k}★={v}" for k, v in sorted(era_dist.items()))
        lines.append(f"  era_judge ({n_era} scored): avg {era_avg:.2f}  {dist_str}")
    else:
        lines.append("  era_judge: n/a")
    lines.append("")

    f = s["followups"]
    lines.append("── Followup signals (correction-rate proxy) ──")
    lines.append(f"  classified: {f['n']}  by rating: {f['by_rating']}")
    lines.append("")

    r = s["retrieval"]
    lines.append("── query_memory/read_memory ──")
    lines.append(f"  calls: {r['n_calls']}  by fn: {r['by_fn']}")
    lines.append(f"  fallback: {r['fallback_pct']}%  latency p50/p95: {_fmt_ms(r['latency_p50'])}/{_fmt_ms(r['latency_p95'])}")
    lines.append("")

    if s["by_topic"]:
        lines.append("── Turns by topic ──")
        for tid, c in s["by_topic"].items():
            lines.append(f"  {tid or '(unknown)':>10s}: {c}")

    # ── Track D: Cognitive Mode Segmentation ──────────────────────────────
    lines.append("")
    lines.append("── Track D: Cognitive Mode Segmentation ─────────────────────────────")
    _meas_path = LOGS_DIR / "measurement.jsonl"
    if not _meas_path.exists():
        lines.append("  (measurement.jsonl not found)")
    else:
        # Load all rows from measurement.jsonl (both turn_measurement and turn_classification)
        _all_rows: list[dict] = []
        for _line in _meas_path.read_text().splitlines():
            _line = _line.strip()
            if not _line:
                continue
            try:
                _all_rows.append(json.loads(_line))
            except json.JSONDecodeError:
                continue

        # Build indexes on turn_id (last-row-wins = latest classification supersedes old)
        _clf_by_turn: dict[str, dict] = {}
        for r in _all_rows:
            if r.get("kind") == "turn_classification" and r.get("turn_id"):
                _clf_by_turn[r["turn_id"]] = r  # overwrite → last row wins

        _meas_by_turn: dict[str, dict] = {}
        for r in _all_rows:
            if r.get("kind") == "turn_measurement" and r.get("turn_id"):
                _meas_by_turn[r["turn_id"]] = r

        _v2_count = sum(1 for r in _clf_by_turn.values() if r.get("taxonomy_version") == "cognitive-mode-v2")
        _v1_count = len(_clf_by_turn) - _v2_count
        _version_note = ""
        if _v1_count and _v2_count:
            _version_note = f" (v2={_v2_count} · v1-legacy={_v1_count})"
        elif _v2_count:
            _version_note = f" (v2={_v2_count})"
        elif _v1_count:
            _version_note = f" (v1-legacy={_v1_count})"

        lines.append(f"  Classification companion rows: {len(_clf_by_turn)}{_version_note}")

        if not _clf_by_turn:
            lines.append("  (no classification rows yet — run classify_backfill.py or wait for live turns)")
        else:
            # Also build utility index (utility.jsonl rows keyed by turn_id for
            # match_rate + composite). Codex r1 P1: index ONLY kind=="turn"
            # rows — composite_update rows carry their own turn_id AND a stale
            # schema_version:1, so indexing them would overwrite the real
            # schema-6 turn row and the era-gate would then null its match_rate.
            _util_path = LOGS_DIR / "utility.jsonl"
            _util_by_turn: dict[str, dict] = {}
            if _util_path.exists():
                for _line in _util_path.read_text().splitlines():
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _ur = json.loads(_line)
                    except json.JSONDecodeError:
                        continue
                    if _ur.get("kind") != "turn":
                        continue
                    _utid = _ur.get("turn_id", "")
                    if _utid:
                        _util_by_turn[_utid] = _ur

            # Accumulate per-mode metrics using join: measurement → classification → utility
            _mode_buckets: dict[str, list[dict]] = {}
            for _tid, _clf in _clf_by_turn.items():
                _mode = _clf.get("turn_mode", "unclassified")
                _mrow = _meas_by_turn.get(_tid, {})
                _urow = _util_by_turn.get(_tid, {})
                # Era-gate match_rate: pre-schema-6 rows use incomparable
                # (non-classifier-aware) eligibility semantics.
                _usv = _urow.get("schema_version")
                _match_rate = (
                    _urow.get("match_rate")
                    if isinstance(_usv, int) and _usv >= 6 else None
                )
                _composite = _urow.get("composite_score") or _mrow.get("utility_composite_score")
                if _mode not in _mode_buckets:
                    _mode_buckets[_mode] = []
                _mode_buckets[_mode].append({
                    "match_rate": _match_rate,
                    "composite_score": _composite,
                    "input_mode": _clf.get("input_mode", "chat"),
                    "transition_flag": _clf.get("transition_flag", False),
                    "confidence": _clf.get("confidence", 0.0),
                })

            lines.append(f"  {'Mode':<15} {'Turns':>6}  {'match_rate':>12}  {'composite':>12}  {'transitions':>12}  {'avg_conf':>9}")
            lines.append(f"  {'-'*15} {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*9}")
            for _mode in sorted(_mode_buckets.keys()):
                _entries = _mode_buckets[_mode]
                _n = len(_entries)
                _mrs = [e["match_rate"] for e in _entries if e["match_rate"] is not None]
                _cs = [e["composite_score"] for e in _entries if e["composite_score"] is not None]
                _transitions = sum(1 for e in _entries if e["transition_flag"])
                _confs = [e["confidence"] for e in _entries if e["confidence"] is not None]
                _mr_avg = f"{sum(_mrs)/len(_mrs):.3f}" if _mrs else "n/a"
                _cs_avg = f"{sum(_cs)/len(_cs):.3f}" if _cs else "n/a"
                _conf_avg = f"{sum(_confs)/len(_confs):.2f}" if _confs else "n/a"
                lines.append(
                    f"  {_mode:<15} {_n:>6}  {_mr_avg:>12}  {_cs_avg:>12}  {_transitions:>12}  {_conf_avg:>9}"
                )

    return "\n".join(lines)


def _resolve_judge_db() -> Path:
    """Resolve judge.db path — same logic as daemon_adapter._resolve_db_path().

    Priority: PDCT_JUDGE_DB env var → DCT_DATA_DIR/judge.db → default data/ path.
    """
    env_path = os.environ.get("PDCT_JUDGE_DB", "").strip()
    if env_path:
        return Path(env_path)
    data_dir_env = os.environ.get("DCT_DATA_DIR", "").strip()
    if data_dir_env:
        return Path(data_dir_env) / "judge.db"
    # Default: sibling data/ dir relative to logs/
    return LOGS_DIR.parent / "data" / "judge.db"


def judge_report(judge_db_path: Path) -> str:
    """Read judge.db and return formatted summary. Never raises."""
    import sqlite3 as _sqlite3
    if not judge_db_path.exists():
        return (
            f"  judge.db not found at {judge_db_path}\n"
            "  Run with PDCT_JUDGE_ENQUEUE=1 to start collecting."
        )
    try:
        # Re-assert 0600 mode per judge schema contract (DB stores redacted user text)
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
            from dct.judge.schema import init_db  # type: ignore
            init_db(judge_db_path)
        except Exception:
            pass  # best-effort; proceed to read regardless
        conn = _sqlite3.connect(str(judge_db_path), timeout=5)
        conn.row_factory = _sqlite3.Row

        # Schema version
        sv_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        schema_version = sv_row["value"] if sv_row else "unknown"

        # Job queue stats
        status_rows = conn.execute(
            "SELECT status, COUNT(*) as n FROM judge_jobs GROUP BY status"
        ).fetchall()
        status_counts = {r["status"]: r["n"] for r in status_rows}
        total_jobs = sum(status_counts.values())

        # Stale claimed (stuck > 10min)
        import time as _time
        stale_claimed = conn.execute(
            "SELECT COUNT(*) FROM judge_jobs WHERE status='claimed' AND claimed_at < ?",
            (_time.time() - 600,),
        ).fetchone()[0]

        # Daily cap state
        from datetime import date as _date
        today = _date.today().isoformat()
        cap_row = None
        try:
            cap_row = conn.execute(
                "SELECT enqueued_count, daily_cap FROM judge_daily_counters WHERE day=?",
                (today,),
            ).fetchone()
        except _sqlite3.OperationalError:
            pass  # table may not exist in older schema versions

        # Score distribution
        score_rows = conn.execute(
            "SELECT score, COUNT(*) as n FROM judge_results "
            "WHERE score IS NOT NULL GROUP BY score ORDER BY score"
        ).fetchall()
        score_dist = {r["score"]: r["n"] for r in score_rows}
        total_scored = sum(score_dist.values())

        # Fail reasons
        fail_rows = conn.execute(
            "SELECT fail_reason, COUNT(*) as n FROM judge_results "
            "WHERE fail_reason IS NOT NULL GROUP BY fail_reason ORDER BY n DESC LIMIT 5"
        ).fetchall()
        conn.close()
    except Exception as e:
        return f"  judge.db read error: {e}"

    lines = [f"  schema_version={schema_version}  total_jobs={total_jobs}"]

    # Queue status
    for status in ("pending", "claimed", "completed", "failed", "skipped"):
        cnt = status_counts.get(status, 0)
        if cnt:
            lines.append(f"    {status}: {cnt}")
    if stale_claimed:
        lines.append(f"    ⚠ stale claimed (>10min): {stale_claimed}")

    # Daily cap
    if cap_row:
        lines.append(
            f"  today: {cap_row['enqueued_count']}/{cap_row['daily_cap']} enqueued"
        )

    # Completion rate
    completed = status_counts.get("completed", 0)
    if total_jobs:
        lines.append(
            f"  completion rate: {completed}/{total_jobs} ({100*completed//total_jobs}%)"
        )

    # Score distribution
    if score_dist:
        lines.append(f"  scores ({total_scored} turns):")
        max_n = max(score_dist.values())
        for score in sorted(score_dist):
            cnt = score_dist[score]
            bar = "█" * max(1, cnt * 15 // max_n)
            lines.append(f"    {score}: {bar} {cnt}")

    if fail_rows:
        lines.append("  top fail reasons:")
        for r in fail_rows:
            lines.append(f"    {r['fail_reason']}: {r['n']}")

    if total_jobs == 0:
        lines.append("  (no jobs yet — enable with PDCT_JUDGE_ENQUEUE=1)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=None, help="Window in days (default: all-time)")
    ap.add_argument("--topic", type=str, default=None, help="Filter to a single thread_id/topic_id")
    ap.add_argument("--json", action="store_true", help="Emit raw JSON instead of text")
    ap.add_argument("--judge", action="store_true", help="Show judge.db stats (P1.3a)")
    args = ap.parse_args()

    summary = report(args.since, args.topic, args.json)
    if args.json:
        # Include Track B feedback in JSON output
        since_dt: datetime | None = None
        if args.since is not None:
            since_dt = datetime.now(tz=timezone.utc) - timedelta(days=args.since)
        summary["feedback"] = feedback_section(since=since_dt, topic=args.topic)
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(render(summary))
        # Track B section (text mode)
        since_dt = None
        if args.since is not None:
            since_dt = datetime.now(tz=timezone.utc) - timedelta(days=args.since)
        fb = feedback_section(since=since_dt, topic=args.topic)
        total_fb = fb["total_feedback_events"]
        traj_dist = fb["trajectory_length_dist"]
        print()
        print("── Track B: Feedback Events ────────────────────────────────────")
        print(f"  Total feedback events:  {total_fb}")
        if traj_dist:
            dist_parts = "  ".join(
                f"{hops}-hop: {traj_dist[hops]}"
                for hops in sorted(traj_dist)
            )
            print(f"  Trajectory dist:        {dist_parts}")
        else:
            print("  Trajectory dist:        (no data)")

        # ── Track C: Directed transitions + VEC_NEAR (live graph snapshot) ──────
        try:
            import sys as _sys
            _dct_src = Path(__file__).resolve().parent.parent / "src"
            if str(_dct_src) not in _sys.path:
                _sys.path.insert(0, str(_dct_src))
            from dct.heat import build_concept_graph
            from dct.event_log import EventLog
            _events = Path(__file__).resolve().parent.parent / "events.jsonl"
            if _events.exists():
                _log = EventLog(_events)
                _cg = build_concept_graph(_log)
                n_trans = len(_cg.transitions)
                co_edges = sum(1 for e in _cg.typed_edges if e[3] == "co_occur")
                # Asymmetry ratio: for each forward pair, check if reverse exists with different count
                _asym = sum(
                    1 for (a, b), c in _cg.transitions.items()
                    if c != _cg.transitions.get((b, a), 0)
                )
                # VEC_NEAR edges are added by service._load_or_build_graph(),
                # not build_concept_graph() — report from typed_edges field only.
                vec_edges = sum(1 for e in _cg.typed_edges if e[3] == "vec_near")
                # Note: when called from pdct_report, VEC_NEAR is not added (no
                # service call). The count will be 0 here but the build path shows them.
                print()
                print("── Track C: Directed Transitions + Graph ───────────────────────")
                print(f"  Directed transition pairs:  {n_trans:,}")
                print(f"  Asymmetric pairs:           {_asym:,}  ({100*_asym/max(n_trans,1):.0f}% of total)")
                print(f"  CO_OCCUR edges:             {co_edges:,}")
                if vec_edges:
                    print(f"  VEC_NEAR edges:             {vec_edges:,}")
        except Exception as _e:
            print(f"  [Track C snapshot unavailable: {_e}]")

    if args.judge and not args.json:
        # --judge is incompatible with --json (would break JSON parsability)
        print("\n── Judge ──────────────────────────────────────────────────────")
        print(judge_report(_resolve_judge_db()))
    elif args.judge and args.json:
        # Include judge stats inside the JSON object instead
        jdb = _resolve_judge_db()
        judge_text = judge_report(jdb)
        # Re-emit as JSON with judge field (already printed JSON above — re-run)
        summary["judge_db_path"] = str(jdb)
        summary["judge_note"] = "use --judge without --json for formatted output"
        print(json.dumps({"_note": "judge stats only available in text mode; re-run without --json --judge"}, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
