"""PDCT Read-Side Health Probe — immediate cascade-failure alarms.

Born 2026-07-16: PDCT retrieval (the cascade) silently failed on 94% of
turns for 24h+ (pdct_skipped_reason=cascade_timeout) while the daily
write-side digest reported healthy. Alex: "if something like pdct entirely
breaks reading, that needs to be brought up immediately to the PDCT channel."

This probe is the read-side counterpart to dct_health_digest.py:
  - scans measurement.jsonl for turns in the last WINDOW_MIN minutes
  - computes cascade success/timeout rates + latency percentiles
  - posts a CRITICAL alert to Telegram topic 0 when the read side is
    broken, and a RECOVERED notice when it heals
  - state-change dedup via a state file: one alert per transition, with a
    re-alert every REALERT_H hours while the breakage persists (so an
    unfixed alarm can't scroll away silently)

Runs every 15 min via launchd (com.exampleco.pdct-read-health). Designed to
be cheap: single pass over the tail of measurement.jsonl, no model loads,
no daemon interaction.

Usage:
    python3 scripts/dct_read_health.py [--dry-run] [--topic N] [--window-min N]

Alarm rules (turns = rows with pdct_skipped_reason in window):
  - turns < MIN_TURNS            → no verdict (quiet hours), state unchanged
  - timeout_rate >= BROKEN_RATE  → BROKEN
  - timeout_rate <= HEALTHY_RATE → HEALTHY
  - in between                   → DEGRADED (alert once, like BROKEN, softer copy)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_PT = ZoneInfo("America/Los_Angeles")

_REPO = Path(__file__).resolve().parent.parent
MEASUREMENT_JSONL = _REPO / "logs" / "measurement.jsonl"
STATE_PATH = _REPO / "logs" / "read_health_state.json"
STACK_CFG = Path.home() / "example-stack" / "config" / "stack.json"
DAEMON_ERR_LOG = (Path.home() / "example-stack" / "tools" / "telegram-dispatch"
                  / "daemon.err.log")
EVENTS_JSONL = _REPO / "events.jsonl"  # retrieval substrate — must keep growing
# Self-heal escalation: heal-requests consumed by health_watchdog (which
# owns the sanctioned kick authority + rate limit + audit trail — this
# probe never restarts anything itself).
HEAL_REQUEST_DIR = (Path.home() / "example-stack" / "tools" / "telegram-dispatch"
                    / "runtime" / "heal-requests")

# Thresholds (module-level so tests can reference them)
WINDOW_MIN = 60          # look-back window for the verdict
MIN_TURNS = 3            # below this, not enough signal — stay quiet
BROKEN_RATE = 0.50       # >=50% cascade_timeout → BROKEN
HEALTHY_RATE = 0.20      # <=20% → HEALTHY (hysteresis gap vs BROKEN_RATE)
REALERT_H = 6.0          # re-alert cadence while broken
TAIL_BYTES = 4_000_000   # read at most this much of the ledger tail
ESCALATE_AFTER_TICKS = 3     # consecutive BROKEN ticks before heal-request
HEAL_REQUEST_COOLDOWN_H = 1.0  # at most one heal-request per hour
META_TENSOR_DEGRADE = 1  # any meta-tensor error in window → at least DEGRADED
CONSTRUCT_WARN = 4       # model constructions in window beyond this → warn
GRAPH_STALE_H = 26.0     # graph.pkl older than this → warn (daily rebuild)


def _now_pt_str() -> str:
    """Current time as '3:45 PM PT' (12-hour, Alex's format rules)."""
    return datetime.now(_PT).strftime("%-I:%M %p PT")


def _parse_ts(ts) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def scan_window(path: Path, window_min: int, now_unix: float) -> dict:
    """Single tail-pass over measurement.jsonl → read-side stats for the window."""
    reasons: dict[str, int] = {}
    latencies: list[int] = []
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > TAIL_BYTES:
                f.seek(size - TAIL_BYTES)
                f.readline()  # discard partial line
            for raw in f:
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "pdct_skipped_reason" not in row:
                    continue
                t = _parse_ts(row.get("ts"))
                if t is None or now_unix - t > window_min * 60:
                    continue
                reason = str(row.get("pdct_skipped_reason") or "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1
                lat = row.get("cascade_latency_ms")
                if isinstance(lat, (int, float)):
                    latencies.append(int(lat))
    except OSError as e:
        return {"error": f"ledger unreadable: {e}", "turns": 0, "reasons": {}}

    turns = sum(reasons.values())
    timeouts = reasons.get("cascade_timeout", 0)
    latencies.sort()
    return {
        "turns": turns,
        "reasons": reasons,
        "timeout_rate": (timeouts / turns) if turns else 0.0,
        "p50_ms": latencies[len(latencies) // 2] if latencies else None,
        "p95_ms": latencies[int(len(latencies) * 0.95)] if latencies else None,
    }


def scan_model_health(log_path: Path, window_min: int, now_unix: float) -> dict:
    """Layer-3 stage: model construction churn + meta-tensor errors from the
    daemon log tail. Healthy steady-state is ~zero constructions per hour
    (singleton) and zero meta-tensor errors. Timestamps in the log are
    HH:MM:SS local; we only look at the tail and accept lines whose clock
    time falls within the window — clock-only timestamps are wrap-safe ONLY
    for windows under 12h, so the window is hard-capped at 11h (Codex:
    a 24h caller would count everything in the tail). An unreadable log
    returns an explicit error instead of a silently-green zero count."""
    window_min = min(window_min, 660)
    counts = {"constructions": 0, "meta_tensor": 0, "error": None}
    if not log_path.exists():
        counts["error"] = f"daemon log missing: {log_path}"
        return counts
    try:
        st = log_path.stat()
        # Pass 1: collect (clock_seconds, kind) for matching lines in tail
        # order. Log timestamps are HH:MM:SS with no date, so a naive
        # same-clock-time line from YESTERDAY would look brand new (Codex
        # r2). Pass 2 disambiguates by anchoring the LAST line to the log
        # file's mtime and walking backwards — every backwards clock jump
        # is a midnight crossing and subtracts a day.
        # Track EVERY parseable-timestamp line (kind=None for non-matches):
        # midnight crossings are often only visible in the lines BETWEEN two
        # matches, so the rollover walk needs the full sequence.
        entries: list[tuple[int, str | None]] = []
        with open(log_path, "rb") as f:
            if st.st_size > TAIL_BYTES:
                f.seek(st.st_size - TAIL_BYTES)
                f.readline()
            for raw in f:
                line = raw.decode("utf-8", errors="ignore")
                try:
                    clock = (int(line[0:2]) * 3600 + int(line[3:5]) * 60
                             + int(line[6:8]))
                except (ValueError, IndexError):
                    continue
                if "Load pretrained SentenceTransformer" in line:
                    kind = "constructions"
                elif "meta tensor" in line:
                    kind = "meta_tensor"
                else:
                    kind = None
                entries.append((clock, kind))
        if entries:
            # Anchor: last line is no newer than the file mtime; walk
            # backwards, treating every backwards clock jump as a midnight
            # crossing (append-only log ⇒ times within a day are monotonic).
            abs_t = st.st_mtime
            prev_clock = entries[-1][0]
            if entries[-1][1] and now_unix - abs_t <= window_min * 60:
                counts[entries[-1][1]] += 1
            for clock, kind in reversed(entries[:-1]):
                delta = prev_clock - clock
                if delta < 0:
                    delta += 86400  # crossed midnight walking backwards
                abs_t -= delta
                prev_clock = clock
                if kind and now_unix - abs_t <= window_min * 60:
                    counts[kind] += 1
    except OSError as e:
        counts["error"] = f"daemon log unreadable: {e}"
    return counts


def events_age_hours(now_unix: float) -> float | None:
    """Age of the newest retrieval-substrate event. The concept graph is
    rebuilt in-memory from events.jsonl; if this file stops growing, the
    read side goes quietly stale even with a healthy cascade."""
    try:
        return (now_unix - EVENTS_JSONL.stat().st_mtime) / 3600.0
    except OSError:
        return None


def write_heal_request(reason: str, now_unix: float, target: str = "daemon") -> bool:
    """Drop a heal-request for health_watchdog to consume (its kick authority,
    its rate limit, its audit trail). Atomic: written to a non-.json temp
    name then os.replace()d, so the watchdog can never read partial JSON
    (Codex). Returns True if written."""
    try:
        import os
        import uuid
        HEAL_REQUEST_DIR.mkdir(parents=True, exist_ok=True)
        # uuid in BOTH names: unique tmp so overlapping writers can't clobber
        # each other mid-write, unique target so same-second requests don't
        # overwrite (Codex r2 — second-precision names collided).
        uniq = uuid.uuid4().hex[:8]
        path = HEAL_REQUEST_DIR / f"pdct-read-{int(now_unix)}-{uniq}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"target": target, "ts": now_unix, "reason": reason,
             "source": "dct_read_health"}), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as e:
        print(f"[read-health] heal-request write failed: {e}", file=sys.stderr)
        return False


def apply_stage_verdicts(verdict: str | None, model: dict,
                         ev_age: float | None) -> str | None:
    """Layer-3 stages ESCALATE the verdict (never soften it) — without
    this they were observability-only dead configuration (Codex r1).

    Quiet-window nuance (Codex r2): when verdict is None (< MIN_TURNS,
    e.g. overnight or bootstrap), only HARD failures escalate — meta-tensor
    errors mean the model is actively poisoned regardless of traffic.
    Soft signals (construction churn, stale events, unreadable log) only
    escalate a window with real turns; on quiet windows they'd fire
    recurring alerts for ordinary idleness and they surface on the next
    active window anyway."""
    if verdict == "BROKEN":
        return verdict
    hard_bad = model.get("meta_tensor", 0) >= META_TENSOR_DEGRADE
    soft_bad = (
        model.get("error") is not None
        or model.get("constructions", 0) > CONSTRUCT_WARN
        or ev_age is None
        or ev_age > GRAPH_STALE_H
    )
    if hard_bad:
        return "DEGRADED"
    if verdict is None:
        return None
    return "DEGRADED" if soft_bad else verdict


WARMUP_S = 600  # daemon uptime below this → cascade timings are cold-start
                # noise; bad verdicts neutralized to None (no alert, no heal)
DAEMON_SOCK = "/tmp/valence-daemon.sock"


def daemon_uptime_s() -> float | None:
    """Uptime via socket op:health; None = unreachable (no warm-up grace —
    an unreachable daemon must still be reportable as BROKEN)."""
    import socket as _socket
    try:
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(DAEMON_SOCK)
        s.sendall(json.dumps({"op": "health"}).encode())
        s.shutdown(_socket.SHUT_WR)
        chunks = []
        while True:
            c = s.recv(65536)
            if not c:
                break
            chunks.append(c)
        s.close()
        return float(json.loads(b"".join(chunks).decode())["uptime_s"])
    except Exception:
        return None


def effective_verdict(verdict: str | None) -> str | None:
    """Neutralize bad verdicts measured against a warming daemon (H1,
    2026-07-17: post-restart cold caches produced 18-27s preloads that
    read as BROKEN, triggering heal restarts that re-created the cold
    start — the churn loop). HEALTHY passes through: good news is good
    news even during warm-up. One probe tick of blindness max; the next
    tick sees a warm daemon."""
    if verdict in ("BROKEN", "DEGRADED"):
        up = daemon_uptime_s()
        if up is not None and up < WARMUP_S:
            print(f"[read-health] verdict={verdict} neutralized: daemon "
                  f"warming (uptime {up:.0f}s < {WARMUP_S}s)", file=sys.stderr)
            return None
    return verdict


def verdict_for(stats: dict) -> str | None:
    """BROKEN / DEGRADED / HEALTHY, or None when there's too little signal."""
    if stats.get("error"):
        return "BROKEN"  # can't read the ledger at all → treat as broken
    if stats["turns"] < MIN_TURNS:
        return None
    rate = stats["timeout_rate"]
    if rate >= BROKEN_RATE:
        return "BROKEN"
    if rate <= HEALTHY_RATE:
        return "HEALTHY"
    return "DEGRADED"


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"verdict": "HEALTHY", "last_alert_unix": 0.0,
                "consecutive_broken": 0, "last_heal_request_unix": 0.0}


def save_state(state: dict) -> None:
    try:
        import os
        # PID-unique tmp: a manual run overlapping the launchd tick must not
        # clobber the other writer's tmp mid-write (Codex r2). Last replace
        # wins, which is correct for a monotonic monitoring state.
        tmp = STATE_PATH.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, STATE_PATH)  # atomic — no partial state on crash
    except OSError as e:
        print(f"[read-health] WARN state write failed: {e}", file=sys.stderr)


def should_alert(prev: dict, verdict: str, now_unix: float) -> bool:
    """Alert on state transition; re-alert every REALERT_H while not HEALTHY."""
    if verdict != prev.get("verdict"):
        return True
    if verdict in ("BROKEN", "DEGRADED"):
        return now_unix - float(prev.get("last_alert_unix", 0)) >= REALERT_H * 3600
    return False


def build_message(verdict: str, stats: dict, window_min: int,
                  stages: dict | None = None, escalation: str = "") -> str:
    reasons = " ".join(f"{k}={v}" for k, v in
                       sorted(stats.get("reasons", {}).items())) or "none"
    lat = ""
    if stats.get("p50_ms") is not None:
        lat = f"\ncascade latency p50 {stats['p50_ms']}ms / p95 {stats['p95_ms']}ms"
    if verdict == "BROKEN":
        head = "🔴 <b>PDCT READ SIDE BROKEN</b>"
        body = ("cascade retrieval is failing — replies are going out "
                "WITHOUT jogged memory and each one eats the full timeout.")
    elif verdict == "DEGRADED":
        head = "🟠 <b>PDCT read side degraded</b>"
        body = "cascade retrieval is partially failing."
    else:
        head = "🟢 <b>PDCT read side recovered</b>"
        body = "cascade retrieval is landing again."
    import html as _html
    err = f"\n⚠️ {_html.escape(str(stats['error']))}" if stats.get("error") else ""
    stage_lines = ""
    for label, (ok, detail) in (stages or {}).items():
        stage_lines += f"\n{'✅' if ok else '⚠️'} {label}: {_html.escape(str(detail))}"
    esc = f"\n{_html.escape(escalation)}" if escalation else ""
    return (
        f"{head} — {_now_pt_str()}\n"
        f"{body}\n"
        f"last {window_min}m: turns={stats['turns']}, "
        f"timeout_rate={stats['timeout_rate']:.0%}\n"
        f"reasons: {reasons}{lat}{err}{stage_lines}{esc}\n"
        f"<i>probe: dct_read_health.py (every 15m)</i>"
    )


def post_to_telegram(message: str, topic_id: int, chat_id: int = 0) -> bool:
    try:
        cfg = json.loads(STACK_CFG.read_text(encoding="utf-8"))
        bot_token = cfg["channels"]["telegram"]["botToken"]
    except Exception as e:
        print(f"[read-health] ERROR loading bot token: {e}", file=sys.stderr)
        return False
    payload = json.dumps({
        "chat_id": chat_id,
        "message_thread_id": topic_id,
        "text": message,
        "parse_mode": "HTML",
    }).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        if result.get("ok"):
            print(f"[read-health] posted to topic {topic_id}", file=sys.stderr)
            return True
        print(f"[read-health] Telegram error: {result.get('description')}",
              file=sys.stderr)
        return False
    except Exception as e:
        print(f"[read-health] HTTP error: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="PDCT read-side health probe")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print verdict/message, don't post or save state")
    parser.add_argument("--topic", type=int, default=0)
    parser.add_argument("--window-min", type=int, default=WINDOW_MIN)
    args = parser.parse_args()

    now_unix = datetime.now(timezone.utc).timestamp()
    stats = scan_window(MEASUREMENT_JSONL, args.window_min, now_unix)
    verdict = verdict_for(stats)
    prev = load_state()

    # ── Layer-3 stages: model health + graph freshness ──
    model = scan_model_health(DAEMON_ERR_LOG, args.window_min, now_unix)
    ev_age = events_age_hours(now_unix)
    stages: dict = {}
    model_ok = (model.get("error") is None
                and model["meta_tensor"] < META_TENSOR_DEGRADE
                and model["constructions"] <= CONSTRUCT_WARN)
    model_detail = (model["error"] if model.get("error")
                    else f"constructions={model['constructions']} "
                         f"meta_tensor={model['meta_tensor']} "
                         f"({min(args.window_min, 660)}m)")
    stages["model"] = (model_ok, model_detail)
    events_ok = ev_age is not None and ev_age <= GRAPH_STALE_H
    stages["events"] = (events_ok,
                        "events.jsonl missing" if ev_age is None
                        else f"events.jsonl age {ev_age:.1f}h")
    # Stages escalate the verdict — a failing stage is a read-side problem
    # even when the turn window is too quiet for a cascade verdict.
    verdict = apply_stage_verdicts(verdict, model, ev_age)
    verdict = effective_verdict(verdict)  # H1 warm-up grace (None → early return below)

    print(f"[read-health] stats={stats} model={model} events_age={ev_age} "
          f"verdict={verdict} prev={prev.get('verdict')}", file=sys.stderr)

    if verdict is None:
        return  # not enough signal; keep previous state untouched

    # ── Layer-2 escalation: sustained BROKEN → heal-request to watchdog ──
    consecutive = (prev.get("consecutive_broken", 0) + 1
                   if verdict == "BROKEN" else 0)
    last_heal = float(prev.get("last_heal_request_unix", 0))
    escalation = ""
    heal_requested_at = last_heal
    if (consecutive >= ESCALATE_AFTER_TICKS
            and now_unix - last_heal >= HEAL_REQUEST_COOLDOWN_H * 3600):
        reason = (f"cascade timeout_rate={stats['timeout_rate']:.0%} for "
                  f"{consecutive} consecutive ticks; in-process heal failed")
        if args.dry_run:
            escalation = f"🩹 would write heal-request: {reason}"
        elif write_heal_request(reason, now_unix):
            heal_requested_at = now_unix
            # Persist the heal timestamp IMMEDIATELY — before any Telegram
            # attempt. Codex: saving it only after a successful post let a
            # Telegram outage defeat the cooldown (a new heal-request every
            # 15m tick during a sustained failure).
            save_state({"verdict": verdict, "consecutive_broken": consecutive,
                        "last_heal_request_unix": heal_requested_at,
                        "last_alert_unix": float(prev.get("last_alert_unix", 0))})
            escalation = ("🩹 in-process healing failed — heal-request handed "
                          "to health-watchdog (its kick authority + rate "
                          "limit + audit trail)")

    alert = should_alert(prev, verdict, now_unix)
    # A HEALTHY→HEALTHY tick never alerts; recovery only fires on transition.
    if verdict == "HEALTHY" and prev.get("verdict") == "HEALTHY":
        alert = False
    if escalation and not args.dry_run:
        alert = True  # an escalation is always worth a line in the channel

    new_state = {"verdict": verdict, "consecutive_broken": consecutive,
                 "last_heal_request_unix": heal_requested_at,
                 "last_alert_unix": float(prev.get("last_alert_unix", 0))}
    msg = build_message(verdict, stats, args.window_min, stages, escalation)
    if args.dry_run:
        print(f"--- would_alert={alert} consecutive={consecutive} ---\n{msg}")
        return

    if alert:
        if post_to_telegram(msg, args.topic):
            new_state["last_alert_unix"] = now_unix
            save_state(new_state)
        else:
            # Alert delivery failed: persist the escalation counter and heal
            # timestamp (the cooldown must survive a Telegram outage) but
            # keep the OLD verdict — the state transition hasn't been
            # announced yet, so the next tick must retry the alert.
            save_state({"verdict": prev.get("verdict", "HEALTHY"),
                        "consecutive_broken": consecutive,
                        "last_heal_request_unix": heal_requested_at,
                        "last_alert_unix": float(prev.get("last_alert_unix", 0))})
    else:
        save_state(new_state)


if __name__ == "__main__":
    main()
