# tests/test_read_health.py
"""PDCT read-side health probe — verdicts, stages, escalation, persistence.

Covers the Codex-audit findings on the 2026-07-16 health work:
  - unreadable ledger is a monitoring failure, never silently ok
  - Layer-3 stages (model churn, stale events, scan errors) escalate the
    verdict instead of being observability-only dead config
  - model scan hard-caps its window (clock-only timestamps valid < 12h)
  - heal-request files are written atomically (no partial JSON visible)
  - heal cooldown survives a Telegram outage (timestamp persisted before
    any post attempt)
"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import dct_read_health as rh


# ── verdicts ──────────────────────────────────────────────────────────

def test_unreadable_ledger_is_broken():
    stats = rh.scan_window(Path("/nonexistent/measurement.jsonl"), 60, time.time())
    assert stats["error"]
    assert rh.verdict_for(stats) == "BROKEN"


def test_stage_escalation_meta_tensor():
    model = {"constructions": 0, "meta_tensor": 3, "error": None}
    assert rh.apply_stage_verdicts("HEALTHY", model, 1.0) == "DEGRADED"
    assert rh.apply_stage_verdicts(None, model, 1.0) == "DEGRADED"


def test_stage_escalation_construction_churn():
    model = {"constructions": rh.CONSTRUCT_WARN + 1, "meta_tensor": 0, "error": None}
    assert rh.apply_stage_verdicts("HEALTHY", model, 1.0) == "DEGRADED"


def test_stage_escalation_stale_events():
    model = {"constructions": 0, "meta_tensor": 0, "error": None}
    assert rh.apply_stage_verdicts("HEALTHY", model, rh.GRAPH_STALE_H + 1) == "DEGRADED"
    assert rh.apply_stage_verdicts("HEALTHY", model, None) == "DEGRADED"


def test_stage_escalation_scan_error():
    model = {"constructions": 0, "meta_tensor": 0, "error": "daemon log missing"}
    assert rh.apply_stage_verdicts("HEALTHY", model, 1.0) == "DEGRADED"


def test_quiet_window_soft_signals_stay_quiet():
    """Codex r2: on a no-verdict (quiet) window, soft signals (churn, stale
    events, unreadable log) must NOT alert — only hard meta-tensor does."""
    soft = {"constructions": rh.CONSTRUCT_WARN + 1, "meta_tensor": 0,
            "error": "daemon log missing"}
    assert rh.apply_stage_verdicts(None, soft, None) is None
    hard = {"constructions": 0, "meta_tensor": 1, "error": None}
    assert rh.apply_stage_verdicts(None, hard, 1.0) == "DEGRADED"


def test_stages_never_soften_broken():
    model = {"constructions": 0, "meta_tensor": 0, "error": None}
    assert rh.apply_stage_verdicts("BROKEN", model, 1.0) == "BROKEN"


def test_healthy_when_all_stages_clean():
    model = {"constructions": 0, "meta_tensor": 0, "error": None}
    assert rh.apply_stage_verdicts("HEALTHY", model, 1.0) == "HEALTHY"


# ── model scan ────────────────────────────────────────────────────────

def test_model_scan_missing_log_reports_error():
    counts = rh.scan_model_health(Path("/nonexistent/daemon.err.log"), 60, time.time())
    assert counts["error"]


def test_model_scan_caps_window(tmp_path):
    """A 24h request must scan at most 11h — clock-only timestamps make
    longer windows count everything in the tail (wrap ambiguity)."""
    log = tmp_path / "daemon.err.log"
    # A line stamped 12h "ago" by clock time must NOT be counted even
    # when the caller asks for a 24h window.
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(hours=12)).strftime("%H:%M:%S")
    recent = datetime.now().strftime("%H:%M:%S")
    log.write_text(
        f"{old} [daemon] Load pretrained SentenceTransformer: x\n"
        f"{recent} [daemon] Load pretrained SentenceTransformer: x\n")
    counts = rh.scan_model_health(log, 24 * 60, time.time())
    assert counts["constructions"] == 1
    assert counts["error"] is None


def test_model_scan_previous_day_same_clock_not_counted(tmp_path):
    """Codex r2: a line from YESTERDAY with a clock time near now must not
    look brand new. The mtime-anchored backwards walk counts the midnight
    crossing between it and the newest line."""
    log = tmp_path / "daemon.err.log"
    from datetime import datetime, timedelta
    now = datetime.now()
    yesterday_same_clock = (now - timedelta(minutes=2)).strftime("%H:%M:%S")
    # Sequence spans a midnight: yesterday's line (clock ≈ now), then a
    # line late yesterday evening, then today's fresh line.
    log.write_text(
        f"{yesterday_same_clock} [daemon] Load pretrained SentenceTransformer: x\n"
        f"23:59:00 [daemon] noise line\n"
        f"00:10:00 [daemon] noise line\n"
        f"{now.strftime('%H:%M:%S')} [daemon] Load pretrained SentenceTransformer: x\n")
    # The scanner only tracks MATCHED lines; the midnight crossing shows up
    # as a backwards clock jump between the two matched lines.
    counts = rh.scan_model_health(log, 60, time.time())
    assert counts["constructions"] == 1, \
        "yesterday's same-clock-time line must be aged a full day back"


# ── heal requests ─────────────────────────────────────────────────────

def test_heal_request_atomic_and_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(rh, "HEAL_REQUEST_DIR", tmp_path)
    now = time.time()
    assert rh.write_heal_request("test reason", now)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    req = json.loads(files[0].read_text())  # complete JSON, never partial
    assert req["target"] == "daemon" and req["ts"] == now
    assert not list(tmp_path.glob("*.tmp")), "temp file must not linger"


def test_heal_cooldown_survives_post_failure(tmp_path, monkeypatch, capsys):
    """Sustained BROKEN + Telegram down: exactly ONE heal-request per
    cooldown window, because the timestamp persists before any post."""
    meas = tmp_path / "measurement.jsonl"
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = [json.dumps({"ts": now_iso, "pdct_skipped_reason": "cascade_timeout",
                        "cascade_latency_ms": 3000}) for _ in range(5)]
    meas.write_text("\n".join(rows) + "\n")

    heal_dir = tmp_path / "heal"
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(
        {"verdict": "BROKEN", "last_alert_unix": 0,
         "consecutive_broken": rh.ESCALATE_AFTER_TICKS - 1,
         "last_heal_request_unix": 0}))

    monkeypatch.setattr(rh, "MEASUREMENT_JSONL", meas)
    monkeypatch.setattr(rh, "STATE_PATH", state_path)
    monkeypatch.setattr(rh, "HEAL_REQUEST_DIR", heal_dir)
    monkeypatch.setattr(rh, "DAEMON_ERR_LOG", tmp_path / "no.log")
    monkeypatch.setattr(rh, "EVENTS_JSONL", meas)  # any fresh file
    monkeypatch.setattr(rh, "post_to_telegram", lambda *a, **k: False)
    monkeypatch.setattr(sys, "argv", ["dct_read_health.py"])

    rh.main()  # tick 1: escalates, post fails
    assert len(list(heal_dir.glob("*.json"))) == 1
    st = json.loads(state_path.read_text())
    assert st["last_heal_request_unix"] > 0, \
        "heal timestamp must persist despite Telegram failure"

    rh.main()  # tick 2 inside cooldown: must NOT write another request
    assert len(list(heal_dir.glob("*.json"))) == 1


# ── H1: warm-up grace (2026-07-17 hardening campaign) ─────────────────

def test_warmup_neutralizes_verdict(monkeypatch):
    import dct_read_health as rh
    monkeypatch.setattr(rh, "daemon_uptime_s", lambda: 120.0)
    assert rh.effective_verdict("BROKEN") is None
    assert rh.effective_verdict("DEGRADED") is None
    assert rh.effective_verdict("HEALTHY") == "HEALTHY"


def test_no_warmup_when_uptime_unknown(monkeypatch):
    import dct_read_health as rh
    monkeypatch.setattr(rh, "daemon_uptime_s", lambda: None)
    assert rh.effective_verdict("BROKEN") == "BROKEN"


def test_no_warmup_when_warm(monkeypatch):
    import dct_read_health as rh
    monkeypatch.setattr(rh, "daemon_uptime_s", lambda: 999.0 + rh.WARMUP_S)
    assert rh.effective_verdict("BROKEN") == "BROKEN"
