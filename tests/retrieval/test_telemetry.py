import json
from pathlib import Path

from dct.retrieval import telemetry


def test_log_call_appends_jsonl(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "retrieval.jsonl"
    monkeypatch.setattr(telemetry, "LOG_PATH", log_path)
    telemetry.log_call(surface="voice", fn="query_memory",
                       seed="morphogenetic fields", result_count=4,
                       used_fallback=False, latency_ms=127)
    telemetry.log_call(surface="telegram", fn="read_memory",
                       seed="abcd-1234", result_count=1,
                       used_fallback=False, latency_ms=12)

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["surface"] == "voice"
    assert rec["fn"] == "query_memory"
    assert rec["seed"] == "morphogenetic fields"
    assert rec["result_count"] == 4
    assert rec["used_fallback"] is False
    assert rec["latency_ms"] == 127
    assert "ts" in rec  # ISO timestamp


def test_log_call_swallows_io_errors(tmp_path: Path, monkeypatch) -> None:
    # Point at a path inside a nonexistent dir; logger should auto-mkdir.
    log_path = tmp_path / "deep" / "nested" / "retrieval.jsonl"
    monkeypatch.setattr(telemetry, "LOG_PATH", log_path)
    telemetry.log_call(surface="cc", fn="query_memory", seed="x",
                       result_count=0, used_fallback=True, latency_ms=5)
    assert log_path.exists()


def test_log_call_truncates_long_seed(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "retrieval.jsonl"
    monkeypatch.setattr(telemetry, "LOG_PATH", log_path)
    long_seed = "x" * 5000
    telemetry.log_call(surface="cc", fn="query_memory", seed=long_seed,
                       result_count=0, used_fallback=False, latency_ms=1)
    rec = json.loads(log_path.read_text().splitlines()[0])
    assert len(rec["seed"]) <= 512
