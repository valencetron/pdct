import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_replays_log_and_prints_snapshot(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    # Two events: alpha at t=0, beta at t=30.
    lines = [
        {"ts": 0.0, "source": "telegram", "op": "write", "concepts": ["alpha"], "metadata": {}},
        {"ts": 30.0, "source": "telegram", "op": "write", "concepts": ["beta"], "metadata": {}},
    ]
    log_path.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dct",
            "--log",
            str(log_path),
            "--now",
            "60",
            "--half-life",
            "60",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out = proc.stdout
    # heat formula: 0.5 ** (elapsed / half_life)
    # beta: ts=30, now=60, elapsed=30, half_life=60  → 0.5^(0.5) ≈ 0.7071
    # alpha: ts=0,  now=60, elapsed=60, half_life=60 → 0.5^(1.0) = 0.5000
    lines_out = [l for l in out.strip().splitlines() if l.strip()]
    assert len(lines_out) == 2
    beta_name, beta_heat = lines_out[0].split("\t")
    alpha_name, alpha_heat = lines_out[1].split("\t")
    assert beta_name == "beta"
    assert alpha_name == "alpha"
    assert float(beta_heat) == pytest.approx(0.5 ** 0.5, rel=1e-4)
    assert float(alpha_heat) == pytest.approx(0.5, rel=1e-4)


def test_cli_exits_nonzero_on_missing_log(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    proc = subprocess.run(
        [sys.executable, "-m", "dct", "--log", str(missing), "--now", "1", "--half-life", "1"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "not found" in proc.stderr
