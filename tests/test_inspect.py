import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_DIR_TG = Path(__file__).parent / "fixtures" / "telegram"
FIXTURE_DIR_CC = Path(__file__).parent / "fixtures" / "claude_code"


def test_inspect_telegram_prints_per_turn_lines(tmp_path):
    src = FIXTURE_DIR_TG / "mixed_signals.messages.json"
    target = tmp_path / "1_2.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable, "-m", "dct.inspect",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, result.stderr
    # At least one per-turn line present.
    assert "concepts=" in result.stdout
    assert "role=" in result.stdout
    # Summary footer present.
    assert "turns parsed" in result.stdout


def test_inspect_claude_code_prints_per_turn_lines(tmp_path):
    src = FIXTURE_DIR_CC / "simple_session.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable, "-m", "dct.inspect",
            "--source", "claude-code",
            "--input", str(tmp_path / "*.jsonl"),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert "context-driven-traversal" in result.stdout
    assert "turns parsed" in result.stdout


def test_inspect_limit_respected(tmp_path):
    src = FIXTURE_DIR_TG / "mixed_signals.messages.json"
    target = tmp_path / "1_2.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable, "-m", "dct.inspect",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
            "--limit", "1",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    # Exactly one per-turn line (counted by "concepts=" substring).
    assert result.stdout.count("concepts=") == 1


def test_inspect_min_concepts_filters(tmp_path):
    src = FIXTURE_DIR_TG / "mixed_signals.messages.json"
    target = tmp_path / "1_2.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")

    # --min-concepts 2 excludes turns with 0 or 1 concepts.
    result = subprocess.run(
        [
            sys.executable, "-m", "dct.inspect",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
            "--min-concepts", "2",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    lines = [ln for ln in result.stdout.splitlines() if "concepts=" in ln]
    for ln in lines:
        bracket = ln[ln.index("concepts=") + len("concepts=") :]
        bracket = bracket.split("text=")[0].strip()
        inner = bracket.strip("[]").strip()
        if inner:
            assert inner.count(",") >= 1


def test_inspect_empty_glob_exits_zero(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "dct.inspect",
            "--source", "telegram",
            "--input", str(tmp_path / "no-such-*.json"),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    assert "no files matched" in result.stderr.lower()


def test_main_success_telegram(tmp_path, monkeypatch, capsys):
    """Test main() with valid telegram fixture."""
    from dct.inspect import main

    src = FIXTURE_DIR_TG / "mixed_signals.messages.json"
    target = tmp_path / "1_2.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.inspect",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
        ],
    )
    ret = main()
    assert ret == 0
    captured = capsys.readouterr()
    assert "concepts=" in captured.out
    assert "turns parsed" in captured.out


def test_main_success_claude_code(tmp_path, monkeypatch, capsys):
    """Test main() with valid claude-code fixture."""
    from dct.inspect import main

    src = FIXTURE_DIR_CC / "simple_session.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.inspect",
            "--source", "claude-code",
            "--input", str(tmp_path / "*.jsonl"),
        ],
    )
    ret = main()
    assert ret == 0
    captured = capsys.readouterr()
    assert "context-driven-traversal" in captured.out
    assert "turns parsed" in captured.out


def test_main_empty_glob_returns_zero(tmp_path, monkeypatch, capsys):
    """Test main() with glob that matches nothing."""
    from dct.inspect import main

    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.inspect",
            "--source", "telegram",
            "--input", str(tmp_path / "no-such-*.json"),
        ],
    )
    ret = main()
    assert ret == 0
    captured = capsys.readouterr()
    assert "no files matched" in captured.err.lower()


def test_main_limit_respected(tmp_path, monkeypatch, capsys):
    """Test main() respects --limit flag."""
    from dct.inspect import main

    src = FIXTURE_DIR_TG / "mixed_signals.messages.json"
    target = tmp_path / "1_2.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.inspect",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
            "--limit", "1",
        ],
    )
    ret = main()
    assert ret == 0
    captured = capsys.readouterr()
    assert captured.out.count("concepts=") == 1


def test_main_min_concepts_filters(tmp_path, monkeypatch, capsys):
    """Test main() respects --min-concepts flag."""
    from dct.inspect import main

    src = FIXTURE_DIR_TG / "mixed_signals.messages.json"
    target = tmp_path / "1_2.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.inspect",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
            "--min-concepts", "2",
        ],
    )
    ret = main()
    assert ret == 0
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if "concepts=" in ln]
    for ln in lines:
        bracket = ln[ln.index("concepts=") + len("concepts=") :]
        bracket = bracket.split("text=")[0].strip()
        inner = bracket.strip("[]").strip()
        if inner:
            assert inner.count(",") >= 1


def test_main_skips_malformed_file(tmp_path, monkeypatch, capsys):
    """Test main() skips malformed files and continues."""
    from dct.inspect import main

    bad = tmp_path / "1_1.messages.json"
    bad.write_text(FIXTURE_DIR_TG.joinpath("malformed.json").read_text(), encoding="utf-8")
    good = tmp_path / "2_2.messages.json"
    good.write_text(FIXTURE_DIR_TG.joinpath("mixed_signals.messages.json").read_text(), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.inspect",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
        ],
    )
    ret = main()
    assert ret == 0
    captured = capsys.readouterr()
    # Bad file skipped, good file processed.
    assert "skipping" in captured.err.lower()
    assert "concepts=" in captured.out


def test_inspect_shows_per_source_summary(tmp_path, monkeypatch, capsys):
    from dct.inspect import main
    cc_src = Path(__file__).parent / "fixtures" / "claude_code" / "with_tool_uses.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(cc_src.read_text(), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.inspect",
            "--source", "claude-code",
            "--input", str(tmp_path / "*.jsonl"),
        ],
    )
    ret = main()
    assert ret == 0
    out = capsys.readouterr().out
    # Footer per-source breakdown appears
    assert "prose:" in out
    assert "tool_input_path:" in out
    assert "tool_input_structured:" in out


def test_inspect_source_only_filter(tmp_path, monkeypatch, capsys):
    from dct.inspect import main
    cc_src = Path(__file__).parent / "fixtures" / "claude_code" / "with_tool_uses.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(cc_src.read_text(), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.inspect",
            "--source", "claude-code",
            "--input", str(tmp_path / "*.jsonl"),
            "--source-only", "tool_input_path",
        ],
    )
    ret = main()
    assert ret == 0
    out = capsys.readouterr().out
    # Only tool_input_path lines appear in body; no prose lines.
    assert "src=tool_input_path" in out
    assert "src=prose" not in out


def test_inspect_src_column_on_body_lines(tmp_path, monkeypatch, capsys):
    from dct.inspect import main
    cc_src = Path(__file__).parent / "fixtures" / "claude_code" / "with_tool_uses.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(cc_src.read_text(), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.inspect",
            "--source", "claude-code",
            "--input", str(tmp_path / "*.jsonl"),
        ],
    )
    ret = main()
    assert ret == 0
    out = capsys.readouterr().out
    assert "src=prose" in out
    assert "src=tool_input_path" in out


def test_inspect_reports_vault_source(tmp_path, monkeypatch, capsys):
    from dct.inspect import main

    md = tmp_path / "note.md"
    md.write_text("""---
concepts: [retell, voice-pipeline]
---

Body [[MCP Bridge]].
""", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["dct.inspect", "--source", "vault", "--input", str(md)],
    )
    assert main() == 0
    out = capsys.readouterr().out
    assert "src=vault" in out
    assert "retell" in out
    assert "mcp-bridge" in out
