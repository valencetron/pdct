import os
import subprocess
import sys
from pathlib import Path

import pytest

from dct.event_log import EventLog
from dct.events import EventOp, EventSource
from dct.ingest import IngestStats, ingest_files

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "telegram"
CC_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "claude_code"


def _copy_fixture(src_name: str, tmp_path: Path, chat: str, thread: str) -> Path:
    src = FIXTURE_DIR / src_name
    target = tmp_path / f"{chat}_{thread}.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")
    os.utime(target, (1_700_000_000.0, 1_700_000_000.0))
    return target


def test_ingest_files_writes_expected_events(tmp_path):
    _copy_fixture("mixed_signals.messages.json", tmp_path, "1", "2")
    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)

    stats = ingest_files(
        [tmp_path / "1_2.messages.json"], log,
        source=EventSource.TELEGRAM, dedupe=False,
    )

    events = log.read_all()
    # Turn 0: alpha, beta (non-empty concepts)
    # Turn 1: gamma
    # Turn 2: whitespace only -> no concepts -> no event
    # Turn 3: alpha
    assert len(events) == 3
    assert [e.concepts for e in events] == [["alpha", "beta"], ["gamma"], ["alpha"]]
    assert all(e.source == EventSource.TELEGRAM for e in events)
    assert all(e.op == EventOp.TRAVERSAL for e in events)
    assert stats.files_processed == 1
    assert stats.turns_parsed == 4
    assert stats.events_written == 3
    assert stats.turns_skipped_dedupe == 0


def test_ingest_files_event_metadata_populated(tmp_path):
    target = _copy_fixture("mixed_signals.messages.json", tmp_path, "1", "2")
    log = EventLog(tmp_path / "events.jsonl")
    ingest_files([target], log, source=EventSource.TELEGRAM, dedupe=False)

    events = log.read_all()
    md = events[0].metadata
    assert md["role"] == "user"
    assert md["source_file"] == str(target.resolve())
    assert md["turn_index"] == "0"
    assert md["chat_id"] == "1"
    assert md["thread_id"] == "2"


def test_ingest_files_zero_concept_turns_not_written(tmp_path):
    target = _copy_fixture("mixed_signals.messages.json", tmp_path, "1", "2")
    log = EventLog(tmp_path / "events.jsonl")
    ingest_files([target], log, source=EventSource.TELEGRAM, dedupe=False)

    events = log.read_all()
    # Turn 2 is whitespace-only — excluded
    turn_indices = [int(e.metadata["turn_index"]) for e in events]
    assert 2 not in turn_indices


def test_ingest_dedupe_skips_already_ingested(tmp_path):
    target = _copy_fixture("mixed_signals.messages.json", tmp_path, "1", "2")
    log = EventLog(tmp_path / "events.jsonl")

    # First run writes 3 events.
    first = ingest_files([target], log, source=EventSource.TELEGRAM, dedupe=False)
    assert first.events_written == 3

    # Second run with dedupe skips the 3 turns that were in the log.
    second = ingest_files([target], log, source=EventSource.TELEGRAM, dedupe=True)
    assert second.events_written == 0
    assert second.turns_skipped_dedupe == 3
    assert second.turns_parsed == 4
    # Log still has only 3 events.
    assert len(log.read_all()) == 3


def test_ingest_without_dedupe_appends_duplicates(tmp_path):
    target = _copy_fixture("mixed_signals.messages.json", tmp_path, "1", "2")
    log = EventLog(tmp_path / "events.jsonl")

    ingest_files([target], log, source=EventSource.TELEGRAM, dedupe=False)
    ingest_files([target], log, source=EventSource.TELEGRAM, dedupe=False)

    # Duplicates appended since dedupe is off.
    assert len(log.read_all()) == 6


def test_cli_ingests_telegram_fixture(tmp_path):
    _copy_fixture("mixed_signals.messages.json", tmp_path, "7", "8")
    log_path = tmp_path / "events.jsonl"

    result = subprocess.run(
        [
            sys.executable, "-m", "dct.ingest",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
            "--log", str(log_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, result.stderr

    log = EventLog(log_path)
    events = log.read_all()
    assert len(events) == 3
    assert events[0].concepts == ["alpha", "beta"]


def test_cli_prints_summary_line(tmp_path):
    _copy_fixture("mixed_signals.messages.json", tmp_path, "7", "8")
    log_path = tmp_path / "events.jsonl"
    result = subprocess.run(
        [
            sys.executable, "-m", "dct.ingest",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
            "--log", str(log_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    assert "1 files" in result.stdout
    assert "3 events" in result.stdout


def test_cli_empty_glob_warns_but_exits_zero(tmp_path):
    log_path = tmp_path / "events.jsonl"
    result = subprocess.run(
        [
            sys.executable, "-m", "dct.ingest",
            "--source", "telegram",
            "--input", str(tmp_path / "nothing-matches-*.json"),
            "--log", str(log_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    assert "no files matched" in result.stderr.lower()


def test_cli_unknown_source_exits_nonzero(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "dct.ingest",
            "--source", "mystery-source",
            "--input", str(tmp_path / "*.json"),
            "--log", str(tmp_path / "events.jsonl"),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode != 0
    assert "source" in result.stderr.lower()


def test_cli_malformed_json_exits_nonzero(tmp_path):
    src = FIXTURE_DIR / "malformed.json"
    target = tmp_path / "1_1.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")
    log_path = tmp_path / "events.jsonl"

    result = subprocess.run(
        [
            sys.executable, "-m", "dct.ingest",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
            "--log", str(log_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    assert "skipping" in result.stderr.lower()
    # Log should not exist or be empty (no events written since file skipped)
    if log_path.exists():
        assert log_path.stat().st_size == 0


def test_ingest_dedupe_skips_malformed_turn_index(tmp_path):
    """Test that dedupe gracefully skips entries with malformed turn_index."""
    from dct.events import Event, EventOp, EventSource

    target = _copy_fixture("mixed_signals.messages.json", tmp_path, "1", "2")
    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)

    # Write an event with a non-numeric turn_index to the log.
    bad_event = Event(
        ts=1_700_000_000,
        source=EventSource.TELEGRAM,
        op=EventOp.TRAVERSAL,
        concepts=["alpha"],
        metadata={
            "role": "user",
            "source_file": str(target.resolve()),
            "turn_index": "not-a-number",
            "chat_id": "1",
            "thread_id": "2",
        },
    )
    log.append(bad_event)

    # Ingest with dedupe=True should not crash; it skips the malformed entry.
    stats = ingest_files([target], log, source=EventSource.TELEGRAM, dedupe=True)
    assert stats.files_processed == 1
    assert stats.turns_parsed == 4
    # All 3 turns with valid concepts should be written (the bad entry is skipped in dedupe init)
    assert stats.events_written == 3


def test_main_success(tmp_path, monkeypatch):
    """Test main() with valid arguments."""
    from dct.ingest import main

    target = _copy_fixture("mixed_signals.messages.json", tmp_path, "1", "2")
    log_path = tmp_path / "events.jsonl"

    # Simulate CLI invocation with sys.argv.
    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.ingest",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
            "--log", str(log_path),
        ],
    )
    ret = main()
    assert ret == 0
    assert log_path.exists()
    log = EventLog(log_path)
    assert len(log.read_all()) == 3


def test_main_no_files_matched(tmp_path, monkeypatch):
    """Test main() with glob that matches nothing."""
    from dct.ingest import main

    log_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.ingest",
            "--source", "telegram",
            "--input", str(tmp_path / "no-such-*.json"),
            "--log", str(log_path),
        ],
    )
    ret = main()
    assert ret == 0  # Exits cleanly with 0 for no matches.


def test_main_malformed_input_error(tmp_path, monkeypatch, capsys):
    """Test main() with malformed JSON file — now skips with warning."""
    from dct.ingest import main

    src = FIXTURE_DIR / "malformed.json"
    target = tmp_path / "1_1.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")
    log_path = tmp_path / "events.jsonl"

    monkeypatch.setattr(
        sys, "argv",
        [
            "dct.ingest",
            "--source", "telegram",
            "--input", str(tmp_path / "*.messages.json"),
            "--log", str(log_path),
        ],
    )
    ret = main()
    assert ret == 0  # Exits cleanly; file skipped.
    captured = capsys.readouterr()
    assert "skipping" in captured.err.lower()


def test_ingest_files_skips_malformed_in_batch(tmp_path, capsys):
    """Test that ingest_files skips malformed files and continues."""
    good = _copy_fixture("mixed_signals.messages.json", tmp_path, "1", "2")
    bad = tmp_path / "3_4.messages.json"
    bad.write_text("{not valid json", encoding="utf-8")
    log = EventLog(tmp_path / "events.jsonl")

    stats = ingest_files([bad, good], log, source=EventSource.TELEGRAM, dedupe=False)

    # Bad file skipped, good file still processed.
    assert stats.files_processed == 1
    assert stats.events_written == 3
    captured = capsys.readouterr()
    assert "skipping" in captured.err
    assert "3_4.messages.json" in captured.err


def test_cli_expands_tilde_in_input(tmp_path, monkeypatch):
    """Test that --input expands ~ before globbing."""
    # Point HOME at tmp_path so ~ resolves there, drop a fixture in it.
    monkeypatch.setenv("HOME", str(tmp_path))
    _copy_fixture("mixed_signals.messages.json", tmp_path, "5", "6")
    log_path = tmp_path / "events.jsonl"

    result = subprocess.run(
        [
            sys.executable, "-m", "dct.ingest",
            "--source", "telegram",
            "--input", "~/*.messages.json",
            "--log", str(log_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert log_path.exists()
    log = EventLog(log_path)
    assert len(log.read_all()) == 3


def test_ingest_files_claude_code_source(tmp_path):
    src = CC_FIXTURE_DIR / "simple_session.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")
    log = EventLog(tmp_path / "events.jsonl")

    stats = ingest_files([target], log, source=EventSource.CLAUDE_CODE, dedupe=False)
    events = log.read_all()

    # Two non-empty turns -> two events (the third is empty content).
    assert stats.events_written == 2
    assert all(e.source == EventSource.CLAUDE_CODE for e in events)
    # First turn has a wikilink concept.
    assert "context-driven-traversal" in events[0].concepts


def test_ingest_files_claude_code_event_metadata(tmp_path):
    src = CC_FIXTURE_DIR / "simple_session.jsonl"
    proj = tmp_path / "-Users-user-myproject"
    proj.mkdir()
    target = proj / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")
    log = EventLog(tmp_path / "events.jsonl")

    ingest_files([target], log, source=EventSource.CLAUDE_CODE)
    md = log.read_all()[0].metadata
    assert md["role"] == "user"
    assert md["session_id"] == "aaaa-1111"
    assert md["project_slug"] == "-Users-user-myproject"
    assert md["line_idx"] == "0"


def test_cli_claude_code_source(tmp_path):
    src = CC_FIXTURE_DIR / "simple_session.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")
    log_path = tmp_path / "events.jsonl"

    result = subprocess.run(
        [
            sys.executable, "-m", "dct.ingest",
            "--source", "claude-code",
            "--input", str(tmp_path / "*.jsonl"),
            "--log", str(log_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, result.stderr
    log = EventLog(log_path)
    assert len(log.read_all()) == 2


def test_cli_mixed_sources_to_same_log(tmp_path):
    # Telegram ingest first, then Claude Code — both into the same log.
    tg_src = FIXTURE_DIR / "mixed_signals.messages.json"
    tg_target = tmp_path / "9_9.messages.json"
    tg_target.write_text(tg_src.read_text(), encoding="utf-8")

    cc_src = CC_FIXTURE_DIR / "simple_session.jsonl"
    cc_target = tmp_path / "aaaa-1111.jsonl"
    cc_target.write_text(cc_src.read_text(), encoding="utf-8")

    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)
    ingest_files([tg_target], log, source=EventSource.TELEGRAM)
    ingest_files([cc_target], log, source=EventSource.CLAUDE_CODE)

    events = log.read_all()
    sources = {e.source for e in events}
    assert EventSource.TELEGRAM in sources
    assert EventSource.CLAUDE_CODE in sources


def test_ingest_claude_code_emits_per_source_events(tmp_path):
    src = CC_FIXTURE_DIR / "with_tool_uses.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")
    log = EventLog(tmp_path / "events.jsonl")

    ingest_files([target], log, source=EventSource.CLAUDE_CODE)
    events = log.read_all()
    sources = {e.metadata.get("extraction_source") for e in events}
    assert "prose" in sources
    assert "tool_input_path" in sources
    assert "tool_input_structured" in sources


def test_ingest_prose_event_has_extraction_source_metadata(tmp_path):
    tg_target = _copy_fixture("mixed_signals.messages.json", tmp_path, "1", "2")
    log = EventLog(tmp_path / "events.jsonl")
    ingest_files([tg_target], log, source=EventSource.TELEGRAM)
    events = log.read_all()
    assert all(e.metadata["extraction_source"] == "prose" for e in events)


def test_ingest_path_event_carries_denylist_filtered_concepts(tmp_path):
    src = CC_FIXTURE_DIR / "with_tool_uses.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")
    log = EventLog(tmp_path / "events.jsonl")

    ingest_files([target], log, source=EventSource.CLAUDE_CODE)
    events = log.read_all()
    path_events = [e for e in events if e.metadata.get("extraction_source") == "tool_input_path"]
    assert len(path_events) == 1
    # "example-stack", "tools" denylisted; "telegram-dispatch" + "daemon" survive.
    assert "telegram-dispatch" in path_events[0].concepts
    assert "daemon" in path_events[0].concepts
    assert "tools" not in path_events[0].concepts
    assert "example-stack" not in path_events[0].concepts


def test_ingest_structured_event_has_tool_name_metadata(tmp_path):
    src = CC_FIXTURE_DIR / "with_tool_uses.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")
    log = EventLog(tmp_path / "events.jsonl")

    ingest_files([target], log, source=EventSource.CLAUDE_CODE)
    events = log.read_all()
    structured = [
        e for e in events
        if e.metadata.get("extraction_source") == "tool_input_structured"
    ]
    assert len(structured) == 2  # mc_card_update + Skill = 2 turns
    tool_names = {e.metadata["tool_name"] for e in structured}
    assert "Skill" in tool_names
    assert any("mc_card_update" in tn for tn in tool_names)


def test_dedupe_key_includes_extraction_source(tmp_path):
    # Re-ingest with dedupe: events already present should not duplicate.
    src = CC_FIXTURE_DIR / "with_tool_uses.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")
    log = EventLog(tmp_path / "events.jsonl")

    first = ingest_files([target], log, source=EventSource.CLAUDE_CODE)
    before = len(log.read_all())
    assert before == first.events_written

    second = ingest_files([target], log, source=EventSource.CLAUDE_CODE, dedupe=True)
    assert second.events_written == 0
    after = len(log.read_all())
    assert after == before  # nothing added


class TestVoiceIngest:
    def test_voice_ingest_end_to_end(self, tmp_path: Path) -> None:
        import json
        from dct.event_log import EventLog
        from dct.events import EventSource
        from dct.ingest import ingest_files

        transcript_path = tmp_path / "20260315T194243Z_conv_abc.json"
        transcript_path.write_text(
            json.dumps({
                "conversation_id": "abc",
                "timestamp": "2026-03-15T19:42:43+00:00",
                "transcript": [
                    {"role": "agent",
                     "message": "Checking the [[voice pipeline]] tonight.",
                     "time_in_call_secs": 0},
                    {"role": "user",
                     "message": "What about the #mission-control-app?",
                     "time_in_call_secs": 3},
                ],
            }),
            encoding="utf-8",
        )
        log_path = tmp_path / "events.jsonl"
        log = EventLog(log_path)
        stats = ingest_files([transcript_path], log, source=EventSource.VOICE)
        assert stats.files_processed == 1
        assert stats.events_written >= 1

        events = list(log.read_all())
        assert events, "expected at least one event"
        assert all(e.source == EventSource.VOICE for e in events)
        assert all(e.metadata.get("extraction_source") == "prose" for e in events)

    def test_voice_ingest_dedupe(self, tmp_path: Path) -> None:
        import json
        from dct.event_log import EventLog
        from dct.events import EventSource
        from dct.ingest import ingest_files

        transcript_path = tmp_path / "20260315T194243Z_conv_abc.json"
        transcript_path.write_text(
            json.dumps({
                "conversation_id": "abc",
                "timestamp": "2026-03-15T19:42:43+00:00",
                "transcript": [
                    {"role": "user",
                     "message": "Discuss the [[voice pipeline]].",
                     "time_in_call_secs": 0},
                ],
            }),
            encoding="utf-8",
        )
        log_path = tmp_path / "events.jsonl"
        log = EventLog(log_path)

        first = ingest_files([transcript_path], log, source=EventSource.VOICE)
        second = ingest_files([transcript_path], log, source=EventSource.VOICE, dedupe=True)

        assert first.events_written >= 1
        assert second.events_written == 0
        assert second.turns_skipped_dedupe >= 1


def test_dedupe_old_events_without_extraction_source_treated_as_prose(tmp_path):
    # Write an old-style event (no extraction_source in metadata), then re-ingest.
    tg_target = _copy_fixture("mixed_signals.messages.json", tmp_path, "1", "2")
    log = EventLog(tmp_path / "events.jsonl")

    from dct.events import Event
    legacy = Event(
        ts=1_700_000_000.0,
        source=EventSource.TELEGRAM,
        op=EventOp.TRAVERSAL,
        concepts=["alpha", "beta"],
        metadata={
            "role": "user",
            "source_file": str(tg_target.resolve()),
            "turn_index": "0",
            "chat_id": "1",
            "thread_id": "2",
            # no extraction_source key
        },
    )
    log.append(legacy)

    stats = ingest_files([tg_target], log, source=EventSource.TELEGRAM, dedupe=True)
    # Turn 0 prose re-derived concepts match legacy's; it MUST be considered
    # duplicate even though legacy lacks extraction_source.
    turn0_count = sum(
        1 for e in log.read_all()
        if e.metadata.get("turn_index") == "0"
    )
    assert turn0_count == 1
    assert stats.turns_skipped_dedupe >= 1


def test_ingest_vault_source_emits_vault_event(tmp_path):
    from dct.event_log import EventLog
    from dct.events import EventSource
    from dct.ingest import ingest_files

    vault_md = tmp_path / "note.md"
    vault_md.write_text("""---
concepts: [voice-pipeline, retell]
---

Body mentions [[MCP Bridge]].
""", encoding="utf-8")

    log = EventLog(tmp_path / "events.jsonl")
    stats = ingest_files([vault_md], log, source=EventSource.VAULT)
    assert stats.files_processed == 1
    assert stats.events_written == 1

    ev = log.read_all()[0]
    assert ev.source == EventSource.VAULT
    assert ev.metadata["extraction_source"] == "vault"
    assert ev.metadata["source_file"] == str(vault_md.resolve())
    assert set(ev.concepts) >= {"voice-pipeline", "retell", "mcp-bridge"}


def test_ingest_vault_skips_empty_file(tmp_path):
    from dct.event_log import EventLog
    from dct.events import EventSource
    from dct.ingest import ingest_files

    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    log = EventLog(tmp_path / "events.jsonl")
    stats = ingest_files([empty], log, source=EventSource.VAULT)
    assert stats.events_written == 0


def test_ingest_cli_accepts_vault_choice(tmp_path, monkeypatch, capsys):
    from dct.ingest import main

    md = tmp_path / "x.md"
    md.write_text("---\nconcepts: [foo-bar]\n---\n\nbody.\n", encoding="utf-8")
    log_path = tmp_path / "events.jsonl"

    monkeypatch.setattr(
        "sys.argv",
        ["dct.ingest", "--source", "vault", "--input", str(md), "--log", str(log_path)],
    )
    assert main() == 0
    out = capsys.readouterr().out
    assert "1 files" in out
