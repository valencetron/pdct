from pathlib import Path

import pytest

from dct.adapters.claude_code import flatten_content, _parse_timestamp, parse_file

CC_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "claude_code"


def test_flatten_plain_string():
    assert flatten_content("hello") == "hello"


def test_flatten_empty_string():
    assert flatten_content("") == ""


def test_flatten_list_text_blocks_only():
    content = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert flatten_content(content) == "first\nsecond"


def test_flatten_skips_tool_use():
    content = [
        {"type": "text", "text": "before"},
        {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"cmd": "ls"}},
        {"type": "text", "text": "after"},
    ]
    assert flatten_content(content) == "before\nafter"


def test_flatten_skips_tool_result():
    # Divergent from Telegram: claude_code skips tool_result entirely.
    content = [
        {"type": "text", "text": "prose"},
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "100KB of code"},
    ]
    assert flatten_content(content) == "prose"


def test_flatten_empty_list_returns_empty():
    assert flatten_content([]) == ""


def test_flatten_unknown_block_types_skipped():
    content = [
        {"type": "thinking", "text": "..."},
        {"type": "text", "text": "visible"},
    ]
    assert flatten_content(content) == "visible"


def test_flatten_non_list_non_string_returns_empty():
    assert flatten_content(None) == ""
    assert flatten_content(42) == ""


def test_parse_timestamp_iso_with_z():
    # 2026-04-18T10:00:00Z = unix 1776506400
    ts = _parse_timestamp("2026-04-18T10:00:00.000Z")
    assert ts == 1776506400.0


def test_parse_timestamp_iso_with_offset():
    ts = _parse_timestamp("2026-04-18T10:00:00+00:00")
    assert ts == 1776506400.0


def test_parse_timestamp_missing_returns_none():
    assert _parse_timestamp(None) is None


def test_parse_timestamp_malformed_returns_none():
    assert _parse_timestamp("not-a-date") is None


def test_parse_timestamp_empty_returns_none():
    assert _parse_timestamp("") is None


def test_parse_file_simple_session(tmp_path):
    src = CC_FIXTURE_DIR / "simple_session.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    # Third record has empty content -> skipped.
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].text == "working on [[Context-Driven Traversal]] today"
    assert turns[0].turn_index == 0
    assert turns[1].role == "assistant"
    assert turns[1].turn_index == 1


def test_parse_file_uses_real_timestamps(tmp_path):
    src = CC_FIXTURE_DIR / "simple_session.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    # 2026-04-18T10:00:00Z -> 1776506400.0
    assert turns[0].ts == 1776506400.0
    assert turns[1].ts == 1776506405.0  # +5 seconds


def test_parse_file_source_meta_populated(tmp_path):
    src = CC_FIXTURE_DIR / "simple_session.jsonl"
    # Parent dir of fixture copy acts as "project slug".
    proj = tmp_path / "-Users-user-myproject"
    proj.mkdir()
    target = proj / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    meta = turns[0].source_meta
    assert meta["session_id"] == "aaaa-1111"
    assert meta["project_slug"] == "-Users-user-myproject"
    assert meta["line_idx"] == "0"


def test_parse_file_mixed_blocks(tmp_path):
    src = CC_FIXTURE_DIR / "mixed_blocks.jsonl"
    target = tmp_path / "bbbb-2222.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    # Turn 0: text blocks only (tool_use dropped).
    assert turns[0].text == "About to run the tool\nand then continue"
    # Turn 1: tool_result dropped; only text block kept.
    assert turns[1].text == "done with #inspection"


def test_parse_file_filters_sidechain_and_noise(tmp_path):
    src = CC_FIXTURE_DIR / "filtered_noise.jsonl"
    target = tmp_path / "cccc-3333.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    # Only the non-sidechain user + assistant turns survive (lines 1 and 4).
    assert len(turns) == 2
    assert turns[0].text == "real user turn about [[alpha]]"
    assert turns[0].turn_index == 1  # physical line index preserved
    assert turns[1].text == "response with #gamma"
    assert turns[1].turn_index == 4


def test_parse_file_source_file_is_filename_not_path(tmp_path):
    src = CC_FIXTURE_DIR / "simple_session.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    assert turns[0].source_file == "aaaa-1111.jsonl"


def test_parse_file_missing_file_raises(tmp_path):
    missing = tmp_path / "no-such.jsonl"
    with pytest.raises(ValueError, match="file not found"):
        parse_file(missing)


def test_parse_file_rejects_wrong_suffix(tmp_path):
    target = tmp_path / "aaaa.txt"
    target.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="expected.*jsonl"):
        parse_file(target)


def test_parse_file_malformed_tail_skipped(tmp_path, capsys):
    src = CC_FIXTURE_DIR / "malformed_tail.jsonl"
    target = tmp_path / "dddd-4444.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    # Two valid records + one skipped malformed tail.
    assert len(turns) == 2
    assert turns[0].text == "first"
    assert turns[1].text == "second with [[delta]]"
    captured = capsys.readouterr()
    assert "malformed JSON" in captured.err


def test_parse_file_missing_timestamp_skipped(tmp_path, capsys):
    src = CC_FIXTURE_DIR / "missing_timestamp.jsonl"
    target = tmp_path / "eeee-5555.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    # Only the first record has a valid timestamp.
    assert len(turns) == 1
    assert turns[0].text == "has ts [[epsilon]]"
    captured = capsys.readouterr()
    assert "missing or malformed timestamp" in captured.err


def test_parse_file_empty_file(tmp_path):
    target = tmp_path / "empty-session.jsonl"
    target.write_text("", encoding="utf-8")
    turns = parse_file(target)
    assert turns == []


def test_parse_file_missing_message_skipped(tmp_path, capsys):
    target = tmp_path / "ffff-6666.jsonl"
    target.write_text(
        '{"type":"user","isSidechain":false,"timestamp":"2026-04-18T10:00:00Z"}\n',
        encoding="utf-8",
    )
    turns = parse_file(target)
    assert turns == []
    captured = capsys.readouterr()
    assert "missing message" in captured.err


def test_parse_file_populates_tool_uses_for_allowlisted(tmp_path):
    src = CC_FIXTURE_DIR / "with_tool_uses.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    # Lines 1, 2, 3 produce tool_uses (Read, mc_card_update, Skill).
    # Line 4 (Bash) has no allowlisted tool_uses. Line 5 is sidechain — dropped.
    assert len(turns) == 5  # user + 4 assistant (one Bash survives with empty tool_uses)

    # Line 1: Read tool_use captured
    read_turn = turns[1]
    assert read_turn.turn_index == 1
    assert len(read_turn.tool_uses) == 1
    assert read_turn.tool_uses[0].tool_name == "Read"
    assert read_turn.tool_uses[0].inputs["file_path"].endswith("daemon.py")

    # Line 2: mc_card_update captured via allowlist suffix
    mc_turn = turns[2]
    assert len(mc_turn.tool_uses) == 1
    assert mc_turn.tool_uses[0].tool_name.endswith("mc_card_update")
    assert mc_turn.tool_uses[0].inputs["slug"] == "voice-onboarding"

    # Line 3: Skill captured
    skill_turn = turns[3]
    assert len(skill_turn.tool_uses) == 1
    assert skill_turn.tool_uses[0].tool_name == "Skill"
    assert skill_turn.tool_uses[0].inputs == {"skill": "brainstorming"}

    # Line 4: Bash — disallowed, tool_uses empty
    bash_turn = turns[4]
    assert bash_turn.tool_uses == ()


def test_parse_file_user_turn_tool_uses_always_empty(tmp_path):
    src = CC_FIXTURE_DIR / "with_tool_uses.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    user_turn = turns[0]
    assert user_turn.role == "user"
    assert user_turn.tool_uses == ()


def test_parse_file_sidechain_filter_includes_tool_uses(tmp_path):
    src = CC_FIXTURE_DIR / "with_tool_uses.jsonl"
    target = tmp_path / "aaaa-1111.jsonl"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    # No turn from line 5 (sidechain) should be present.
    assert all(t.turn_index != 5 for t in turns)


def test_parse_file_malformed_tool_use_input_tolerated(tmp_path):
    # A tool_use block with missing `input` yields a ToolUseRef with empty inputs.
    content = (
        '{"type":"assistant","isSidechain":false,'
        '"timestamp":"2026-04-19T10:00:00Z",'
        '"message":{"role":"assistant","content":['
        '{"type":"tool_use","id":"t1","name":"Read"}'
        ']}}'
    )
    target = tmp_path / "bbbb.jsonl"
    target.write_text(content, encoding="utf-8")
    turns = parse_file(target)
    assert len(turns) == 1
    assert turns[0].text == ""
    assert len(turns[0].tool_uses) == 1
    assert turns[0].tool_uses[0].tool_name == "Read"
    assert turns[0].tool_uses[0].inputs == {}


def test_parse_file_tool_use_alongside_text_keeps_turn(tmp_path):
    # Text block present -> turn survives -> tool_uses populated.
    content = (
        '{"type":"assistant","isSidechain":false,'
        '"timestamp":"2026-04-19T10:00:00Z",'
        '"message":{"role":"assistant","content":['
        '{"type":"text","text":"hello"},'
        '{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/x.py"}}'
        ']}}'
    )
    target = tmp_path / "cccc.jsonl"
    target.write_text(content, encoding="utf-8")
    turns = parse_file(target)
    assert len(turns) == 1
    assert turns[0].text == "hello"
    assert len(turns[0].tool_uses) == 1
    assert turns[0].tool_uses[0].tool_name == "Read"
