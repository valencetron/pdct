import pytest

from dct.adapters.telegram import ParsedTurn, parse_filename


def test_parse_filename_standard():
    chat, thread = parse_filename("1003690648082_92.messages.json")
    assert chat == "1003690648082"
    assert thread == "92"


def test_parse_filename_none_thread():
    chat, thread = parse_filename("1003690648082_None.messages.json")
    assert chat == "1003690648082"
    assert thread == "None"


def test_parse_filename_multiple_underscores_uses_last():
    # Per spec: "chat_id: everything before the last `_`"
    chat, thread = parse_filename("1003_999_55.messages.json")
    assert chat == "1003_999"
    assert thread == "55"


def test_parse_filename_rejects_wrong_suffix():
    with pytest.raises(ValueError, match="messages.json"):
        parse_filename("1003690648082_92.json")


def test_parsed_turn_requires_non_negative_turn_index():
    with pytest.raises(ValueError, match="turn_index"):
        ParsedTurn(
            role="user", text="x", turn_index=-1,
            source_file="/f.messages.json", ts=1.0,
            source_meta={"chat_id": "1", "thread_id": "2"},
        )


def test_parsed_turn_requires_non_empty_role():
    with pytest.raises(ValueError, match="role"):
        ParsedTurn(
            role="", text="x", turn_index=0,
            source_file="/f.messages.json", ts=1.0,
            source_meta={"chat_id": "1", "thread_id": "2"},
        )


def test_parsed_turn_requires_finite_ts():
    import math
    with pytest.raises(ValueError, match="ts"):
        ParsedTurn(
            role="user", text="x", turn_index=0,
            source_file="/f.messages.json",
            ts=float("nan"),
            source_meta={"chat_id": "1", "thread_id": "2"},
        )


def test_parsed_turn_allows_empty_text():
    turn = ParsedTurn(
        role="user", text="", turn_index=0,
        source_file="/f.messages.json", ts=1.0,
        source_meta={"chat_id": "1", "thread_id": "2"},
    )
    assert turn.text == ""


def test_parsed_turn_exposes_source_meta():
    turn = ParsedTurn(
        role="user", text="hi", turn_index=0,
        source_file="/f.messages.json", ts=1.0,
        source_meta={"chat_id": "123", "thread_id": "456"},
    )
    assert turn.source_meta == {"chat_id": "123", "thread_id": "456"}


from dct.adapters.telegram import flatten_content


def test_flatten_plain_string():
    assert flatten_content("hello world") == "hello world"


def test_flatten_empty_string():
    assert flatten_content("") == ""


def test_flatten_list_of_text_blocks():
    content = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert flatten_content(content) == "first\nsecond"


def test_flatten_ignores_tool_use_blocks():
    content = [
        {"type": "text", "text": "before"},
        {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"cmd": "ls"}},
        {"type": "text", "text": "after"},
    ]
    assert flatten_content(content) == "before\nafter"


def test_flatten_tool_result_with_string_content():
    content = [
        {"type": "text", "text": "prose"},
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "output text"},
    ]
    assert flatten_content(content) == "prose\noutput text"


def test_flatten_tool_result_with_list_content():
    # Some tool_result entries wrap content in another block list.
    content = [
        {"type": "tool_result", "content": [{"type": "text", "text": "nested"}]},
    ]
    assert flatten_content(content) == "nested"


def test_flatten_empty_list_returns_empty():
    assert flatten_content([]) == ""


def test_flatten_unknown_block_types_skipped():
    content = [
        {"type": "image", "source": "..."},
        {"type": "text", "text": "only this"},
    ]
    assert flatten_content(content) == "only this"


import os
from pathlib import Path

from dct.adapters.telegram import parse_file

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "telegram"


def test_parse_file_simple_string_content(tmp_path):
    src = FIXTURE_DIR / "simple_string_content.messages.json"
    target = tmp_path / "1003_92.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    assert len(turns) == 3
    assert [t.role for t in turns] == ["user", "assistant", "user"]
    assert turns[0].text == "working on [[Context-Driven Traversal]] today"
    assert turns[1].text == "Let's talk about #consciousness"
    assert turns[2].text == ""
    assert [t.turn_index for t in turns] == [0, 1, 2]
    assert all(t.source_meta["chat_id"] == "1003" for t in turns)
    assert all(t.source_meta["thread_id"] == "92" for t in turns)


def test_parse_file_assigns_ordered_timestamps(tmp_path):
    src = FIXTURE_DIR / "simple_string_content.messages.json"
    target = tmp_path / "1_2.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")
    os.utime(target, (1_700_000_000.0, 1_700_000_000.0))

    turns = parse_file(target)
    assert turns[0].ts == 1_700_000_000.0
    assert turns[1].ts == pytest.approx(1_700_000_000.0 + 1e-3)
    assert turns[2].ts == pytest.approx(1_700_000_000.0 + 2e-3)


def test_parse_file_block_content(tmp_path):
    src = FIXTURE_DIR / "block_content.messages.json"
    target = tmp_path / "5_6.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    assert len(turns) == 2
    assert turns[0].text == "First block\nAfter tool call"
    assert turns[1].text == "some output"


def test_parse_file_source_file_is_absolute_path(tmp_path):
    src = FIXTURE_DIR / "simple_string_content.messages.json"
    target = tmp_path / "9_9.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")

    turns = parse_file(target)
    assert Path(turns[0].source_file).is_absolute()
    assert turns[0].source_file == str(target.resolve())


def test_parse_file_malformed_raises(tmp_path):
    src = FIXTURE_DIR / "malformed.json"
    target = tmp_path / "1_1.messages.json"
    target.write_text(src.read_text(), encoding="utf-8")

    with pytest.raises(ValueError, match="1_1.messages.json"):
        parse_file(target)


def test_parse_file_rejects_bak_suffix(tmp_path):
    target = tmp_path / "1_1.messages.json.bak"
    target.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="messages.json"):
        parse_file(target)


def test_parse_file_missing_file_raises(tmp_path):
    missing = tmp_path / "9999_1.messages.json"
    with pytest.raises(ValueError, match="file not found"):
        parse_file(missing)


def test_parse_file_non_list_raises(tmp_path):
    p = tmp_path / "100_400.messages.json"
    p.write_text('{"role": "user", "content": "hi"}', encoding="utf-8")
    with pytest.raises(ValueError, match="expected top-level array"):
        parse_file(p)


def test_parsed_turn_default_tool_uses_empty_tuple():
    from dct.adapters.telegram import ParsedTurn
    pt = ParsedTurn(
        role="user", text="x", turn_index=0,
        source_file="/f.messages.json", ts=1.0,
        source_meta={"chat_id": "1", "thread_id": "2"},
    )
    assert pt.tool_uses == ()


def test_tool_use_ref_importable_and_stores_inputs():
    from dct.adapters.telegram import ToolUseRef
    t = ToolUseRef(tool_name="Read", inputs={"file_path": "/x/y.py"})
    assert t.tool_name == "Read"
    assert t.inputs == {"file_path": "/x/y.py"}


def test_parsed_turn_accepts_explicit_tool_uses():
    from dct.adapters.telegram import ParsedTurn, ToolUseRef
    ref = ToolUseRef(tool_name="Read", inputs={"file_path": "/a.py"})
    pt = ParsedTurn(
        role="user", text="x", turn_index=0,
        source_file="/f.messages.json", ts=1.0,
        source_meta={"chat_id": "1", "thread_id": "2"},
        tool_uses=(ref,),
    )
    assert pt.tool_uses == (ref,)
