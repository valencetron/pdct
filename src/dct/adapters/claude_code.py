"""Claude Code session transcript adapter.

Parses `~/.claude/projects/<slug>/<session>.jsonl` session files into ordered
ParsedTurn records. Pure parser: no concept extraction, no event construction.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from dct.adapters.telegram import ParsedTurn, ToolUseRef


_ALLOWED_TOOL_NAMES: frozenset[str] = frozenset({
    "Read", "Edit", "Write", "Grep", "Skill",
})
_ALLOWED_TOOL_SUFFIXES: tuple[str, ...] = (
    "mc_card_create", "mc_card_update", "mc_card_list",
)


def _is_allowed_tool(name: str) -> bool:
    if not isinstance(name, str) or not name:
        return False
    if name in _ALLOWED_TOOL_NAMES:
        return True
    return any(name.endswith(suffix) for suffix in _ALLOWED_TOOL_SUFFIXES)


def _collect_tool_uses(content) -> tuple[ToolUseRef, ...]:
    if not isinstance(content, list):
        return ()
    refs: list[ToolUseRef] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name")
        if not _is_allowed_tool(name):
            continue
        raw_input = block.get("input")
        inputs = raw_input if isinstance(raw_input, dict) else {}
        refs.append(ToolUseRef(tool_name=name, inputs=inputs))
    return tuple(refs)


def flatten_content(content: str | list) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        t = block.get("text")
        if isinstance(t, str) and t:
            parts.append(t)
    return "\n".join(parts)


def _parse_timestamp(raw) -> float | None:
    if not isinstance(raw, str) or not raw:
        return None
    s = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.timestamp()


def parse_file(path: Path) -> list[ParsedTurn]:
    path = Path(path)
    if not path.name.endswith(".jsonl"):
        raise ValueError(f"expected *.jsonl, got {path.name}")
    if not path.exists():
        raise ValueError(f"file not found: {path.name}")

    session_id = path.name[: -len(".jsonl")]
    project_slug = path.parent.name
    source_file = path.name

    turns: list[ParsedTurn] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(
                    f"dct.claude_code: skipping {path.name} line {line_idx}: "
                    f"malformed JSON ({exc})",
                    file=sys.stderr,
                )
                continue
            if not isinstance(record, dict):
                print(
                    f"dct.claude_code: skipping {path.name} line {line_idx}: "
                    f"not a JSON object",
                    file=sys.stderr,
                )
                continue
            if record.get("type") not in ("user", "assistant"):
                continue
            if record.get("isSidechain"):
                continue
            ts = _parse_timestamp(record.get("timestamp"))
            if ts is None:
                print(
                    f"dct.claude_code: skipping {path.name} line {line_idx}: "
                    f"missing or malformed timestamp",
                    file=sys.stderr,
                )
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                print(
                    f"dct.claude_code: skipping {path.name} line {line_idx}: "
                    f"missing message",
                    file=sys.stderr,
                )
                continue
            role = message.get("role")
            if not isinstance(role, str) or not role:
                print(
                    f"dct.claude_code: skipping {path.name} line {line_idx}: "
                    f"missing role",
                    file=sys.stderr,
                )
                continue
            text = flatten_content(message.get("content", ""))
            tool_uses = (
                _collect_tool_uses(message.get("content"))
                if role == "assistant" else ()
            )
            if not text.strip() and not tool_uses:
                continue
            turns.append(
                ParsedTurn(
                    role=role,
                    text=text,
                    turn_index=line_idx,
                    source_file=source_file,
                    ts=ts,
                    source_meta={
                        "session_id": session_id,
                        "project_slug": project_slug,
                        "line_idx": str(line_idx),
                    },
                    tool_uses=tool_uses,
                )
            )
    return turns
