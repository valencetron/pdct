"""Telegram-dispatch messages.json adapter.

Parses `<chat_id>_<thread_id>.messages.json` into ordered ParsedTurn records.
Pure parser: no concept extraction, no event construction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolUseRef:
    tool_name: str
    inputs: dict


@dataclass(frozen=True)
class ParsedTurn:
    role: str
    text: str
    turn_index: int
    source_file: str
    ts: float
    source_meta: dict
    tool_uses: tuple[ToolUseRef, ...] = ()

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError(f"turn_index must be >= 0, got {self.turn_index}")
        if not self.role:
            raise ValueError("role must not be empty")
        if not math.isfinite(self.ts) or self.ts < 0:
            raise ValueError(f"ts must be finite and non-negative, got {self.ts}")


def parse_filename(name: str) -> tuple[str, str]:
    suffix = ".messages.json"
    if not name.endswith(suffix):
        raise ValueError(f"expected *.messages.json, got {name}")
    stem = name[: -len(suffix)]
    if "_" not in stem:
        raise ValueError(f"expected <chat>_<thread>.messages.json, got {name}")
    chat, _, thread = stem.rpartition("_")
    return chat, thread


def flatten_content(content: str | list) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text")
            if isinstance(t, str) and t:
                parts.append(t)
        elif btype == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str) and inner:
                parts.append(inner)
            elif isinstance(inner, list):
                parts.append(flatten_content(inner))
        # tool_use and all other block types contribute nothing.
    return "\n".join(p for p in parts if p)


def parse_file(path: Path) -> list[ParsedTurn]:
    path = Path(path).resolve()
    chat_id, thread_id = parse_filename(path.name)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path.name}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: expected top-level array, got {type(raw).__name__}")

    mtime = path.stat().st_mtime
    source_file = str(path)
    turns: list[ParsedTurn] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path.name}: entry {idx} is not an object")
        role = entry.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{path.name}: entry {idx} missing role")
        text = flatten_content(entry.get("content", ""))
        turns.append(
            ParsedTurn(
                role=role,
                text=text,
                turn_index=idx,
                source_file=source_file,
                ts=mtime + idx * 1e-3,
                source_meta={"chat_id": chat_id, "thread_id": thread_id},
            )
        )
    return turns
