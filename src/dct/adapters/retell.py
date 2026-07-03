"""Retell voice transcript adapter.

Parses `~/example-stack/tools/retell-endpoint/vps-source/transcripts/*.json`
into ordered ParsedTurn records. Pure parser: no concept extraction,
no event construction. V1 is text-only — `tool_uses` is always empty.
MCP tool_call extraction is deferred (see DP-28).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from dct.adapters.telegram import ParsedTurn


def parse_filename(name: str) -> tuple[str, str]:
    """Return ``(call_type, conversation_id)`` parsed from a transcript basename.

    Accepts three shapes of basename:

    - ``<ts>_conv_<id>.json``  → ``("conv", "<id>")``
    - ``<ts>_call_<id>.json``  → ``("call", "<id>")``
    - ``<ts>_<anything-else>.json`` → ``("test", "<anything-else>")``

    Where ``<ts>`` is a compact ISO-like timestamp (ignored here; the top-level
    ``timestamp`` field inside the JSON is authoritative).
    """
    suffix = ".json"
    if not name.endswith(suffix):
        raise ValueError(f"expected *.json, got {name!r}")
    stem = name[: -len(suffix)]
    if "_" not in stem:
        raise ValueError(f"expected <ts>_<rest>.json (no underscore found): {name!r}")
    _, _, rest = stem.partition("_")
    if rest.startswith("conv_"):
        return ("conv", rest[len("conv_"):])
    if rest.startswith("call_"):
        return ("call", rest[len("call_"):])
    return ("test", rest)


def _parse_iso_timestamp(raw: object) -> float | None:
    """Return epoch seconds for an ISO8601 string, or ``None`` on failure."""
    if not isinstance(raw, str) or not raw:
        return None
    s = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.timestamp()


def parse_file(path: Path) -> list[ParsedTurn]:
    """Parse a Retell transcript JSON into ordered ParsedTurn records.

    V1 is text-only: ``tool_uses`` is always ``()``. Empty messages are
    silently skipped; non-dict entries are silently skipped.
    """
    path = Path(path).resolve()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path.name}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}: expected top-level object, got {type(payload).__name__}")

    call_start_ts = _parse_iso_timestamp(payload.get("timestamp"))
    if call_start_ts is None:
        raise ValueError(f"{path.name}: missing or malformed top-level 'timestamp'")

    transcript = payload.get("transcript")
    if not isinstance(transcript, list):
        raise ValueError(f"{path.name}: missing or non-list 'transcript'")

    call_type, conversation_id = parse_filename(path.name)

    top_meta = payload.get("metadata")
    meta_extras: dict[str, str] = {}
    if isinstance(top_meta, dict):
        for k in ("chat_id", "topic_id"):
            v = top_meta.get(k)
            if isinstance(v, str) and v:
                meta_extras[k] = v

    source_file = str(path)
    turns: list[ParsedTurn] = []
    for idx, entry in enumerate(transcript):
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if not isinstance(role, str) or not role:
            continue
        text = entry.get("message")
        if not isinstance(text, str) or not text.strip():
            continue
        offset = entry.get("time_in_call_secs")
        if isinstance(offset, (int, float)) and math.isfinite(offset) and offset >= 0:
            ts_offset = float(offset)
        else:
            ts_offset = idx * 1e-3
        emit_role = "assistant" if role == "agent" else role
        turns.append(ParsedTurn(
            role=emit_role,
            text=text,
            turn_index=idx,
            source_file=source_file,
            ts=call_start_ts + ts_offset,
            source_meta={
                "conversation_id": conversation_id,
                "call_type": call_type,
                **meta_extras,
            },
        ))
    return turns
