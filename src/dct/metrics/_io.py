"""Shared JSONL reading helpers — corrupt-line-tolerant, days-filter-aware."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

# Era boundary for match-rate / eligibility aggregates. The node_kinds
# classifier-aware scoring (Code/Concept Layer Split, 2026-06-14) changed
# concepts_eligible / concepts_matched / match_rate semantics, so rows
# emitted before schema 6 are NOT comparable and must be excluded from
# match-quality trends by default. Bumped 1→6 at the daemon emit site.
UTILITY_MATCH_SCHEMA_MIN = 6


def parse_ts(ts_str: str) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware UTC datetime, or None."""
    if not ts_str:
        return None
    try:
        # Python 3.11+ handles trailing Z natively; older needs replace.
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def iter_rows_jsonl(
    path: Path,
    *,
    since: datetime | None = None,
    kind: str | None = None,
    min_schema: int | None = None,
) -> Iterator[dict]:
    """Yield rows from a JSONL file, dropping corrupt or malformed lines.

    - Missing/unreadable file → empty iterator (no exception).
    - Lines that don't parse as JSON object → skipped.
    - If `since` is set, drops rows whose `ts` field is < `since`.
    - If `kind` is set, drops rows where row.get("kind") != kind.
    - If `min_schema` is set, drops rows whose `schema_version` is missing
      or < min_schema (era-gate for incomparable pre-bump rows). Rows with
      no schema_version are treated as pre-era and excluded.
    """
    try:
        f = open(path, "r", encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if kind is not None and row.get("kind") != kind:
                continue
            if min_schema is not None:
                sv = row.get("schema_version")
                if not isinstance(sv, int) or sv < min_schema:
                    continue
            if since is not None:
                ts = parse_ts(row.get("ts", ""))
                if ts is None or ts < since:
                    continue
            yield row


def days_ago(n: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=n)
