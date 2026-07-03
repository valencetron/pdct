"""Append-only JSONL event log for Dynamic Context Traversal.

The log is the authoritative record. Activation state is derived by
replaying the log through the activation engine.
"""

from __future__ import annotations

import json
from pathlib import Path

from dct.events import Event


class EventLog:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: Event) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def read_all(self) -> list[Event]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        events: list[Event] = []
        for idx, raw in enumerate(raw_lines):
            if not raw.endswith("\n"):
                # Trailing line without newline = crash mid-write. Drop it silently.
                # Fully-written lines always end with \n by contract of append().
                if idx == len(raw_lines) - 1:
                    continue
                # A non-terminal line missing \n should not happen; fail loud.
                raise ValueError(f"event log corrupted at line {idx + 1}")
            stripped = raw.strip()
            if not stripped:
                continue
            event = Event.from_dict(json.loads(stripped))
            if event is not None:
                events.append(event)
        events.sort(key=lambda e: e.ts)
        return events
