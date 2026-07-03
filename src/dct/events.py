"""Event types for Dynamic Context Traversal.

An Event is the atomic record of a graph interaction. Events are append-only
and immutable once written to the log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EventSource(str, Enum):
    TELEGRAM = "telegram"
    VOICE = "voice"
    CLAUDE_CODE = "claude-code"
    VAULT = "vault"


class EventOp(str, Enum):
    READ = "read"
    WRITE = "write"
    TRAVERSAL = "traversal"
    TURN = "turn"  # raw conversation turn — not yet a vault op
    FEEDBACK = "feedback"  # cascade reinforcement event (Track B)
    PRUNE = "prune"  # context pruning event written by memory_manager


@dataclass(frozen=True)
class Event:
    ts: float
    source: EventSource
    op: EventOp
    concepts: list[str]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.concepts:
            raise ValueError("Event.concepts must not be empty")

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "source": self.source.value,
            "op": self.op.value,
            "concepts": list(self.concepts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event | None":
        """Deserialise an event dict. Returns None for unknown op/source values.

        Tolerates future enum additions gracefully — callers must filter None.
        """
        try:
            source = EventSource(d["source"])
        except ValueError:
            return None
        try:
            op = EventOp(d["op"])
        except ValueError:
            return None
        return cls(
            ts=float(d["ts"]),
            source=source,
            op=op,
            concepts=list(d["concepts"]),
            metadata=dict(d.get("metadata", {})),
        )
