"""DCT retrieval engine — cascade, preload, inject.

Consumed by Telegram daemon, Retell server, and MCP bridge to assemble
context for live conversations per the DCT invariant: every READ cascades.
"""
from .types import ConceptHit, PreloadBundle, RetrievalConfig
from .cascade import cascade
from .preload import preload, DistilledNote
from .inject import (
    format_for_telegram,
    format_for_telegram_with_sections,
    format_for_retell,
    format_for_claude_code,
)

__all__ = [
    "ConceptHit",
    "PreloadBundle",
    "RetrievalConfig",
    "DistilledNote",
    "cascade",
    "preload",
    "format_for_telegram",
    "format_for_telegram_with_sections",
    "format_for_retell",
    "format_for_claude_code",
]
