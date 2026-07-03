"""Self-rating tag parser for PDCT utility measurement (P1.4).

Tag format (must be the final standalone content of reply, modulo trailing whitespace):
    <!-- pdct:self_rating=<value> -->

Valid values: useful | partial | noise | absent

STRIPPING CONTRACT: strip_self_rating_tag() is called immediately after
final model text is assembled, before DCT memory, cosine scoring, judge
payload, relay/TTS, trace formatting, or Telegram send. Never called on
partial/streaming text.
"""
from __future__ import annotations

import re

VALID_RATINGS: frozenset[str] = frozenset({"useful", "partial", "noise", "absent"})

# Matches the tag only when it is the last non-whitespace content in the string.
# \\Z anchors to true string end (not just a line end within MULTILINE mode).
_TAG_RE = re.compile(
    r"\s*<!--\s*pdct:self_rating=(\w+)\s*-->\s*\Z",
    re.MULTILINE,
)


def extract_self_rating(text: str) -> str | None:
    """Return the self-rating value if a valid tag is the final content of text.

    Returns None if no tag, tag is mid-text, or value not in VALID_RATINGS.
    """
    m = _TAG_RE.search(text)
    if m is None:
        return None
    value = m.group(1)
    return value if value in VALID_RATINGS else None


def strip_self_rating_tag(text: str) -> str:
    """Remove the trailing self-rating tag (if present). Strips trailing whitespace."""
    return _TAG_RE.sub("", text).rstrip()


__all__ = ["extract_self_rating", "strip_self_rating_tag", "VALID_RATINGS"]
