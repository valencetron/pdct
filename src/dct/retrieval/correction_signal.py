"""Correction-pattern detector for next-user-turn classification.

Heuristic-only (no LLM call). Used to label whether Alex's reply to a
prior assistant turn looks like a correction, continuation, or neutral
message. Headline metric: correction-rate split by ablation arm.

Spec: §Stage 3B. Patterns refined after codex round 2 (dropped overly-broad
'actually' pattern; added word-boundary guards).

Design notes:
  - "actually let me..." is NOT a correction — it's redirect / clarification.
    We drop the 'actually' pattern entirely.
  - "no problem" is NOT a correction. The leading-no pattern requires a
    bare 'no' followed by punctuation/space and not part of a phrase.
  - All matches log the pattern name and a 50-char excerpt for audit.
"""
from __future__ import annotations

import re
from typing import Optional

# (compiled_pattern, name)
# Note: 'no' and 'nope' require a following punctuation mark (not just space) —
# avoids "no problem" / "no idea" false positives. 'wrong' and 'incorrect' can
# stand alone as standalone leading correction words.
_CORRECTION_PATTERNS = [
    # 'no'/'nope' followed by punctuation or EOL — strong correction signal
    (re.compile(r"^(no|nope)(?=[,.!?:;\-]|$)", re.IGNORECASE), "leading-no"),
    # 'wrong'/'incorrect' as leading word — accept space follow-up too
    (re.compile(r"^(wrong|incorrect)(?=[\s,.!?:;\-]|$)", re.IGNORECASE), "leading-no"),
    (re.compile(r"\bthat'?s wrong\b", re.IGNORECASE), "thats-wrong"),
    (re.compile(r"\b(you'?re wrong|got that wrong)\b", re.IGNORECASE), "youre-wrong"),
    (re.compile(r"\b(didn'?t work|doesn'?t work|isn'?t working)\b", re.IGNORECASE), "doesnt-work"),
    (re.compile(r"\b(undo|revert|roll ?back)\b", re.IGNORECASE), "undo"),
]

_CONTINUATION_PATTERNS = [
    # leading approval — anchored
    (re.compile(r"^(ok|okay|good|great|nice|perfect|approved?|yes)\b", re.IGNORECASE), "approve"),
    (re.compile(r"^ship it\b", re.IGNORECASE), "ship-it"),
    (re.compile(r"\b(thanks|thank you)\b", re.IGNORECASE), "thanks"),
]

_TOOL_PREFIXES = ("[tool_result:", "[tool:")


def classify_user_followup(text: str, prev_turn_id: Optional[str]) -> Optional[dict]:
    """Classify the user-turn text as correction/continuation/neutral.

    Returns:
        None — when classification should be skipped (no prev turn, too short,
               tool-result wrapper).
        dict with keys: rating, matched_pattern (None for neutral), excerpt.
    """
    if not prev_turn_id:
        return None
    if not text:
        return None
    stripped = text.lstrip()
    if len(stripped) < 4:
        return None
    if stripped.startswith(_TOOL_PREFIXES):
        return None

    excerpt = stripped[:50]

    for pat, name in _CORRECTION_PATTERNS:
        if pat.search(stripped):
            return {"rating": "correction", "matched_pattern": name, "excerpt": excerpt}

    for pat, name in _CONTINUATION_PATTERNS:
        if pat.search(stripped):
            return {"rating": "continuation", "matched_pattern": name, "excerpt": excerpt}

    return {"rating": "neutral", "matched_pattern": None, "excerpt": excerpt}
