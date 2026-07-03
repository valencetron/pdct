"""Track B — cascade feedback helpers.

Pure, I/O-free utilities for computing useful concepts, anti-leakage
filters, hop-aware multipliers, and feedback event metadata. The daemon
calls into this module to produce one feedback event per useful concept.
"""

from __future__ import annotations

import re as _re
from enum import Enum


class AntiLeakLevel(str, Enum):
    STRICT = "strict"  # all 3 filters: seed-drop + hop-1-penalty + lexical-copy
    LOOSE = "loose"    # only seed-drop
    OFF = "off"        # no filters (ablation arm)


def multipliers_for_path(
    path: list[str],
    *,
    base: float = 3.0,
    hop_penalty_factor: float = 1.0,
) -> list[float]:
    """Compute the per-edge multiplier list for a feedback path.

    Returns floats (R3.6). Rounding to int happens at metadata-write time;
    keeping precision here lets low-base ablations and aggressive hop-1
    penalties survive without being floored to 1 prematurely.

    For an N-edge path (N+1 nodes), returns N multipliers. Edge i (between
    path[i] and path[i+1]) has hop = i+1.

    Multiplier formula: (base + hop) × (hop_penalty_factor if hop == 1 else 1).
    No floor here — caller decides rounding policy.
    """
    if len(path) < 2:
        return []
    out: list[float] = []
    for i in range(len(path) - 1):
        hop = i + 1
        raw = float(base) + hop
        if hop == 1:
            raw = raw * hop_penalty_factor
        out.append(raw)
    return out


def round_multipliers_for_storage(mults: list[float]) -> list[int]:
    """Convert float multipliers to ints for JSONL persistence.

    Uses round() then floors at 1 (an event MUST reinforce by at least 1).
    Loss of float precision happens once, at write time.
    """
    return [max(1, int(round(m))) for m in mults]


def _slug_word_pattern(slug: str) -> _re.Pattern:
    """Compile a word-boundary regex for a slug.

    Multi-token slugs ('context-stream', 'context_stream', 'context stream')
    all match against any of those forms in the target text. Separator is
    one-or-more of: hyphen, underscore, whitespace.

    R3 fix: previous version compiled only `\\s+` between tokens, so the
    hyphenated form 'context-stream' (the actual slug shape emitted in PDCT
    prompt blocks) silently failed to match itself.

    Single-token slug 'phenomenology' compiles to `\\bphenomenology\\b`
    — strict word boundary, no substring match against 'phenomenology123'.
    """
    tokens = [_re.escape(t) for t in slug.lower().replace("-", " ").replace("_", " ").split() if t]
    if not tokens:
        return _re.compile(r"(?!)")  # matches nothing
    body = r"[-_\s]+".join(tokens)
    return _re.compile(rf"\b{body}\b", _re.IGNORECASE)


def _word_boundary_count(text: str, slug: str) -> int:
    """Count whole-word occurrences of slug in text. R3.5: NOT substring."""
    if not text:
        return 0
    return len(_slug_word_pattern(slug).findall(text))


def compute_useful_concepts(
    *,
    reply_concepts: set[str],
    cascade_paths: dict[str, list[str]],
    user_seed_concepts: set[str],
    prompt_block: str,
    reply_text: str,
    anti_leak: AntiLeakLevel = AntiLeakLevel.STRICT,
) -> list[str]:
    """Return useful concepts (intersection of reply_concepts × cascade_paths)
    after anti-leak filters per the requested level.

    Filters:
        1. seed-drop: concept must not be in user_seed_concepts.
        2. hop-1 penalty: applied at multiplier-time, not here.
        3. lexical-copy: word-boundary count(slug in prompt_block) must be
           <= count in reply_text. If prompt has more, the model is just
           parroting prompt content rather than producing new useful retrieval.

    Mode application:
        STRICT — applies seed-drop AND lexical-copy.
        LOOSE  — applies seed-drop only.
        OFF    — no filtering.

    Returns: ordered list of useful concept slugs (preserves cascade_paths order).
    """
    apply_seed = anti_leak in (AntiLeakLevel.STRICT, AntiLeakLevel.LOOSE)
    apply_lex = anti_leak == AntiLeakLevel.STRICT

    candidates = [c for c in cascade_paths.keys() if c in reply_concepts]
    out: list[str] = []
    for c in candidates:
        if apply_seed and c in user_seed_concepts:
            continue
        if apply_lex:
            prompt_count = _word_boundary_count(prompt_block, c)
            reply_count = _word_boundary_count(reply_text, c)
            if prompt_count > reply_count:
                continue
        out.append(c)
    return out


def build_feedback_event_metadata(
    *,
    useful_concept: str,
    path: list[str],
    multipliers: list[int],
    thread_id: str,
    anti_leak_applied: list[str],
) -> dict:
    """Construct the metadata dict for a feedback event.

    Validates that len(multipliers) == len(path) - 1.
    """
    expected = max(0, len(path) - 1)
    if len(multipliers) != expected:
        raise ValueError(
            f"multipliers length {len(multipliers)} != len(path)-1 = {expected}"
        )
    return {
        "useful_concept": useful_concept,
        "path": list(path),
        "multipliers": list(multipliers),
        "thread_id": thread_id,
        "anti_leak_applied": list(anti_leak_applied),
    }
