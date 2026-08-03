"""Structure-aware anchor loading.

Design: docs/superpowers/specs/2026-07-27-structure-aware-anchor-loading-design.md
Motivation: the 2026-07-27 SOUL.md incident — `_load_anchors()` head-chopped a
103 KB soul file at the token cap, silently dropping 92% of it (including every
"Inviolable Rule" section). This module parses anchor markdown into sections,
classifies each into importance tiers, and assembles a capped block that drops
whole sections (least-important first, journal before core before inviolable)
instead of blind mid-sentence truncation.

Dependency-free by design; must stay portable to pdct-public.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "AnchorSection",
    "parse_sections",
    "assemble_anchor_block",
    "DEFAULT_TIER_MARKERS",
    "DEFAULT_TIER_ORDER",
]

DEFAULT_TIER_MARKERS: dict[str, tuple[str, ...]] = {
    "inviolable": ("inviolable", "red line", "kill-switch"),
}
DEFAULT_TIER_ORDER: tuple[str, ...] = ("inviolable", "core", "journal")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
_TIER_TAG_RE = re.compile(r"\[tier:(inviolable|core|journal)\]", re.IGNORECASE)
# Heading ends with a parenthesized date, e.g. "## Fable notes (2026-07-14)"
_DATE_SUFFIX_RE = re.compile(r"\(\s*\d{4}-\d{2}-\d{2}[^)]*\)\s*$")

TRUNCATION_MARKER = "[… truncated]"


@dataclass(frozen=True)
class AnchorSection:
    """One heading-delimited slice of an anchor file."""
    source: Path | None   # which anchor file (None for synthetic input)
    heading: str          # raw heading line ("" for preamble)
    level: int            # 1-6; preamble = 0
    tier: str             # "inviolable" | "core" | "journal"
    text: str             # heading + body, verbatim
    tokens: int           # estimated
    order: int            # global original position, for stable re-assembly


def _estimate_tokens(text: str) -> int:
    """Match preload.py's estimator: ~1 token per 4 chars."""
    return len(text) // 4


def _explicit_tier(heading: str,
                   markers: dict[str, tuple[str, ...]]) -> str | None:
    """Return an explicitly-signaled tier for a heading, or None."""
    tag = _TIER_TAG_RE.search(heading)
    if tag:
        return tag.group(1).lower()
    low = heading.lower()
    for tier, subs in markers.items():
        if any(s in low for s in subs):
            return tier
    if _DATE_SUFFIX_RE.search(heading):
        return "journal"
    return None


def parse_sections(
    text: str,
    *,
    source: Path | None = None,
    markers: dict[str, tuple[str, ...]] | None = None,
    order_start: int = 0,
) -> list[AnchorSection]:
    """Split markdown into heading-delimited sections with tier labels.

    - Content before the first heading becomes a level-0 "preamble" section.
    - `#` lines inside fenced code blocks (``` or ~~~) are NOT headings.
    - Tier: explicit [tier:x] tag or marker substring in the heading wins;
      a `(YYYY-MM-DD…)` heading suffix means journal; otherwise the section
      inherits its nearest ancestor heading's tier, defaulting to core.
    """
    markers = markers if markers is not None else DEFAULT_TIER_MARKERS
    lines = text.split("\n")

    # Pass 1: split into (heading, level, lines) blocks, fence-aware.
    blocks: list[tuple[str, int, list[str]]] = []
    cur_heading, cur_level, cur_lines = "", 0, []
    fence: str | None = None
    for line in lines:
        f = _FENCE_RE.match(line)
        if f:
            tick = f.group(1)
            if fence is None:
                fence = tick
            elif fence == tick:
                fence = None
        h = _HEADING_RE.match(line) if fence is None else None
        if h:
            if cur_lines or cur_heading:
                blocks.append((cur_heading, cur_level, cur_lines))
            cur_heading, cur_level, cur_lines = line, len(h.group(1)), [line]
        else:
            cur_lines.append(line)
    if cur_lines or cur_heading:
        blocks.append((cur_heading, cur_level, cur_lines))

    # Pass 2: classify with ancestor inheritance.
    sections: list[AnchorSection] = []
    # Stack of (level, tier) for ancestor headings.
    stack: list[tuple[int, str]] = []
    for i, (heading, level, blk_lines) in enumerate(blocks):
        explicit = _explicit_tier(heading, markers) if heading else None
        if level > 0:
            while stack and stack[-1][0] >= level:
                stack.pop()
        if explicit is not None:
            tier = explicit
        elif stack:
            tier = stack[-1][1]
        else:
            tier = "core"
        if level > 0:
            stack.append((level, tier))
        body = "\n".join(blk_lines)
        # Drop empty preamble (e.g. file starts with a heading).
        if level == 0 and not body.strip():
            continue
        sections.append(AnchorSection(
            source=source,
            heading=heading,
            level=level,
            tier=tier,
            text=body,
            tokens=_estimate_tokens(body),
            order=order_start + i,
        ))
    return sections


def _truncate_sentence_boundary(text: str, max_tokens: int) -> str:
    """Cut text at the last sentence/paragraph boundary under the char budget.

    Fallback for the pathological case of a single section larger than the
    entire cap. Appends TRUNCATION_MARKER so the cut is visible in-context.
    """
    budget_chars = max(max_tokens * 4 - len(TRUNCATION_MARKER) - 1, 0)
    if len(text) <= budget_chars:
        return text
    head = text[:budget_chars]
    # Prefer paragraph break, then sentence end, then newline, then hard cut.
    for sep in ("\n\n", ". ", "\n"):
        idx = head.rfind(sep)
        if idx > budget_chars // 2:
            head = head[: idx + (1 if sep == ". " else 0)]
            break
    return head.rstrip() + "\n" + TRUNCATION_MARKER


def assemble_anchor_block(
    sections: list[AnchorSection],
    cap_tokens: int,
    *,
    tier_order: tuple[str, ...] = DEFAULT_TIER_ORDER,
) -> tuple[str, list[AnchorSection]]:
    """Assemble a capped anchors block by whole-section keep/drop.

    Sections are admitted tier-by-tier (tier_order = most important first),
    in original document order within each tier, until the budget runs out.
    Kept sections are re-emitted in original document order so the output
    still reads as one coherent document. Returns (text, dropped_sections).

    A truncation notice is appended whenever anything drops, so the model
    knows its anchors are incomplete (never silent again).
    """
    total = sum(s.tokens for s in sections)
    if total <= cap_tokens:
        return "\n".join(s.text for s in sections), []

    # Reserve headroom for the truncation notice.
    budget = max(cap_tokens - 64, 0)
    kept: list[AnchorSection] = []
    dropped: list[AnchorSection] = []
    known = set(tier_order)
    remainder = [s for s in sections if s.tier not in known]

    for tier in tier_order:
        for s in (x for x in sections if x.tier == tier):
            if s.tokens <= budget:
                kept.append(s)
                budget -= s.tokens
            elif s.tokens > cap_tokens and not kept and budget > 64:
                # Spec §3 exception: a single section larger than the whole
                # cap, with nothing else admitted yet (i.e. it's the most
                # important content available) — sentence-boundary truncate
                # rather than losing it entirely. Oversized low-tier sections
                # competing with kept higher-tier content still drop whole.
                cut = _truncate_sentence_boundary(s.text, budget)
                kept.append(AnchorSection(
                    source=s.source, heading=s.heading, level=s.level,
                    tier=s.tier, text=cut,
                    tokens=_estimate_tokens(cut), order=s.order,
                ))
                budget = 0
            else:
                dropped.append(s)
    dropped.extend(remainder)

    kept.sort(key=lambda s: s.order)
    dropped.sort(key=lambda s: s.order)

    by_tier: dict[str, int] = {}
    for s in dropped:
        by_tier[s.tier] = by_tier.get(s.tier, 0) + 1
    drop_tokens = sum(s.tokens for s in dropped)
    detail = ", ".join(f"{n} {t}" for t, n in sorted(by_tier.items()))
    notice = (
        f"\n---\n[anchor loader: {len(dropped)} section(s) dropped to fit "
        f"the {cap_tokens}-token cap (~{drop_tokens} tokens: {detail}). "
        f"Run `dct doctor` for details.]"
    )
    return "\n".join(s.text for s in kept) + notice, dropped
