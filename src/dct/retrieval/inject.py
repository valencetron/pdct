"""Inject — per-surface formatters for cascade+preload output."""
from __future__ import annotations
from typing import Any

from .types import ConceptHit, PreloadBundle


def _render_hits(hits: list[ConceptHit]) -> str:
    if not hits:
        return "(no cascade)"
    lines = []
    for h in hits:
        lines.append(f"- [{h.hop}] {h.concept} (score={h.score:.2f})")
    return "\n".join(lines)


def _render_recent(recent: dict[str, str]) -> str:
    parts = []
    for surface, text in recent.items():
        if text.strip():
            parts.append(f"### {surface}\n{text}")
    return "\n\n".join(parts) if parts else "(no recent per-surface history)"


def format_for_telegram(bundle: PreloadBundle, hits: list[ConceptHit]) -> str:
    """Plain-text block to prepend to the Telegram system prompt."""
    return (
        "## Anchors\n"
        f"{bundle.anchors}\n\n"
        "## Today\n"
        f"{bundle.today_summaries or '(empty)'}\n\n"
        "## Recent (per surface)\n"
        f"{_render_recent(bundle.recent_summaries)}\n\n"
        "## Jogged (cascade)\n"
        f"{_render_hits(hits)}\n"
    )


def format_for_telegram_with_sections(
    bundle: PreloadBundle,
    hits: list[ConceptHit],
) -> dict[str, Any]:
    """Section-aware variant of format_for_telegram.

    Returns:
        {
          "full": str,                    # concat of all rendered sections
          "sections": {
              "anchors": {"payload": str, "rendered": str},
              "today":   {"payload": str, "rendered": str},
              "recent":  {"payload": str, "rendered": str},
              "jogged":  {"payload": str, "rendered": str},
          }
        }

    Empty sections (no payload) render to an empty string — no header, no
    "(empty)" placeholder. This is the contract used by the prelim-metrics
    measurement code: payload_chars == 0 ⟺ rendered_chars == 0, so the
    ablation arm cleanly subtracts retrieval-eligible sections without
    leaving header bytes behind.

    Old format_for_telegram (with placeholders) is preserved for back-compat
    with Retell + integration tests.

    Spec: docs/superpowers/specs/2026-04-29-pdct-prelim-metrics-spec.md (v4)
    """
    a = bundle.anchors or ""
    t = bundle.today_summaries or ""
    r = _render_recent_strict(bundle.recent_summaries)
    j = _render_hits_strict(hits)

    sections = {
        "anchors": {
            "payload": a,
            "rendered": f"## Anchors\n{a}\n\n" if a else "",
        },
        "today": {
            "payload": t,
            "rendered": f"## Today\n{t}\n\n" if t else "",
        },
        "recent": {
            "payload": r,
            "rendered": f"## Recent (per surface)\n{r}\n\n" if r else "",
        },
        "jogged": {
            "payload": j,
            "rendered": f"## Jogged (cascade)\n{j}\n" if j else "",
        },
    }
    full = "".join(s["rendered"] for s in sections.values())
    return {"full": full, "sections": sections}


def _render_recent_strict(recent: dict[str, str]) -> str:
    """Like _render_recent but returns "" instead of placeholder when empty."""
    parts = []
    for surface, text in (recent or {}).items():
        if text and text.strip():
            parts.append(f"### {surface}\n{text}")
    return "\n\n".join(parts)


def _render_hits_strict(hits: list[ConceptHit]) -> str:
    """Like _render_hits but returns "" instead of "(no cascade)" when empty."""
    if not hits:
        return ""
    return "\n".join(f"- [{h.hop}] {h.concept} (score={h.score:.2f})" for h in hits)


def format_for_retell(bundle: PreloadBundle, hits: list[ConceptHit]) -> str:
    """System prompt addition for Retell voice sessions.

    Currently uses the same shape as Telegram; split in future if divergence needed.
    """
    return format_for_telegram(bundle, hits)


def format_for_claude_code(hits: list[ConceptHit]) -> dict[str, Any]:
    """jogged: block for enriching MCP tool responses.

    Claude Code's preload is handled by CLAUDE.md (project convention), so this
    formatter only emits the cascade hits. Returns a JSON-safe dict.
    """
    return {
        "jogged": [
            {
                "concept": h.concept,
                "score": round(h.score, 4),
                "hop": h.hop,
                "source_slug": h.source_slug,
                "snippet": h.snippet,
            }
            for h in hits
        ]
    }
