"""Classify a concept-graph node as a jog-able concept or a non-jog action.

Deterministic, no LLM. Three tiers, in order:
  1. Action stoplist (overrides frequency) — verbs/imperatives + inflections.
  2. Frequency-promoted single-token domain nouns.
  3. Token-count default: multi-token=concept; sub-threshold single=action.
"""
from __future__ import annotations

Kind = str  # "concept" | "action"

# Seeded from the live node census (2026-06-13). Tier 1: checked BEFORE
# frequency, because high-frequency actions ("restart"=153) are still actions.
# Includes common inflections (Codex r1 #9: "started"/"running" leaked).
ACTION_STOPLIST: frozenset[str] = frozenset({
    "fix", "fixed", "build", "built", "restart", "restarted", "run", "ran",
    "running", "rerun", "retry", "sent", "send", "read", "start", "started",
    "starting", "stop", "stopped", "do", "done", "make", "made", "get", "got",
    "deploy", "deployed", "deploying", "ship", "shipped", "kill", "killed",
    "push", "pushed", "pull", "pulled", "merge", "merged", "generate",
    "generated", "list", "status", "command", "add", "added", "remove",
    "removed", "delete", "deleted", "update", "updated", "create", "created",
    "check", "checked", "verify", "verified", "test", "tested", "commit",
    "committed", "reset", "open", "opened", "close", "closed", "set", "write",
    "wrote", "show", "showed", "review", "reviewed", "debug", "patch",
    "patched", "ask", "asked", "call", "called",
})

# Tier-2 frequency threshold: a single-token node seen at least this many
# times is treated as a recurring topic (domain noun), not a fragment.
# Tunable lever — start at 10.
DOMAIN_NOUN_MIN_COUNT: int = 10


def classify_node(slug: str, graph_count: int) -> Kind:
    s = (slug or "").strip().lower()
    if not s:
        return "action"
    if s in ACTION_STOPLIST:          # tier 1 — overrides everything
        return "action"
    # multi-token (hyphen OR space OR underscore) → concept. Space-separated
    # entity names ("exampleco labs", "mission control") are concepts, not
    # actions — caught by the audit regression list 2026-06-13.
    if "-" in s or " " in s or "_" in s:
        return "concept"
    if graph_count >= DOMAIN_NOUN_MIN_COUNT:   # tier 2 — frequency promotion
        return "concept"
    return "action"                    # tier 3 — low-freq single token
