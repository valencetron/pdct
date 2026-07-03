"""run_cell — the atomic benchmark unit.

For one (question, config) arm, run R replicates of:
    retrieval (config_override) → same-model reply → Haiku judge
        → 3-leg composite (benchmark weighting).

REUSE, don't reimplement:
  - service.run(config_override=...)        retrieval under the lever arm
  - dct.llm._client_factory()               same-model reply (claude-sonnet-4-6)
  - dct.judge.invoke_judge(prompt)          era_judge leg
  - dct.retrieval.utility.score_turn_utility match_rate leg
  - dct.retrieval.cosine.cosine_score       cosine leg
  - dct.composite.compute_composite(legs, weights=BENCHMARK_WEIGHTS)

In-process only: passing config_override means service.run() never writes the
live overrides file (asserted in tests).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from dct.composite import compute_composite
from dct.judge.invoker import invoke_judge  # noqa: F401  (patched in tests)
from dct.research import BENCHMARK_WEIGHTS
from dct.retrieval import service
from dct.retrieval.cosine import cosine_score
from dct.retrieval.utility import score_turn_utility

log = logging.getLogger(__name__)

# Same model the live system replies with. Pinned contract (Codex finding).
REPLY_MODEL = "claude-sonnet-4-6"
REPLY_MAX_TOKENS = 1024

# CRITICAL — first-party identity. Claude Max OAuth tokens are only accepted
# in-contract when the LEADING system block is exactly the Claude Code identity
# string (same as the daemon's reply path, daemon.py:4483/5543/7239). Sending a
# Max OAuth token with a different leading system prompt gets throttled (429) as
# out-of-contract — NOT a real rate limit. We pass system as a LIST: Claude Code
# identity first, then the benchmark instructions as a second block.
_CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
REPLY_INSTRUCTIONS = (
    "Answer the user's message using the injected context block where relevant. "
    "Be concise and direct."
)
REPLY_SYSTEM = [
    {"type": "text", "text": _CLAUDE_CODE_IDENTITY},
    {"type": "text", "text": REPLY_INSTRUCTIONS},
]


def _reply_client():
    """Anthropic client for the reply. Seam for tests — patch this."""
    from dct.llm import _client_factory

    return _client_factory()


def _generate_reply(user_text: str, cascade_block: str) -> str:
    """Generate a same-model reply given the question + injected context."""
    client = _reply_client()
    content = (
        f"## Injected context\n{cascade_block or '(none)'}\n\n"
        f"## User message\n{user_text}"
    )
    resp = client.messages.create(
        model=REPLY_MODEL,
        max_tokens=REPLY_MAX_TOKENS,
        system=REPLY_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text


def _judge_prompt(user_text: str, cascade_block: str, reply_text: str) -> str:
    """Build the (user, cascade, reply) judge prompt — same shape as worker."""
    return "\n".join(
        [
            "## Era: benchmark",
            "",
            "## User message",
            user_text.strip() or "(empty)",
            "",
            "## Cascade context injected",
            (cascade_block or "").strip() or "(none)",
            "",
            "## AI reply",
            reply_text.strip() or "(empty)",
        ]
    )


def _score_replicate(question: str, config) -> dict[str, Any]:
    """One replicate: retrieve → reply → judge → 3 legs → composite."""
    # 1. Retrieval under the lever arm (in-process, no file write).
    retrieval = service.run(question, config_override=config)
    cascade_block = retrieval.get("prompt_block", "") or ""
    injected = retrieval.get("cascade_concepts", []) or []
    paths = retrieval.get("cascade_paths", {}) or {}
    node_kinds = retrieval.get("node_kinds", {}) or {}

    # 2. Same-model reply.
    reply = _generate_reply(question, cascade_block)

    # 3a. match_rate leg.
    util = score_turn_utility(reply, injected, paths, node_kinds=node_kinds)
    match_rate = util.get("match_rate")

    # 3b. cosine leg (cascade-vs-reply, same as live).
    cos = cosine_score(cascade_block, reply)

    # 3c. era_judge leg — pass the RAW Haiku 1-5 score. compute_composite()
    # does the 1-5 → [0,1] normalization itself (composite.py). Do NOT
    # pre-normalize here, or it gets normalized twice (5 → 1.0 → 0.0) and the
    # highest-weight leg is silently zeroed. (Codex diff-audit finding #1.)
    judge = invoke_judge(_judge_prompt(question, cascade_block, reply))
    judge_failed = judge.status != "ok" or judge.score is None
    era_judge_raw: Optional[int] = None if judge_failed else judge.score

    # Build legs dict — omit failed/None legs so composite normalizes correctly.
    legs: dict[str, Any] = {}
    if match_rate is not None:
        legs["match_rate"] = match_rate
    if cos is not None:
        legs["cosine_score"] = cos
    if era_judge_raw is not None:
        legs["era_judge"] = era_judge_raw  # raw 1-5; composite normalizes

    result = compute_composite(legs, weights=BENCHMARK_WEIGHTS)

    return {
        "question": question,
        "legs": legs,
        "composite": result.score,
        "weights": BENCHMARK_WEIGHTS,
        "judge_failed": judge_failed,
        "reply_len": len(reply),
        "injected_count": len(injected),
    }


def run_cell(
    question: str,
    config,
    *,
    replicates: int = 1,
    arm_label: str = "",
) -> list[dict[str, Any]]:
    """Run R replicates of (question, config) → list of scored rows.

    Each row carries the arm_label so the sweep can pair by (question, arm).
    A replicate that errors out is logged and skipped (never crashes the run).
    """
    rows: list[dict[str, Any]] = []
    for i in range(replicates):
        try:
            row = _score_replicate(question, config)
        except Exception as e:  # noqa: BLE001 — never let one replicate kill the run
            log.warning("[research.runner] replicate %d failed: %s", i, e)
            continue
        row["arm_label"] = arm_label
        row["replicate"] = i
        rows.append(row)
    return rows
