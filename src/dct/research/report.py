"""Report/archive layer → Obsidian.

Every experiment (sweep, deploy, revert) writes a structured dated markdown
report to the vault for pattern-mining — both Alex and Orion review these
over time for cross-lever interactions.

- Vault root resolved from arg/env, NOT hardcoded (Codex finding).
- Atomic write (temp + rename).
- Same-day collision → numbered suffix (never overwrite a prior report).
- Write failure raises after retries (never silently dropped).
"""
from __future__ import annotations

import datetime
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

EXPERIMENTS_SUBDIR = "pdct-experiments"


def _resolve_vault_root(vault_root: Optional[Path]) -> Path:
    """Resolve the vault root: explicit arg > env > default. Never hardcoded inline."""
    if vault_root is not None:
        return Path(vault_root)
    env = os.environ.get("PDCT_VAULT_ROOT") or os.environ.get("OBSIDIAN_VAULT")
    if env:
        return Path(env)
    return Path.home() / "Documents" / "OBSIDIAN"


def _fmt(v: Any) -> str:
    """Format a numeric composite to 2 decimals; pass through non-numerics."""
    if isinstance(v, (int, float)):
        return f"{v:.2f}"
    return str(v)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s or "experiment"


def _render(exp: dict[str, Any]) -> str:
    lever = exp.get("lever", "?")
    trigger = exp.get("trigger", "?")
    lines = [
        f"# PDCT Experiment — {lever}",
        "",
        f"- **When:** {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"- **Lever:** `{lever}`",
        f"- **Trigger:** {trigger}",
        f"- **Incumbent:** {exp.get('incumbent')}",
        f"- **Winner / candidate:** {exp.get('winner')}",
        "",
        "## Composite",
        f"- before: **{_fmt(exp.get('before_composite'))}**",
        f"- after:  **{_fmt(exp.get('after_composite'))}**",
        "",
        "## Per-leg delta",
    ]
    for leg, d in (exp.get("per_leg_delta") or {}).items():
        lines.append(f"- {leg}: {d:+.4f}" if isinstance(d, (int, float)) else f"- {leg}: {d}")
    lines += ["", "## Top-moving questions"]
    for q in (exp.get("top_moving_questions") or []):
        lines.append(f"- ({q.get('delta'):+.3f}) {q.get('question')}")
    ns = exp.get("north_star") or {}
    lines += [
        "",
        "## North Star",
        f"- blocked: {ns.get('blocked')}",
        f"- vetoed: {ns.get('vetoed')}",
        f"- per-criterion: {ns.get('per_criterion')}",
        "",
        "## Verdict",
        f"- label: **{exp.get('verdict_label', '?')}**",
        f"- reason: `{exp.get('reason', '?')}`",
        "",
        str(exp.get("verdict", "")),
        "",
        "---",
        "_Auto-written by dct.research.report — PDCT benchmark research engine._",
    ]
    return "\n".join(lines)


def write_report(
    exp: dict[str, Any],
    *,
    vault_root: Optional[Path] = None,
    max_retries: int = 3,
) -> Path:
    """Write the experiment report. Returns the path. Raises on persistent failure."""
    root = _resolve_vault_root(vault_root)
    out_dir = root / EXPERIMENTS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    date = datetime.date.today().isoformat()
    base = f"{date}-{_slug(exp.get('lever', 'experiment'))}"

    # Same-day collision → numbered suffix, never overwrite.
    candidate = out_dir / f"{base}.md"
    n = 2
    while candidate.exists():
        candidate = out_dir / f"{base}-{n}.md"
        n += 1

    body = _render(exp)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            tmp = candidate.with_suffix(".md.tmp")
            tmp.write_text(body)
            tmp.rename(candidate)
            return candidate
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("[research.report] write attempt %d failed: %s", attempt + 1, e)
            time.sleep(0.1 * (attempt + 1))
    raise RuntimeError(f"failed to write report after {max_retries} retries: {last_err}")
