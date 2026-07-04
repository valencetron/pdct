"""Obsidian vault markdown adapter.

Parses an MD file (with optional YAML frontmatter) into a single ParsedTurn
record. Used by dct.watch + dct.ingest for the vault ingest path.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dct.adapters.telegram import ParsedTurn
from dct.rules import _filter_slugs, extract, to_slug


_FM_DELIM = "---"


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    """Return ``(frontmatter_dict, body)``. Empty dict if no frontmatter."""
    if not raw.startswith(_FM_DELIM + "\n") and not raw.startswith(_FM_DELIM + "\r\n"):
        return {}, raw
    rest = raw.split("\n", 1)[1] if raw.startswith(_FM_DELIM + "\n") else raw.split("\r\n", 1)[1]
    end_idx = rest.find("\n" + _FM_DELIM)
    if end_idx < 0:
        return {}, raw
    fm_text = rest[:end_idx]
    body = rest[end_idx + len("\n" + _FM_DELIM):].lstrip("\r\n")
    try:
        parsed = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed, body


def is_ignored_path(path: Path, *, root: Path | None = None) -> bool:
    """Return True for paths the vault adapter should skip.

    Ignores any path containing a segment starting with '.' — covers
    .obsidian/, .trash/, .DS_Store, .git/, etc.

    When ``root`` is given and ``path`` is inside it, only the segments
    BELOW the root are checked. This matters because the default
    PDCT_HOME is ``~/.pdct`` — the watch root itself contains a dotted
    segment, and filtering on the full path would silently ignore every
    note in a default install (found live on the VPS, 2026-07-04).
    """
    p = Path(path)
    if root is not None:
        try:
            parts = p.resolve().relative_to(Path(root).resolve()).parts
        except (ValueError, OSError):
            parts = p.parts
    else:
        parts = p.parts
    for part in parts:
        if part.startswith("."):
            return True
    return False


def parse_file(path: Path) -> list[ParsedTurn]:
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        raise ValueError(f"file not found: {p.name}")
    try:
        raw = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"non-utf8 vault file: {p.name}: {exc}") from exc

    frontmatter, body = _split_frontmatter(raw)
    body_stripped = body.strip()
    if not body_stripped and not frontmatter:
        return []
    if not body_stripped and frontmatter and not frontmatter.get("concepts"):
        return []

    return [ParsedTurn(
        role="assistant",
        text=body,
        turn_index=0,
        source_file=str(p),
        ts=p.stat().st_mtime,
        source_meta={"frontmatter": frontmatter},
    )]


def extract_vault_concepts(turn: ParsedTurn) -> list[str]:
    """Union of frontmatter-listed + body-extracted concepts, hygiene-filtered.

    If explicit extraction (wikilinks/hashtags/frontmatter) yields nothing,
    falls back to graph-backed prose matching so that plain-text notes
    still emit events. Falls silent on any error (hot path; never block).
    """
    raw: list[str] = []
    fm = turn.source_meta.get("frontmatter", {}) or {}
    fm_concepts = fm.get("concepts")
    if isinstance(fm_concepts, list):
        for item in fm_concepts:
            if isinstance(item, str) and item:
                raw.append(to_slug(item))
    body_concepts = extract(turn.text)
    raw.extend(body_concepts)
    result = _filter_slugs(raw)
    if result:
        return result

    # Fallback — graph-backed prose matching for free-text notes
    # (same matcher the daemon uses). Loading the graph is ~30ms warm.
    try:
        from dct.retrieval.service import extract_concepts as _prose_extract
        return _prose_extract(turn.text)
    except Exception:
        return []
