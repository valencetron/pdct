"""Batch distiller CLI.

Groups events in the log by (source_channel, session_id), skips already-
distilled sessions (unless --force), re-reads raw source files via existing
adapters, calls the LLM, writes concept-anchored MD notes to
<vault>/dct-distillations/<channel>/<session_id>.md idempotently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dct.events import Event, EventSource
from dct.llm import DistilledNote


import re as _re

_GIST_CAP = 200
_SENT_END_RE = _re.compile(r"(?<=[.!?])\s+")


def _gist_from_summary(summary: str) -> str:
    """First sentence(s) of the summary, extended until the gist meets the
    contract minimum (80 chars) or the summary is exhausted, capped at 200."""
    if not summary:
        return ""
    text = summary.strip().replace("\n", " ")
    sentences = _SENT_END_RE.split(text)
    gist = ""
    for sent in sentences:
        candidate = (gist + " " + sent).strip() if gist else sent.strip()
        gist = candidate
        if len(gist) >= 80:
            break
    gist = gist.rstrip(".!?").strip()
    if len(gist) <= _GIST_CAP:
        return gist
    return gist[:_GIST_CAP - 1].rstrip() + "…"


def _session_id_from_source_file(source_file: str) -> str:
    """Extract session ID from source filename (stem, not including extensions)."""
    name = Path(source_file).name
    # Remove all trailing extensions (e.g., .messages.json -> base, .jsonl -> base)
    while "." in name:
        name = name.rsplit(".", 1)[0]
    return name


@dataclass
class SessionGroup:
    channel: str
    session_id: str
    source_file: str
    ts_start: float
    ts_end: float
    rules_concepts: list[str] = field(default_factory=list)
    metadata_sample: dict = field(default_factory=dict)

    @property
    def session_key(self) -> tuple[str, str]:
        return (self.channel, self.session_id)


def group_events_by_session(events: list[Event]) -> list[SessionGroup]:
    bins: dict[tuple[str, str], SessionGroup] = {}
    for ev in events:
        if ev.source == EventSource.VAULT:
            continue
        channel = ev.source.value
        sf = ev.metadata.get("source_file")
        if not isinstance(sf, str) or not sf:
            continue
        sid = _session_id_from_source_file(sf)
        key = (channel, sid)
        grp = bins.get(key)
        if grp is None:
            grp = SessionGroup(
                channel=channel,
                session_id=sid,
                source_file=sf,
                ts_start=ev.ts,
                ts_end=ev.ts,
                rules_concepts=[],
                metadata_sample=dict(ev.metadata),
            )
            bins[key] = grp
        grp.ts_start = min(grp.ts_start, ev.ts)
        grp.ts_end = max(grp.ts_end, ev.ts)
        for c in ev.concepts:
            if c not in grp.rules_concepts:
                grp.rules_concepts.append(c)
    return list(bins.values())


_DEFAULT_SOURCE_ROOTS = {
    "claude-code": Path.home() / ".claude" / "projects",
}


def resolve_source_path(
    channel: str,
    metadata: dict,
    source_roots: dict[str, Path] | None = None,
) -> Path:
    roots = {**_DEFAULT_SOURCE_ROOTS, **(source_roots or {})}
    sf = metadata.get("source_file", "")
    if not sf:
        raise ValueError(f"no source_file in metadata for channel {channel}")

    sf_path = Path(sf)
    if sf_path.is_absolute():
        return sf_path

    if channel == "claude-code":
        project_slug = metadata.get("project_slug")
        if not project_slug:
            raise ValueError("claude-code source requires project_slug metadata")
        root = roots.get("claude-code")
        if root is None:
            raise ValueError("no source_root configured for claude-code")
        return root / project_slug / sf

    root = roots.get(channel)
    if root is None:
        raise ValueError(f"{channel}: source_file is relative but no root configured")
    return root / sf


_DISTILL_SUBDIR = "dct-distillations"


def output_path(group: SessionGroup, *, vault_root: Path, part: int | None = None) -> Path:
    stem = group.session_id if part is None else f"{group.session_id}--part{part}"
    return Path(vault_root) / _DISTILL_SUBDIR / group.channel / f"{stem}.md"


def is_already_distilled(group: SessionGroup, *, vault_root: Path) -> bool:
    return output_path(group, vault_root=vault_root).exists()


def _slug_to_title_case(slug: str) -> str:
    """Convert hyphen-separated slug to Title Case."""
    return " ".join(part.capitalize() for part in slug.split("-"))


def _yaml_list(items) -> str:
    """Format items as YAML list syntax: [item1, item2, ...]."""
    return "[" + ", ".join(items) + "]"


def render_note(
    *,
    group: SessionGroup,
    note: DistilledNote,
    turn_count: int,
    distilled_at: datetime,
    distilled_model: str,
) -> str:
    """Render frontmatter + body for a distilled note.

    Returns markdown text with YAML frontmatter and sections for summary,
    key concepts, and key quotes.
    """
    # Compute union of concepts (rules + LLM) preserving order
    concepts_rules = list(group.rules_concepts)
    concepts_llm = list(note.concepts)
    union: list[str] = []
    seen: set[str] = set()
    for c in concepts_rules + concepts_llm:
        if c not in seen:
            seen.add(c)
            union.append(c)

    # Build YAML frontmatter
    fm_lines = [
        "---",
        f"title: {note.title}",
        f"source_channel: {group.channel}",
        f"session_id: {group.session_id}",
        f"source_file: {group.source_file}",
        f"session_ts_start: {group.ts_start}",
        f"session_ts_end: {group.ts_end}",
        f"session_turn_count: {turn_count}",
        f"distilled_at: {distilled_at.isoformat()}",
        f"distilled_model: {distilled_model}",
        f"gist: {_gist_from_summary(note.summary)}",
        f"concepts: {_yaml_list(union)}",
        f"concepts_rules: {_yaml_list(concepts_rules)}",
        f"concepts_llm: {_yaml_list(concepts_llm)}",
        "---",
        "",
    ]

    # Build body
    body_lines: list[str] = []
    body_lines.append("## Summary")
    body_lines.append("")
    body_lines.append(note.summary.strip())
    body_lines.append("")

    body_lines.append("## Key concepts")
    body_lines.append("")
    if union:
        for c in union:
            body_lines.append(f"- [[{_slug_to_title_case(c)}]]")
    else:
        body_lines.append("_(none)_")
    body_lines.append("")

    body_lines.append("## Key quotes")
    body_lines.append("")
    if note.key_quotes:
        for q in note.key_quotes:
            role = q.get("role", "unknown")
            text = q.get("text", "").strip().replace("\n", " ")
            body_lines.append(f"- _{role}:_ \"{text}\"")
    else:
        body_lines.append("_(none)_")
    body_lines.append("")

    return "\n".join(fm_lines + body_lines)


import argparse
import os
import sys
import tempfile

from dct.adapters.claude_code import parse_file as _cc_parse_file
from dct.adapters.retell import parse_file as _retell_parse_file
from dct.adapters.telegram import parse_file as _tg_parse_file
from dct.event_log import EventLog
from dct.llm import call_distiller, resolve_model_id


_PARSERS = {
    "telegram":    _tg_parse_file,
    "claude-code": _cc_parse_file,
    "voice":       _retell_parse_file,
}


def write_note(target: Path, content: str) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def distill_one(
    *,
    group: SessionGroup,
    vault_root: Path,
    source_roots: dict[str, Path] | None,
    model: str,
    force: bool = False,
) -> str:
    """Return one of: 'written', 'skipped', 'empty', 'error:<reason>'."""
    if not force and is_already_distilled(group, vault_root=vault_root):
        return "skipped"

    parse_file = _PARSERS.get(group.channel)
    if parse_file is None:
        return f"error:no-parser-for-{group.channel}"

    try:
        source_path = resolve_source_path(
            channel=group.channel,
            metadata={"source_file": group.source_file, **group.metadata_sample},
            source_roots=source_roots,
        )
    except ValueError as exc:
        return f"error:resolve-{exc}"

    if not source_path.exists():
        return f"error:missing-source:{source_path}"

    try:
        turns = parse_file(source_path)
    except ValueError as exc:
        return f"error:parse-{exc}"
    if not turns:
        return "empty"

    llm_turns = [{"role": t.role, "text": t.text} for t in turns if t.text.strip()]
    if not llm_turns:
        return "empty"

    session_meta = {
        "source_channel": group.channel,
        "session_id": group.session_id,
        "ts_start": group.ts_start,
        "ts_end": group.ts_end,
        "turn_count": len(turns),
    }

    try:
        note = call_distiller(
            turns=llm_turns,
            session_meta=session_meta,
            rules_concepts=group.rules_concepts,
            model=model,
        )
    except Exception as exc:
        return f"error:llm-{type(exc).__name__}:{exc}"

    rendered = render_note(
        group=group,
        note=note,
        turn_count=len(turns),
        distilled_at=datetime.now(timezone.utc),
        distilled_model=resolve_model_id(model),
    )
    from dct.contract import check_fields
    report = check_fields(
        title=note.title,
        gist=_gist_from_summary(note.summary),
        concepts=list(group.rules_concepts) + list(note.concepts),
        date=group.ts_start or group.ts_end,
    )
    if not report.ok:
        print(
            f"[contract] {group.session_id}: "
            + "; ".join(report.violations),
            file=sys.stderr,
        )

    target = output_path(group, vault_root=vault_root)
    try:
        write_note(target, rendered)
    except OSError as exc:
        return f"error:write-{exc}"
    return "written" if report.ok else "written-contract-violations"


def main() -> int:
    p = argparse.ArgumentParser(prog="dct.distiller")
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--vault", required=True, type=Path,
                   help="vault root, e.g. ~/example-stack/vault")
    p.add_argument("--source",
                   choices=["telegram", "claude-code", "voice"],
                   default=None,
                   help="filter to one channel (default: all)")
    p.add_argument("--force", action="store_true",
                   help="re-distill even if output MD exists")
    p.add_argument("--model", default="haiku",
                   help="haiku|sonnet|opus (or full claude-*-4-* ID)")
    p.add_argument("--limit", type=int, default=0,
                   help="process at most N groups (0 = unbounded)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    vault_root = Path(args.vault).expanduser().resolve()
    events = EventLog(args.log).read_all()
    groups = group_events_by_session(events)
    if args.source:
        groups = [g for g in groups if g.channel == args.source]
    groups.sort(key=lambda g: (g.channel, g.session_id))

    written = skipped = empty = errors = 0
    for idx, g in enumerate(groups):
        if args.limit and idx >= args.limit:
            break
        result = distill_one(
            group=g, vault_root=vault_root,
            source_roots=None, model=args.model, force=args.force,
        )
        if args.verbose:
            print(f"  {g.channel}/{g.session_id}: {result}", file=sys.stderr)
        if result == "written":
            written += 1
        elif result == "skipped":
            skipped += 1
        elif result == "empty":
            empty += 1
        elif result.startswith("error:"):
            errors += 1

    print(f"distilled: {written} written, {skipped} skipped, {empty} empty, {errors} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
