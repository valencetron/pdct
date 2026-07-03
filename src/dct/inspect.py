"""Read-only audit tool for the DCT ingest pipeline.

Runs adapters + the rules layer without writing to the event log. Phase 3:
reports prose / tool_input_path / tool_input_structured extraction sources.

Usage:
    python -m dct.inspect --source claude-code --input "~/.claude/projects/**/*.jsonl"
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter
from pathlib import Path

from dct.adapters.claude_code import parse_file as _claude_code_parse_file
from dct.adapters.retell import parse_file as _retell_parse_file
from dct.adapters.telegram import parse_file as _telegram_parse_file
from dct.adapters.vault import (
    extract_vault_concepts,
    parse_file as _vault_parse_file,
)
from dct.rules import extract, extract_from_paths, extract_from_structured_fields

_ADAPTERS = {
    "telegram":    _telegram_parse_file,
    "claude-code": _claude_code_parse_file,
    "voice":       _retell_parse_file,
    "vault":       _vault_parse_file,
}

_EXTRACTION_SOURCES = ("prose", "tool_input_path", "tool_input_structured", "vault")


def _gather_extractions(turn) -> list[tuple[str, list[str]]]:
    results: list[tuple[str, list[str]]] = []

    fm = turn.source_meta.get("frontmatter")
    if isinstance(fm, dict):
        vault_concepts = extract_vault_concepts(turn)
        if vault_concepts:
            results.append(("vault", vault_concepts))
        return results  # vault turns only emit vault; skip prose/path/struct

    prose = extract(turn.text)
    if prose:
        results.append(("prose", prose))

    path_inputs: list[str] = []
    for tu in getattr(turn, "tool_uses", ()):
        if tu.tool_name in ("Read", "Edit", "Write"):
            v = tu.inputs.get("file_path")
            if isinstance(v, str):
                path_inputs.append(v)
        elif tu.tool_name == "Grep":
            v = tu.inputs.get("path")
            if isinstance(v, str):
                path_inputs.append(v)
    paths = extract_from_paths(path_inputs)
    if paths:
        results.append(("tool_input_path", paths))

    struct_fields: dict[str, str] = {}
    for tu in getattr(turn, "tool_uses", ()):
        if any(tu.tool_name.endswith(s) for s in ("mc_card_create", "mc_card_update", "mc_card_list")):
            for k in ("slug", "title", "status", "tags"):
                v = tu.inputs.get(k)
                if isinstance(v, str) and v:
                    struct_fields[k] = v
        elif tu.tool_name == "Skill":
            v = tu.inputs.get("skill")
            if isinstance(v, str) and v:
                struct_fields["skill"] = v
    structured = extract_from_structured_fields(struct_fields)
    if structured:
        results.append(("tool_input_structured", structured))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(prog="dct.inspect")
    parser.add_argument("--source", required=True, choices=list(_ADAPTERS.keys()))
    parser.add_argument("--input", required=True, help="glob pattern for input files")
    parser.add_argument("--limit", type=int, default=0, help="stop after N body lines (0 = unbounded)")
    parser.add_argument(
        "--min-concepts", type=int, default=1,
        help="only print extractions with at least K concepts (0 prints all)",
    )
    parser.add_argument(
        "--source-only", choices=_EXTRACTION_SOURCES, default=None,
        help="restrict body lines to one extraction source",
    )
    args = parser.parse_args()

    expanded = os.path.expanduser(args.input)
    matches = [Path(p) for p in sorted(glob.glob(expanded))]
    if not matches:
        print(f"dct.inspect: no files matched: {expanded}", file=sys.stderr)
        return 0

    parse_file = _ADAPTERS[args.source]
    printed = 0
    turns_parsed = 0
    events_by_source: Counter[str] = Counter()
    concepts_by_source: dict[str, Counter[str]] = {s: Counter() for s in _EXTRACTION_SOURCES}

    for path in matches:
        try:
            turns = parse_file(path)
        except ValueError as exc:
            print(f"dct.inspect: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        for turn in turns:
            turns_parsed += 1
            for src, concepts in _gather_extractions(turn):
                events_by_source[src] += 1
                concepts_by_source[src].update(concepts)
                if args.source_only and src != args.source_only:
                    continue
                if len(concepts) < args.min_concepts:
                    continue
                if args.limit and printed >= args.limit:
                    break
                text_preview = turn.text.strip().replace("\n", " ")
                if len(text_preview) > 100:
                    text_preview = text_preview[:100] + "..."
                print(
                    f"[{turn.ts:.3f}]  {turn.source_file}#{turn.turn_index}  "
                    f"role={turn.role}  src={src}  concepts={concepts}  "
                    f"text=\"{text_preview}\""
                )
                printed += 1
            if args.limit and printed >= args.limit:
                break
        if args.limit and printed >= args.limit:
            break

    total_events = sum(events_by_source.values())
    print(
        f"\n{turns_parsed} turns parsed, {total_events} events across "
        f"{len([s for s, n in events_by_source.items() if n])} sources:"
    )
    for s in _EXTRACTION_SOURCES:
        print(f"  {s}: {events_by_source[s]} events")
    print("Top per source:")
    for s in _EXTRACTION_SOURCES:
        top = ", ".join(f"{c} ({n})" for c, n in concepts_by_source[s].most_common(10))
        print(f"  {s}: {top or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
