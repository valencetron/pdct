"""Batch ingest from stream adapters into the DCT event log.

Phase 2a: Telegram adapter only. Reads .messages.json files, extracts concepts
via rules, writes one Event per non-empty concept set.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dct.adapters.claude_code import parse_file as _claude_code_parse_file
from dct.adapters.retell import parse_file as _retell_parse_file
from dct.adapters.telegram import parse_file as _telegram_parse_file
from dct.adapters.vault import (
    extract_vault_concepts,
    parse_file as _vault_parse_file,
)
from dct.event_log import EventLog
from dct.events import Event, EventOp, EventSource
from dct.rules import extract, extract_from_paths, extract_from_structured_fields


@dataclass
class IngestStats:
    files_processed: int = 0
    turns_parsed: int = 0
    events_written: int = 0
    turns_skipped_dedupe: int = 0


_ADAPTERS = {
    EventSource.TELEGRAM:    _telegram_parse_file,
    EventSource.CLAUDE_CODE: _claude_code_parse_file,
    EventSource.VOICE:       _retell_parse_file,
    EventSource.VAULT:       _vault_parse_file,
}


def ingest_files(
    paths: list[Path],
    log: EventLog,
    *,
    source: EventSource,
    dedupe: bool = False,
) -> IngestStats:
    stats = IngestStats()
    seen_keys: set[tuple[str, int, str]] = set()
    if dedupe:
        for prior in log.read_all():
            md = prior.metadata
            sf = md.get("source_file")
            ti = md.get("turn_index")
            es = md.get("extraction_source") or "prose"
            if sf is not None and ti is not None:
                try:
                    seen_keys.add((sf, int(ti), es))
                except ValueError:
                    continue

    adapter_parse = _ADAPTERS[source]

    for path in paths:
        try:
            turns = adapter_parse(path)
        except ValueError as exc:
            print(f"dct.ingest: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        stats.files_processed += 1
        stats.turns_parsed += len(turns)
        for turn in turns:
            base_metadata = {
                "role": turn.role,
                "source_file": turn.source_file,
                "turn_index": str(turn.turn_index),
                **{k: v for k, v in turn.source_meta.items() if isinstance(v, str)},
            }

            if source == EventSource.VAULT:
                vault_concepts = extract_vault_concepts(turn)
                if not vault_concepts:
                    continue
                key = (turn.source_file, turn.turn_index, "vault")
                if dedupe and key in seen_keys:
                    stats.turns_skipped_dedupe += 1
                    continue
                log.append(Event(
                    ts=turn.ts,
                    source=source,
                    op=EventOp.WRITE,
                    concepts=vault_concepts,
                    metadata={**base_metadata, "extraction_source": "vault"},
                ))
                stats.events_written += 1
                seen_keys.add(key)
                continue

            # Prose event
            prose_concepts = extract(turn.text)
            if prose_concepts:
                key = (turn.source_file, turn.turn_index, "prose")
                if dedupe and key in seen_keys:
                    stats.turns_skipped_dedupe += 1
                else:
                    log.append(Event(
                        ts=turn.ts,
                        source=source,
                        op=EventOp.TRAVERSAL,
                        concepts=prose_concepts,
                        metadata={**base_metadata, "extraction_source": "prose"},
                    ))
                    stats.events_written += 1
                    seen_keys.add(key)

            # Path-derived event
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
            path_concepts = extract_from_paths(path_inputs)
            if path_concepts:
                key = (turn.source_file, turn.turn_index, "tool_input_path")
                if dedupe and key in seen_keys:
                    stats.turns_skipped_dedupe += 1
                else:
                    log.append(Event(
                        ts=turn.ts,
                        source=source,
                        op=EventOp.TRAVERSAL,
                        concepts=path_concepts,
                        metadata={**base_metadata, "extraction_source": "tool_input_path"},
                    ))
                    stats.events_written += 1
                    seen_keys.add(key)

            # Structured-field event
            struct_fields: dict[str, str] = {}
            contributing_tools: set[str] = set()
            for tu in getattr(turn, "tool_uses", ()):
                if any(tu.tool_name.endswith(s) for s in ("mc_card_create", "mc_card_update", "mc_card_list")):
                    for k in ("slug", "title", "status", "tags"):
                        v = tu.inputs.get(k)
                        if isinstance(v, str) and v:
                            struct_fields[k] = v
                    contributing_tools.add(tu.tool_name)
                elif tu.tool_name == "Skill":
                    v = tu.inputs.get("skill")
                    if isinstance(v, str) and v:
                        struct_fields["skill"] = v
                    contributing_tools.add(tu.tool_name)
            struct_concepts = extract_from_structured_fields(struct_fields)
            if struct_concepts:
                key = (turn.source_file, turn.turn_index, "tool_input_structured")
                if dedupe and key in seen_keys:
                    stats.turns_skipped_dedupe += 1
                else:
                    log.append(Event(
                        ts=turn.ts,
                        source=source,
                        op=EventOp.TRAVERSAL,
                        concepts=struct_concepts,
                        metadata={
                            **base_metadata,
                            "extraction_source": "tool_input_structured",
                            "tool_name": ",".join(sorted(contributing_tools)),
                        },
                    ))
                    stats.events_written += 1
                    seen_keys.add(key)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(prog="dct.ingest")
    parser.add_argument("--source", required=True,
                        choices=["telegram", "claude-code", "voice", "vault"])
    parser.add_argument("--input", required=True, help="glob pattern for input files")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--dedupe", action="store_true", help="skip turns already in log")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    source_enum = EventSource(args.source)

    expanded = os.path.expanduser(args.input)
    matches = [Path(p) for p in sorted(glob.glob(expanded))]
    if not matches:
        print(f"dct.ingest: no files matched: {expanded}", file=sys.stderr)
        return 0

    log = EventLog(args.log)
    try:
        if args.verbose:
            for p in matches:
                print(f"  processing {p}", file=sys.stderr)
        stats = ingest_files(matches, log, source=source_enum, dedupe=args.dedupe)
    except ValueError as exc:
        print(f"dct.ingest: {exc}", file=sys.stderr)
        return 1

    print(
        f"{stats.files_processed} files, {stats.turns_parsed} turns, "
        f"{stats.events_written} events, "
        f"{stats.turns_skipped_dedupe} duplicates skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
