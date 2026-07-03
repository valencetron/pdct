"""Vault filesystem watcher daemon.

Observes MD changes under the Obsidian vault root and appends one
EventSource.VAULT event per stable change. Debounces rapid repeats
from Obsidian's multi-write-per-edit pattern.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from dct.adapters.vault import (
    extract_vault_concepts,
    is_ignored_path,
    parse_file as _vault_parse_file,
)
from dct.event_log import EventLog
from dct.events import Event, EventOp, EventSource


class VaultEventHandler:
    def __init__(self, *, log: EventLog, debounce_secs: float = 0.5) -> None:
        self._log = log
        self._debounce = float(debounce_secs)
        self._last_fired: dict[str, float] = {}

    def handle_path(self, path: Path, *, fs_event: str) -> None:
        p = Path(path)
        if p.suffix != ".md":
            return
        if is_ignored_path(p):
            return

        key = str(p.resolve())
        now = time.monotonic()
        last = self._last_fired.get(key, 0.0)
        if self._debounce > 0 and (now - last) < self._debounce:
            return

        try:
            turns = _vault_parse_file(p)
        except ValueError:
            return
        if not turns:
            return

        turn = turns[0]
        concepts = extract_vault_concepts(turn)
        if not concepts:
            return

        meta = {
            "role": turn.role,
            "source_file": turn.source_file,
            "turn_index": str(turn.turn_index),
            "extraction_source": "vault",
            "fs_event": fs_event,
        }
        # Propagate the distilled note's title so the Context Stream rail
        # can show a human-readable label instead of "write exampleco chat".
        fm = turn.source_meta.get("frontmatter", {}) or {}
        title = fm.get("title")
        if isinstance(title, str) and title.strip():
            # Skip if title is just an ID-like string (digits/underscores only
            # — e.g., telegram chat_id topic_id). Falls back to preview logic.
            if not all(ch.isdigit() or ch in "_-" for ch in title.strip()):
                meta["title"] = title.strip()

        # Text preview — prefer the ## Summary section (the distilled note's
        # actual content), fall back to raw body. So the rail shows what
        # the distillation actually captured instead of being a black box.
        body = turn.text or ""
        preview = ""
        summary_idx = body.find("## Summary")
        if summary_idx >= 0:
            # Skip the "## Summary" heading line itself
            after_heading = body.find("\n", summary_idx)
            if after_heading >= 0:
                start = after_heading + 1
                next_section = body.find("\n## ", start)
                section = body[start:next_section] if next_section >= 0 else body[start:]
                preview = section.strip()
        if not preview:
            preview = body.strip()
        if preview:
            meta["text_preview"] = preview[:400]
        self._log.append(Event(
            ts=turn.ts,
            source=EventSource.VAULT,
            op=EventOp.WRITE,
            concepts=concepts,
            metadata=meta,
        ))
        self._last_fired[key] = now


class _WatchdogBridge(FileSystemEventHandler):
    def __init__(self, handler: VaultEventHandler) -> None:
        self._handler = handler

    def on_created(self, event):
        if not event.is_directory:
            self._handler.handle_path(Path(event.src_path), fs_event="created")

    def on_modified(self, event):
        if not event.is_directory:
            self._handler.handle_path(Path(event.src_path), fs_event="modified")

    def on_moved(self, event):
        if not event.is_directory:
            self._handler.handle_path(Path(event.dest_path), fs_event="moved")


def run_watcher_until(
    *,
    vault_root: Path,
    log: EventLog,
    debounce_secs: float = 0.5,
    deadline_secs: float = 0.0,
    until: Callable[[], bool] | None = None,
) -> None:
    handler = VaultEventHandler(log=log, debounce_secs=debounce_secs)
    observer = Observer()
    observer.schedule(_WatchdogBridge(handler), str(vault_root), recursive=True)
    observer.start()
    try:
        start = time.monotonic()
        while True:
            if until is not None and until():
                break
            if deadline_secs > 0 and (time.monotonic() - start) >= deadline_secs:
                break
            time.sleep(0.05)
    finally:
        observer.stop()
        observer.join(timeout=2.0)


def main() -> int:
    p = argparse.ArgumentParser(prog="dct.watch")
    p.add_argument("--vault", required=True, type=Path)
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--debounce-secs", type=float, default=0.5)
    p.add_argument("--pidfile", type=Path, default=Path("/tmp/dct-vault-watcher.pid"))
    args = p.parse_args()

    args.pidfile.write_text(str(__import__("os").getpid()) + "\n", encoding="utf-8")
    print(
        f"dct.watch: observing {args.vault} → {args.log} "
        f"(debounce={args.debounce_secs}s)",
        file=sys.stderr,
    )
    try:
        run_watcher_until(
            vault_root=args.vault,
            log=EventLog(args.log),
            debounce_secs=args.debounce_secs,
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
