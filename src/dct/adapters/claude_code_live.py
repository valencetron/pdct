"""Live tailer for Claude Code session JSONL files.

Watches `~/.claude/projects/**/*.jsonl` with watchdog. On each new line
appended to a session file, parses and emits an Event to DCT's events.jsonl.

What gets emitted:
  - user message text       → op=write, role=user
  - assistant text reply    → op=write, role=assistant
  - assistant tool_use      → op=read (if tool name matches READ_TOOLS)
                               op=write (if tool name matches WRITE_TOOLS)

What is skipped:
  - queue-operation entries (noise)
  - attachment entries (hook success/failure, summary fetches)
  - tool_result entries (redundant with tool_use)

On startup, tracks existing files at EOF (skip historical). Files created
after startup are tailed from byte 0 (all content is new).

Offsets live in-memory only — daemon restart resumes from current EOF.
Acceptable for a live observability feed; historical data is covered
separately by `dct ingest --source claude-code`.

Installation:
  Invoked as a launchd service via
  `launchd/com.exampleco.dct-claude-code-watcher.plist`.

CLI entry:
  python -m dct.adapters.claude_code_live
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from dct.events import Event, EventSource, EventOp
from dct.event_log import EventLog
from dct.retrieval.service import extract_concepts


CLAUDE_PROJECTS_ROOT = Path.home() / ".claude" / "projects"
from dct import config as _cfg

EVENTS_JSONL = _cfg.events_path()

# Tool-name → direction mapping (Claude Code built-in tools)
READ_TOOLS = frozenset({
    "Read", "Grep", "Glob", "WebFetch", "WebSearch", "NotebookRead",
    "BashOutput", "TodoList", "Task",  # Task is more read-like (dispatches)
})
WRITE_TOOLS = frozenset({
    "Write", "Edit", "NotebookEdit", "TodoWrite", "Bash",  # Bash is side-effect-heavy
    "ExitPlanMode", "KillShell",
})

logger = logging.getLogger("dct.claude_code_live")


def _classify_tool(tool_name: str) -> EventOp:
    if tool_name in READ_TOOLS:
        return EventOp.READ
    if tool_name in WRITE_TOOLS:
        return EventOp.WRITE
    # Unknown tools default to READ — safer assumption for observability.
    # Custom MCP tools often fall here (mcp__something__foo).
    return EventOp.READ


def _message_text(message: dict[str, Any]) -> str:
    """Extract plain text from a Claude message's content blocks."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return ""


def _tool_use_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def _parse_line(line: str, session_id: str, project_dir: str) -> list[Event]:
    """Parse one CC session log line → 0-N events."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return []

    if not isinstance(record, dict):
        return []

    # Skip obvious noise
    if record.get("type") in ("queue-operation", "summary"):
        return []
    if "attachment" in record and "message" not in record:
        return []

    message = record.get("message")
    if not isinstance(message, dict):
        return []

    role = message.get("role") or record.get("type") or ""
    if role not in ("user", "assistant"):
        return []

    # Parse timestamp
    ts_raw = record.get("timestamp") or record.get("ts")
    if isinstance(ts_raw, str):
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            ts = time.time()
    elif isinstance(ts_raw, (int, float)):
        ts = float(ts_raw)
    else:
        ts = time.time()

    events: list[Event] = []

    # Text content → one write event per user/assistant turn.
    text = _message_text(message)
    if text.strip():
        concepts = extract_concepts(text)
        if concepts:
            # op=turn (not write) — CC session turns are raw conversation,
            # not yet persisted to vault. Distillation produces the op=write.
            events.append(Event(
                ts=ts,
                source=EventSource.CLAUDE_CODE,
                op=EventOp.TURN,
                concepts=concepts,
                metadata={
                    "role": role,
                    "session_id": session_id,
                    "project_dir": project_dir,
                    "extraction_source": "prose",
                    "model": str(message.get("model", "")),
                },
            ))

    # Tool calls — emit events for Read/Grep AND Write/Edit tools touching
    # the Obsidian vault. Reads are knowledge retrieval; writes are real
    # persistent-knowledge ops (memory mode wants both). Vault writes are
    # ALSO captured by the vault-watcher (filesystem side); we de-dupe by
    # tagging extraction_source="vault_write" + the consumer can filter
    # by op+source_file+ts proximity if needed. In practice the two
    # adapters fire on different signals (tool_use record vs file mtime)
    # at slightly different times, and both are useful: the tool_use
    # captures the *intent* (CC about to write), while vault-watcher
    # captures the *result* (file actually changed).
    #
    # Other built-in tool calls (Bash, Task, Skill, etc.) stay silent —
    # they're ambient code work, not knowledge ops.
    #
    # Post-DP-42 (mcp-bridge source restore), we'll also emit events for
    # mcp__* tool calls since those are the other class of knowledge ops.
    if role == "assistant":
        for block in _tool_use_blocks(message):
            tool_name = str(block.get("name", ""))
            tool_input = block.get("input") or {}
            if not isinstance(tool_input, dict):
                continue
            path = str(tool_input.get("file_path") or tool_input.get("path") or "")
            pattern = str(tool_input.get("pattern") or "")

            if tool_name in ("Read", "Grep", "Glob"):
                if not _is_vault_path(path):
                    continue
                probe_text = f"{Path(path).name} {pattern}"
                concepts = extract_concepts(probe_text)
                if not concepts:
                    concepts = [Path(path).stem.lower().replace(" ", "-")[:40]]
                events.append(Event(
                    ts=ts,
                    source=EventSource.CLAUDE_CODE,
                    op=EventOp.READ,
                    concepts=concepts,
                    metadata={
                        "role": role,
                        "session_id": session_id,
                        "project_dir": project_dir,
                        "extraction_source": "vault_read",
                        "tool_name": tool_name,
                        "source_file": path,
                    },
                ))
            elif tool_name in ("Write", "Edit", "NotebookEdit"):
                if not _is_vault_path(path):
                    continue
                # Probe text: filename + edit content (Write→content,
                # Edit→new_string). Concepts from real prose if available.
                content = str(
                    tool_input.get("content")
                    or tool_input.get("new_string")
                    or tool_input.get("new_source")
                    or ""
                )
                probe_text = f"{Path(path).name}\n{content[:2000]}"
                concepts = extract_concepts(probe_text)
                if not concepts:
                    concepts = [Path(path).stem.lower().replace(" ", "-")[:40]]
                events.append(Event(
                    ts=ts,
                    source=EventSource.CLAUDE_CODE,
                    op=EventOp.WRITE,
                    concepts=concepts,
                    metadata={
                        "role": role,
                        "session_id": session_id,
                        "project_dir": project_dir,
                        "extraction_source": "vault_write",
                        "tool_name": tool_name,
                        "source_file": path,
                    },
                ))

    return events


_VAULT_PATH_ROOTS = (
    str(Path.home() / "example-stack" / "vault"),
)


def _is_vault_path(path: str) -> bool:
    if not path:
        return False
    for root in _VAULT_PATH_ROOTS:
        if path.startswith(root):
            return True
    return False


class JsonlWatcher:
    """Tails all `.jsonl` files under a root directory.

    Not thread-safe per file; relies on watchdog's single-threaded dispatcher.
    """

    def __init__(self, root: Path, events_log: EventLog) -> None:
        self.root = root
        self.events_log = events_log
        self._offsets: dict[str, int] = {}
        self._lock = threading.Lock()

    def seed_from_existing(self) -> int:
        """Seek all existing .jsonl files to EOF. Returns count."""
        count = 0
        if not self.root.is_dir():
            return 0
        for p in self.root.rglob("*.jsonl"):
            try:
                self._offsets[str(p)] = p.stat().st_size
                count += 1
            except OSError:
                pass
        return count

    def handle_modified(self, path: str) -> None:
        with self._lock:
            offset = self._offsets.get(path, 0)
            try:
                size = os.stat(path).st_size
            except OSError:
                return
            if size == offset:
                return
            if size < offset:
                # Truncation — reset to EOF to avoid re-emitting.
                self._offsets[path] = size
                return
            try:
                with open(path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read(size - offset)
                self._offsets[path] = size
            except OSError as e:
                logger.warning("tail failed for %s: %s", path, e)
                return

        # Parse outside the lock — parsing is the slow part.
        try:
            lines = chunk.decode("utf-8", errors="replace").splitlines()
        except Exception:
            return

        session_id = Path(path).stem
        project_dir = Path(path).parent.name

        for line in lines:
            if not line.strip():
                continue
            events = _parse_line(line, session_id=session_id, project_dir=project_dir)
            for ev in events:
                try:
                    self.events_log.append(ev)
                except Exception as e:
                    logger.warning("append failed: %s", e)

    def handle_created(self, path: str) -> None:
        # New session file — start at byte 0 so all content is captured.
        with self._lock:
            self._offsets[path] = 0
        self.handle_modified(path)


class _Dispatcher(FileSystemEventHandler):
    def __init__(self, watcher: JsonlWatcher) -> None:
        self.watcher = watcher

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if str(event.src_path).endswith(".jsonl"):
            self.watcher.handle_modified(str(event.src_path))

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if str(event.src_path).endswith(".jsonl"):
            self.watcher.handle_created(str(event.src_path))


def main(argv: list[str] | None = None) -> int:
    del argv
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [cc-watcher] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not CLAUDE_PROJECTS_ROOT.is_dir():
        logger.error("claude projects root does not exist: %s", CLAUDE_PROJECTS_ROOT)
        return 1

    events_log = EventLog(EVENTS_JSONL)
    watcher = JsonlWatcher(CLAUDE_PROJECTS_ROOT, events_log)
    seeded = watcher.seed_from_existing()
    logger.info("seeded %d existing files at EOF (skipping historical)", seeded)

    observer = Observer()
    observer.schedule(_Dispatcher(watcher), str(CLAUDE_PROJECTS_ROOT), recursive=True)
    observer.start()
    logger.info("watching %s (recursive)", CLAUDE_PROJECTS_ROOT)

    stop = threading.Event()

    def _shutdown(signum: int, _frame: Any) -> None:
        logger.info("signal %s received, stopping", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        while not stop.is_set():
            stop.wait(timeout=5.0)
    finally:
        observer.stop()
        observer.join(timeout=5.0)
        logger.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
