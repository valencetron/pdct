"""PDCT observability events that align with lean-ctx gain reporting.

Emits JSONL events to ~/.lean-ctx/pdct-events.log so a future `pdct gain`
dashboard can correlate distillation read volume with model token usage
across all surfaces (telegram, retell, claude-code).

Disabled when LEANCTX_PDCT_INSTRUMENT=0 (default '1' = on).

This module is fail-closed for PDCT: any error during emit is swallowed
silently. Observability MUST NOT break the read path.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

LOG_PATH = Path.home() / ".lean-ctx" / "pdct-events.log"


def emit(event_type: str, **fields: Any) -> None:
    """Append one JSONL event. Silent on every error."""
    if os.environ.get("LEANCTX_PDCT_INSTRUMENT", "1") == "0":
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), "type": event_type, **fields}
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        # Never break PDCT for observability.
        pass
