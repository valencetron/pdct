"""Central path/config resolution for PDCT.

Every filesystem location the system touches resolves through here, so a
single env var relocates the whole installation. Precedence per path:

    1. Specific env var (e.g. PDCT_EVENTS_PATH)
    2. PDCT_HOME env var (all defaults nest under it)
    3. Legacy default (~/example-stack/... — preserved for existing installs)

Public installs set PDCT_HOME (install.sh does this) and everything lands
under it:

    $PDCT_HOME/
      events.jsonl          conversation event log
      vault/distillations/  distilled memory notes (or point at Obsidian)
      runtime/              overrides, regions, tuning state
      logs/                 telemetry, ledgers
      data/                 judge.db and other databases

Vault roots may instead point at an Obsidian vault via OBSIDIAN_VAULT or
PDCT_VAULT_ROOT.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


def pdct_home() -> Path:
    """Root of the PDCT installation's mutable state."""
    p = _env_path("PDCT_HOME")
    if p is not None:
        return p
    # Legacy default: the repo checkout itself under ~/example-stack.
    return Path.home() / "example-stack" / "dynamic-context-traversal"


def events_path() -> Path:
    return _env_path("PDCT_EVENTS_PATH") or pdct_home() / "events.jsonl"


def runtime_dir() -> Path:
    return _env_path("PDCT_RUNTIME_DIR") or pdct_home() / "runtime"


def logs_dir() -> Path:
    return _env_path("PDCT_LOGS_DIR") or pdct_home() / "logs"


def data_dir() -> Path:
    return _env_path("DCT_DATA_DIR") or pdct_home() / "data"


def overrides_path() -> Path:
    return _env_path("PDCT_OVERRIDES_PATH") or runtime_dir() / "pdct-overrides.json"


def vault_roots() -> list[Path]:
    """Distillation roots, in walk order.

    PDCT_VAULT_ROOT / OBSIDIAN_VAULT (first match) override entirely; the
    legacy multi-root default is preserved otherwise.
    """
    for var in ("PDCT_VAULT_ROOT", "OBSIDIAN_VAULT"):
        p = _env_path(var)
        if p is not None:
            return [p / "distillations"] if (p / "distillations").is_dir() else [p]
    if _env_path("PDCT_HOME") is not None:
        return [pdct_home() / "vault" / "distillations"]
    return [
        Path.home() / "example-stack" / "vault" / "distillations",
        Path.home() / "example-stack" / "vault" / "dct-distillations",
        Path.home() / "example-stack" / "memory" / "distillations",
    ]


def archive_root() -> Path:
    p = _env_path("PDCT_ARCHIVE_ROOT")
    if p is not None:
        return p
    if _env_path("PDCT_HOME") is not None:
        return pdct_home() / "vault" / "compaction-archive"
    return Path.home() / "example-stack" / "vault" / "compaction-archive"


def anchor_candidates() -> list[Path]:
    """Optional always-on context anchor files (soul/CLAUDE docs)."""
    v = os.environ.get("PDCT_ANCHOR_PATHS")
    if v:
        return [Path(x).expanduser() for x in v.split(os.pathsep) if x.strip()]
    if _env_path("PDCT_HOME") is not None:
        return [pdct_home() / "ANCHOR.md"]
    return [
        Path.home() / "CLAUDE.md",
        Path.home() / "example-stack" / "CLAUDE.md",
        Path.home() / "example-stack" / "soul.md",
    ]
