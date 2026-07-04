"""family.py — sibling detection for the valence agent harness (Build 105).

PDCT and valence (github.com/valencetron/valence) are independent packages
that light up small integrations when co-installed ("family package").
This module is PDCT's half: detect a valence install and surface an
ADVISORY doctor check about it.

Contract (mirrors valence's pdct probe):
  - advisory-only: the check is never `required`, so a broken sibling can
    never change PDCT's doctor exit code;
  - silent when absent: no valence home → no check emitted at all (most
    installs won't have valence — absence is not a warning);
  - schema-tolerant: a malformed/missing fleet-status.json degrades to a
    warn-style advisory, never an exception;
  - staleness of fleet-status.json is reported as detail only — valence
    writes it on `valence doctor --fleet` runs, not from a heartbeat, so
    age must not drive the ok/warn verdict.

NOTE (export): this file is exempt from the "valence" identifier rewrite in
scripts/export_public.sh and check_sanitized.sh — here "valence" refers to
the public sibling product, not a private identity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def valence_home() -> Path:
    """Resolve the sibling harness home: $VALENCE_HOME or ~/.valence."""
    return Path(os.environ.get("VALENCE_HOME", "~/.valence")).expanduser()


def sibling_checks(home: Path | None = None, *, check_cls):
    """Return 0 or 1 advisory Check objects describing the valence sibling.

    ``check_cls`` is dct.doctor.Check (injected to avoid an import cycle).
    """
    home = home if home is not None else valence_home()
    if not home.is_dir():
        return []
    status_path = home / "fleet-status.json"
    if not status_path.exists():
        return [check_cls("sibling:valence", True,
                          f"install detected at {home} (no fleet-status.json"
                          " — run `valence doctor --fleet` to populate)",
                          required=False, id="env.sibling")]
    try:
        data = json.loads(status_path.read_text())
        probes = data.get("probes") if isinstance(data, dict) else None
        if not isinstance(probes, list):
            return [check_cls("sibling:valence", False,
                              "fleet-status.json present but has no probes"
                              " list", required=False, id="env.sibling")]
        fails = [p.get("id", "?") for p in probes
                 if isinstance(p, dict) and p.get("status") == "fail"]
        age = ""
        gen = data.get("generatedAt")
        if isinstance(gen, str):
            age = f", generated {gen}"
        if fails:
            return [check_cls("sibling:valence", False,
                              "valence fleet reports failing probes: "
                              f"{', '.join(str(f) for f in fails[:5])}{age}",
                              required=False, id="env.sibling")]
        return [check_cls("sibling:valence", True,
                          f"healthy ({len(probes)} probes{age})",
                          required=False, id="env.sibling")]
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as e:
        return [check_cls("sibling:valence", False,
                          f"fleet-status.json unreadable: "
                          f"{type(e).__name__}: {e}",
                          required=False, id="env.sibling")]
