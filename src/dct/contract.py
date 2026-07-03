"""Distillation contract — the write-path half of PDCT.

Retrieval quality is a function of write-path discipline (measured:
April-era distillations with empty gists score far worse than June-era
contract-meeting ones). This module defines the minimum a distillation
must contain for the retrieval stack to do its job, and lints notes at
write time so garbage never accumulates silently again.

Contract v1:
  - title: present, >= 8 chars, not a bare UUID/session-id
  - gist:  present, >= 80 chars (one real sentence of substance)
  - concepts: >= 3
  - date: parseable session/distilled timestamp

check_note() returns a list of violations (empty = passes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

GIST_MIN_CHARS = 80
CONCEPTS_MIN = 3
TITLE_MIN_CHARS = 8

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_SESSION_ID_RE = re.compile(r"^\d{6,}[_-]\d+$")


@dataclass
class ContractReport:
    violations: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations


def check_fields(
    *,
    title: str | None,
    gist: str | None,
    concepts: list[str] | None,
    date: str | None,
) -> ContractReport:
    v: list[str] = []
    t = (title or "").strip()
    if len(t) < TITLE_MIN_CHARS:
        v.append(f"title too short (<{TITLE_MIN_CHARS} chars)")
    elif _UUID_RE.match(t) or _SESSION_ID_RE.match(t):
        v.append("title is a bare UUID/session-id, not descriptive")

    g = (gist or "").strip()
    if not g:
        v.append("gist missing")
    elif len(g) < GIST_MIN_CHARS:
        v.append(f"gist too short ({len(g)} < {GIST_MIN_CHARS} chars)")

    if len(concepts or []) < CONCEPTS_MIN:
        v.append(f"fewer than {CONCEPTS_MIN} concepts")

    if not str(date or "").strip():
        v.append("no date")

    return ContractReport(violations=v)
