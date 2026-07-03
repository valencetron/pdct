"""Index of distillation files across the vault.

Walks one or more distillation roots (daemon-compaction `distillations/` and
DCT batch `dct-distillations/`), parses YAML frontmatter, and exposes a
{id -> DistillationRef} map. ID is the filename stem.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dct import config as _cfg

_DEFAULT_ROOTS = _cfg.vault_roots()
_FM_DELIM = "---"
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class DistillationRef:
    id: str
    path: Path
    date: str
    title: str
    concepts: list[str] = field(default_factory=list)
    gist: str = ""


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    if not raw.startswith(_FM_DELIM + "\n"):
        return {}, raw
    rest = raw.split("\n", 1)[1]
    end = rest.find("\n" + _FM_DELIM)
    if end < 0:
        return {}, raw
    fm_text = rest[:end]
    body = rest[end + len("\n" + _FM_DELIM):].lstrip("\r\n")
    try:
        parsed = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        # Frontmatter writers sometimes emit unquoted values containing
        # colons (e.g. `gist: bug: ...`), which breaks the whole YAML doc.
        # Fall back to line-level extraction so one bad value doesn't
        # blank every field (concepts especially — they gate retrieval).
        parsed = _parse_fm_lines(fm_text)
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed, body


_FM_LINE_RE = re.compile(r"^([A-Za-z_][\w-]*):\s*(.*)$")


def _parse_fm_lines(fm_text: str) -> dict:
    """Lossy per-line frontmatter parse used when full-document YAML fails.

    Each `key: value` line is parsed independently: try YAML on the value;
    on failure keep the raw string. Inline lists like `[a, b]` parse fine
    via YAML. Multi-line values are not supported (treated as flat lines)."""
    out: dict = {}
    for line in fm_text.splitlines():
        m = _FM_LINE_RE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        try:
            out[key] = yaml.safe_load(val) if val else ""
        except yaml.YAMLError:
            out[key] = val
    return out


def _id_from_path(path: Path) -> str:
    return path.stem


def _date_from(fm: dict, path: Path) -> str:
    d = fm.get("date") or fm.get("compacted_at") or fm.get("distilled_at") or ""
    if isinstance(d, str) and _DATE_RE.match(d):
        return d[:10]
    m = _DATE_RE.search(path.stem)
    return m.group(1) if m else ""


def _coerce_str_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if x]
    return []


def _ref_from_file(path: Path) -> DistillationRef:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return DistillationRef(id=_id_from_path(path), path=path, date="", title=path.stem)
    fm, _ = _split_frontmatter(raw)
    return DistillationRef(
        id=_id_from_path(path),
        path=path,
        date=_date_from(fm, path),
        title=str(fm.get("title") or path.stem),
        concepts=_coerce_str_list(fm.get("concepts")),
        gist=str(fm.get("gist") or ""),
    )


# ---------------------------------------------------------------------------
# Index cache (perf). build_index() re-walks the vault and re-parses YAML
# frontmatter for every distillation on every call — ~6s per call, and it was
# invoked once per query_memory(). That made a 100-question eval run ~18 min,
# blowing past pdct_ledger's 900s subprocess timeout → benchmark_status=run_error.
#
# We memoize the result keyed by (resolved roots, include_ineligible, resolved
# exclude_roots, PDCT_DISABLE_ELIGIBILITY, vault-mtime-signature). The mtime
# signature is max(st_mtime) over all *.md under the roots, so adding/editing a
# distillation invalidates the cache automatically — same invalidation contract
# the graph cache uses in service._load_or_build_graph. The audit path
# (reason_counts is not None) always does a fresh uncached walk so its counts
# reflect the current corpus exactly.
_INDEX_CACHE: dict[tuple, dict[str, "DistillationRef"]] = {}


def _vault_mtime_signature(roots: list[Path]) -> tuple[int, float]:
    """(file_count, max_mtime) — count catches deletions that max-mtime misses."""
    sig = 0.0
    count = 0
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("*.md"):
                try:
                    m = p.stat().st_mtime
                    count += 1
                    if m > sig:
                        sig = m
                except OSError:
                    pass
        except OSError:
            pass
    return (count, sig)


def build_index(
    roots: list[Path] | None = None,
    *,
    include_ineligible: bool = False,
    reason_counts: dict[str, int] | None = None,
    exclude_roots: list[Path] | None = None,
) -> dict[str, DistillationRef]:
    """Build the {id -> DistillationRef} index.

    By default, low-value distillations (raw transcript dumps, no-concept, thin,
    pruned-recap, bare-id-title) are filtered out via the shared eligibility gate
    so that LIVE retrieval and the eval harness see the SAME eligible corpus.

    Results are memoized keyed by the resolved roots + flags + a vault mtime
    signature, so repeated calls (e.g. one per query_memory()) are near-free
    until a distillation file changes. The audit path (reason_counts is not
    None) bypasses the cache and always does a fresh walk.

    Args:
        roots: distillation roots to walk (defaults to _DEFAULT_ROOTS).
        include_ineligible: escape hatch — keep every file regardless of the gate.
        reason_counts: optional dict that, if provided, is populated with
            {exclusion_reason: count} for audit/observability.
        exclude_roots: paths whose subtree is excluded even if reachable from
            `roots`. Used for per-speaker scoping (e.g. exclude vault/distillations/ayan
            when querying for Alex).
    """
    import os

    from dct.retrieval.eligibility import is_eligible  # local: avoid import cycle

    resolved_roots = roots if roots is not None else _DEFAULT_ROOTS

    # Cache lookup — skip entirely on the audit path (needs a fresh count walk).
    cache_key: tuple | None = None
    if reason_counts is None:
        disable_elig = os.environ.get("PDCT_DISABLE_ELIGIBILITY") == "1"
        excl_sig = tuple(sorted(str(Path(e).resolve()) for e in (exclude_roots or [])))
        roots_sig = tuple(str(Path(r).resolve()) for r in resolved_roots)
        mtime_sig = _vault_mtime_signature(resolved_roots)
        cache_key = (
            roots_sig, include_ineligible, disable_elig, excl_sig, mtime_sig,
        )
        cached = _INDEX_CACHE.get(cache_key)
        if cached is not None:
            return dict(cached)  # shallow copy — caller mutation must not poison cache

    result = _build_index_uncached(
        roots=roots,
        include_ineligible=include_ineligible,
        reason_counts=reason_counts,
        exclude_roots=exclude_roots,
        is_eligible=is_eligible,
    )
    if cache_key is not None:
        _INDEX_CACHE[cache_key] = dict(result)
    return result


def _build_index_uncached(
    roots: list[Path] | None = None,
    *,
    include_ineligible: bool = False,
    reason_counts: dict[str, int] | None = None,
    exclude_roots: list[Path] | None = None,
    is_eligible=None,
) -> dict[str, DistillationRef]:
    import os

    if is_eligible is None:
        from dct.retrieval.eligibility import is_eligible

    # Ops escape hatch: PDCT_DISABLE_ELIGIBILITY=1 forces the unfiltered corpus
    # (used to measure filter-off baselines without code changes).
    if os.environ.get("PDCT_DISABLE_ELIGIBILITY") == "1":
        include_ineligible = True

    roots = roots if roots is not None else _DEFAULT_ROOTS
    excluded_resolved = []
    if exclude_roots:
        for er in exclude_roots:
            try:
                if er.exists():
                    excluded_resolved.append(er.resolve())
            except OSError:
                pass
    idx: dict[str, DistillationRef] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.md"):
            if not p.is_file():
                continue
            if excluded_resolved:
                try:
                    p_res = p.resolve()
                    if any(p_res.is_relative_to(er) for er in excluded_resolved):
                        continue
                except OSError:
                    pass
            ref = _ref_from_file(p)
            if not include_ineligible:
                try:
                    raw = p.read_text(encoding="utf-8", errors="replace")
                    _, body = _split_frontmatter(raw)
                except OSError:
                    body = ""
                ok, reason = is_eligible(ref, body)
                if reason_counts is not None and reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if not ok:
                    continue
            # On id collision, prefer most-recently-modified file.
            prev = idx.get(ref.id)
            if prev is None or p.stat().st_mtime > prev.path.stat().st_mtime:
                idx[ref.id] = ref
    return idx


def find_by_id(id: str, index: dict[str, DistillationRef] | None = None) -> DistillationRef | None:
    idx = index if index is not None else build_index()
    return idx.get(id)
