"""Index of distillation files across the vault.

Walks one or more distillation roots (daemon-compaction `distillations/` and
DCT batch `dct-distillations/`), parses YAML frontmatter, and exposes a
{id -> DistillationRef} map. ID is the filename stem.
"""
from __future__ import annotations

import re
import time as _time
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
    except (yaml.YAMLError, ValueError, TypeError):
        # YAMLError: frontmatter writers sometimes emit unquoted values
        # containing colons (e.g. `gist: bug: ...`), breaking the whole doc.
        # ValueError/TypeError: PyYAML's implicit timestamp resolver raises
        # (not YAMLError) on an impossible date — hand-written notes really
        # do contain these. Two exist in the 1998-2004 Experience Journal
        # (1999-02-30.md, 1998-00-00.md) from the original digitization, and
        # before this catch a single one aborted the ENTIRE index build.
        # Fall back to line-level extraction so one bad value doesn't blank
        # every field (concepts especially — they gate retrieval).
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
        except (yaml.YAMLError, ValueError, TypeError):
            # ValueError/TypeError: PyYAML's implicit timestamp resolver raises
            # these (not YAMLError) on an impossible date like 1999-02-30. This
            # is the LAST-RESORT parser, so swallowing here is what keeps one
            # bad line from aborting the whole index build.
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


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")
_HEADING_RE = re.compile(r"^#{1,3}\s+(.{3,60})$", re.MULTILINE)


def _ref_from_note_file(path: Path) -> DistillationRef:
    """Read `path` and build a note ref. Convenience wrapper for callers that
    don't already have the parsed frontmatter (tests, ad-hoc tooling); the index
    walk uses _ref_from_note_fm to avoid a second read."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return DistillationRef(id=_id_from_path(path), path=path, date="", title=path.stem)
    fm, body = _split_frontmatter(raw)
    return _ref_from_note_fm(path, fm, body)


def _ref_from_note_fm(path: Path, fm: dict, body: str) -> DistillationRef:
    """Build a ref for a hand-written note from already-parsed frontmatter.

    Notes rarely carry `concepts:` — they carry `tags:`, wikilinks, and headings.
    The cascade jogs off concepts, so a note with none is unreachable; derive
    them here rather than dropping the note. Order is most- to least-curated:
    explicit concepts, then tags, then wikilink targets, then headings, then the
    path/title words. Deduped, lowercased, capped so one long note cannot
    dominate the concept graph.
    """
    derived: list[str] = list(_coerce_str_list(fm.get("concepts")))
    if not derived:
        derived += _coerce_str_list(fm.get("tags"))
    for m in _WIKILINK_RE.findall(body)[:8]:
        derived.append(m.strip().split("/")[-1])
    for h in _HEADING_RE.findall(body)[:6]:
        derived.append(h.strip().rstrip(":"))
    # Folder + filename carry real signal in this vault ("people/Ayan.md",
    # "Alex's Life/Cultivation Journal/2026-07-17.md").
    derived.append(path.stem)
    if path.parent.name:
        derived.append(path.parent.name)

    seen: set[str] = set()
    concepts: list[str] = []
    for c in derived:
        c = re.sub(r"\s+", " ", str(c)).strip().lower()
        # Drop pure dates/ids — they are not concepts to jog off.
        if not c or len(c) < 2 or re.fullmatch(r"[\d\-_.]+", c):
            continue
        if c not in seen:
            seen.add(c)
            concepts.append(c)
        if len(concepts) >= 24:
            break

    return DistillationRef(
        id=_id_from_path(path),
        path=path,
        date=_date_from(fm, path),
        title=str(fm.get("title") or path.stem),
        concepts=concepts,
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
    include_notes: bool | None = None,
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
        include_notes: also index hand-written vault notes (people/, projects/,
            journals, top-level wiki) under their own signal-based gate. None =>
            read PDCT_INCLUDE_NOTES, default ON. Set 0 to reproduce the
            distillations-only corpus that pre-2026-07-27 benchmarks measured.
    """
    import os

    from dct.retrieval.eligibility import is_eligible  # local: avoid import cycle

    resolved_roots = roots if roots is not None else _DEFAULT_ROOTS
    if include_notes is None:
        include_notes = os.environ.get("PDCT_INCLUDE_NOTES", "1") != "0"

    # Cache lookup — skip entirely on the audit path (needs a fresh count walk).
    cache_key: tuple | None = None
    if reason_counts is None:
        disable_elig = os.environ.get("PDCT_DISABLE_ELIGIBILITY") == "1"
        excl_sig = tuple(sorted(str(Path(e).resolve()) for e in (exclude_roots or [])))
        roots_sig = tuple(str(Path(r).resolve()) for r in resolved_roots)
        mtime_sig = _vault_mtime_signature(resolved_roots)
        note_sig = _notes_mtime_signature() if include_notes else ()
        cache_key = (
            roots_sig, include_ineligible, disable_elig, excl_sig, mtime_sig,
            include_notes, note_sig,
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
    if include_notes:
        result.update(_build_notes_index(
            include_ineligible=include_ineligible,
            reason_counts=reason_counts,
            exclude_roots=exclude_roots,
        ))
    if cache_key is not None:
        _INDEX_CACHE[cache_key] = dict(result)
    return result


_NOTES_SIG_CACHE = {"checked_mono": float("-inf"), "val": (0, 0.0)}
_NOTES_SIG_TTL_S = 15.0


def _notes_excluded_prefixes() -> tuple[str, ...]:
    """Exclusion prefixes as plain strings.

    Deliberately NOT Path.resolve() per file: resolve() is a syscall-heavy
    stat/readlink walk, and doing it for every one of ~1k notes on every
    build_index call cost ~30s (measured 2026-07-27 while wiring notes in).
    The vault has no symlinked subtrees, so prefix matching is equivalent here
    and ~free.
    """
    return tuple(str(e) + "/" for e in _cfg.notes_exclude_roots())


def _notes_mtime_signature() -> tuple:
    """Cheap change signal for the notes corpus (count + newest mtime).

    Mirrors _vault_mtime_signature's contract but over the note roots. Behind a
    15s TTL because this runs on EVERY build_index() call to compute the cache
    key — without the TTL the "cheap" key was itself a full stat-walk.
    """
    now_m = _time.monotonic()
    if now_m - _NOTES_SIG_CACHE["checked_mono"] < _NOTES_SIG_TTL_S:
        return _NOTES_SIG_CACHE["val"]
    newest = 0.0
    count = 0
    excl = _notes_excluded_prefixes()
    for root in _cfg.notes_roots():
        if not root.exists():
            continue
        try:
            for p in root.rglob("*.md"):
                sp = str(p)
                if sp.startswith(excl):
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                count += 1
                if st.st_mtime > newest:
                    newest = st.st_mtime
        except OSError:
            pass
    val = (count, newest)
    _NOTES_SIG_CACHE["checked_mono"] = now_m
    _NOTES_SIG_CACHE["val"] = val
    return val


def _build_notes_index(
    *,
    include_ineligible: bool = False,
    reason_counts: dict[str, int] | None = None,
    exclude_roots: list[Path] | None = None,
) -> dict[str, DistillationRef]:
    """Index hand-written vault notes as retrievable refs.

    Separate walk from the distillation index because notes need a different
    ref builder (concepts derived from tags/wikilinks/path) AND a different
    eligibility gate (signal-based, not the 400-char distillation floor). Before
    2026-07-27 notes were not in the corpus at all, so a fleshed-out
    people/Ayan.md could never be retrieved no matter how good it was.
    """
    from dct.retrieval.eligibility import is_eligible_note

    excluded = list(_notes_excluded_prefixes())
    excluded += [str(e) + "/" for e in (exclude_roots or [])]
    excl = tuple(excluded)

    idx: dict[str, DistillationRef] = {}
    mtimes: dict[str, float] = {}
    for root in _cfg.notes_roots():
        if not root.exists():
            continue
        for p in root.rglob("*.md"):
            sp = str(p)
            if sp.startswith(excl):
                continue
            # Obsidian/vcs plumbing: any dotted path segment.
            if any(part.startswith(".") for part in p.parts):
                continue
            # ONE read + ONE frontmatter parse per note. The first cut read and
            # YAML-parsed every file twice (once for the ref, once for the
            # eligibility body), which doubled a ~30s walk for no reason.
            try:
                st = p.stat()
                raw = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            fm, body = _split_frontmatter(raw)
            ref = _ref_from_note_fm(p, fm, body)
            if not include_ineligible:
                ok, reason = is_eligible_note(ref, body)
                if reason_counts is not None and reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if not ok:
                    continue
            prev_mtime = mtimes.get(ref.id)
            if prev_mtime is None or st.st_mtime > prev_mtime:
                idx[ref.id] = ref
                mtimes[ref.id] = st.st_mtime
    return idx


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
