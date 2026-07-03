"""Memory retrieval primitives — query_memory + read_memory.

Surface-agnostic: same interface for Telegram, Voice, Claude Code.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from dataclasses import asdict
import json
import sys

from dct.retrieval import telemetry
from dct.retrieval.cascade import cascade
from dct.retrieval.distill_index import DistillationRef, build_index
from dct.retrieval.related import RelatedRef, related_distillations
from dct.retrieval.types import ConceptHit, RetrievalConfig

_TOP_K = 5
_RERANK_POOL = 25


@dataclass(frozen=True)
class DistillationRow:
    id: str
    path: str
    date: str
    title: str
    concepts: list[str] = field(default_factory=list)
    gist: str = ""
    score: float = 0.0
    source: str = "graph"  # "graph" or "fallback"


def _cascade_for_seed(seed: str) -> list[ConceptHit]:
    """Run DCT cascade for a single seed phrase. Returns ConceptHits or []."""
    from dct.retrieval.service import _load_or_build_graph, _derive_seeds, build_config
    graph = _load_or_build_graph()
    seeds = _derive_seeds(seed, graph)
    if not seeds:
        return []
    return cascade(seed_concepts=seeds, graph=graph, heat={},
                   config=build_config(), current_context=set())


from functools import lru_cache


@lru_cache(maxsize=262144)
def _concept_match_strength(cascade_concept: str, ref_concept: str) -> float:
    """Score how well a cascade concept matches a frontmatter concept. 0.0–1.0.

    Handles vocabulary mismatch between event-graph concepts (old heuristic slugs
    like 'ayan') and frontmatter concepts (new LLM-extracted like 'ayan-iep-meeting').

    Returns:
      1.0  — exact match
      0.7  — cascade concept is a full prefix/suffix of ref concept
      0.3  — single token overlap (e.g. 'ayan' in 'ayan-name-correction')
      0.0  — no match
    """
    if cascade_concept == ref_concept:
        return 1.0

    parts_ref = ref_concept.split("-")
    parts_cas = cascade_concept.split("-")

    # Full prefix/suffix match (e.g. 'ayan-iep' matches 'ayan-iep-meeting')
    if ref_concept.startswith(cascade_concept + "-") or ref_concept.endswith("-" + cascade_concept):
        return 0.7
    if cascade_concept.startswith(ref_concept + "-") or cascade_concept.endswith("-" + ref_concept):
        return 0.7

    # Token overlap — count how many cascade tokens appear in ref
    overlap = sum(1 for p in parts_cas if len(p) >= 3 and p in parts_ref)
    if overlap == 0:
        # Reverse: ref tokens in cascade
        overlap = sum(1 for p in parts_ref if len(p) >= 3 and p in parts_cas)
    if overlap == 0:
        return 0.0

    # Scale by how much of the shorter concept is covered
    shorter = min(len(parts_cas), len(parts_ref))
    coverage = overlap / max(shorter, 1)
    # Single-token overlap on a multi-word concept = weak signal (0.3)
    # Full coverage = strong signal (0.7)
    return 0.3 + (0.4 * coverage)


def _recency_boost(date_str: str) -> float:
    """Score boost for recent distillations. Returns 0.0–0.3.

    Today = 0.3, yesterday = 0.25, last week = 0.15, older = 0.0–0.1.
    """
    if not date_str:
        return 0.0
    try:
        import datetime
        d = datetime.date.fromisoformat(date_str[:10])
        today = datetime.date.today()
        age_days = (today - d).days
        if age_days <= 0:
            return 0.30
        elif age_days <= 1:
            return 0.25
        elif age_days <= 3:
            return 0.20
        elif age_days <= 7:
            return 0.15
        elif age_days <= 14:
            return 0.10
        elif age_days <= 30:
            return 0.05
        return 0.0
    except (ValueError, TypeError):
        return 0.0


_STOPWORDS = frozenset(
    "the a an and or of to in on for with was were is are be this that what "
    "which who how when where why did do does done about from as at by it its "
    "his her their our your my we you i he she they not no".split()
)


def _text_match_boost(query_text: str, ref: DistillationRef) -> float:
    """0.0–0.25 boost for query keywords appearing in title+gist text."""
    if not query_text:
        return 0.0
    tokens = [
        t for t in re.findall(r"[a-z0-9]{3,}", query_text.lower())
        if t not in _STOPWORDS
    ]
    if not tokens:
        return 0.0
    hay = (ref.title + " " + ref.gist + " " + " ".join(ref.concepts)).lower()
    hits = sum(1 for t in tokens if t in hay)
    return 0.25 * (hits / len(tokens))


_MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
_MONTHS.update({m[:3].lower(): v for m, v in list(_MONTHS.items())})


def _query_dates(query_text: str) -> list:
    """Extract explicit dates referenced in the query. Returns date objects."""
    import datetime
    out = []
    today = datetime.date.today()
    # ISO: 2026-05-26
    for m in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", query_text):
        try:
            out.append(datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    # "May 26" / "May 26, 2026"
    for m in re.finditer(
        r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d{2}))?\b",
        query_text,
    ):
        mon = _MONTHS.get(m.group(1).lower())
        if not mon:
            continue
        year = int(m.group(3)) if m.group(3) else today.year
        try:
            d = datetime.date(year, mon, int(m.group(2)))
        except ValueError:
            continue
        if not m.group(3) and d > today:
            d = d.replace(year=year - 1)
        out.append(d)
    return out


def _temporal_boost(query_dates: list, ref_date: str) -> float:
    """0.0-0.3 boost when the doc's date is near a date the query mentions.

    Events get discussed on or shortly AFTER the referenced date, so the
    window is asymmetric: same day=0.3, within 7 days after=0.2, within
    14 days either side=0.1.
    """
    if not query_dates or not ref_date:
        return 0.0
    import datetime
    try:
        d = datetime.date.fromisoformat(ref_date[:10])
    except (ValueError, TypeError):
        return 0.0
    best = 0.0
    for qd in query_dates:
        delta = (d - qd).days
        if delta == 0:
            best = max(best, 0.30)
        elif 0 < delta <= 7:
            best = max(best, 0.20)
        elif -14 <= delta <= 14:
            best = max(best, 0.10)
    return best


def _aggregate(
    hits_per_seed: list[list[ConceptHit]],
    index: dict[str, DistillationRef],
    query_text: str = "",
) -> list[DistillationRow]:
    """Map concept hits → distillation rows with multi-signal scoring.

    Scoring formula per distillation:
      overlap_score = (matched_concepts / total_ref_concepts) × best_cascade_score
      recency_boost = 0.0–0.3 based on distillation date
      final_score   = overlap_score + recency_boost

    Uses fuzzy concept matching so event-graph concepts (e.g. 'ayan')
    can match frontmatter concepts (e.g. 'ayan-iep-meeting').
    """
    # Best score per concept across all seeds.
    concept_score: dict[str, float] = {}
    for hits in hits_per_seed:
        for h in hits:
            cur = concept_score.get(h.concept, 0.0)
            if h.score > cur:
                concept_score[h.concept] = h.score

    if not concept_score:
        return []

    # Prune near-zero cascade concepts: a typical cascade returns thousands of
    # concepts but only a few dozen carry signal. Dropping score<=0.005 cuts
    # pairwise fuzzy matching from ~43M calls to ~400K (>100x speedup) with
    # no ranking change (their contribution was numerically negligible).
    concept_score = {c: s for c, s in concept_score.items() if s > 0.005}
    if not concept_score:
        return []

    # Hard cap: keep only the strongest cascade concepts. After the
    # word-boundary seed fix (2026-06-12), seeds are higher quality and
    # cascades grew 2-6x (up to ~6000 concepts), blowing up the pairwise
    # fuzzy-match stage. Measured: top-300 by score is ranking-equivalent
    # (tail scores are ≤1% of head) and keeps query p50 in budget.
    _CASCADE_CONCEPT_CAP = 300
    if len(concept_score) > _CASCADE_CONCEPT_CAP:
        top = sorted(concept_score.items(), key=lambda kv: -kv[1])[:_CASCADE_CONCEPT_CAP]
        concept_score = dict(top)

    # IDF weighting: a cascade concept that matches half the vault carries
    # almost no signal; a rare concept is highly discriminative. Weight each
    # cascade concept by log-scaled inverse document frequency over the
    # distillation index (token-overlap counted as a "hit").
    import math
    n_docs = max(len(index), 1)
    df: dict[str, int] = {}
    for cc in concept_score:
        count = 0
        for ref in index.values():
            for rc in ref.concepts:
                if _concept_match_strength(cc, rc) > 0:
                    count += 1
                    break
        df[cc] = count
    idf: dict[str, float] = {
        cc: math.log((n_docs + 1) / (df[cc] + 1)) / math.log(n_docs + 1)
        for cc in concept_score
    }
    # Fold IDF into the cascade score so generic concepts stop dominating.
    concept_score = {cc: s * max(idf[cc], 0.05) for cc, s in concept_score.items()}

    cascade_concepts = list(concept_score.keys())
    query_dates = _query_dates(query_text)

    # Dense semantic channel — covers vocabulary mismatch the concept graph
    # can't (returns {} if embeddings unavailable; ranking degrades gracefully).
    try:
        from dct.retrieval.embed_index import semantic_scores
        sem = semantic_scores(query_text, index) if query_text else {}
    except Exception:
        sem = {}

    scored: list[tuple[float, DistillationRef]] = []
    for ref in index.values():
        if not ref.concepts:
            continue

        # Score each ref concept against cascade hits, weighting by match strength
        weighted_scores: list[float] = []
        for rc in ref.concepts:
            best_match = 0.0
            for cc in cascade_concepts:
                strength = _concept_match_strength(cc, rc)
                if strength > 0:
                    # Combined: cascade score × match strength
                    combined = concept_score[cc] * strength
                    if combined > best_match:
                        best_match = combined
            if best_match > 0:
                weighted_scores.append(best_match)

        if weighted_scores:
            # Multi-concept overlap: reward distillations matching many concepts
            overlap_ratio = len(weighted_scores) / max(len(ref.concepts), 1)
            avg_match = sum(weighted_scores) / len(weighted_scores)
            best_match = max(weighted_scores)
            # Weighted: 40% best match, 30% avg match quality, 30% overlap breadth
            base_score = (0.4 * best_match) + (0.3 * avg_match) + (0.3 * overlap_ratio)
        else:
            # No concept-graph match. Don't drop the doc yet — strong text or
            # temporal evidence below can still surface it (the cascade
            # vocabulary often misses LLM-extracted frontmatter concepts).
            base_score = 0.0

        # Recency boost
        # Recency: halved for explicit memory queries — deep recall should
        # not bury month-old sources under today's chatter.
        recency = 0.5 * _recency_boost(ref.date)
        text_boost = _text_match_boost(query_text, ref)
        temporal = _temporal_boost(query_dates, ref.date)

        # Semantic boost: bge cosines run ~0.45 baseline / ~0.75 strong on
        # this corpus. Map 0.55..0.80 -> 0.0..0.45 so only genuinely similar
        # docs get lift, with enough range to outrank concept-only matches.
        sem_raw = sem.get(ref.id, 0.0)
        sem_boost = max(0.0, min(0.45, (sem_raw - 0.55) * 1.8)) if sem_raw else 0.0

        if base_score == 0.0:
            # Concept-missed doc: needs real text or semantic evidence to
            # enter ranking; recency alone can't float it (temporal is fine —
            # it is query-anchored, not "newer is better").
            if text_boost < 0.10 and sem_boost < 0.10:
                continue
            final_score = min(text_boost + temporal + sem_boost, 1.0)
        else:
            final_score = min(
                base_score + recency + text_boost + temporal + sem_boost, 1.0)

        scored.append((final_score, ref))

    scored.sort(key=lambda sr: (-sr[0], sr[1].date or "", sr[1].id))

    # Cross-encoder rerank over a UNION candidate pool. The prior channel
    # alone buries answer docs under topical neighbors (measured: canary
    # misses had semantic rank 1-340 but never made the prior top-25).
    # Union three recall channels so the cross-encoder gets to see every
    # doc any single channel believes in:
    #   - top _RERANK_POOL by blended prior
    #   - top 20 by dense semantic cosine
    #   - top 10 by keyword text match
    pool = scored[:_RERANK_POOL]
    if query_text:
        pool_ids = {r.id for _, r in pool}
        extras: list[tuple[float, DistillationRef]] = []
        if sem:
            for cid, _sc in sorted(sem.items(), key=lambda kv: -kv[1])[:20]:
                ref = index.get(cid)
                if ref is not None and cid not in pool_ids:
                    pool_ids.add(cid)
                    extras.append((0.0, ref))
        tm = sorted(
            ((_text_match_boost(query_text, ref), ref)
             for ref in index.values() if ref.id not in pool_ids),
            key=lambda t: -t[0])[:10]
        for tb, ref in tm:
            if tb >= 0.10:
                pool_ids.add(ref.id)
                extras.append((0.0, ref))
        # BM25 full-text channel: catches "many medium-rare words"
        # questions whose answer doc has no distinctive identifier and
        # weak title/gist match. Measured 2026-06-11: BM25 ranked both
        # remaining canary misses #1 while no other channel pooled them.
        try:
            from dct.retrieval.bm25_index import bm25_top
            for cid in bm25_top(query_text, index, k=10):
                ref = index.get(cid)
                if ref is not None and cid not in pool_ids:
                    pool_ids.add(cid)
                    extras.append((0.0, ref))
        except Exception:
            pass
        # Rare-token body grep: distinctive query tokens (CamelCase,
        # snake_case, long technicals) often live only in the body — not
        # in title/gist/concepts/embedding head. Grep them so docs whose
        # body contains the literal answer term reach the cross-encoder.
        for ref in _rare_token_hits(query_text, index, exclude=pool_ids):
            pool_ids.add(ref.id)
            extras.append((0.0, ref))
        pool = pool + extras
    if query_text and len(pool) > _TOP_K:
        try:
            from dct.retrieval.rerank import rerank
            by_id = {r.id: (sc, r) for sc, r in pool}
            cands = [
                (r.id, _rerank_text(r, query_text), sc)
                for sc, r in pool
            ]
            reranked = rerank(query_text, cands)
            pool = [(blended, by_id[cid][1]) for cid, blended in reranked
                    if cid in by_id]
        except Exception:
            pass

    return [
        DistillationRow(
            id=r.id, path=str(r.path), date=r.date, title=r.title,
            concepts=r.concepts, gist=r.gist, score=round(s, 3), source="graph",
        )
        for s, r in pool[:_TOP_K]
    ]


def _rerank_text(ref: DistillationRef, query_text: str = "") -> str:
    """Searchable text for cross-encoder scoring: title + gist + concepts +
    body head + query-anchored body snippets. The body head alone misses
    answers that live deep in the doc (e.g. a rejection rationale at char
    3000); snippets around rare query-token occurrences put the actual
    answer text in front of the cross-encoder."""
    parts = [
        ref.title, ref.gist,
        " ".join(c.replace("-", " ") for c in ref.concepts),
    ]
    try:
        raw = ref.path.read_text(encoding="utf-8", errors="ignore")
        if raw.startswith("---"):
            end = raw.find("---", 3)
            if end != -1:
                raw = raw[end + 3:]
        body = raw.strip()
        # Query-anchored snippets go BEFORE the body head: rerank() caps
        # input at 1500 chars, and title+gist+concepts+900-char head
        # already total ~1200 — snippets appended after were silently
        # truncated away (measured 2026-06-11: answer text at char 2765
        # never reached the CE; its score was 0.0001 vs rival 0.70).
        snippets: list[str] = []
        if query_text and len(body) > 900:
            toks = [
                t for t in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{4,}", query_text)
                if t.lower() not in _STOPWORDS
            ]
            low = body.lower()
            seen_spans: list[tuple[int, int]] = []
            for t in toks:
                i = low.find(t.lower(), 900)
                if i == -1:
                    continue
                a, b = max(0, i - 120), min(len(body), i + 180)
                if any(a < se and b > ss for ss, se in seen_spans):
                    continue
                seen_spans.append((a, b))
                if len(seen_spans) >= 3:
                    break
            for a, b in seen_spans:
                snippets.append(body[a:b])
        parts.extend(snippets)
        head_budget = 900 - min(600, sum(len(x) for x in snippets))
        parts.append(body[:head_budget])
    except OSError:
        pass
    return " ".join(p for p in parts if p)


def _rare_token_hits(
    query_text: str,
    index: dict[str, DistillationRef],
    exclude: set[str],
    limit: int = 8,
) -> list[DistillationRef]:
    """Grep distinctive query tokens against distillation bodies.

    Distinctive = CamelCase, snake_case/dotted identifiers, or rare long
    words. Returns up to `limit` refs whose body contains any of them.
    """
    if not query_text:
        return []
    # Identifier-shaped tokens only (CamelCase / snake_case / dotted) —
    # plain long English words ("alternative") match half the vault and
    # just flood the pool with recent chatter.
    tokens: list[str] = []
    for t in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{5,}", query_text):
        if re.search(r"[a-z][A-Z]", t) or "_" in t or "." in t:
            tokens.append(t)
    if not tokens:
        return []
    tokens = list(dict.fromkeys(tokens))[:6]
    rg = shutil.which("rg")
    if rg is None:
        return []
    paths = [str(r.path) for r in index.values()
             if r.id not in exclude and r.path.exists()]
    if not paths:
        return []
    by_path = {str(r.path): r for r in index.values()}
    # Per-token grep; rank docs by how many distinct rare tokens they hit,
    # preferring tokens that match few docs (rarity = signal).
    doc_hits: dict[str, float] = {}
    for tok in tokens:
        try:
            proc = subprocess.run(
                [rg, "-l", "-i", "-F", "--", tok, *paths],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        matched = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        if not matched or len(matched) > 40:
            continue  # token too common to discriminate
        w = 1.0 / len(matched)
        for mp in matched:
            doc_hits[mp] = doc_hits.get(mp, 0.0) + w
    if not doc_hits:
        return []
    ranked = sorted(doc_hits.items(), key=lambda kv: -kv[1])[:limit]
    return [by_path[p] for p, _ in ranked if p in by_path]


def _ripgrep_fallback(
    seed: str,
    index: dict[str, DistillationRef],
) -> list[DistillationRow]:
    """When cascade returns nothing, brute-force grep distillation files."""
    if not seed.strip():
        return []
    rg = shutil.which("rg")
    if rg is None:
        return []
    paths = [str(r.path) for r in index.values() if r.path.exists()]
    if not paths:
        return []
    try:
        proc = subprocess.run(
            [rg, "-l", "-i", "--", seed, *paths],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    matched_paths = {ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()}
    if not matched_paths:
        return []
    rows: list[DistillationRow] = []
    for ref in index.values():
        if str(ref.path) in matched_paths:
            rows.append(DistillationRow(
                id=ref.id, path=str(ref.path), date=ref.date, title=ref.title,
                concepts=ref.concepts, gist=ref.gist, score=0.5, source="fallback",
            ))
    rows.sort(key=lambda r: (r.date or "", r.id), reverse=True)
    return rows[:_TOP_K]


def query_memory(
    seed: str | list[str],
    *,
    _surface: str = "unknown",
    roots: list[Path] | None = None,
    exclude_roots: list[Path] | None = None,
) -> list[DistillationRow]:
    """Search the vault of distillations for content matching `seed`.

    Args:
      seed: free text or list of free-text strings.
      _surface: caller identifier for telemetry only ('voice', 'telegram', 'cc').
      roots: restrict the corpus to these distillation roots (defaults to PDCT's
        full _DEFAULT_ROOTS). Used for per-speaker scoping by callers that
        want a small corpus to dominate ranking.
      exclude_roots: subtract these subtrees from the corpus even if reachable
        from `roots`. Together with `roots` this enables pre-top-k speaker
        scoping (the bridge passes exclude_roots=[ayan/] for Alex queries).
    """
    t0 = time.monotonic()
    seeds: list[str] = [seed] if isinstance(seed, str) else list(seed or [])
    seeds = [s for s in (s.strip() for s in seeds) if s]
    seed_for_log = " | ".join(seeds)

    index = build_index(roots=roots, exclude_roots=exclude_roots)
    hits_per_seed = [_cascade_for_seed(s) for s in seeds]
    rows = _aggregate(hits_per_seed, index, query_text=seed_for_log)
    used_fallback = False

    if not rows and seeds:
        used_fallback = True
        # Fallback only on first seed (rg accepts one pattern); union semantics
        # via repeated calls if we ever need it.
        rows = _ripgrep_fallback(seeds[0], index)

    latency_ms = int((time.monotonic() - t0) * 1000)
    telemetry.log_call(
        surface=_surface, fn="query_memory", seed=seed_for_log,
        result_count=len(rows), used_fallback=used_fallback,
        latency_ms=latency_ms,
    )
    return rows


@dataclass(frozen=True)
class MemoryRead:
    id: str
    date: str
    title: str
    related_distillations: list[RelatedRef]
    body: str


def read_memory(
    id: str,
    *,
    _surface: str = "unknown",
) -> MemoryRead:
    """Return full distillation markdown + header with top-3 related distillations.

    Raises KeyError if id is not in the vault index.
    """
    t0 = time.monotonic()
    # Read-by-id is an INSPECTION path, not a retrieval-candidate path: an
    # otherwise-ineligible distillation must still be readable by id (the
    # eligibility gate governs what surfaces, not what exists). Codex P1.
    index = build_index(include_ineligible=True)
    ref = index.get(id)
    if ref is None:
        telemetry.log_call(
            surface=_surface, fn="read_memory", seed=id,
            result_count=0, used_fallback=False,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        raise KeyError(f"distillation not found: {id}")

    try:
        body = ref.path.read_text(encoding="utf-8", errors="replace")
        try:
            from ..leanctx_obs import emit as _lc_emit
            _lc_emit(
                "pdct_read_distillation",
                path=str(ref.path),
                bytes=len(body),
                surface=_surface,
            )
        except Exception:
            pass
    except OSError as e:
        telemetry.log_call(
            surface=_surface, fn="read_memory", seed=id,
            result_count=0, used_fallback=False,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        raise KeyError(f"distillation file unreadable: {id}: {e}") from e

    related = related_distillations(id, k=3, index=index)
    result = MemoryRead(
        id=ref.id, date=ref.date, title=ref.title,
        related_distillations=related, body=body,
    )
    telemetry.log_call(
        surface=_surface, fn="read_memory", seed=id,
        result_count=1, used_fallback=False,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    return result


# ── CLI for cross-venv callers (Telegram daemon, voice action server) ──
#
# Usage:
#   stdin (JSON): {"mode": "query", "seed": "<str|list>", "surface": "telegram"}
#   stdin (JSON): {"mode": "read",  "id":   "<str>",       "surface": "voice"}
# stdout (JSON):
#   query: {"rows": [DistillationRow as dict, ...]}
#   read:  {"id", "date", "title", "related_distillations": [...], "body"}
# on error: stderr {"error": str, "error_type": str}; exit 1.

def _row_to_dict(r) -> dict:
    return {
        "id": r.id, "path": r.path, "date": r.date, "title": r.title,
        "concepts": list(r.concepts), "gist": r.gist,
        "score": r.score, "source": r.source,
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        req = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"invalid JSON: {e}", "error_type": "JSONDecodeError"}), file=sys.stderr)
        return 1
    mode = req.get("mode", "query")
    surface = req.get("surface") or "unknown"
    try:
        if mode == "query":
            raw_roots = req.get("roots") or None
            raw_excl = req.get("exclude_roots") or None
            kw = {"_surface": surface}
            if raw_roots:
                kw["roots"] = [Path(p) for p in raw_roots]
            if raw_excl:
                kw["exclude_roots"] = [Path(p) for p in raw_excl]
            rows = query_memory(req.get("seed", ""), **kw)
            print(json.dumps({"rows": [_row_to_dict(r) for r in rows]}, ensure_ascii=False))
        elif mode == "eligible":
            # Validate a freshly-written distillation against the same gate
            # retrieval applies. Caller passes {"mode":"eligible","path":"<abs>"}
            # and gets back {"ok":bool, "reason":str}. Used by claude-mcp-bridge's
            # pdct_writeback to refuse writes that would be silently filtered out.
            from dct.retrieval.eligibility import is_eligible
            from dct.retrieval.distill_index import _ref_from_file, _split_frontmatter
            p = Path(req.get("path", ""))
            if not p.is_file():
                print(json.dumps({"ok": False, "reason": "no-such-file"}))
                return 0
            ref = _ref_from_file(p)
            raw = p.read_text(encoding="utf-8", errors="replace")
            _, body = _split_frontmatter(raw)
            ok, reason = is_eligible(ref, body)
            print(json.dumps({"ok": ok, "reason": reason}))
            return 0
        elif mode == "read":
            result = read_memory(req.get("id", ""), _surface=surface)
            print(json.dumps({
                "id": result.id, "date": result.date, "title": result.title,
                "related_distillations": [
                    {"id": r.id, "title": r.title, "score": r.score}
                    for r in result.related_distillations
                ],
                "body": result.body,
            }, ensure_ascii=False))
        else:
            print(json.dumps({"error": f"unknown mode: {mode}", "error_type": "ValueError"}), file=sys.stderr)
            return 1
    except KeyError as e:
        print(json.dumps({"error": str(e), "error_type": "KeyError"}), file=sys.stderr)
        return 2
    except Exception as e:
        print(json.dumps({"error": str(e), "error_type": type(e).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
