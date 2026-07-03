"""Benchmark v3 generator: mine distillations for ground-truth Q/A pairs.

Pipeline: stratified sample -> LLM Q/A extraction -> negative generation ->
dedup/balance -> freeze pdct-questions-v3.json.

Usage:
    python3 benchmark/build_v3.py sample        # write candidate pool
    python3 benchmark/build_v3.py extract       # call LLM per candidate
    python3 benchmark/build_v3.py negatives     # generate negative questions
    python3 benchmark/build_v3.py freeze        # dedup, balance, write v3 json
    python3 benchmark/build_v3.py all
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dct.retrieval.distill_index import build_index  # noqa: E402

WORK = ROOT / "benchmark" / ".v3-work"
WORK.mkdir(exist_ok=True)
POOL_F = WORK / "pool.json"
EXTRACTED_F = WORK / "extracted.jsonl"
NEGATIVES_F = WORK / "negatives.json"
OUT_F = ROOT / "benchmark" / "pdct-questions-v3.json"

POOL_SIZE = 150
TARGET_TOTAL = 100
QUOTAS = {
    "factual-recall": 30,
    "temporal": 20,
    "cross-thread": 15,
    "decision-recall": 20,
    "negative": 15,
}
EXTRACT_MODEL = "claude-haiku-4-5-20251001"  # sonnet 429-limited 2026-06-10; haiku confirmed working
MAX_BODY_CHARS = 6000
SEED = 20260610

EXTRACT_SYSTEM = """You extract benchmark questions for a personal-memory retrieval system.

You will be given one "distillation" (a summary of a past conversation between Alex and his AI assistant Orion). Extract 0-2 question/answer pairs that test whether a memory system can retrieve THIS document.

HARD REQUIREMENTS for every question:
1. MEMORY-ONLY ANSWERABLE: a model WITHOUT access to Alex's history must be unable to answer it. Never ask anything answerable from general knowledge.
2. SELF-CONTAINED: the question must make sense on its own ("What database did Alex choose for the judge cache in the PDCT project?" not "What did we choose?").
3. GROUND-TRUTH: expected_answer must be a specific fact (name, number, date, decision, reason) stated verbatim-or-near in the document.
4. Include a short verbatim excerpt (<=200 chars) from the document that contains the answer.

Categories (pick the best fit per question):
- factual-recall: specific name/number/path/config stated in the doc
- temporal: when something happened or what order events occurred
- cross-thread: connects this doc's project to another project/system mentioned in it
- decision-recall: why a choice was made, what alternatives were rejected

Difficulty:
- easy: answer sits verbatim in one obvious place
- medium: answer requires combining 2+ statements in the doc
- hard: answer requires inference across the doc or knowledge of how this connects to other work

If the document is too thin, vague, or purely conversational to yield a question meeting ALL requirements, return an empty list. Quality over quantity — most documents should yield 0 or 1 questions.

Respond with ONLY a JSON array (possibly empty):
[{"question": "...", "expected_answer": "...", "acceptable_keywords": ["k1","k2","k3"], "excerpt": "...", "category": "factual-recall|temporal|cross-thread|decision-recall", "difficulty": "easy|medium|hard"}]"""

NEGATIVE_SYSTEM = """You generate "negative" benchmark questions for a personal-memory retrieval system: questions about plausible-sounding facts that are NOT in Alex's memory. These test hallucination resistance — the correct answer is "I don't have that in memory."

You will be given a list of real project names, people, and systems from Alex's actual history. Generate questions that:
1. Use REAL entities from the list (so they sound plausible and trigger retrieval)
2. Ask about specific facts/events that NEVER happened (fabricated meetings, decisions, configs, dates)
3. Are specific enough that a hallucinating system would confidently invent an answer

Respond with ONLY a JSON array:
[{"question": "...", "trap_type": "fabricated-event|fabricated-decision|fabricated-config|fabricated-person-link", "real_entities_used": ["..."]}]"""


def _client():
    from dct.llm import _client_factory
    return _client_factory()


def _qid(question: str) -> str:
    return "q3_" + hashlib.sha256(question.encode()).hexdigest()[:12]


def _norm_tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {
        "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is",
        "was", "what", "which", "did", "does", "user", "assistant", "when",
        "why", "how", "that", "with", "his", "he",
    }


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── stage 1: sample ──────────────────────────────────────────────────────────

def stage_sample() -> list:
    refs = list(build_index().values())
    # keep only docs with real bodies
    usable = []
    for r in refs:
        try:
            body = r.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(body) >= 800:
            usable.append((r, len(body)))
    rng = random.Random(SEED)
    # stratify by month so eras are covered
    by_month = defaultdict(list)
    for r, n in usable:
        by_month[(r.date or "0000-00")[:7]].append(r)
    months = sorted(by_month)
    pool, per_month = [], max(2, POOL_SIZE // max(1, len(months)))
    for m in months:
        docs = by_month[m]
        rng.shuffle(docs)
        pool.extend(docs[:per_month])
    rng.shuffle(pool)
    pool = pool[:POOL_SIZE]
    POOL_F.write_text(json.dumps(
        [{"id": r.id, "path": str(r.path), "date": r.date, "title": r.title}
         for r in pool], indent=1))
    print(f"sampled {len(pool)} candidates across {len(months)} months "
          f"(usable corpus: {len(usable)})")
    return pool


# ── stage 2: extract ─────────────────────────────────────────────────────────

def _parse_json_array(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?\s*|\s*```$", "", raw)
    start = raw.find("[")
    if start < 0:
        return []
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "[":
            depth += 1
        elif raw[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    out = json.loads(raw[start:i + 1])
                    return out if isinstance(out, list) else []
                except json.JSONDecodeError:
                    return []
    return []


def stage_extract():
    pool = json.loads(POOL_F.read_text())
    done_ids = set()
    if EXTRACTED_F.exists():
        for line in EXTRACTED_F.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["source_distillation_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    client = _client()
    n_q = 0
    with EXTRACTED_F.open("a") as out:
        for i, cand in enumerate(pool):
            if cand["id"] in done_ids:
                continue
            try:
                body = Path(cand["path"]).read_text(
                    encoding="utf-8", errors="replace")[:MAX_BODY_CHARS]
            except OSError:
                continue
            prompt = (f"Document id: {cand['id']}\nDate: {cand['date']}\n"
                      f"Title: {cand['title']}\n\n---\n{body}")
            try:
                resp = client.messages.create(
                    model=EXTRACT_MODEL, max_tokens=1200,
                    system=EXTRACT_SYSTEM,
                    messages=[{"role": "user", "content": prompt}])
                items = _parse_json_array(resp.content[0].text)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}] {cand['id']}: API error {e!r} — backoff 30s, one retry")
                time.sleep(30)
                try:
                    resp = client.messages.create(
                        model=EXTRACT_MODEL, max_tokens=1200,
                        system=EXTRACT_SYSTEM,
                        messages=[{"role": "user", "content": prompt}])
                    items = _parse_json_array(resp.content[0].text)
                except Exception as e2:  # noqa: BLE001
                    print(f"  [{i}] {cand['id']}: retry failed {e2!r} — skipping")
                    continue
            for it in items:
                if not all(k in it for k in
                           ("question", "expected_answer", "excerpt",
                            "category", "difficulty")):
                    continue
                # provenance check: excerpt must actually be in the doc
                # (whitespace-normalized)
                norm = " ".join(body.split())
                if " ".join(str(it["excerpt"]).split()) not in norm:
                    continue
                row = {
                    "id": _qid(it["question"]),
                    "question": it["question"],
                    "expected_answer": it["expected_answer"],
                    "acceptable_keywords": it.get("acceptable_keywords", []),
                    "category": it["category"],
                    "difficulty": it["difficulty"],
                    "source_distillation_id": cand["id"],
                    "source_path": cand["path"],
                    "source_date": cand["date"],
                    "frozen_excerpt": it["excerpt"],
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                n_q += 1
            time.sleep(1.0)
            if i % 20 == 0:
                print(f"  [{i}/{len(pool)}] extracted so far: {n_q}")
    print(f"extraction complete: +{n_q} new questions")


# ── stage 3: negatives ───────────────────────────────────────────────────────

def stage_negatives():
    refs = list(build_index().values())
    concepts = Counter()
    for r in refs:
        for c in r.concepts:
            concepts[c] += 1
    top = [c for c, n in concepts.most_common(80) if n >= 3]
    client = _client()
    resp = client.messages.create(
        model=EXTRACT_MODEL, max_tokens=2500, system=NEGATIVE_SYSTEM,
        messages=[{"role": "user", "content":
                   "Real entities/concepts from Alex's history:\n"
                   + "\n".join(f"- {c}" for c in top)
                   + f"\n\nGenerate {QUOTAS['negative'] + 5} negative questions."}])
    items = _parse_json_array(resp.content[0].text)
    rows = []
    for it in items:
        if "question" not in it:
            continue
        rows.append({
            "id": _qid(it["question"]),
            "question": it["question"],
            "expected_answer": "NOT_IN_MEMORY — correct behavior is to say this is not in memory",
            "acceptable_keywords": [],
            "category": "negative",
            "difficulty": "medium",
            "trap_type": it.get("trap_type", "unknown"),
            "real_entities_used": it.get("real_entities_used", []),
            "source_distillation_id": None,
            "frozen_excerpt": None,
        })
    NEGATIVES_F.write_text(json.dumps(rows, indent=1))
    print(f"generated {len(rows)} negative questions")


# ── stage 4: freeze ──────────────────────────────────────────────────────────

def stage_freeze():
    rows = []
    seen_ids = set()
    for line in EXTRACTED_F.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            rows.append(r)
    negatives = json.loads(NEGATIVES_F.read_text()) if NEGATIVES_F.exists() else []

    # dedup near-duplicates by token jaccard
    kept = []
    toks = []
    for r in sorted(rows, key=lambda r: r["source_date"] or ""):
        t = _norm_tokens(r["question"])
        if any(_jaccard(t, t2) > 0.6 for t2 in toks):
            continue
        kept.append(r)
        toks.append(t)

    # balance to quotas (positives)
    by_cat = defaultdict(list)
    for r in kept:
        by_cat[r["category"]].append(r)
    rng = random.Random(SEED)
    final = []
    for cat, quota in QUOTAS.items():
        if cat == "negative":
            continue
        docs = by_cat.get(cat, [])
        rng.shuffle(docs)
        final.extend(docs[:quota])
    # backfill shortfall from surplus categories
    shortfall = (TARGET_TOTAL - QUOTAS["negative"]) - len(final)
    if shortfall > 0:
        surplus = [r for r in kept if r not in final]
        rng.shuffle(surplus)
        final.extend(surplus[:shortfall])
    # negatives
    rng.shuffle(negatives)
    final.extend(negatives[:QUOTAS["negative"]])

    out = {
        "version": "v3",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "benchmark/build_v3.py",
        "extract_model": EXTRACT_MODEL,
        "seed": SEED,
        "corpus": "distill_index roots (~770 distillations)",
        "counts": dict(Counter(r["category"] for r in final)),
        "questions": final,
    }
    OUT_F.write_text(json.dumps(out, indent=1))
    print(f"froze {len(final)} questions -> {OUT_F}")
    print(json.dumps(out["counts"], indent=1))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("sample", "all"):
        stage_sample()
    if stage in ("extract", "all"):
        stage_extract()
    if stage in ("negatives", "all"):
        stage_negatives()
    if stage in ("freeze", "all"):
        stage_freeze()
