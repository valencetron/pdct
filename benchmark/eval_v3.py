"""eval_v3 — score the v3 benchmark question set end-to-end.

Per question:
    service.run(question) -> prompt_block (cascade context)
    -> same-model reply (Claude Code identity system block, runner.py contract)
    -> grade:
        positive: keyword grading against acceptable_keywords (any-match per
                  keyword group; score = fraction of keyword groups present)
        negative: PASS iff the reply abstains (says it doesn't know / can't
                  find it) — any confident fabricated answer = FAIL.

Usage:
    PYTHONPATH=src python3 benchmark/eval_v3.py --smoke 20
    PYTHONPATH=src python3 benchmark/eval_v3.py            # full 100
    PYTHONPATH=src python3 benchmark/eval_v3.py --no-context   # ablation arm

Output: benchmark/.v3-work/eval-<run_id>.jsonl + summary to stdout.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dct.llm import _client_factory  # noqa: E402
from dct.retrieval import service  # noqa: E402
from dct.retrieval import memory_api  # noqa: E402
from dct.retrieval.retrieval_metrics import gold_ids, recall_at_k  # noqa: E402

ASSET = ROOT / "benchmark" / "pdct-questions-v3.json"
WORK = ROOT / "benchmark" / ".v3-work"

# Same contract as src/dct/research/runner.py — leading Claude Code identity
# block is required for Max OAuth in-contract calls.
_CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
REPLY_INSTRUCTIONS = (
    "Answer the user's question using the injected context block where relevant. "
    "If the context does not contain the answer and you do not know it, say so "
    "plainly — do not guess or fabricate. Be concise and direct."
)
REPLY_SYSTEM = [
    {"type": "text", "text": _CLAUDE_CODE_IDENTITY},
    {"type": "text", "text": REPLY_INSTRUCTIONS},
]
REPLY_MODEL = "claude-haiku-4-5"  # Sonnet 429-limited today; haiku keeps arms comparable
REPLY_MAX_TOKENS = 512
_TOP_K_RECALL = 5  # recall@k cutoff for the retrieval-only arm + metrics probe

ABSTAIN_PAT = re.compile(
    r"\b(don'?t know|do not know|don'?t have|do not have|"
    r"no (record|information|context|mention)|"
    r"not (in|found in|present in|mentioned|aware|sure)|can'?t find|cannot find|"
    r"unable to (find|locate|verify)|no such|never (happened|occurred|approved|stored)|"
    r"doesn'?t (exist|appear|mention)|does not (exist|appear|mention)|"
    r"i have no|there (is|was) no|didn'?t (happen|occur)|isn'?t any|no evidence)\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def grade_positive(reply: str, q: dict) -> dict:
    """Fraction of keyword groups matched. acceptable_keywords is a list where
    each item may be 'foo' or 'foo|bar' (alternatives)."""
    body = _norm(reply)
    groups = q.get("acceptable_keywords") or []
    if not groups:
        return {"score": None, "matched": [], "missed": [], "note": "no keywords"}
    matched, missed = [], []
    for g in groups:
        alts = [a.strip() for a in str(g).split("|") if a.strip()]
        if any(_norm(a).strip() in body for a in alts):
            matched.append(g)
        else:
            missed.append(g)
    return {"score": len(matched) / len(groups), "matched": matched, "missed": missed}


def grade_negative(reply: str) -> dict:
    abstained = bool(ABSTAIN_PAT.search(reply))
    return {"score": 1.0 if abstained else 0.0, "abstained": abstained}


def redact_exc(e: BaseException) -> str:
    """Sanitized representation of an exception for durable artifacts/logs.
    Class name only — raw exception text can embed paths/URLs/credentials
    from lower-level clients (Codex diff r3 #4 / r4)."""
    return type(e).__name__


def ceiling_adjusted(score: float, support) -> float:
    """Generation score divided by the gold doc's keyword supportability,
    capped at 1.0. When support is missing/zero, returns the raw score
    (no adjustment). Only meaningful for retrieved@5 rows — the caller is
    responsible for that gating."""
    if support and support > 0:
        return min(1.0, score / support)
    return score


# Known positive (answerable) categories. Used only as a backward-compat
# fallback for legacy rows lacking an explicit is_positive flag.
_POSITIVE_CATEGORIES = frozenset(
    {"factual-recall", "decision-recall", "temporal"})


def _has_support_cap(q: dict) -> bool:
    """True when the question carries a numeric gold_keyword_support strictly
    below 1.0. Must NOT use `or` short-circuit: support==0.0 is a real (worst)
    cap, not a missing annotation (Codex diff r1 #4). Predicate is `< 1.0` to
    match the contract/label exactly — a rounded 0.999 still counts (r2 #2)."""
    sup = q.get("gold_keyword_support")
    return isinstance(sup, (int, float)) and not isinstance(sup, bool) and sup < 1.0


def honest_axes(rows: list, qmap: dict) -> dict:
    """Compute the 5 honest numbers from graded rows. Pure — no I/O.

    Exclusion contract:
      - recall@5: rows where retrieval_hit5 is not None (has gold)
      - GEN raw: POSITIVE rows only (is_positive True). Negative/abstain rows
        are excluded even though grade_negative emits a 0/1 score, so the
        "positives" label is truthful (Codex diff #1).
      - GEN when retrieved@5: positives where retrieval_hit5 is True
      - ceiling-adj: same subset, each score / gold_keyword_support capped 1.0
      - benchmark_unsupported: questions whose gold_keyword_support < 1.0
        (0.0 counts; see _has_support_cap).
      - retrieval_probe_errors: rows whose metric probe raised (counted as
        misses but surfaced separately so an outage is distinguishable from
        genuine misses).
    """
    # Backward-compat: rows from before is_positive existed fall back to
    # category (Codex diff r2 #1). To avoid silently counting corrupt/unknown
    # rows as positive (r3 #3), the fallback requires a KNOWN positive
    # category; unknown/missing/misspelled categories are excluded.
    def _is_pos(r):
        if "is_positive" in r:
            return bool(r["is_positive"])
        return r.get("category") in _POSITIVE_CATEGORIES

    pos = [r for r in rows if r.get("score") is not None and _is_pos(r)]
    retr = [r for r in pos if r.get("retrieval_hit5") is True]
    recall_rows = [r for r in rows if r.get("retrieval_hit5") is not None]
    adj = [ceiling_adjusted(r["score"],
                            qmap.get(r["id"], {}).get("gold_keyword_support"))
           for r in retr]

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    # Probe errors counted by KEY PRESENCE (r3 #1) — an exception with an
    # empty message must still register. Split answerable (counted as misses)
    # from total so the label is truthful (r3 #2).
    probe_err_total = sum(1 for r in rows if "retrieval_probe_error" in r)
    probe_err_answerable = sum(1 for r in rows if "retrieval_probe_error" in r
                               and r.get("retrieval_hit5") is not None)

    return {
        "retrieval_recall_at5": mean([1 if r["retrieval_hit5"] else 0
                                      for r in recall_rows]),
        "retrieval_n": len(recall_rows),
        "gen_raw": mean([r["score"] for r in pos]),
        "gen_n": len(pos),
        "gen_when_retrieved5": mean([r["score"] for r in retr]),
        "gen_retrieved_n": len(retr),
        "gen_when_retrieved5_ceiling_adj": mean(adj),
        "benchmark_unsupported": sum(1 for q in qmap.values()
                                     if _has_support_cap(q)),
        "retrieval_probe_errors": probe_err_total,
        "retrieval_probe_errors_answerable": probe_err_answerable,
    }


HYDRATE_BUDGET_CHARS = 24000  # ~6k tokens of distillation bodies
TOOL_LOOP_MAX_CALLS = 6

TOOLS = [
    {
        "name": "query_memory",
        "description": (
            "Search the vault of past-conversation distillations. Returns up to "
            "5 rows (id, date, title, gist) ranked by graph relevance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"seed": {"type": "string", "description": "free-text query"}},
            "required": ["seed"],
        },
    },
    {
        "name": "read_memory",
        "description": "Read the full content of one distillation by id from query_memory.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
]


def hydrate_block(question: str) -> tuple[str, int]:
    """Single-shot arm B: query_memory on the question, inject top distillation
    bodies up to budget. Returns (block, n_docs)."""
    rows = memory_api.query_memory(question, _surface="eval_v3")
    parts, used = [], 0
    for row in rows:
        try:
            body = memory_api.read_memory(row.id, _surface="eval_v3").body
        except KeyError:
            continue
        take = body[: max(0, HYDRATE_BUDGET_CHARS - used)]
        if not take:
            break
        parts.append(f"### {row.date} — {row.title}\n{take}")
        used += len(take)
        if used >= HYDRATE_BUDGET_CHARS:
            break
    return "\n\n".join(parts), len(parts)


def _tool_result(name: str, inp: dict) -> str:
    if name == "query_memory":
        rows = memory_api.query_memory(inp.get("seed", ""), _surface="eval_v3_tools")
        return json.dumps([
            {"id": r.id, "date": r.date, "title": r.title, "gist": r.gist[:300]}
            for r in rows
        ])
    if name == "read_memory":
        try:
            m = memory_api.read_memory(inp.get("id", ""), _surface="eval_v3_tools")
            return m.body[:16000]
        except KeyError as e:
            return f"ERROR: {e}"
    return f"ERROR: unknown tool {name}"


def _call_with_retry(client, **kw):
    for attempt in range(4):
        try:
            return client.messages.create(**kw)
        except Exception as e:
            if attempt == 3:
                raise
            # OAuth token can rotate mid-run -> 401; rebuild the client.
            if "authentication" in str(e).lower() or "401" in str(e):
                try:
                    client = _client_factory()
                except Exception:
                    pass
            wait = 15 * (attempt + 1)
            print(f"  [retry] {redact_exc(e)} — sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)


def run_tool_loop(client, question: str, block: str) -> tuple[str, int]:
    """Arm C: cascade block + query_memory/read_memory tools, agentic loop.
    Returns (final_reply, n_tool_calls)."""
    content = (
        f"## Injected context\n{block or '(none)'}\n\n"
        f"## User question\n{question}\n\n"
        "Use query_memory/read_memory to verify before answering. If you cannot "
        "find evidence after searching, say you don't know — never fabricate."
    )
    messages = [{"role": "user", "content": content}]
    calls = 0
    while True:
        resp = _call_with_retry(
            client, model=REPLY_MODEL, max_tokens=1024,
            system=REPLY_SYSTEM, tools=TOOLS, messages=messages,
        )
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses or calls >= TOOL_LOOP_MAX_CALLS:
            text = "".join(b.text for b in resp.content if b.type == "text")
            return text, calls
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in tool_uses:
            calls += 1
            results.append({
                "type": "tool_result", "tool_use_id": tu.id,
                "content": _tool_result(tu.name, tu.input or {}),
            })
        messages.append({"role": "user", "content": results})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="sample N questions")
    ap.add_argument("--no-context", action="store_true", help="ablation: skip retrieval")
    ap.add_argument("--mode", choices=["block", "hydrated", "tools"], default="block")
    ap.add_argument("--seed", type=int, default=63)
    ap.add_argument("--retrieval-only", action="store_true",
                    help="no LLM: only measure gold-doc recall@5 (fast arm)")
    args = ap.parse_args()

    qs = json.load(open(ASSET))["questions"]
    if args.smoke:
        rng = random.Random(args.seed)
        # stratified smoke: keep category mix
        by_cat: dict[str, list] = {}
        for q in qs:
            by_cat.setdefault(q["category"], []).append(q)
        take = []
        n = args.smoke
        for cat, items in by_cat.items():
            k = max(1, round(n * len(items) / len(qs)))
            take.extend(rng.sample(items, min(k, len(items))))
        qs = take[:n] if len(take) > n else take

    if args.retrieval_only:
        applicable = 0
        hits = 0
        errors = 0
        per_cat: dict = {}
        for i, q in enumerate(qs, 1):
            gold = gold_ids(q)
            try:
                rows = memory_api.query_memory(q["question"], _surface="eval_v3_retrieval")
            except Exception as e:  # noqa: BLE001 — count, never abort the run
                errors += 1
                print(f"[{i}/{len(qs)}] RETRIEVAL ERROR: {redact_exc(e)}",
                      file=sys.stderr)
                rows = []
            r = recall_at_k(rows, gold, k=_TOP_K_RECALL)
            if r is None:
                continue  # negative/no-gold question: excluded from recall
            applicable += 1
            hits += 1 if r else 0
            per_cat.setdefault(q["category"], []).append(1 if r else 0)
            print(f"[{i}/{len(qs)}] {q['category'][:12]:<12} recall@5={'HIT' if r else 'miss'}  {q['question'][:60]}")
        print("\n=== RETRIEVAL-ONLY SUMMARY ===")
        for cat, v in sorted(per_cat.items()):
            print(f"{cat:<16} n={len(v):<3} recall@5={sum(v)/len(v):.2f}")
        overall = (hits / applicable) if applicable else None
        if applicable:
            print(f"{'OVERALL':<16} n={applicable:<3} recall@5={overall:.2f}")
        if errors:
            print(f"(retrieval errors: {errors} — counted as misses)")
        WORK.mkdir(exist_ok=True)
        rid = time.strftime("%Y%m%d-%H%M%S")
        rsum = {"run_id": rid, "arm": "retrieval_only",
                "retrieval_recall_at5": overall, "applicable_n": applicable,
                "errors": errors,
                "by_category": {c: sum(v) / len(v) for c, v in per_cat.items()}}
        (WORK / f"retrieval-only-{rid}.summary.json").write_text(json.dumps(rsum, indent=2))
        print(f"summary -> {WORK / ('retrieval-only-' + rid + '.summary.json')}")
        return

    WORK.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{args.mode}" + ("-noctx" if args.no_context else "")
    out_f = WORK / f"eval-{run_id}.jsonl"
    client = _client_factory()

    rows = []
    for i, q in enumerate(qs, 1):
        block = ""
        cascade_count = 0
        tool_calls = 0
        if not args.no_context:
            try:
                if args.mode == "hydrated":
                    block, cascade_count = hydrate_block(q["question"])
                else:
                    r = service.run(q["question"])
                    block = r.get("prompt_block", "") or ""
                    cascade_count = r.get("cascade_count", 0)
            except Exception as e:  # retrieval failure = empty context, note it
                block = ""
                cascade_count = -1
                print(f"  [warn] retrieval failed q{i}: {redact_exc(e)}", file=sys.stderr)
        if args.mode == "tools":
            reply, tool_calls = run_tool_loop(client, q["question"], block)
        else:
            content = (
                f"## Injected context\n{block or '(none)'}\n\n"
                f"## User question\n{q['question']}"
            )
            resp = _call_with_retry(
                client, model=REPLY_MODEL, max_tokens=REPLY_MAX_TOKENS,
                system=REPLY_SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
            reply = resp.content[0].text

        if q["category"] == "negative":
            g = grade_negative(reply)
        else:
            g = grade_positive(reply, q)
        row = {
            "id": q["id"],
            "category": q["category"],
            "difficulty": q.get("difficulty"),
            "question": q["question"],
            "expected": q.get("expected_answer"),
            "reply": reply,
            "cascade_count": cascade_count,
            "context_chars": len(block),
            "tool_calls": tool_calls,
            "mode": args.mode,
            "is_positive": q["category"] != "negative",
            **g,
        }
        # Metrics-only retrieval probe (does not affect generation).
        _gold = gold_ids(q)
        try:
            _mrows = memory_api.query_memory(q["question"], _surface="eval_v3_metric")
            _hit = recall_at_k(_mrows, _gold, k=_TOP_K_RECALL)
        except Exception as e:  # noqa: BLE001
            # A probe failure on an ANSWERABLE question is a miss, not an
            # exclusion — otherwise retrieval errors silently vanish from
            # the denominator (Codex diff #2). No-gold questions stay None.
            # Record the error on the row so an outage is distinguishable from
            # genuine misses in the summary (Codex diff r2 #3).
            # Sanitized on stderr too — stderr is often persisted as durable
            # CI logs, so raw exception text (paths/URLs/creds) must not leak
            # there either (Codex diff r4). Class name only.
            print(f"  [warn] retrieval probe failed q{i}: {redact_exc(e)}",
                  file=sys.stderr)
            _hit = False if _gold else None
            # Persist only the exception CLASS, not raw text — lower-level
            # client errors can embed paths/URLs/credentials, and the JSONL is
            # a durable artifact (Codex diff r3 #4 / r4). stderr is redacted
            # the same way (durable CI logs), via redact_exc().
            row["retrieval_probe_error"] = redact_exc(e)
        row["retrieval_hit5"] = _hit  # True / False / None(no gold)
        rows.append(row)
        out_f.open("a").write(json.dumps(row) + "\n")
        s = row["score"]
        print(f"[{i}/{len(qs)}] {q['category'][:12]:<12} score={s if s is None else round(s,2)}  {q['question'][:70]}")

    # summary
    print("\n=== SUMMARY ===")
    by_cat: dict[str, list] = {}
    for r in rows:
        if r["score"] is not None:
            by_cat.setdefault(r["category"], []).append(r["score"])
    for cat, scores in sorted(by_cat.items()):
        print(f"{cat:<16} n={len(scores):<3} mean={sum(scores)/len(scores):.2f}")
    all_s = [r["score"] for r in rows if r["score"] is not None]
    if all_s:
        print(f"{'OVERALL':<16} n={len(all_s):<3} mean={sum(all_s)/len(all_s):.2f}")
    print(f"\nrows -> {out_f}")

    # Honest, separated axes — single source of truth in honest_axes()
    # (tested in tests/retrieval/test_ceiling_math.py). Ceiling-adjustment is
    # applied ONLY on the retrieved@5 subset (Codex #11): dividing a generation
    # score by full-gold supportability when the answer may not have been
    # injected would let injection gaps masquerade as recovered generation.
    qmap = {q["id"]: q for q in json.load(open(ASSET))["questions"]}
    ax = honest_axes(rows, qmap)

    def _f(x):
        return "n/a" if x is None else f"{x:.2f}"

    have_ceiling = any(q.get("gold_keyword_support") is not None for q in qmap.values())
    print("\n=== HONEST AXES (5 numbers) ===")
    print(f"  RETRIEVAL recall@5            n={ax['retrieval_n']:<3} {_f(ax['retrieval_recall_at5'])}")
    print(f"  GEN raw (all positives)       n={ax['gen_n']:<3} {_f(ax['gen_raw'])}")
    print(f"  GEN when retrieved@5          n={ax['gen_retrieved_n']:<3} {_f(ax['gen_when_retrieved5'])}")
    print(f"  GEN when retrieved@5 (ceil)   n={ax['gen_retrieved_n']:<3} {_f(ax['gen_when_retrieved5_ceiling_adj'])}")
    print(f"  BENCHMARK unsupported (<1.0)  {ax['benchmark_unsupported']}")
    if ax.get("retrieval_probe_errors"):
        print(f"  RETRIEVAL probe errors        {ax['retrieval_probe_errors']} "
              f"({ax.get('retrieval_probe_errors_answerable', 0)} answerable, "
              "counted as misses)")
    if not have_ceiling:
        print("  (no gold_keyword_support annotations yet — run "
              "scripts/benchmark_keyword_audit.py --write; ceiling == raw)")
    if args.mode != "hydrated":
        # Codex diff #3: retrieval_hit5 is measured via query_memory(), but in
        # block/tools mode generation is fed by service.run()'s cascade block,
        # NOT the query_memory rows. So "retrieved@5" here means "gold was
        # query_memory-retrievable", not "gold was injected into generation".
        # Only --mode hydrated injects query_memory bodies; trust the
        # retrieved@5 generation split fully only there.
        print(f"  [caveat] mode={args.mode}: retrieved@5 measures query_memory, "
              "not what generation saw. Use --mode hydrated for a faithful split.")

    summary = {"run_id": run_id, "mode": args.mode, **ax}
    (WORK / f"eval-{run_id}.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {WORK / ('eval-' + run_id + '.summary.json')}")


if __name__ == "__main__":
    main()
