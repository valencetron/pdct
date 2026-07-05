"""Load MTRAG corpora, conversations, qrels gold, and query variants.

Empirical facts (verified 2026-06-16 against vendored FiQA fixtures):
- query `_id` format = '<32hex><::>turn' (separator chars ord 60,58,58,62; turn 1-based).
- qrels/dev.tsv query-id column == the questions `_id` EXACTLY → gold joins by id.
- conversation→corpus comes from retriever.collection.name (3rd '-' token;
  'ibmcloud' maps to the 'cloud' retrieval_tasks dir).
"""
from __future__ import annotations
import json, zipfile, io
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
_CORPUS_DIR = {"clapnq": "clapnq", "fiqa": "fiqa", "govt": "govt",
               "ibmcloud": "cloud", "cloud": "cloud"}


def _corpus_of(convo: dict) -> str:
    name = convo.get("retriever", {}).get("collection", {}).get("name", "")
    parts = name.split("-")
    tok = parts[2] if len(parts) >= 3 else ""
    return _CORPUS_DIR.get(tok, tok)


def load_passages(corpus: str, limit: int | None = None) -> list[dict]:
    zp = DATA / f"{corpus}.jsonl.zip"
    out: list[dict] = []
    with zipfile.ZipFile(zp) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            for i, raw in enumerate(io.TextIOWrapper(fh, encoding="utf-8")):
                if not raw.strip():
                    continue
                d = json.loads(raw)
                pid = str(d.get("_id") or d.get("id") or i)
                out.append({"id": pid, "title": d.get("title", ""),
                            "text": d.get("text", "")})
                if limit and len(out) >= limit:
                    break
    return out


def load_conversations(corpus: str | None = None) -> list[dict]:
    allc = json.loads((DATA / "conversations" / "conversations.json").read_text())
    if corpus is None:
        return allc
    target = _CORPUS_DIR.get(corpus, corpus)
    return [c for c in allc if _corpus_of(c) == target]


def load_retrieval_tasks(corpus: str, variant: str) -> list[dict]:
    fp = DATA / "retrieval_tasks" / corpus / f"{corpus}_{variant}.jsonl"
    return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]


def load_qrels(corpus: str) -> dict[str, set[str]]:
    """query-id -> set(gold corpus-id). Skips header row."""
    fp = DATA / "retrieval_tasks" / corpus / "qrels" / "dev.tsv"
    out: dict[str, set[str]] = {}
    for i, line in enumerate(fp.read_text().splitlines()):
        if i == 0 or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        qid, cid = parts[0], parts[1]
        out.setdefault(qid, set()).add(cid)
    return out


def load_passages_with_gold(corpus: str):
    """FULL corpus + gold-coverage check. Returns (passages, missing_gold_ids)."""
    passages = load_passages(corpus)
    qrels = load_qrels(corpus)
    gold = set().union(*qrels.values()) if qrels else set()
    ids = {p["id"] for p in passages}
    missing = gold - ids
    return passages, missing
