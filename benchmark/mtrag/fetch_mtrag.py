"""fetch_mtrag.py — download the MTRAG benchmark data from IBM's official repo.

MTRAG (Katsis et al. 2025, arXiv:2501.03468) is distributed by IBM at
https://github.com/IBM/mt-rag-benchmark under its own license. We do not
redistribute the data; this script fetches exactly the files the PDCT
cross-corpus harness needs into benchmark/mtrag/data/.

Usage:
    python -m benchmark.mtrag.fetch_mtrag            # all corpora
    python -m benchmark.mtrag.fetch_mtrag fiqa govt  # subset

Then run e.g.:
    python -m benchmark.mtrag.run_mtrag --corpus govt
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/IBM/mt-rag-benchmark/main"
DATA = Path(__file__).parent / "data"
CORPORA = ["fiqa", "govt", "cloud", "clapnq"]
VARIANTS = ["questions", "lastturn", "rewrite"]


def _get(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ {dest.relative_to(DATA.parent)} (cached)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def fetch(corpora: list[str]) -> None:
    _get(f"{BASE}/mtrag-human/conversations/conversations.json",
         DATA / "conversations" / "conversations.json")
    for c in corpora:
        _get(f"{BASE}/corpora/passage_level/{c}.jsonl.zip", DATA / f"{c}.jsonl.zip")
        for v in VARIANTS:
            _get(f"{BASE}/mtrag-human/retrieval_tasks/{c}/{c}_{v}.jsonl",
                 DATA / "retrieval_tasks" / c / f"{c}_{v}.jsonl")
        _get(f"{BASE}/mtrag-human/retrieval_tasks/{c}/qrels/dev.tsv",
             DATA / "retrieval_tasks" / c / "qrels" / "dev.tsv")
    print("done.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a in CORPORA]
    fetch(args or CORPORA)
