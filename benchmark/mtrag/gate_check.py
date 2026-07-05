"""GO/NO-GO gate: build concepts over FULL FiQA, print sanity stats.

GO (all hold): avg/passage in [3,12]; top-40 are recognizable finance phrases;
singleton% < 80. NO-GO -> STOP, card note, do not proceed.
"""
from collections import Counter
from benchmark.mtrag import ingest, keyphrase


def main():
    passages = ingest.load_passages("fiqa")  # FULL corpus
    docs = [p["text"] for p in passages]
    ex = keyphrase.CorpusExtractor(docs, top_k=8)
    allc = Counter()
    per = []
    for d in docs:
        cs = ex.extract(d)
        per.append(len(cs))
        allc.update(cs)
    n = len(docs)
    avg = sum(per) / n if n else 0
    print(f"passages={n}  unique_concepts={len(allc)}  avg/passage={avg:.1f}")
    print("TOP 40 (should be finance topics, not stopword sludge):")
    for c, ct in allc.most_common(40):
        print(f"  {ct:6d}  {c}")
    singles = sum(1 for _, ct in allc.items() if ct == 1)
    singleton_pct = 100 * singles / max(len(allc), 1)
    print(f"singleton%={singleton_pct:.0f}")
    go = (3 <= avg <= 12) and singleton_pct < 80
    print(f"\nGATE: {'GO' if go else 'NO-GO'}  (avg in [3,12]={3<=avg<=12}, singleton%<80={singleton_pct<80})")


if __name__ == "__main__":
    main()
