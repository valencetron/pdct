from __future__ import annotations
import math
import random


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def divergence(a: set, b: set) -> float:
    return 1.0 - jaccard(a, b)


def recall_at_k(ranked, gold, k):
    if not gold:
        return 1.0
    return len(set(ranked[:k]) & gold) / len(gold)


def ndcg_at_k(ranked, gold, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in gold)
    ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked, gold):
    """Reciprocal rank of the first gold passage (0 if none in the list)."""
    if not gold:
        return 0.0
    for i, p in enumerate(ranked):
        if p in gold:
            return 1.0 / (i + 1)
    return 0.0


def rank_overlap_at_k(a, b, k):
    """Top-k SET (membership) Jaccard of two arms' results — measures how many
    passages enter/leave the top-k, NOT their order. Two top-k sets that are
    reordered return 1.0. We report 1 - this as 'path-memory magnitude' = the
    fraction of the top-k set that changed; rank-position movement is captured
    separately by MRR. (Named *rank* for historical reasons; it is set overlap.)"""
    sa, sb = set(a[:k]), set(b[:k])
    if not sa and not sb:
        return 1.0
    u = sa | sb
    return len(sa & sb) / len(u) if u else 1.0


def percentile(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def permutation_null_divergence(play_fn, middle, opener, closer,
                                max_perms=120, seed=0):
    """Same-content/different-ORDER null (Codex P0/P1 fix).

    Plays a fixed reference permutation (the given `middle` order) and compares
    its final set against every OTHER ordering of the SAME middle multiset.
    EXHAUSTIVE when the number of DISTINCT non-reference permutations <= max_perms
    (exact null); otherwise samples max_perms of them. Duplicate permutations
    (when middle has repeated texts) are de-duped so n_perms_used is honest. The
    reference order is EXCLUDED from the null (it contributes divergence 0 and
    would bias the percentile down). `opener` and `closer` MUST be lists (they are
    concatenated with the middle list). Returns (null_divergences, n_perms_used)."""
    import itertools
    rng = random.Random(seed)
    ref_mid = list(middle)
    ref = play_fn(list(opener) + ref_mid + list(closer))
    if len(ref_mid) <= 1:
        return [], 0
    seen = set()
    others = []
    for p in itertools.permutations(ref_mid):
        pl = list(p)
        if pl == ref_mid:
            continue
        key = tuple(pl)
        if key in seen:
            continue
        seen.add(key)
        others.append(pl)
    if len(others) > max_perms:
        rng.shuffle(others)
        others = others[:max_perms]
    return [divergence(ref, play_fn(list(opener) + p + list(closer))) for p in others], len(others)
