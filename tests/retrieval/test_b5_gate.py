"""B5 concept-token gate invariant (2026-07-23).

_aggregate() gates the per-ref fuzzy loop on shared tokens instead of scanning
every cascade concept. That is byte-identical ONLY if _concept_match_strength()
returns >0 exclusively when the two concepts share a token. This randomized
property test guards that invariant (its failure would mean the gate silently
drops real matches)."""
import random

from dct.retrieval.memory_api import _concept_match_strength, _concept_tokens

_TOKS = ["ayan", "iep", "meeting", "ab", "cd", "x", "orion", "pdct", "co",
         "note", "correction", "name", "abc", "de", "fghij"]


def test_positive_match_implies_shared_token():
    rng = random.Random(20260723)
    positives = 0
    for _ in range(8000):
        cc = "-".join(rng.choice(_TOKS) for _ in range(rng.randint(1, 4)))
        rc = "-".join(rng.choice(_TOKS) for _ in range(rng.randint(1, 4)))
        if _concept_match_strength(cc, rc) > 0:
            positives += 1
            assert _concept_tokens(cc) & _concept_tokens(rc), (
                f"{cc!r} vs {rc!r} scores >0 but shares no token — the B5 gate "
                f"would wrongly drop this pair")
    assert positives > 200  # sanity: positive matches were actually exercised


def test_gate_matches_bruteforce():
    """Directly: the token-gated best-match equals the scan-all best-match."""
    rng = random.Random(99)
    for _ in range(400):
        cascade = list({"-".join(rng.choice(_TOKS) for _ in range(rng.randint(1, 3)))
                        for _ in range(rng.randint(1, 12))})
        score = {cc: rng.random() for cc in cascade}
        rc = "-".join(rng.choice(_TOKS) for _ in range(rng.randint(1, 3)))

        brute = max((score[cc] * _concept_match_strength(cc, rc)
                     for cc in cascade), default=0.0)
        brute = max(brute, 0.0)

        by_tok: dict[str, list[str]] = {}
        for cc in cascade:
            for t in _concept_tokens(cc):
                by_tok.setdefault(t, []).append(cc)
        gated, seen = 0.0, set()
        for t in _concept_tokens(rc):
            for cc in by_tok.get(t, ()):
                if cc in seen:
                    continue
                seen.add(cc)
                s = _concept_match_strength(cc, rc)
                if s > 0:
                    gated = max(gated, score[cc] * s)

        assert abs(brute - gated) < 1e-12, f"gate != bruteforce for rc={rc!r}"
