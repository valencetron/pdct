"""Construct divergent-middle/same-end arms from SAME-corpus conversations.

NOTE: the HEADLINE path-dependence statistic (Leg 1) uses within-conversation
same-content/different-order permutations (see run_mtrag.leg1). This cross-
conversation A/B builder is retained only for the QUALITATIVE different-topic
demo in the report — it is explicitly NOT the headline number."""
from __future__ import annotations


def _user_turns(c):
    return [m["text"] for m in c["messages"] if m["speaker"] == "user"]


def construct_arms(convos, opener, closer, max_middle=3):
    if len(convos) < 2:
        raise ValueError("construct_arms needs >= 2 conversations")
    mids = [_user_turns(c)[:max_middle] for c in convos[:2]]
    if not mids[0] or not mids[1]:
        raise ValueError("both conversations must have >= 1 user turn")
    L = min(len(mids[0]), len(mids[1]))
    mid_a, mid_b = mids[0][:L], mids[1][:L]
    return {"A": [opener] + mid_a + [closer],
            "B": [opener] + mid_b + [closer],
            "C": [closer]}
