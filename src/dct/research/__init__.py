"""dct.research — PDCT benchmark research engine (build #56).

The scientific measurement instrument for lever tuning: re-ask a frozen
question set through live retrieval + same-model reply + Haiku judge under
each lever setting, score on a 3-leg composite, sweep with paired stats.

In-process only — uses service.run(config_override=...) so a sweep NEVER
writes the live overrides file. The live deploy controller is a follow-on build.
"""

BENCHMARK_WEIGHTS = {
    "era_judge": 0.4,
    "match_rate": 0.3,
    "cosine_score": 0.3,
}
