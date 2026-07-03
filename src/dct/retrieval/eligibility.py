"""Retrieval eligibility filter for distillations.

A single, pure gate that decides whether a distillation is eligible to be
retrieved / injected. Applied inside `build_index()` so that LIVE retrieval and
the eval harness share the EXACT same eligible corpus — honest scores, better
live context, one lever.

The dominant noise class is the pre-summarizer crystallize output: raw tool-call
transcript dumps whose body is `[tool:Bash cmd=...]` / `tool_result` traces with
no prose summary. Cascade jogs off these and injects pure noise. We also exclude
no-concept refs (nothing to jog off), thin bodies, bare-topic-key titles with no
prose, and the rare pruned-recap marker.

`is_eligible(ref, body) -> (bool, reason)` is pure and side-effect-free.
`reason` is "" when eligible, else a short stable code used for audit counters.
"""
from __future__ import annotations

import re

# --- thresholds (tuned against the 2026-06-12 corpus profile) ---
_MIN_BODY_CHARS = 400          # bodies shorter than this carry no retrievable signal
_TRANSCRIPT_MARKER_MIN = 3     # >= this many trace markers to even consider it a dump
# Prose density: real summaries carry >= ~1 natural sentence per this many chars.
# A raw tool dump (even one wrapping 10KB of command output behind 3 delimiters)
# has near-zero prose density, so this catches the low-marker-density case that a
# marker/char ratio misses. Codex P1.
_MIN_PROSE_DENSITY = 1 / 900   # sentences-per-char floor below which body is a dump

# Trace markers that betray a raw transcript dump rather than a prose summary.
_TRACE_RE = re.compile(r"\[tool:|tool_result|\[rc=|assistant:\s|\buser:\s")

# Bare topic-key titles like "1003690648082_19971", "1003690648082_None".
_BARE_ID_TITLE_RE = re.compile(r"^[\d_]+$|_none$|^\d{6,}$", re.IGNORECASE)

# Pruned-recap / epistemic-status lossy summaries (rare but cheap to drop).
_RECAP_RE = re.compile(r"recap \(pruned\)|EPISTEMIC STATUS|memory_manager:recap")

# Prose heuristic: a "real" sentence is a run of >=5 natural words (alphabetic,
# space-separated) terminated by sentence punctuation. Tight enough that code
# lines like `def fn_0(): return 0]` do NOT count as prose.
_SENTENCE_RE = re.compile(r"(?:[A-Za-z']+\s+){4,}[A-Za-z']+\s*[.!?]")


def _looks_like_transcript(body: str) -> bool:
    """True when the body is raw tool-call traces rather than a prose summary.

    Gate on the PRESENCE of trace markers (so we never touch pure-prose files),
    then decide on PROSE DENSITY rather than marker density. Marker-per-char
    misses a dump that wraps a huge tool output behind only a few delimiters;
    prose-per-char does not — a real summary has sentences throughout, a dump
    has almost none regardless of how large the embedded output is. (Codex P1.)
    """
    markers = len(_TRACE_RE.findall(body))
    if markers < _TRANSCRIPT_MARKER_MIN:
        return False
    sentences = len(_SENTENCE_RE.findall(body))
    # Substantial real prose anywhere => keep it even with embedded traces.
    if sentences >= 5:
        return False
    prose_density = sentences / max(len(body), 1)
    return prose_density < _MIN_PROSE_DENSITY


def is_eligible(ref, body: str) -> tuple[bool, str]:
    """Decide retrieval eligibility for one distillation.

    Args:
        ref:  a DistillationRef (uses .concepts and .title).
        body: the post-frontmatter body text.

    Returns:
        (eligible, reason). reason == "" iff eligible.
    """
    stripped = (body or "").strip()

    # 1. thin — no retrievable signal regardless of anything else.
    if len(stripped) < _MIN_BODY_CHARS:
        return False, "thin"

    # 2. no-concepts — cascade has nothing to jog off.
    if not getattr(ref, "concepts", None):
        return False, "no-concepts"

    # 3. pruned-recap / epistemic-status lossy summary.
    if _RECAP_RE.search(stripped):
        return False, "pruned-recap"

    # 4. raw transcript dump (the dominant noise class).
    if _looks_like_transcript(stripped):
        return False, "transcript-dump"

    # 5. bare-id title AND no real prose (belt-and-suspenders for the
    #    pre-summarizer files that slipped past the transcript check).
    title = (getattr(ref, "title", "") or "").strip()
    if _BARE_ID_TITLE_RE.search(title):
        if len(_SENTENCE_RE.findall(stripped)) < 3:
            return False, "bare-id-title"

    return True, ""
