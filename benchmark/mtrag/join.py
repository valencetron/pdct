"""Join MTRAG retrieval-task queries to conversation standalone flags.

EMPIRICAL FACTS (verified 2026-06-16):
- qid = '<32hex><::>turn'; separator is the literal 4 chars '<::>', turn 1-based.
- qrels query-id == questions _id EXACTLY -> gold joins by id (no hash lookup).
- The `questions` variant `text` IS the cumulative user history: lines joined by
  '\\n', each prefixed '|user|: '. Split to recover per-turn user messages.
- Standalone flag lives per-MESSAGE: messages[*].enrichments['Multi-Turn'].
  ['N/A'] => standalone (first turn); 'Follow-up'/'Clarification' => non-standalone.
"""
from __future__ import annotations
import re

_SEP = "<::>"
_WS = re.compile(r"\s+")


def parse_qid(qid: str):
    cid, _, turn = qid.partition(_SEP)
    return cid, int(turn)


def split_user_turns(questions_text: str) -> list[str]:
    parts = re.split(r"\|user\|:\s*", questions_text)
    return [p.strip() for p in parts if p.strip()]


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def is_standalone(multi_turn_vals) -> bool:
    vals = set(multi_turn_vals or [])
    return vals == {"N/A"} or not vals


def _user_msgs(convo: dict) -> list[dict]:
    return [m for m in convo["messages"] if m["speaker"] == "user"]


def build_standalone_index(convos: list[dict]):
    """Map normalized-first-user-turn -> {"flags":[...], "users":[norm,...]}.
    Codex P1: detect collisions. If two conversations share a normalized first
    user turn with DIFFERENT prefixes, mark that key AMBIGUOUS."""
    idx: dict[str, dict] = {}
    ambiguous: set[str] = set()
    for c in convos:
        ums = _user_msgs(c)
        if not ums:
            continue
        key = _norm(ums[0]["text"])
        flags = [is_standalone(m.get("enrichments", {}).get("Multi-Turn")) for m in ums]
        users = [_norm(m["text"]) for m in ums]
        if key in idx and idx[key]["users"] != users:
            ambiguous.add(key)
        idx[key] = {"flags": flags, "users": users}
    for k in ambiguous:
        idx[k]["ambiguous"] = True
    return idx


def standalone_for(questions_text: str, turn: int, standalone_index: dict):
    """Standalone bool for `turn` (1-based), or None if unjoined/ambiguous.
    Codex P1: require the query's user-turn PREFIX to match the matched
    conversation's prefix, and refuse ambiguous keys."""
    users = split_user_turns(questions_text)
    if not users:
        return None
    entry = standalone_index.get(_norm(users[0]))
    if not entry or entry.get("ambiguous"):
        return None
    qprefix = [_norm(u) for u in users]
    cprefix = entry["users"][:len(qprefix)]
    if cprefix != qprefix:
        return None
    flags = entry["flags"]
    if turn - 1 >= len(flags):
        return None
    return flags[turn - 1]
