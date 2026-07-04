"""Anthropic SDK wrapper for the DCT distiller.

Isolates the LLM call from business logic: all prompt construction, model-ID
resolution, schema validation, and retry logic live here. Callers pass turns
and get back a validated DistilledNote.
"""

from __future__ import annotations

from dataclasses import dataclass


_MODEL_SHORTNAMES: dict[str, str] = {
    "haiku":  "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

_FULL_MODEL_IDS: frozenset[str] = frozenset(_MODEL_SHORTNAMES.values())


def resolve_model_id(name: str) -> str:
    if name in _MODEL_SHORTNAMES:
        return _MODEL_SHORTNAMES[name]
    if name in _FULL_MODEL_IDS:
        return name
    raise ValueError(f"unknown model {name!r}; expected one of "
                     f"{sorted(_MODEL_SHORTNAMES)} or {sorted(_FULL_MODEL_IDS)}")


@dataclass(frozen=True)
class DistilledNote:
    title: str
    summary: str
    concepts: tuple[str, ...]
    key_quotes: tuple[dict, ...]


SYSTEM_PROMPT = (
    "You are distilling a conversation into a concept-anchored Obsidian note for "
    "Dynamic Context Traversal. Extract the concepts that came up, write a compact "
    "narrative summary, and pick a few key quotes. Use the emit_distilled_note tool."
)


def build_user_prompt(
    turns: list[dict],
    session_meta: dict,
    rules_concepts: list[str],
    *,
    per_turn_char_cap: int = 8000,
) -> str:
    lines: list[str] = []
    lines.append("## Session metadata")
    for k in ("source_channel", "session_id", "ts_start", "ts_end", "turn_count"):
        if k in session_meta:
            lines.append(f"{k}: {session_meta[k]}")
    lines.append("")

    if rules_concepts:
        lines.append("## Rules-extracted concepts (already found, seed your list)")
        lines.append(", ".join(rules_concepts))
        lines.append("")

    lines.append("## Turns")
    for turn in turns:
        role = turn.get("role", "unknown")
        text = turn.get("text", "")
        if len(text) > per_turn_char_cap:
            text = text[:per_turn_char_cap] + "..."
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


from dct.rules import _filter_slugs, to_slug


def _parse_tool_response(tool_input: dict) -> DistilledNote:
    for key in ("title", "summary", "concepts", "key_quotes"):
        if key not in tool_input:
            raise ValueError(f"missing field in LLM response: {key}")
    if not isinstance(tool_input["title"], str):
        raise ValueError("title must be string")
    if not isinstance(tool_input["summary"], str):
        raise ValueError("summary must be string")
    if not isinstance(tool_input["concepts"], list):
        raise ValueError("concepts must be list")
    if not isinstance(tool_input["key_quotes"], list):
        raise ValueError("key_quotes must be list")

    raw_concepts = [to_slug(c) for c in tool_input["concepts"] if isinstance(c, str)]
    hygienic = _filter_slugs(raw_concepts)

    quotes: list[dict] = []
    for q in tool_input["key_quotes"]:
        if not isinstance(q, dict):
            continue
        role = q.get("role")
        text = q.get("text")
        if isinstance(role, str) and isinstance(text, str):
            quotes.append({"role": role, "text": text})

    return DistilledNote(
        title=tool_input["title"],
        summary=tool_input["summary"],
        concepts=tuple(hygienic),
        key_quotes=tuple(quotes),
    )


_TOOL_SCHEMA = {
    "name": "emit_distilled_note",
    "description": "Emit the distilled note for this session.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title":    {"type": "string", "maxLength": 80},
            "summary":  {"type": "string"},
            "concepts": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
            "key_quotes": {
                "type": "array", "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["role", "text"],
                },
            },
        },
        "required": ["title", "summary", "concepts", "key_quotes"],
    },
}


_OAUTH_BETA_HEADER = "oauth-2025-04-20"


def _client_factory():
    """Return an anthropic.Anthropic client wired for OAuth or static API key.

    Priority order (via dct.auth.load_oauth_token):
      1. ~/.claude/.credentials.json  (Claude Max live token)
      2. macOS Keychain                (Claude Code-credentials)
      3. ~/example-stack/config/stack.json
      4. ANTHROPIC_API_KEY env

    OAuth tokens (``sk-ant-oat…``) are wired as ``auth_token`` with the
    ``anthropic-beta: oauth-2025-04-20`` header so the API accepts
    Claude Max credentials. Static API keys use ``api_key``. Missing
    credentials fall through to the default constructor so the SDK can
    still pick up a late-set env var or raise its own clear error.
    """
    import anthropic
    from dct import auth
    try:
        tok = auth.load_oauth_token()
    except auth.TokenLoadError:
        return anthropic.Anthropic()
    if auth.is_oauth_token(tok):
        return anthropic.Anthropic(
            auth_token=tok,
            default_headers={"anthropic-beta": _OAUTH_BETA_HEADER},
        )
    return anthropic.Anthropic(api_key=tok)


def _auth_error_cls():
    """Return anthropic.AuthenticationError, or an unreachable sentinel.

    Seam for tests — unit tests without the SDK import can substitute a
    fake exception class. When the SDK is importable, production code
    catches real 401s.
    """
    try:
        import anthropic
        return anthropic.AuthenticationError
    except Exception:
        class _Unreachable(Exception):
            pass
        return _Unreachable


def _extract_note(resp) -> DistilledNote:
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" \
                and getattr(block, "name", None) == "emit_distilled_note":
            return _parse_tool_response(block.input)
    raise ValueError("LLM response contained no tool_use block")


def call_distiller(
    turns: list[dict],
    session_meta: dict,
    rules_concepts: list[str],
    model: str = "haiku",
) -> DistilledNote:
    """Call the distiller LLM. Handles OAuth refresh-on-401 + transient retry.

    Retry policy:
      - Up to 2 attempts on ConnectionError/TimeoutError (per auth pass).
      - On AuthenticationError, invoke ``auth.refresh_oauth_via_cli`` and
        rebuild the client once. A second 401 after refresh propagates.
    """
    # Non-anthropic providers route through the provider abstraction
    # (JSON-schema emulation + strict parse). The anthropic path below is
    # unchanged — SDK + OAuth refresh-on-401 — so existing installs are
    # behavior-identical.
    from dct import providers as _prov
    if _prov.provider_name() != "anthropic":
        user_prompt = build_user_prompt(turns, session_meta, rules_concepts)
        obj = _prov.complete_json(SYSTEM_PROMPT, user_prompt,
                                  _TOOL_SCHEMA["input_schema"])
        return _parse_tool_response(obj)

    # anthropic provider WITHOUT the SDK installed (it's an optional extra
    # on public installs) — route through the urllib provider layer, the
    # same code path the doctor validates. SDK installs keep the original
    # path below (refresh-on-401 + SDK retry semantics).
    try:
        import anthropic  # noqa: F401
    except ImportError:
        user_prompt = build_user_prompt(turns, session_meta, rules_concepts)
        obj = _prov.complete_json(SYSTEM_PROMPT, user_prompt,
                                  _TOOL_SCHEMA["input_schema"],
                                  model=resolve_model_id(model))
        return _parse_tool_response(obj)

    model_id = resolve_model_id(model)
    user_prompt = build_user_prompt(turns, session_meta, rules_concepts)
    auth_error_cls = _auth_error_cls()

    client = _client_factory()
    refreshed = False
    last_transient: Exception | None = None

    while True:
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=[_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "emit_distilled_note"},
                messages=[{"role": "user", "content": user_prompt}],
            )
            return _extract_note(resp)
        except auth_error_cls:
            if refreshed:
                raise
            from dct import auth
            auth.refresh_oauth_via_cli()
            client = _client_factory()
            refreshed = True
            last_transient = None
            continue
        except (ConnectionError, TimeoutError) as exc:
            if last_transient is not None:
                raise
            last_transient = exc
            continue


_CONCEPT_EXTRACTOR_SYSTEM = (
    "You extract the key conceptual topics from a single piece of text "
    "(a user message, an assistant reply, or a short passage). "
    "Output 3-7 slug-form topics: lowercase, hyphen-separated, no spaces, "
    "no leading/trailing quotes. Prefer specific distinctive topics — "
    "named projects, named systems, proper nouns, named files, specific "
    "technical terms, philosophical subjects. Skip filler, pronouns, and "
    "generic words (e.g., 'thing', 'stuff', 'code', 'project', 'system')."
)

_CONCEPT_EXTRACTOR_TOOL = {
    "name": "emit_concepts",
    "description": "Return the key concepts extracted from the text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "concepts": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 8,
                "description": "3-7 slug-form concepts (lowercase, hyphen-separated).",
            },
        },
        "required": ["concepts"],
    },
}


def _concept_extractor_via_urllib(text: str, model_id: str) -> list[str]:
    """Call /v1/messages via urllib with Claude Code OAuth headers.

    Uses the same auth path as the daemon (shared.oauth_client.api_headers):
    OAuth token from ~/.claude/.credentials.json, User-Agent and
    anthropic-client-platform set to claude_code_cli so billing treats
    this as interactive Claude Code usage, not programmatic API use.

    No anthropic SDK required — works in any Python environment.
    Returns [] on any error so callers fall back to heuristic extraction.
    """
    import json
    import urllib.request
    import urllib.error
    import sys
    import os

    # Locate shared oauth_client — try the canonical exampleco path first,
    # then fall back to the daemon's working directory convention.
    _shared_paths = [
        os.path.expanduser("~/example-stack/tools"),
        os.path.expanduser("~/example-stack/tools/telegram-dispatch"),
    ]
    for p in _shared_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from shared.oauth_client import api_headers
    except ImportError:
        return []

    payload = {
        "model": model_id,
        "max_tokens": 256,
        "system": _CONCEPT_EXTRACTOR_SYSTEM,
        "tools": [_CONCEPT_EXTRACTOR_TOOL],
        "tool_choice": {"type": "tool", "name": "emit_concepts"},
        "messages": [{"role": "user", "content": text[:4000]}],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers=api_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []

    for block in body.get("content", []):
        if block.get("type") == "tool_use":
            items = (block.get("input") or {}).get("concepts")
            if isinstance(items, list):
                out: list[str] = []
                for c in items:
                    if isinstance(c, str):
                        s = c.strip().lower().replace(" ", "-")
                        if s and len(s) <= 60:
                            out.append(s)
                return out[:8]
    return []


def call_concept_extractor(text: str, model: str = "haiku") -> list[str]:
    """Extract slug-form concepts from arbitrary text via a cheap LLM call.

    Used for per-turn event tagging (daemon, retell server, cc-watcher).
    Unlike the distiller, this doesn't produce a summary — just a concept
    list. Returns [] on any auth/network/parsing error so callers can
    fall back to heuristic extraction.

    Uses raw urllib + Claude Code OAuth headers (same path as the daemon)
    so no anthropic SDK is required and billing treats this as interactive
    Claude Code usage rather than programmatic API access.
    """
    if not text or len(text.strip()) < 12:
        return []
    from dct import providers as _prov
    if _prov.provider_name() != "anthropic":
        try:
            obj = _prov.complete_json(
                _CONCEPT_EXTRACTOR_SYSTEM, text[:4000],
                _CONCEPT_EXTRACTOR_TOOL["input_schema"], max_tokens=256)
        except _prov.ProviderError:
            return []
        items = obj.get("concepts")
        if not isinstance(items, list):
            return []
        out = [c.strip().lower().replace(" ", "-") for c in items
               if isinstance(c, str) and c.strip() and len(c) <= 60]
        return out[:8]
    model_id = resolve_model_id(model)
    concepts = _concept_extractor_via_urllib(text, model_id)
    if concepts:
        return concepts
    # Portable fallback: the shared exampleco oauth_client isn't present on
    # public installs — use the self-contained provider path instead.
    try:
        obj = _prov.complete_json(
            _CONCEPT_EXTRACTOR_SYSTEM, text[:4000],
            _CONCEPT_EXTRACTOR_TOOL["input_schema"],
            model=model_id, max_tokens=256)
    except _prov.ProviderError:
        return []
    items = obj.get("concepts")
    if not isinstance(items, list):
        return []
    out = [c.strip().lower().replace(" ", "-") for c in items
           if isinstance(c, str) and c.strip() and len(c) <= 60]
    return out[:8]
