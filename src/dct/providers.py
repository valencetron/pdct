"""LLM provider abstraction — PDCT works with whatever LLM the user has.

Two backends behind one interface:

  anthropic          (default) Anthropic API via OAuth (Claude Max/Pro
                     subscription — zero API key) or ANTHROPIC_API_KEY.
                     OAuth traffic sends the full first-party Claude Code
                     header shape so it is billed/treated as interactive
                     Claude Code usage.

  openai-compatible  any /v1/chat/completions endpoint: OpenAI, OpenRouter,
                     Groq, Together, and local models via Ollama or
                     LM Studio. Structured output is emulated with a
                     JSON-schema prompt + strict parse.

  codex-oauth        (experimental) ChatGPT Plus/Pro subscription via the
                     Codex CLI's OAuth login (~/.codex/auth.json). Speaks
                     the Responses API at chatgpt.com/backend-api/codex,
                     auto-refreshes tokens, sends the first-party Codex CLI
                     header shape. Zero API spend. Structured output is
                     emulated (prompt + parse), same as openai-compatible.

Config (pdct.env or exported):
    PDCT_LLM_PROVIDER      anthropic | openai-compatible | codex-oauth
    PDCT_LLM_BASE_URL      e.g. http://localhost:11434/v1 (openai-compatible)
    PDCT_LLM_MODEL         model name for the endpoint
    PDCT_LLM_API_KEY       bearer key if the endpoint needs one
    PDCT_LLM_API_KEY_ENV   name of another env var holding the key
                           (indirection — no secret written to pdct.env)
    PDCT_CODEX_AUTH_PATH   override ~/.codex/auth.json      (codex-oauth)
    PDCT_CODEX_BASE_URL    override backend URL (tests)     (codex-oauth)

The interface is deliberately tiny — the two shapes PDCT actually needs:
    complete_json(system, user, schema)  -> dict   (distiller-style)
    complete_text(system, user)          -> str    (judge-style)
Both raise ProviderError with an actionable message on failure.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class ProviderError(RuntimeError):
    """LLM call failed — message is actionable for the operator."""


# ── first-party Claude Code header shape (OAuth traffic only) ──────────────
# Mirrors the battle-tested shape from the ExampleCo stack: a claude-cli UA
# with a REAL version (local CLI → cached last-good → pinned constant,
# never 0.0.0 — that UA is a tell), anthropic-client-platform, and the
# oauth beta header. Without these, OAuth traffic gets flagged/rate-limited
# as non-first-party.

_UA_VERSION_CACHE = Path.home() / ".pdct" / "ua-version"
_UA_VERSION_PIN = "2.1.183"
_ua_cache: dict = {"ver": None, "ts": 0.0}


def _claude_cli_version() -> str:
    if _ua_cache["ver"] and time.time() - _ua_cache["ts"] < 3600:
        return _ua_cache["ver"]
    ver = ""
    try:
        r = subprocess.run(["claude", "--version"], capture_output=True,
                           text=True, timeout=5)
        m = re.search(r"(\d+\.\d+\.\d+)", r.stdout or "")
        if m:
            ver = m.group(1)
            try:
                _UA_VERSION_CACHE.parent.mkdir(parents=True, exist_ok=True)
                _UA_VERSION_CACHE.write_text(ver)
            except OSError:
                pass
    except (OSError, subprocess.TimeoutExpired):
        pass
    if not ver:
        try:
            ver = _UA_VERSION_CACHE.read_text().strip()
        except OSError:
            ver = ""
    if not re.fullmatch(r"\d+\.\d+\.\d+", ver or ""):
        ver = _UA_VERSION_PIN
    _ua_cache.update(ver=ver, ts=time.time())
    return ver


def first_party_headers(token: str) -> dict[str, str]:
    """Full Claude Code first-party header shape for OAuth bearer traffic."""
    ver = _claude_cli_version()
    return {
        "Authorization": f"Bearer {token}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "oauth-2025-04-20",
        "Content-Type": "application/json",
        "User-Agent": f"claude-cli/{ver} (external, cli)",
        "anthropic-client-platform": "claude_code_cli",
    }


# ── config resolution ───────────────────────────────────────────────────────

def provider_name() -> str:
    return os.environ.get("PDCT_LLM_PROVIDER", "anthropic").strip().lower()


def resolve_api_key() -> str | None:
    """Bearer key for openai-compatible endpoints, with indirection support.

    Order: PDCT_LLM_API_KEY (literal) → PDCT_LLM_API_KEY_ENV (name of
    another env var — lets `pdct configure --key-env NAME` avoid writing
    secrets into pdct.env) → OPENAI_API_KEY.
    """
    key = os.environ.get("PDCT_LLM_API_KEY")
    if key:
        return key
    ref = os.environ.get("PDCT_LLM_API_KEY_ENV")
    if ref:
        val = os.environ.get(ref)
        if val:
            return val
    return os.environ.get("OPENAI_API_KEY")


def provider_available() -> tuple[bool, str]:
    """(usable, detail) — can the configured provider be called at all?"""
    p = provider_name()
    if p == "anthropic":
        from dct import auth
        try:
            auth.load_oauth_token()
            return True, "anthropic credentials found"
        except auth.TokenLoadError:
            return False, ("no Claude credentials — set ANTHROPIC_API_KEY, log "
                           "into Claude Code, or configure PDCT_LLM_PROVIDER="
                           "openai-compatible")
    if p == "openai-compatible":
        base = os.environ.get("PDCT_LLM_BASE_URL", "")
        model = os.environ.get("PDCT_LLM_MODEL", "")
        if not base:
            return False, "PDCT_LLM_BASE_URL not set"
        if not model:
            return False, "PDCT_LLM_MODEL not set"
        return True, f"{base} model={model}"
    if p == "codex-oauth":
        from dct import codex_auth
        return codex_auth.default_store().status()
    return False, f"unknown PDCT_LLM_PROVIDER={p!r}"


def probe_endpoint(timeout: float = 5.0) -> tuple[bool, str]:
    """Actually reach the configured endpoint and validate auth.

    Distinguishes connectivity/auth failures from model-capability failures
    (doctor contract: llm.endpoint = "reachable + auth valid").
    """
    p = provider_name()
    usable, detail = provider_available()
    if not usable:
        return False, detail
    if p == "anthropic":
        try:
            _anthropic_request({
                "model": "claude-haiku-4-5", "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}],
            }, timeout=timeout)
            return True, "anthropic API reachable, auth valid"
        except ProviderError as e:
            msg = str(e)
            # 400s mean we reached the API and auth passed (bad request
            # shape is fine for a probe); 401/403 = auth failure.
            if "HTTP 400" in msg:
                return True, "anthropic API reachable, auth valid"
            return False, msg[:200]
    if p == "codex-oauth":
        # Token validity/refreshability is the auth probe; then a 1-token
        # round-trip proves the backend is reachable with this account.
        from dct import codex_auth
        try:
            codex_auth.default_store().get_access_token()
        except codex_auth.CodexAuthError as e:
            return False, f"codex-oauth: {e}"[:200]
        try:
            _codex_request("Reply with the single word: pong", "ping",
                           _codex_model(), 16, timeout=timeout)
            return True, "codex backend reachable, auth valid"
        except ProviderError as e:
            return False, str(e)[:200]
    # openai-compatible: hit /models (universally supported, cheap)
    base = os.environ.get("PDCT_LLM_BASE_URL", "").rstrip("/")
    headers = {}
    key = resolve_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"{base}/models", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True, f"{base} reachable, auth accepted"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, f"auth rejected by {base} (HTTP {e.code})"
        if e.code in (404, 405, 501):
            # Server doesn't implement /models — reachable is enough.
            return True, f"{base} reachable (HTTP {e.code} on /models probe)"
        return False, f"endpoint error: {base} (HTTP {e.code} on /models)"
    except (urllib.error.URLError, OSError) as e:
        return False, f"endpoint unreachable: {base} — {e}"


# ── anthropic backend ───────────────────────────────────────────────────────

def _anthropic_request(payload: dict, timeout: float = 60.0) -> dict:
    """POST /v1/messages via urllib — OAuth first-party headers or API key."""
    from dct import auth
    try:
        tok = auth.load_oauth_token()
    except auth.TokenLoadError as e:
        raise ProviderError(f"anthropic: no credentials ({e})") from e
    if auth.is_oauth_token(tok):
        headers = first_party_headers(tok)
    else:
        headers = {
            "x-api-key": tok,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise ProviderError(f"anthropic: HTTP {e.code} — {body}") from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise ProviderError(f"anthropic: {type(e).__name__}: {e}") from e


def _anthropic_json(system: str, user: str, schema: dict,
                    model: str, max_tokens: int) -> dict:
    tool = {"name": "emit", "description": "Emit the structured result.",
            "input_schema": schema}
    body = _anthropic_request({
        "model": model, "max_tokens": max_tokens, "system": system,
        "tools": [tool], "tool_choice": {"type": "tool", "name": "emit"},
        "messages": [{"role": "user", "content": user}],
    })
    for block in body.get("content", []):
        if block.get("type") == "tool_use":
            inp = block.get("input")
            if isinstance(inp, dict):
                return inp
    raise ProviderError("anthropic: response contained no tool_use block")


def _anthropic_text(system: str, user: str, model: str, max_tokens: int) -> str:
    body = _anthropic_request({
        "model": model, "max_tokens": max_tokens, "system": system,
        "messages": [{"role": "user", "content": user}],
    })
    parts = [b.get("text", "") for b in body.get("content", [])
             if b.get("type") == "text"]
    if not parts:
        raise ProviderError("anthropic: response contained no text block")
    return "".join(parts)


# ── openai-compatible backend ───────────────────────────────────────────────

def _openai_request(messages: list[dict], model: str, max_tokens: int,
                    timeout: float = 60.0, force_json: bool = False) -> str:
    base = os.environ.get("PDCT_LLM_BASE_URL", "").rstrip("/")
    if not base:
        raise ProviderError("openai-compatible: PDCT_LLM_BASE_URL not set")
    headers = {"Content-Type": "application/json"}
    key = resolve_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload: dict = {"model": model, "max_tokens": max_tokens,
                     "messages": messages}
    if force_json:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(f"{base}/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        # Some servers reject response_format — retry without it once.
        if force_json and e.code in (400, 422):
            payload.pop("response_format", None)
            req2 = urllib.request.Request(f"{base}/chat/completions",
                                          data=json.dumps(payload).encode(),
                                          headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req2, timeout=timeout) as resp:
                    body = json.loads(resp.read())
            except Exception as e2:  # noqa: BLE001
                raise ProviderError(f"openai-compatible: {e2}") from e2
        else:
            raise ProviderError(
                f"openai-compatible: HTTP {e.code} — {raw[:300]}") from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise ProviderError(f"openai-compatible: {type(e).__name__}: {e}") from e
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(f"openai-compatible: malformed response: {e}") from e


def _extract_json_object(text: str) -> dict:
    """Parse a JSON object out of model text (handles code fences + prose)."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # salvage the first balanced {...}
    start = t.find("{")
    depth = 0
    for i in range(start, len(t)) if start >= 0 else []:
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    break
    raise ProviderError(
        "openai-compatible: model did not return parseable JSON — "
        "it may be below PDCT's minimum capability (see INSTALL.md)")


def _openai_json(system: str, user: str, schema: dict,
                 model: str, max_tokens: int) -> dict:
    sys_prompt = (
        f"{system}\n\nRespond with ONLY a JSON object (no prose, no code "
        f"fences) that validates against this JSON schema:\n"
        f"{json.dumps(schema)}"
    )
    text = _openai_request(
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": user}],
        model, max_tokens, force_json=True)
    obj = _extract_json_object(text)
    missing = [k for k in schema.get("required", []) if k not in obj]
    if missing:
        raise ProviderError(
            f"openai-compatible: JSON missing required fields {missing} — "
            "model may be below PDCT's minimum capability")
    return obj


# ── codex-oauth backend (experimental) ──────────────────────────────────────

_CODEX_BACKEND_URL = "https://chatgpt.com/backend-api/codex/responses"
_CODEX_MODEL_DEFAULT = "gpt-5.5"
_CODEX_UA = "OpenAI/Codex-CLI/0.125.0"


def _codex_model() -> str:
    return os.environ.get("PDCT_LLM_MODEL") or _CODEX_MODEL_DEFAULT


def _codex_backend_url() -> str:
    """Backend URL. The override exists for tests only and is restricted to
    loopback — otherwise a poisoned pdct.env could redirect OAuth bearer
    tokens to an arbitrary server."""
    override = os.environ.get("PDCT_CODEX_BASE_URL")
    if not override:
        return _CODEX_BACKEND_URL
    from urllib.parse import urlparse
    host = (urlparse(override).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return override
    raise ProviderError(
        "codex-oauth: PDCT_CODEX_BASE_URL override is restricted to "
        f"loopback (got {host!r}) — refusing to send OAuth tokens there")


def _codex_sse_text(resp) -> str:
    """Collect output_text deltas from a Responses-API SSE stream."""
    buf: list[str] = []
    current_event = ""
    for raw in resp:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line:
            current_event = ""
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
            continue
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            etype = current_event or obj.get("type", "")
            if etype == "response.output_text.delta":
                buf.append(obj.get("delta", "") or "")
            elif etype == "error":
                raise ProviderError(
                    f"codex-oauth: stream error: {obj.get('message') or obj}")
    return "".join(buf)


def _codex_request(system: str, user: str, model: str, max_tokens: int,
                   timeout: float = 120.0) -> str:
    """One text round-trip via the ChatGPT/Codex Responses API (SSE)."""
    from dct import codex_auth
    store = codex_auth.default_store()
    try:
        token = store.get_access_token()
    except codex_auth.CodexAuthError as e:
        raise ProviderError(f"codex-oauth: {e}") from e

    body = json.dumps({
        "model": model,
        "input": [{"role": "user", "content": user}],
        "instructions": system or "You are a helpful assistant.",
        "store": False,       # required by backend
        "stream": True,       # required by backend
        # NOTE: the ChatGPT Codex backend rejects max_output_tokens with
        # HTTP 400 "Unsupported parameter" (verified live 2026-07-04).
        # max_tokens is advisory-only for this backend — do not send it.
        "reasoning": {"effort": "low", "summary": "auto"},
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }).encode()

    def _attempt(tok: str):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {tok}",
            "Accept": "text/event-stream",
            "User-Agent": _CODEX_UA,   # first-party Codex CLI shape
        }
        account_id = codex_auth.extract_account_id(tok)
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id
        req = urllib.request.Request(_codex_backend_url(), data=body,
                                     headers=headers, method="POST")
        return urllib.request.urlopen(req, timeout=timeout)

    try:
        resp = _attempt(token)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            # Force refresh and retry once (token may be freshly revoked).
            try:
                token = store.force_refresh_and_get()
                resp = _attempt(token)
            except (codex_auth.CodexAuthError, urllib.error.HTTPError) as e2:
                raise ProviderError(
                    f"codex-oauth: auth failed after refresh: {e2}") from e2
            except (urllib.error.URLError, OSError) as e2:
                raise ProviderError(f"codex-oauth: {e2}") from e2
        else:
            raw = e.read().decode("utf-8", errors="replace")[:300]
            raise ProviderError(f"codex-oauth: HTTP {e.code} — {raw}") from e
    except (urllib.error.URLError, OSError) as e:
        raise ProviderError(f"codex-oauth: network error: {e}") from e

    with resp:
        text = _codex_sse_text(resp)
    if not text.strip():
        raise ProviderError("codex-oauth: empty response from backend")
    return text


def _codex_json(system: str, user: str, schema: dict,
                model: str, max_tokens: int) -> dict:
    sys_prompt = (
        f"{system}\n\nRespond with ONLY a JSON object (no prose, no code "
        f"fences) that validates against this JSON schema:\n"
        f"{json.dumps(schema)}"
    )
    text = _codex_request(sys_prompt, user, model, max_tokens)
    try:
        obj = _extract_json_object(text)
    except ProviderError as e:
        raise ProviderError(str(e).replace("openai-compatible", "codex-oauth")) from e
    missing = [k for k in schema.get("required", []) if k not in obj]
    if missing:
        raise ProviderError(
            f"codex-oauth: JSON missing required fields {missing} — "
            "model may be below PDCT's minimum capability")
    return obj


# ── public interface ────────────────────────────────────────────────────────

def _default_model(purpose: str) -> str:
    m = os.environ.get("PDCT_LLM_MODEL")
    if m:
        return m
    if provider_name() == "anthropic":
        from dct.llm import resolve_model_id
        return resolve_model_id("haiku")
    if provider_name() == "codex-oauth":
        return _CODEX_MODEL_DEFAULT
    raise ProviderError("PDCT_LLM_MODEL must be set for openai-compatible")


def complete_json(system: str, user: str, schema: dict, *,
                  model: str | None = None, max_tokens: int = 2048) -> dict:
    """Structured completion → dict validated for required schema fields."""
    p = provider_name()
    m = model or _default_model("json")
    if p == "anthropic":
        return _anthropic_json(system, user, schema, m, max_tokens)
    if p == "openai-compatible":
        return _openai_json(system, user, schema, m, max_tokens)
    if p == "codex-oauth":
        return _codex_json(system, user, schema, m, max_tokens)
    raise ProviderError(f"unknown PDCT_LLM_PROVIDER={p!r}")


def complete_text(system: str, user: str, *,
                  model: str | None = None, max_tokens: int = 512) -> str:
    """Plain text completion."""
    p = provider_name()
    m = model or _default_model("text")
    if p == "anthropic":
        return _anthropic_text(system, user, m, max_tokens)
    if p == "openai-compatible":
        return _openai_request(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], m, max_tokens)
    if p == "codex-oauth":
        return _codex_request(system, user, m, max_tokens)
    raise ProviderError(f"unknown PDCT_LLM_PROVIDER={p!r}")


# ── capability probe (shared by configure + doctor stage 6) ─────────────────

class CapabilityResult:
    """Plain result object — no doctor Check formatting coupled in."""

    def __init__(self):
        self.endpoint_ok = False
        self.endpoint_detail = ""
        self.structured_ok = False
        self.structured_detail = ""
        self.concepts_ok = False
        self.concepts_detail = ""
        self.judge_ok = False
        self.judge_detail = ""
        self.provider = ""
        self.model = ""

    @property
    def ok(self) -> bool:
        # full minimum capability gate — same bar as doctor stage 6
        return (self.endpoint_ok and self.structured_ok
                and self.concepts_ok and self.judge_ok)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "provider": self.provider, "model": self.model,
                "endpoint": {"ok": self.endpoint_ok,
                             "detail": self.endpoint_detail},
                "structured": {"ok": self.structured_ok,
                               "detail": self.structured_detail},
                "concepts": {"ok": self.concepts_ok,
                             "detail": self.concepts_detail},
                "judge": {"ok": self.judge_ok,
                          "detail": self.judge_detail}}


class env_overlay:
    """Temporarily apply an env snapshot so probes run against a JUST-WRITTEN
    config instead of whatever the shell happens to export (exported vars
    must not shadow what `pdct configure` just wrote).

    ``overlay`` maps VAR → value; VAR → None means *remove* it.
    """

    def __init__(self, overlay: dict[str, str | None]):
        self.overlay = overlay
        self._saved: dict[str, str | None] = {}

    def __enter__(self):
        for k, v in self.overlay.items():
            self._saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, old in self._saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        return False


def check_capability(overlay: dict[str, str | None] | None = None,
                     timeout: float = 30.0) -> CapabilityResult:
    """Endpoint reachability + structured-JSON round-trip against the
    configured (or overlaid) provider. The minimum bar for distillation."""
    res = CapabilityResult()
    ctx = env_overlay(overlay) if overlay else None
    if ctx:
        ctx.__enter__()
    try:
        res.provider = provider_name()
        res.endpoint_ok, res.endpoint_detail = probe_endpoint(timeout=min(timeout, 10.0))
        if not res.endpoint_ok:
            res.structured_detail = "skipped — endpoint unreachable/auth invalid"
            return res
        schema = {"type": "object",
                  "properties": {"summary": {"type": "string"},
                                 "concepts": {"type": "array",
                                              "items": {"type": "string"}}},
                  "required": ["summary", "concepts"]}
        expected = {"pgvector", "vector-database", "hnsw", "benchmarking",
                    "retrieval", "vector-search", "recall", "database"}
        try:
            res.model = _default_model("json")
            obj = complete_json(
                "Distill this exchange into a note. concepts are lowercase "
                "hyphen-separated slugs.",
                "user: We benchmarked pgvector against a dedicated vector "
                "database and chose pgvector for operational simplicity.\n"
                "assistant: Sensible — HNSW index?\nuser: Yes, HNSW with "
                "m=16; recall at 10 was 0.94.",
                schema, max_tokens=512)
            res.structured_ok = isinstance(obj.get("concepts"), list)
            res.structured_detail = (f"valid JSON with {sorted(obj.keys())}"
                                     if res.structured_ok
                                     else "JSON returned but concepts missing")
            got = {str(c).strip().lower().replace(" ", "-")
                   for c in (obj.get("concepts") or [])}
            overlap = {g for g in got
                       if any(e in g or g in e for e in expected)}
            res.concepts_ok = len(overlap) >= 2
            res.concepts_detail = (f"matched {sorted(overlap)[:4]}"
                                   if res.concepts_ok else
                                   f"only matched {sorted(overlap)} — below "
                                   "minimum capability")
        except ProviderError as e:
            res.structured_detail = str(e)[:300]
            res.concepts_detail = "skipped — structured output failed"
        # judge round-trip (same bar as doctor llm.judge)
        try:
            text = complete_text(
                "You are a relevance judge. Respond with ONLY a JSON object "
                '{"score": <0-10 integer>, "rationale": "<one sentence>"}.',
                "Question: which vector database did the team choose?\n"
                "Retrieved note: The team benchmarked pgvector and chose it "
                "for operational simplicity.", max_tokens=128)
            t = text.strip()
            if t.startswith("```"):
                lines = t.splitlines()
                t = "\n".join(lines[1:-1] if lines[-1].strip() == "```"
                              else lines[1:]).strip()
            verdict = json.loads(t[t.find("{"):t.rfind("}") + 1])
            res.judge_ok = isinstance(verdict.get("score"), (int, float))
            res.judge_detail = f"verdict score={verdict.get('score')}"
        except (ProviderError, json.JSONDecodeError, ValueError) as e:
            res.judge_detail = f"{type(e).__name__}: {str(e)[:200]}"
        return res
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
