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

Config (pdct.env or exported):
    PDCT_LLM_PROVIDER   anthropic | openai-compatible   (default: anthropic)
    PDCT_LLM_BASE_URL   e.g. http://localhost:11434/v1  (openai-compatible)
    PDCT_LLM_MODEL      model name for the endpoint
    PDCT_LLM_API_KEY    bearer key if the endpoint needs one

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
    return False, f"unknown PDCT_LLM_PROVIDER={p!r}"


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
    key = os.environ.get("PDCT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
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


# ── public interface ────────────────────────────────────────────────────────

def _default_model(purpose: str) -> str:
    m = os.environ.get("PDCT_LLM_MODEL")
    if m:
        return m
    if provider_name() == "anthropic":
        from dct.llm import resolve_model_id
        return resolve_model_id("haiku")
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
    raise ProviderError(f"unknown PDCT_LLM_PROVIDER={p!r}")
