"""Provider status core — the shared truth model for configure/doctor/--show.

Typed, separated signals (a Codex plan-audit demand): *detected* (an auth
artifact exists on this machine), *configured* (current env would select this
backend), *auth_valid* (the credential actually resolves), *reachable*
(endpoint answered a cheap probe). Capability (can the model actually
distill?) is deliberately NOT claimed here — that is `providers.check_capability`,
a live round-trip.

Consumed by:
    pdct configure          (detection menu + post-write probe)
    pdct configure --show   (diagnostics view)
    pdct doctor             (stage 6 wraps the same helpers)
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Local endpoints worth sniffing (fast, 1s timeout each).
LOCAL_ENDPOINTS = (
    ("ollama", "http://localhost:11434", "/api/tags"),
    ("lm-studio", "http://localhost:1234", "/v1/models"),
)


@dataclass
class BackendStatus:
    name: str                 # human label: anthropic | openai | codex-oauth | ollama | lm-studio
    provider: str             # PDCT_LLM_PROVIDER value that would drive it
    detected: bool = False    # auth artifact / endpoint exists
    configured: bool = False  # current env selects this backend
    auth_valid: bool = False  # credential resolves (not just present)
    reachable: bool | None = None  # endpoint ping ok (None = not probed)
    detail: str = ""
    source: str = ""          # env | file | oauth | keychain | endpoint
    base_url: str = ""        # for openai-compatible candidates

    def to_dict(self) -> dict:
        return asdict(self)


def _probe_url(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def _anthropic_status(probe: bool) -> BackendStatus:
    """Delegate to the REAL auth resolver chain (credentials.json → keychain
    → stack.json → env) — never a parallel reimplementation."""
    from dct import auth
    st = BackendStatus(name="anthropic", provider="anthropic")
    try:
        tok = auth.load_oauth_token()
        st.detected = True
        st.auth_valid = bool(tok)
        # identify which source won, for --show (re-walk, cheap)
        for fetch, label in ((auth._try_credentials_json, "file"),
                             (auth._try_keychain, "keychain"),
                             (auth._try_stack_json, "file"),
                             (auth._try_env, "env")):
            try:
                if fetch():
                    st.source = label
                    break
            except Exception:  # noqa: BLE001
                continue
        st.detail = "credentials resolve via dct.auth chain"
    except auth.TokenLoadError:
        st.detail = "no Anthropic credentials (checked credentials.json, keychain, stack.json, env)"
    return st


def _openai_status() -> BackendStatus:
    st = BackendStatus(name="openai", provider="openai-compatible",
                       base_url="https://api.openai.com/v1", source="env")
    from dct import providers as prov
    key = prov.resolve_api_key()  # literal → indirection → OPENAI_API_KEY
    if key:
        st.detected = True
        st.auth_valid = True  # presence; live validation is the probe's job
        if os.environ.get("PDCT_LLM_API_KEY"):
            st.detail = "PDCT_LLM_API_KEY present"
        elif os.environ.get("PDCT_LLM_API_KEY_ENV"):
            st.detail = f"key via ${os.environ['PDCT_LLM_API_KEY_ENV']} (indirection)"
        else:
            st.detail = "OPENAI_API_KEY present"
    else:
        st.detail = "no API key (OPENAI_API_KEY / PDCT_LLM_API_KEY[_ENV])"
    return st


def _codex_status() -> BackendStatus:
    st = BackendStatus(name="codex-oauth", provider="codex-oauth", source="oauth")
    auth_path = Path(os.environ.get("PDCT_CODEX_AUTH_PATH",
                                    "~/.codex/auth.json")).expanduser()
    st.detected = auth_path.exists()
    if st.detected:
        try:
            from dct import codex_auth
            ok, detail = codex_auth.default_store().status()
            st.auth_valid = ok
            st.detail = detail
        except Exception as e:  # noqa: BLE001
            st.detail = f"auth.json present but store errored: {type(e).__name__}"
    else:
        st.detail = "no Codex CLI login (~/.codex/auth.json)"
    return st


def _local_statuses(probe: bool) -> list[BackendStatus]:
    out = []
    for name, base, path in LOCAL_ENDPOINTS:
        st = BackendStatus(name=name, provider="openai-compatible",
                           base_url=f"{base}/v1", source="endpoint")
        if probe:
            st.reachable = _probe_url(base + path)
            st.detected = bool(st.reachable)
            st.auth_valid = bool(st.reachable)  # local servers need no key
            st.detail = (f"{base} answering" if st.reachable
                         else f"{base} not answering")
        else:
            st.detail = f"{base} (not probed)"
        out.append(st)
    return out


def detect_backends(probe_local: bool = True) -> list[BackendStatus]:
    """Ordered candidate list. Marks whichever backend current env selects."""
    from dct import providers as prov
    active = prov.provider_name()
    active_base = os.environ.get("PDCT_LLM_BASE_URL", "").rstrip("/")

    cands = [_anthropic_status(probe_local), _codex_status(), _openai_status()]
    cands += _local_statuses(probe_local)

    for st in cands:
        if st.provider != active:
            continue
        if st.provider == "openai-compatible":
            # configured only if base_url matches (or env base is unset/custom)
            st.configured = bool(active_base) and st.base_url.rstrip("/") == active_base
        else:
            st.configured = True
    # custom openai-compatible endpoint from env that isn't in our list
    if active == "openai-compatible" and active_base and \
            not any(c.configured for c in cands):
        st = BackendStatus(name="custom", provider="openai-compatible",
                           base_url=active_base, source="env",
                           detected=True, configured=True,
                           detail=f"custom endpoint from env: {active_base}")
        cands.append(st)

    # order: configured first, then auth_valid, then detected
    cands.sort(key=lambda s: (not s.configured, not s.auth_valid, not s.detected))
    return cands


def best_candidate(cands: list[BackendStatus] | None = None) -> BackendStatus | None:
    """First usable backend (auth_valid, or reachable for local)."""
    for st in (cands or detect_backends()):
        if st.auth_valid or st.reachable:
            return st
    return None
