"""Tests for the codex-oauth provider backend (Build 104).

Everything runs against local mock servers — no network, no real tokens.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dct import codex_auth
from dct import providers as prov


# ── helpers ──────────────────────────────────────────────────────────────────

def _jwt(payload: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJub25lIn0.{body}.sig"


def _write_auth(tmp_path, *, expires_in_s=3600, refresh_token="rt-1",
                account_id="acct-42"):
    p = tmp_path / "auth.json"
    tok = _jwt({"chatgpt_account_id": account_id,
                "exp": int(time.time()) + expires_in_s})
    p.write_text(json.dumps({
        "tokens": {
            "access_token": tok,
            "refresh_token": refresh_token,
            "expires_at": int((time.time() + expires_in_s) * 1000),
        }
    }))
    return p


def _serve(handler_cls):
    srv = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _sse_body(text: str) -> bytes:
    events = [
        ("response.output_text.delta",
         {"type": "response.output_text.delta", "delta": text}),
        ("response.completed",
         {"type": "response.completed",
          "response": {"usage": {"input_tokens": 5, "output_tokens": 3}}}),
    ]
    out = []
    for name, obj in events:
        out.append(f"event: {name}\ndata: {json.dumps(obj)}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode()


# ── token store ──────────────────────────────────────────────────────────────

def test_store_missing_auth_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_CODEX_AUTH_PATH", str(tmp_path / "nope.json"))
    store = codex_auth.TokenStore()
    with pytest.raises(codex_auth.CodexAuthError, match="not found"):
        store.get_access_token()
    ok, detail = store.status()
    assert not ok and "not found" in detail


def test_store_valid_token_no_refresh(tmp_path, monkeypatch):
    p = _write_auth(tmp_path, expires_in_s=3600)
    monkeypatch.setenv("PDCT_CODEX_AUTH_PATH", str(p))
    store = codex_auth.TokenStore()
    tok = store.get_access_token()
    assert codex_auth.extract_account_id(tok) == "acct-42"
    ok, detail = store.status()
    assert ok and "valid" in detail


def test_store_expired_refreshes_and_persists(tmp_path, monkeypatch):
    p = _write_auth(tmp_path, expires_in_s=10)  # inside 60s margin

    new_access = _jwt({"chatgpt_account_id": "acct-42",
                       "exp": int(time.time()) + 7200})
    hits = []

    class _Refresh(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            hits.append(self.rfile.read(length).decode())
            body = json.dumps({
                "access_token": new_access,
                "refresh_token": "rt-2",
                "expires_at": int((time.time() + 7200) * 1000),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = _serve(_Refresh)
    monkeypatch.setenv("PDCT_CODEX_AUTH_PATH", str(p))
    monkeypatch.setattr(codex_auth, "TOKEN_URL",
                        f"http://127.0.0.1:{srv.server_port}/oauth/token")
    store = codex_auth.TokenStore()
    tok = store.get_access_token()
    srv.shutdown()

    assert tok == new_access
    assert "grant_type=refresh_token" in hits[0]
    assert "rt-1" in hits[0]
    # persisted back to disk with the rotated refresh token
    on_disk = json.loads(p.read_text())
    assert on_disk["tokens"]["access_token"] == new_access
    assert on_disk["tokens"]["refresh_token"] == "rt-2"
    assert (p.stat().st_mode & 0o777) == 0o600


def test_store_expired_no_refresh_token(tmp_path, monkeypatch):
    p = _write_auth(tmp_path, expires_in_s=10, refresh_token="")
    monkeypatch.setenv("PDCT_CODEX_AUTH_PATH", str(p))
    store = codex_auth.TokenStore()
    ok, detail = store.status()
    assert not ok and "no refresh_token" in detail


def test_store_refresh_http_error(tmp_path, monkeypatch):
    p = _write_auth(tmp_path, expires_in_s=10)

    class _Deny(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(400)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"no")

        def log_message(self, *a):
            pass

    srv = _serve(_Deny)
    monkeypatch.setenv("PDCT_CODEX_AUTH_PATH", str(p))
    monkeypatch.setattr(codex_auth, "TOKEN_URL",
                        f"http://127.0.0.1:{srv.server_port}/oauth/token")
    store = codex_auth.TokenStore()
    with pytest.raises(codex_auth.CodexAuthError, match="HTTP 400"):
        store.get_access_token()
    srv.shutdown()


# ── provider integration ─────────────────────────────────────────────────────

@pytest.fixture()
def codex_env(tmp_path, monkeypatch):
    p = _write_auth(tmp_path, expires_in_s=3600)
    monkeypatch.setenv("PDCT_CODEX_AUTH_PATH", str(p))
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "codex-oauth")
    monkeypatch.delenv("PDCT_LLM_MODEL", raising=False)
    # reset module-level store cache
    codex_auth._store = None
    yield tmp_path
    codex_auth._store = None


def _backend(text=None, status=200, capture=None):
    class _Backend(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            if capture is not None:
                capture.append({"body": req,
                                "headers": dict(self.headers.items())})
            if status != 200:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = _sse_body(text or "")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    return _Backend


def test_provider_available_codex(codex_env):
    ok, detail = prov.provider_available()
    assert ok and "codex" in detail


def test_complete_text_via_codex(codex_env, monkeypatch):
    seen: list = []
    srv = _serve(_backend(text="pong", capture=seen))
    monkeypatch.setenv("PDCT_CODEX_BASE_URL",
                       f"http://127.0.0.1:{srv.server_port}/responses")
    out = prov.complete_text("sys", "ping")
    srv.shutdown()
    assert out == "pong"
    req = seen[0]
    # first-party header shape + account routing
    assert req["headers"].get("User-Agent", "").startswith("OpenAI/Codex-CLI/")
    assert req["headers"].get("Chatgpt-Account-Id") == "acct-42"
    assert req["body"]["store"] is False and req["body"]["stream"] is True
    assert req["body"]["model"] == prov._CODEX_MODEL_DEFAULT


def test_complete_json_via_codex(codex_env, monkeypatch):
    srv = _serve(_backend(text='{"a": 1, "b": "x"}'))
    monkeypatch.setenv("PDCT_CODEX_BASE_URL",
                       f"http://127.0.0.1:{srv.server_port}/responses")
    obj = prov.complete_json("sys", "u", {"type": "object",
                                          "required": ["a", "b"]})
    srv.shutdown()
    assert obj == {"a": 1, "b": "x"}


def test_complete_json_missing_required_fields(codex_env, monkeypatch):
    srv = _serve(_backend(text='{"a": 1}'))
    monkeypatch.setenv("PDCT_CODEX_BASE_URL",
                       f"http://127.0.0.1:{srv.server_port}/responses")
    with pytest.raises(prov.ProviderError, match="missing required fields"):
        prov.complete_json("sys", "u", {"type": "object",
                                        "required": ["a", "b"]})
    srv.shutdown()


def test_401_triggers_refresh_and_retry(codex_env, monkeypatch, tmp_path):
    calls = {"n": 0}

    class _FirstDeny(BaseHTTPRequestHandler):
        def do_POST(self):
            calls["n"] += 1
            if calls["n"] == 1:
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = _sse_body("recovered")
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    new_access = _jwt({"chatgpt_account_id": "acct-42",
                       "exp": int(time.time()) + 7200})

    class _Refresh(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.dumps({"access_token": new_access,
                               "refresh_token": "rt-2"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    backend = _serve(_FirstDeny)
    refresher = _serve(_Refresh)
    monkeypatch.setenv("PDCT_CODEX_BASE_URL",
                       f"http://127.0.0.1:{backend.server_port}/responses")
    monkeypatch.setattr(codex_auth, "TOKEN_URL",
                        f"http://127.0.0.1:{refresher.server_port}/oauth/token")
    out = prov.complete_text("sys", "ping")
    backend.shutdown()
    refresher.shutdown()
    assert out == "recovered"
    assert calls["n"] == 2


def test_backend_5xx_is_provider_error(codex_env, monkeypatch):
    srv = _serve(_backend(status=502))
    monkeypatch.setenv("PDCT_CODEX_BASE_URL",
                       f"http://127.0.0.1:{srv.server_port}/responses")
    with pytest.raises(prov.ProviderError, match="HTTP 502"):
        prov.complete_text("sys", "ping")
    srv.shutdown()


def test_probe_endpoint_codex(codex_env, monkeypatch):
    srv = _serve(_backend(text="pong"))
    monkeypatch.setenv("PDCT_CODEX_BASE_URL",
                       f"http://127.0.0.1:{srv.server_port}/responses")
    ok, detail = prov.probe_endpoint()
    srv.shutdown()
    assert ok and "reachable" in detail


def test_probe_endpoint_codex_bad_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("PDCT_LLM_PROVIDER", "codex-oauth")
    monkeypatch.setenv("PDCT_CODEX_AUTH_PATH", str(tmp_path / "missing.json"))
    codex_auth._store = None
    ok, detail = prov.probe_endpoint()
    assert not ok and "not found" in detail
    codex_auth._store = None


# ── Codex round-2 findings ───────────────────────────────────────────────────

def test_default_store_recreated_on_env_change(tmp_path, monkeypatch):
    """F1: changing PDCT_CODEX_AUTH_PATH must yield a store bound to the
    new file WITHOUT anyone manually resetting the module global."""
    p1 = _write_auth(tmp_path, account_id="acct-one")
    p2dir = tmp_path / "two"
    p2dir.mkdir()
    p2 = _write_auth(p2dir, account_id="acct-two")

    monkeypatch.setenv("PDCT_CODEX_AUTH_PATH", str(p1))
    s1 = codex_auth.default_store()
    assert codex_auth.extract_account_id(s1.get_access_token()) == "acct-one"

    monkeypatch.setenv("PDCT_CODEX_AUTH_PATH", str(p2))
    s2 = codex_auth.default_store()
    assert s2 is not s1
    assert codex_auth.extract_account_id(s2.get_access_token()) == "acct-two"
    codex_auth._store = None


def test_codex_base_url_override_loopback_only(codex_env, monkeypatch):
    """F5: OAuth bearer tokens must never be sent to a non-loopback
    override URL."""
    monkeypatch.setenv("PDCT_CODEX_BASE_URL", "https://evil.example.com/v1")
    with pytest.raises(prov.ProviderError, match="loopback"):
        prov.complete_text("sys", "ping")


def test_codex_request_omits_max_output_tokens(codex_env, monkeypatch):
    """The ChatGPT Codex backend rejects max_output_tokens with HTTP 400
    (verified live on the real endpoint, 2026-07-04). The battle-tested
    daemon provider never sends it. Regression: keep it OFF the wire;
    max_tokens is advisory-only for this backend."""
    seen: list = []
    srv = _serve(_backend(text="ok", capture=seen))
    monkeypatch.setenv("PDCT_CODEX_BASE_URL",
                       f"http://127.0.0.1:{srv.server_port}/responses")
    prov.complete_text("sys", "ping", max_tokens=99)
    srv.shutdown()
    assert "max_output_tokens" not in seen[0]["body"]
    assert "max_tokens" not in seen[0]["body"]
