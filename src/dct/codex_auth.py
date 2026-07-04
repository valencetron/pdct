"""Codex OAuth token store — ChatGPT subscription auth for PDCT (experimental).

Reads the OAuth tokens written by the Codex CLI (``codex`` → sign in) at
``~/.codex/auth.json`` and keeps them fresh:

  - proactive refresh 60s before expiry via https://auth.openai.com/oauth/token
    (grant_type=refresh_token + the Codex CLI client_id)
  - refreshed tokens are written back atomically (0600) so the Codex CLI
    also picks them up
  - callers can force a refresh on a 401

This gives ChatGPT Plus/Pro subscribers distillation/judge capability with
zero API spend. Third-party OAuth use of an active Codex CLI login is the
same mechanism the CLI itself uses; see CONFIGURATION.md → codex-oauth.

Override the auth file location with ``PDCT_CODEX_AUTH_PATH``.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from threading import Lock

TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # Codex CLI's public client id
REFRESH_MARGIN_S = 60


class CodexAuthError(RuntimeError):
    """Codex OAuth failure — message is actionable for the operator."""


def auth_json_path() -> Path:
    override = os.environ.get("PDCT_CODEX_AUTH_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".codex" / "auth.json"


def decode_jwt_payload(token: str) -> dict:
    """Decode a JWT payload without signature verification (own token store)."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:  # noqa: BLE001
        return {}


def extract_account_id(access_token: str) -> str:
    """ChatGPT account id from the access-token JWT (Codex CLI fallbacks)."""
    payload = decode_jwt_payload(access_token)
    if payload.get("chatgpt_account_id"):
        return payload["chatgpt_account_id"]
    auth_block = payload.get("https://api.openai.com/auth") or {}
    if isinstance(auth_block, dict) and auth_block.get("chatgpt_account_id"):
        return auth_block["chatgpt_account_id"]
    orgs = payload.get("organizations") or []
    if orgs and isinstance(orgs[0], dict):
        return orgs[0].get("id", "")
    return ""


class TokenStore:
    """Thread-safe wrapper around the Codex CLI auth.json with auto-refresh."""

    def __init__(self, path: Path | None = None) -> None:
        self._lock = Lock()
        # Bind the path at construction time — env changes after this
        # create a NEW store via default_store(), never mutate this one.
        self._path = path if path is not None else auth_json_path()
        self._data: dict = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    # ── internals (caller must hold the lock) ────────────────────────────

    def _load(self) -> None:
        p = self.path
        if not p.exists():
            raise CodexAuthError(
                f"Codex auth file not found at {p}. Run 'codex' and sign in "
                "first (npm install -g @openai/codex; codex login)."
            )
        try:
            self._data = json.loads(p.read_text())
            self._loaded = True
        except (OSError, json.JSONDecodeError) as e:
            raise CodexAuthError(f"Failed to read {p}: {e}") from e

    def _tokens(self) -> dict:
        return self._data.get("tokens") or self._data

    def _expires_at(self) -> float:
        ea = self._tokens().get("expires_at")
        if ea:
            return float(ea) / 1000 if float(ea) > 1e12 else float(ea)
        at = self._tokens().get("access_token") or ""
        exp = decode_jwt_payload(at).get("exp") if at else None
        return float(exp) if exp else 0.0

    def _needs_refresh(self) -> bool:
        exp = self._expires_at()
        return exp != 0 and time.time() >= (exp - REFRESH_MARGIN_S)

    def _do_refresh(self) -> None:
        # Interprocess exclusion: PDCT and the Codex CLI (or two PDCT
        # processes) may refresh concurrently; refresh tokens rotate, so
        # a stale writer can clobber a newer token set. Hold an flock for
        # the whole reload → refresh → write-back sequence.
        import fcntl
        lock_path = self.path.with_suffix(".json.lock")
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            lock_fd = -1
        try:
            if lock_fd >= 0:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                # Another process may have refreshed while we waited —
                # reload from disk and re-check before spending our
                # (possibly now-invalid) refresh token.
                try:
                    self._load()
                except CodexAuthError:
                    pass
                if not self._needs_refresh():
                    return
            self._do_refresh_locked()
        finally:
            if lock_fd >= 0:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)

    def _do_refresh_locked(self) -> None:
        refresh_token = self._tokens().get("refresh_token") \
            or self._data.get("refresh_token")
        if not refresh_token:
            raise CodexAuthError(
                "No refresh_token in auth.json — re-run 'codex' and sign in."
            )
        body = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        }).encode()
        req = urllib.request.Request(
            TOKEN_URL, data=body, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                new_tokens = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:300]
            raise CodexAuthError(
                f"Token refresh failed HTTP {e.code}: {err}") from e
        except Exception as e:  # noqa: BLE001
            raise CodexAuthError(f"Token refresh network error: {e}") from e
        if "error" in new_tokens:
            raise CodexAuthError(f"Token refresh error: {new_tokens}")

        if "tokens" in self._data:
            self._data["tokens"].update(new_tokens)
        else:
            self._data.update(new_tokens)
        # Write back atomically; tmp file is 0600 from creation so tokens
        # are never world-readable even transiently or on a failed replace.
        p = self.path
        tmp = p.with_suffix(".json.tmp")
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                         0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(self._data, indent=2))
            os.replace(tmp, p)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass  # in-memory tokens still valid; disk write best-effort

    def _access_token(self) -> str:
        tok = self._tokens().get("access_token") \
            or self._data.get("access_token") or ""
        if not tok:
            raise CodexAuthError("No access_token in auth.json")
        return tok

    # ── public API ────────────────────────────────────────────────────────

    def get_access_token(self) -> str:
        with self._lock:
            if not self._loaded:
                self._load()
            if self._needs_refresh():
                self._do_refresh()
            return self._access_token()

    def force_refresh_and_get(self) -> str:
        with self._lock:
            if not self._loaded:
                self._load()
            self._do_refresh()
            return self._access_token()

    def status(self) -> tuple[bool, str]:
        """(usable, detail) without making a network call unless refresh
        is needed. Used by provider_available()."""
        try:
            with self._lock:
                if not self._loaded:
                    self._load()
                exp = self._expires_at()
            if exp and time.time() < (exp - REFRESH_MARGIN_S):
                remaining = int(exp - time.time())
                return True, f"codex token valid ({remaining}s remaining)"
            # expired or near expiry — usable iff a refresh_token exists
            with self._lock:
                has_refresh = bool(self._tokens().get("refresh_token")
                                   or self._data.get("refresh_token"))
            if has_refresh:
                return True, "codex token expiring — will refresh on use"
            return False, "codex token expired and no refresh_token present"
        except CodexAuthError as e:
            return False, str(e)


# Module-level default store (tests construct their own with a temp path).
_store: TokenStore | None = None
_store_lock = Lock()


def default_store() -> TokenStore:
    global _store
    with _store_lock:
        # Re-create if the env override changed (tests).
        if _store is None or _store.path != auth_json_path():
            _store = TokenStore()
        return _store
