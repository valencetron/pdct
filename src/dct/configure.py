"""`pdct configure` — provider setup front door.

Thin CLI layer over the provider status core (provider_status.py) and the
shared capability probe (providers.check_capability). Writes pdct.env via
an atomic comment-preserving upsert; never requires exported env vars.

Modes:
    pdct configure                      interactive (TTY) / detection report (non-TTY)
    pdct configure --provider X ...     non-interactive, agent/script friendly
    pdct configure --show [--json]      redacted diagnostics view
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import tempfile
from pathlib import Path

from dct import config as dctconfig
from dct import providers as prov
from dct.provider_status import detect_backends, BackendStatus

MANAGED_KEYS = ("PDCT_LLM_PROVIDER", "PDCT_LLM_BASE_URL", "PDCT_LLM_MODEL",
                "PDCT_LLM_API_KEY", "PDCT_LLM_API_KEY_ENV")


# ── pdct.env atomic upsert ──────────────────────────────────────────────────

def env_file_path() -> Path:
    home = Path(os.environ.get("PDCT_HOME") or (Path.home() / ".pdct")).expanduser()
    return home / "pdct.env"


def upsert_env(path: Path, updates: dict[str, str | None]) -> None:
    """Update KEY=VALUE lines in place, append new keys, preserve comments
    and unknown lines. None = remove the key. Atomic tempfile+rename, 0600."""
    lines: list[str] = []
    if path.exists():
        lines = path.read_text().splitlines()
    pending = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            if key in pending:
                val = pending.pop(key)
                if val is not None:
                    out.append(f"{key}={val}")
                continue  # None → drop the line
            # also swallow commented-out managed keys being set now
        elif stripped.startswith("#") and "=" in stripped:
            ck = stripped.lstrip("# ").split("=", 1)[0].strip()
            if ck in pending and pending[ck] is not None:
                out.append(f"{ck}={pending.pop(ck)}")
                continue
        out.append(line)
    for key, val in pending.items():
        if val is not None:
            out.append(f"{key}={val}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".pdct.env.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(out).rstrip("\n") + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_env_file(path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    if not path.exists():
        return vals
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            vals[k.strip()] = v.strip()
    return vals


def snapshot_overlay(path: Path) -> dict[str, str | None]:
    """Env overlay representing EXACTLY the just-written file: managed keys
    present in the file are applied; managed keys absent are REMOVED so
    shell exports can't shadow the new config during the probe."""
    vals = read_env_file(path)
    overlay: dict[str, str | None] = {}
    for k in MANAGED_KEYS:
        overlay[k] = vals.get(k)  # None removes
    # ambient OPENAI_API_KEY must not leak to a keyless/custom endpoint:
    # suppress it unless the written config explicitly references it
    if vals.get("PDCT_LLM_API_KEY_ENV") != "OPENAI_API_KEY" and \
            not vals.get("PDCT_LLM_API_KEY"):
        base = vals.get("PDCT_LLM_BASE_URL", "")
        if "api.openai.com" not in base:
            overlay["OPENAI_API_KEY"] = None
    return overlay


# ── redaction helpers ───────────────────────────────────────────────────────

def _key_status() -> dict:
    """Key presence + source, never the key itself (no prefixes either)."""
    if os.environ.get("PDCT_LLM_API_KEY"):
        return {"present": True, "source": "pdct.env/env",
                "length": len(os.environ["PDCT_LLM_API_KEY"])}
    ref = os.environ.get("PDCT_LLM_API_KEY_ENV")
    if ref and os.environ.get(ref):
        return {"present": True, "source": f"env:{ref}",
                "length": len(os.environ[ref])}
    if os.environ.get("OPENAI_API_KEY"):
        return {"present": True, "source": "env:OPENAI_API_KEY",
                "length": len(os.environ["OPENAI_API_KEY"])}
    return {"present": False, "source": "", "length": 0}


# ── --show ──────────────────────────────────────────────────────────────────

def cmd_show(args: argparse.Namespace) -> int:
    envf = env_file_path()
    cands = detect_backends(probe_local=True)
    active = prov.provider_name()
    info = {
        "provider": active,
        "model": os.environ.get("PDCT_LLM_MODEL", "(provider default)"),
        "base_url": os.environ.get("PDCT_LLM_BASE_URL", ""),
        "key": _key_status(),
        "pdct_env": str(envf) if args.paths else envf.name,
        "pdct_env_exists": envf.exists(),
        "pdct_home": (str(dctconfig.pdct_home()) if args.paths else "set"),
        "vault_root": ([str(v) for v in dctconfig.vault_roots()] if args.paths else "set"),
        "backends": [c.to_dict() for c in cands],
    }
    if not args.paths:
        for b in info["backends"]:
            b.pop("base_url", None)
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    print(f"provider   : {info['provider']}")
    print(f"model      : {info['model']}")
    if info["base_url"]:
        print(f"base_url   : {info['base_url']}")
    k = info["key"]
    print(f"api key    : {'present (' + k['source'] + ', len ' + str(k['length']) + ')' if k['present'] else 'none'}")
    print(f"pdct.env   : {info['pdct_env']}{'' if info['pdct_env_exists'] else ' (missing)'}")
    print("\ndetected backends:")
    for c in cands:
        flags = []
        if c.configured:
            flags.append("configured")
        if c.auth_valid:
            flags.append("auth")
        if c.reachable:
            flags.append("reachable")
        mark = "●" if (c.auth_valid or c.reachable) else "○"
        print(f"  {mark} {c.name:<12} [{', '.join(flags) or 'not available'}] {c.detail}")
    return 0


# ── configure (write) ───────────────────────────────────────────────────────

def _apply(provider: str, base_url: str | None, model: str | None,
           key: str | None, key_env: str | None) -> Path:
    envf = env_file_path()
    updates: dict[str, str | None] = {"PDCT_LLM_PROVIDER": provider}
    updates["PDCT_LLM_BASE_URL"] = base_url or None
    updates["PDCT_LLM_MODEL"] = model or None
    # always clear BOTH key fields first — switching providers or going
    # keyless must never leave a stale literal secret on disk
    updates["PDCT_LLM_API_KEY"] = None
    updates["PDCT_LLM_API_KEY_ENV"] = None
    if key:
        print("⚠ writing a literal secret into pdct.env (0600). Prefer "
              "--key-env NAME to reference an env var instead.",
              file=sys.stderr)
        updates["PDCT_LLM_API_KEY"] = key
    elif key_env:
        updates["PDCT_LLM_API_KEY_ENV"] = key_env
    upsert_env(envf, updates)
    return envf


def _probe(envf: Path) -> int:
    overlay = snapshot_overlay(envf)
    print("probing capability against the just-written config …")
    res = prov.check_capability(overlay)
    print(f"  provider   : {res.provider}")
    if res.model:
        print(f"  model      : {res.model}")
    print(f"  endpoint   : {'✅' if res.endpoint_ok else '❌'} {res.endpoint_detail}")
    print(f"  structured : {'✅' if res.structured_ok else '❌'} {res.structured_detail}")
    print(f"  concepts   : {'✅' if res.concepts_ok else '❌'} {res.concepts_detail}")
    print(f"  judge      : {'✅' if res.judge_ok else '❌'} {res.judge_detail}")
    if res.ok:
        print("✅ provider configured and capable — distillation enabled")
        return 0
    print("❌ provider not capable yet — fix the above, or run "
          "`pdct doctor --live` for the full diagnosis", file=sys.stderr)
    return 1


def cmd_configure(args: argparse.Namespace) -> int:
    if args.show:
        return cmd_show(args)

    if args.key and args.key_env:
        print("error: --key and --key-env are mutually exclusive",
              file=sys.stderr)
        return 2

    if args.provider:  # non-interactive
        if args.provider == "openai-compatible" and not args.base_url:
            print("error: --base-url required for openai-compatible "
                  "(e.g. https://api.openai.com/v1 or http://localhost:11434/v1)",
                  file=sys.stderr)
            return 2
        if args.provider == "openai-compatible" and not args.model:
            print("error: --model required for openai-compatible", file=sys.stderr)
            return 2
        envf = _apply(args.provider, args.base_url, args.model,
                      args.key, args.key_env)
        print(f"wrote {envf}")
        if args.no_probe:
            return 0
        return _probe(envf)

    # bare invocation
    cands = detect_backends(probe_local=True)
    if not sys.stdin.isatty():
        print("detected backends (non-interactive — use flags to configure):")
        for c in cands:
            mark = "●" if (c.auth_valid or c.reachable) else "○"
            print(f"  {mark} {c.name:<12} {c.detail}")
        print("\nusage: pdct configure --provider "
              "{anthropic|openai-compatible|codex-oauth} "
              "[--base-url URL --model M] [--key-env NAME | --key VALUE]")
        return 2

    # interactive
    usable = [c for c in cands if c.auth_valid or c.reachable]
    print("PDCT provider setup — detected on this machine:\n")
    for i, c in enumerate(cands, 1):
        mark = "●" if (c.auth_valid or c.reachable) else "○"
        print(f"  {i}. {mark} {c.name:<12} {c.detail}")
    print("\n(● = usable now, ○ = needs credentials)")
    default = next((str(i) for i, c in enumerate(cands, 1)
                    if c is (usable[0] if usable else None)), "1")
    choice = input(f"\nselect a backend [{default}]: ").strip() or default
    try:
        sel = cands[int(choice) - 1]
    except (ValueError, IndexError):
        print("invalid choice", file=sys.stderr)
        return 2

    base_url = sel.base_url or None
    model = None
    key = None
    key_env = None
    if sel.provider == "openai-compatible":
        base_url = input(f"base URL [{base_url or ''}]: ").strip() or base_url
        if not base_url:
            print("base URL required", file=sys.stderr)
            return 2
        model = input("model name: ").strip()
        if not model:
            print("model required", file=sys.stderr)
            return 2
        if sel.name == "openai" and os.environ.get("OPENAI_API_KEY"):
            key_env = "OPENAI_API_KEY"
            print("  using OPENAI_API_KEY from environment (referenced, not copied)")
        elif sel.source == "endpoint":
            pass  # local server, no key needed
        else:
            ref = input("env var holding the API key (blank to type it): ").strip()
            if ref:
                key_env = ref
            else:
                key = getpass.getpass("API key (hidden): ").strip() or None
    envf = _apply(sel.provider, base_url, model, key, key_env)
    print(f"\nwrote {envf}")
    if args.no_probe:
        return 0
    return _probe(envf)


def add_parser(sub) -> None:
    p = sub.add_parser("configure",
                       help="detect and configure the LLM provider")
    p.add_argument("--provider",
                   choices=["anthropic", "openai-compatible", "codex-oauth"])
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--key", help="API key literal (written to pdct.env, 0600 — "
                                 "prefer --key-env)")
    p.add_argument("--key-env", help="name of an env var holding the API key")
    p.add_argument("--no-probe", action="store_true",
                   help="skip the post-write capability probe")
    p.add_argument("--show", action="store_true",
                   help="show resolved provider diagnostics")
    p.add_argument("--json", action="store_true", help="JSON output (--show)")
    p.add_argument("--paths", action="store_true",
                   help="include full filesystem paths in --show output")
    p.set_defaults(fn=cmd_configure)
