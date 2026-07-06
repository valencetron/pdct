"""`pdct` — the PDCT command-line surface.

Subcommands:
    pdct init                        detect env, scaffold PDCT_HOME, write pdct.env
    pdct daemon start|stop|status|logs|restart
    pdct daemon install-service|uninstall-service   (launchd/systemd upgrade path)
    pdct doctor [--live] [--json]
    pdct ingest <path> [--source S]
    pdct recall "question" [--json]  shell-level retrieval — the integration
                                     surface for users, scripts, and agents

Everything resolves paths through dct.config, so PDCT_HOME (or pdct.env)
relocates the whole install.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dct import config as _cfg


# ── pdct.env loading (so `pdct` works without exporting vars manually) ─────

def _load_pdct_env() -> None:
    """Source $PDCT_HOME/pdct.env (or ~/.pdct/pdct.env) into os.environ.

    Explicit exported env vars win — the file only fills gaps.
    """
    candidates = []
    if os.environ.get("PDCT_HOME"):
        candidates.append(Path(os.environ["PDCT_HOME"]).expanduser() / "pdct.env")
    candidates.append(Path.home() / ".pdct" / "pdct.env")
    for envf in candidates:
        if not envf.is_file():
            continue
        for line in envf.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
        break  # first found wins


# ── init ────────────────────────────────────────────────────────────────────

def _detect_llm() -> tuple[str, str]:
    """Best available LLM auth → (provider, detail)."""
    if Path("~/.claude/.credentials.json").expanduser().exists():
        return "anthropic", "Claude Code OAuth credentials (~/.claude/.credentials.json)"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "ANTHROPIC_API_KEY env var"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai-compatible", "OPENAI_API_KEY env var (api.openai.com)"
    codex_auth = Path(os.environ.get("PDCT_CODEX_AUTH_PATH",
                                     "~/.codex/auth.json")).expanduser()
    if codex_auth.exists():
        return "codex-oauth", f"Codex CLI OAuth login ({codex_auth})"
    # local Ollama?
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1):
            return "openai-compatible", "local Ollama on :11434"
    except Exception:  # noqa: BLE001
        pass
    return "", "none found — retrieval-only mode (distillation disabled)"


def cmd_init(args: argparse.Namespace) -> int:
    home = Path(args.home or os.environ.get("PDCT_HOME")
                or (Path.home() / ".pdct")).expanduser()
    print(f"pdct init → {home}")
    for sub in ("vault/distillations", "runtime", "logs", "data"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    (home / "events.jsonl").touch()

    # vault detection
    vault = os.environ.get("OBSIDIAN_VAULT") or os.environ.get("PDCT_VAULT_ROOT")
    if vault:
        print(f"  vault: using {vault}")
    else:
        vault = str(home / "vault")
        print(f"  vault: none detected — using {vault} "
              f"(set OBSIDIAN_VAULT in pdct.env to use an Obsidian vault)")

    provider, detail = _detect_llm()
    print(f"  llm: {detail}")

    envf = home / "pdct.env"
    if envf.exists() and not args.force:
        print(f"  pdct.env exists, leaving as-is (--force to rewrite)")
    else:
        lines = [
            "# PDCT configuration — sourced by the pdct CLI (exported vars win)",
            f"PDCT_HOME={home}",
            f"PDCT_VAULT_ROOT={vault}",
            "# PDCT_SCHEDULER_INTERVAL=300",
            "# ── LLM provider (distillation/judge; optional) ──",
        ]
        # Never write an UNVERIFIED provider (Build 122): detection here is
        # file-existence only. `pdct configure --auto` probes live and writes
        # the provider only after the capability check passes.
        if provider:
            lines.append(f"# detected: {provider} — run `pdct configure --auto` to verify+enable")
        lines.append("# PDCT_LLM_PROVIDER=anthropic | openai-compatible | codex-oauth")
        lines += [
            "# PDCT_LLM_BASE_URL=http://localhost:11434/v1   # for openai-compatible",
            "# PDCT_LLM_MODEL=claude-3-5-haiku-20241022",
            "# PDCT_LLM_API_KEY=",
        ]
        envf.write_text("\n".join(lines) + "\n")
        print(f"  wrote {envf}")

    print("\nNext steps:")
    print(f"  export PDCT_HOME={home}")
    print("  pdct doctor          # verify the install")
    print("  pdct daemon start    # keep the write path alive")
    return 0


# ── daemon ──────────────────────────────────────────────────────────────────

def cmd_daemon(args: argparse.Namespace) -> int:
    from dct import supervisor as sup
    act = args.action
    if act == "start":
        ok, msg = sup.start_daemon()
        print(f"pdct daemon: {msg}")
        return 0 if ok else 1
    if act == "stop":
        ok, msg = sup.stop_daemon()
        print(f"pdct daemon: {msg}")
        return 0 if ok else 1
    if act == "restart":
        sup.stop_daemon()
        ok, msg = sup.start_daemon()
        print(f"pdct daemon: {msg}")
        return 0 if ok else 1
    if act == "status":
        st = sup.daemon_status()
        if args.json:
            print(json.dumps(st, indent=2))
        else:
            if st["running"]:
                inner = st.get("status", {})
                up = inner.get("uptime_s")
                sched = inner.get("scheduler", {})
                watcher = inner.get("watcher", {})
                print(f"pdct daemon: running (pid {st['pid']}, up {up}s)")
                print(f"  watcher: {'alive' if watcher.get('alive') else 'DOWN'}"
                      f" root={watcher.get('vault_root')}")
                print(f"  scheduler: ticks={sched.get('ticks')} "
                      f"last_rc={sched.get('last_rc')} interval={sched.get('interval_s')}s")
                print(f"  last_event_ts: {inner.get('last_event_ts')}")
            else:
                print("pdct daemon: not running")
        return 0 if st["running"] else 3
    if act == "logs":
        lp = sup.log_path()
        if not lp.exists():
            print(f"no log at {lp}")
            return 1
        lines = lp.read_text(errors="replace").splitlines()
        print("\n".join(lines[-args.lines:]))
        return 0
    if act in ("install-service", "uninstall-service"):
        from dct import service as svc
        fn = svc.install_service if act == "install-service" else svc.uninstall_service
        ok, msg = fn(dry_run=args.dry_run)
        print(msg)
        return 0 if ok else 1
    if act == "service-status":
        from dct import service as svc
        st = svc.service_status()
        if args.json:
            # CONTRACT: --json ALWAYS exits 0; callers (install.sh) branch on
            # ['state'], never on exit code — a set -e script must survive drift.
            print(json.dumps(st, indent=2))
            return 0
        state = st["state"]
        print(f"service: {state}  ({st.get('unit_path')})")
        for k, v in st.get("facts", {}).items():
            if k != "manager":
                print(f"  {k}: {v}")
        remedy = {
            "stale-interpreter": "run: pdct daemon install-service",
            "missing-interpreter": "run: pdct daemon install-service",
            "broken-interpreter": "run: pdct daemon install-service",
            "stale-env": "run: pdct daemon install-service",
            "installed-inactive": "run: pdct daemon install-service",
            "manager-unavailable": st.get("facts", {}).get("remedy", ""),
        }.get(state)
        if remedy:
            print(f"  → {remedy}")
        # installed-disabled only counts as OK when the operator intended it
        # (intent marker) — an unintentionally disabled service is drift.
        ok_states = {"healthy", "not-installed"}
        if state == "installed-disabled" and st.get("facts", {}).get("intentional"):
            ok_states.add("installed-disabled")
        return 0 if state in ok_states else 1
    print(f"unknown daemon action: {act}", file=sys.stderr)
    return 2


# ── doctor / ingest / recall ────────────────────────────────────────────────

def cmd_doctor(args: argparse.Namespace) -> int:
    from dct import doctor
    return doctor.run(json_out=args.json, live=args.live, corpus=args.corpus)


def cmd_ingest(args: argparse.Namespace) -> int:
    import subprocess
    cmd = [sys.executable, "-m", "dct.ingest",
           "--source", args.source, "--input", str(args.path),
           "--log", str(_cfg.events_path()), "--dedupe"]
    return subprocess.run(cmd).returncode


def cmd_recall(args: argparse.Namespace) -> int:
    """Shell-level retrieval — the talker surface for users & their agents."""
    from dct.retrieval.memory_api import query_memory, _row_to_dict
    rows = query_memory(args.question, _surface="pdct-cli")
    if args.json:
        print(json.dumps({"rows": [_row_to_dict(r) for r in rows]},
                         ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("(no matches)")
        return 3
    for r in rows[:args.top]:
        print(f"• [{r.date}] {r.title}  (id={r.id})")
        gist = (r.gist or "").strip()
        if gist:
            print(f"    {gist[:240]}")
    return 0


# ── parser ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pdct",
                                 description="PDCT — path-dependent context traversal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="detect environment and scaffold PDCT_HOME")
    p.add_argument("--home", help="PDCT_HOME location (default ~/.pdct)")
    p.add_argument("--force", action="store_true", help="rewrite pdct.env")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("daemon", help="supervisor control")
    p.add_argument("action", choices=["start", "stop", "restart", "status",
                                      "logs", "install-service",
                                      "uninstall-service", "service-status"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--lines", type=int, default=40)
    p.add_argument("--dry-run", action="store_true",
                   help="print service files without installing")
    p.set_defaults(fn=cmd_daemon)

    p = sub.add_parser("doctor", help="self-diagnosis")
    p.add_argument("--json", action="store_true")
    p.add_argument("--live", action="store_true")
    p.add_argument("--corpus", type=Path, default=None)
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("ingest", help="ingest transcript file(s) into the event log")
    p.add_argument("path", help="file or glob of transcripts")
    p.add_argument("--source", default="claude-code",
                   choices=["claude-code", "telegram", "voice"])
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("recall", help="query memory from the shell")
    p.add_argument("question")
    p.add_argument("--json", action="store_true")
    p.add_argument("--top", type=int, default=5)
    p.set_defaults(fn=cmd_recall)

    from dct.tuning.cli import register as _register_tune
    _register_tune(sub)

    from dct.configure import add_parser as _register_configure
    _register_configure(sub)

    return ap


def main(argv: list[str] | None = None) -> int:
    _load_pdct_env()
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
