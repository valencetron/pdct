"""Sanity CLI: replay an event log and print a heat snapshot.

Usage:
    python -m dct --log events.jsonl --now 1713456789.0 --half-life 3600

Output: one "<concept>\t<heat>" line per non-cold concept, descending by heat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dct.activation import ActivationEngine, DecayConfig
from dct.event_log import EventLog


def main() -> int:
    parser = argparse.ArgumentParser(prog="dct")
    parser.add_argument("--log", required=True, type=Path, help="path to JSONL event log")
    parser.add_argument("--now", required=True, type=float, help="timestamp to evaluate heat at")
    parser.add_argument("--half-life", required=True, type=float, help="decay half-life in seconds")
    parser.add_argument("--hop-cap", type=int, default=0, help="blast radius hop cap (default 0)")
    parser.add_argument("--falloff", type=float, default=0.5, help="blast radius falloff (default 0.5)")
    parser.add_argument("--min-heat", type=float, default=0.01, help="snapshot heat threshold")
    args = parser.parse_args()

    if not args.log.exists():
        print(f"dct: log not found: {args.log}", file=sys.stderr)
        return 1

    log = EventLog(args.log)
    config = DecayConfig(
        half_life_seconds=args.half_life,
        radius_hop_cap=args.hop_cap,
        radius_falloff=args.falloff,
    )
    eng = ActivationEngine.replay(log, config=config)
    snap = eng.snapshot(now=args.now, min_heat=args.min_heat)
    for concept, heat in snap.items():
        print(f"{concept}\t{heat:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
