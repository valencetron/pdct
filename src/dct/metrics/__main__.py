"""Subcommand dispatcher for `python -m dct.metrics`."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dct.metrics",
        description="PDCT prelim metrics CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tokens = sub.add_parser("tokens", help="Token-cost panel")
    p_tokens.add_argument("--days", type=int, default=7)

    p_utility = sub.add_parser("utility", help="Surface-reuse rate panel")
    p_utility.add_argument("--days", type=int, default=7)

    p_ablation = sub.add_parser("ablation", help="Ablation comparison panel")
    p_ablation.add_argument("--days", type=int, default=7)

    args = parser.parse_args(argv)

    if args.cmd == "tokens":
        from . import tokens
        return tokens.run(days=args.days)
    elif args.cmd == "utility":
        try:
            from . import utility as utility_cmd
        except ImportError:
            print("utility CLI not yet implemented (Stage 2)", file=sys.stderr)
            return 2
        return utility_cmd.run(days=args.days)
    elif args.cmd == "ablation":
        try:
            from . import ablation as ablation_cmd
        except ImportError:
            print("ablation CLI not yet implemented (Stage 3)", file=sys.stderr)
            return 2
        return ablation_cmd.run(days=args.days)
    return 2


if __name__ == "__main__":
    sys.exit(main())
