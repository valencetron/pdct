"""Activation validation CLI — Phase 4.

Three modes wrapping the existing ActivationEngine:
    snapshot (default) — single-timestamp heat dump
    --scrub           — trajectory table across the log
    --inspect SLUG    — single-concept detail report

Run: python -m dct.replay --log events.jsonl [options]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dct.activation import ActivationEngine, DecayConfig
from dct.event_log import EventLog
from dct.events import Event


_SLICE_RE = re.compile(r"^(\d+)([dhme])$")
_TIME_UNIT_SECONDS = {"d": 86400.0, "h": 3600.0, "m": 60.0}


def parse_slice_spec(spec: str) -> tuple[str, float]:
    if not isinstance(spec, str) or not spec:
        raise ValueError("slice spec must be a non-empty string like '1d', '6h', '30m', '500e'")
    m = _SLICE_RE.match(spec)
    if not m:
        raise ValueError(f"malformed slice spec: {spec!r} (expected e.g. '1d', '500e')")
    n_str, unit = m.group(1), m.group(2)
    n = int(n_str)
    if n <= 0:
        raise ValueError(f"slice count must be positive: {spec!r}")
    if unit == "e":
        return ("events", float(n))
    return ("time", n * _TIME_UNIT_SECONDS[unit])


def compute_slice_cuts(events: list[Event], spec: tuple[str, float]) -> list[float]:
    if not events:
        return []
    kind, value = spec
    last_ts = events[-1].ts
    if kind == "events":
        step = int(value)
        cuts: list[float] = []
        for idx in range(step - 1, len(events), step):
            cuts.append(events[idx].ts)
        if not cuts or cuts[-1] != last_ts:
            cuts.append(last_ts)
        return cuts
    # kind == "time"
    first_ts = events[0].ts
    step = value
    cuts = []
    k = 1
    while first_ts + k * step < last_ts:
        cuts.append(first_ts + k * step)
        k += 1
    cuts.append(last_ts)
    return cuts


BAR_THRESHOLDS: tuple[float, ...] = (0.25, 0.5, 0.75)
BAR_GLYPHS: tuple[str, ...] = ("▁▁▁▁", "▇▁▁▁", "▇▇▁▁", "▇▇▇▇")
_EMPTY_BAR = "    "


def render_bar(heat: float, min_heat: float) -> str:
    if heat < min_heat:
        return _EMPTY_BAR
    for i, threshold in enumerate(BAR_THRESHOLDS):
        if heat < threshold:
            return BAR_GLYPHS[i]
    return BAR_GLYPHS[-1]


def select_fixed_concepts(
    snapshots: list[tuple[float, dict[str, float]]],
    n: int,
) -> list[str]:
    if not snapshots or n <= 0:
        return []
    max_heat: dict[str, float] = {}
    for _, snap in snapshots:
        for concept, heat in snap.items():
            if heat > max_heat.get(concept, 0.0):
                max_heat[concept] = heat
    ranked = sorted(max_heat.items(), key=lambda kv: (-kv[1], kv[0]))
    return [c for c, _ in ranked[:n]]


def detect_new_ignitions(
    snap: dict[str, float],
    prior: dict[str, float],
    fixed: list[str],
    min_heat: float,
) -> list[str]:
    fixed_set = set(fixed)
    out: list[str] = []
    for concept, heat in snap.items():
        if heat < min_heat:
            continue
        if concept in fixed_set:
            continue
        prior_heat = prior.get(concept, 0.0)
        if prior_heat >= min_heat:
            continue
        out.append(concept)
    out.sort()
    return out


MIN_CONCEPT_WIDTH = 12
MAX_CONCEPT_WIDTH = 20
_LABEL_GUTTER = "  "
_COL_GUTTER = "  "


def format_scrub_label(cut_ts: float, slice_kind: str) -> str:
    if slice_kind == "time":
        return datetime.fromtimestamp(cut_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    return f"t={int(cut_ts)}"


def _column(concept: str) -> str:
    if len(concept) > MAX_CONCEPT_WIDTH:
        return concept[: MAX_CONCEPT_WIDTH - 1] + "…"
    return concept.ljust(MIN_CONCEPT_WIDTH)


def format_concept_header(concepts: list[str]) -> str:
    return _COL_GUTTER.join(_column(c) for c in concepts)


def _cell_text(heat: float, min_heat: float, format_mode: str) -> str:
    if format_mode == "numeric":
        if heat < min_heat:
            return "    "
        return f"{heat:.2f}"
    return render_bar(heat, min_heat)


def format_scrub_row(
    cut_ts: float,
    snap: dict[str, float],
    fixed: list[str],
    min_heat: float,
    slice_kind: str,
    format_mode: str,
) -> str:
    cells: list[str] = []
    for concept in fixed:
        heat = snap.get(concept, 0.0)
        text = _cell_text(heat, min_heat, format_mode)
        if len(concept) > MAX_CONCEPT_WIDTH:
            width = MAX_CONCEPT_WIDTH
        else:
            width = MIN_CONCEPT_WIDTH
        cells.append(text.ljust(width))
    label = format_scrub_label(cut_ts, slice_kind)
    return label + _LABEL_GUTTER + _COL_GUTTER.join(cells)


def run_snapshot(log: EventLog, config: DecayConfig, now: float, min_heat: float) -> int:
    eng = ActivationEngine.replay(log, config=config)
    snap = eng.snapshot(now=now, min_heat=min_heat)
    for concept, heat in snap.items():
        print(f"{concept}\t{heat:.4f}")
    return 0


def _build_snapshot_at(events: list[Event], cut_ts: float, config: DecayConfig) -> dict[str, float]:
    eng = ActivationEngine(config=config)
    for e in events:
        if e.ts > cut_ts:
            break
        eng.consume(e)
    return eng.snapshot(now=cut_ts, min_heat=0.0)  # unthresholded; caller filters


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def run_inspect(
    log: EventLog,
    config: DecayConfig,
    concept: str,
    now: float,
    window: float,
) -> int:
    events = list(log.read_all())
    eng = ActivationEngine.replay(log, config=config)
    last = eng.last_seen_ts(concept)
    if last is None:
        print(f"{concept}: never ignited")
        return 0
    heat = eng.heat(concept, now=now)
    age = now - last
    last_iso = datetime.fromtimestamp(last, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"concept: {concept}")
    print(f"last ignited: {last_iso} ({_fmt_age(age)} ago)")
    print(f"current heat: {heat:.2f} (half-life {config.half_life_seconds:.0f}s)")
    in_window = [
        e for e in events
        if concept in e.concepts and (now - window) < e.ts <= now
    ]
    print(f"ignitions in last {int(window)}s (window): {len(in_window)}")
    for e in sorted(in_window, key=lambda ev: ev.ts, reverse=True):
        ts_iso = datetime.fromtimestamp(e.ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        src_file = e.metadata.get("source_file", "?")
        turn_idx = e.metadata.get("turn_index", "?")
        src_val = e.source.value if hasattr(e.source, "value") else e.source
        print(f"  {ts_iso}  source={src_val}  source_file={src_file}#{turn_idx}")
    print("neighbors (blast radius disabled — hop-cap=0): none")
    return 0


def run_scrub(
    log: EventLog,
    config: DecayConfig,
    slice_spec: tuple[str, float],
    top_n: int,
    min_heat: float,
    format_mode: str,
) -> int:
    events = list(log.read_all())
    if not events:
        print("dct.replay: empty log", file=sys.stderr)
        return 0
    raw_cuts = compute_slice_cuts(events, slice_spec)
    # Prepend first_ts so the table always includes a row for the very first event.
    first_ts = events[0].ts
    cuts = ([first_ts] + raw_cuts) if (not raw_cuts or raw_cuts[0] != first_ts) else raw_cuts
    snapshots: list[tuple[float, dict[str, float]]] = [
        (ts, _build_snapshot_at(events, ts, config)) for ts in cuts
    ]
    fixed = select_fixed_concepts(snapshots, top_n)
    slice_kind = slice_spec[0]
    header = "slice".ljust(12) + _LABEL_GUTTER + format_concept_header(fixed)
    print(header)
    prior: dict[str, float] = {}
    for cut_ts, snap in snapshots:
        print(format_scrub_row(cut_ts, snap, fixed, min_heat, slice_kind, format_mode))
        new_ign = detect_new_ignitions(snap, prior, fixed, min_heat)
        if new_ign:
            print(f"  + new: [{', '.join(new_ign)}]")
        prior = snap
    return 0


DEFAULT_HALF_LIFE = 3600.0
DEFAULT_SLICE = "1d"
DEFAULT_TOP_N = 10
DEFAULT_MIN_HEAT = 0.01
DEFAULT_WINDOW = 86400.0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dct.replay",
                                description="DCT activation validation CLI (Phase 4)")
    p.add_argument("--log", required=True, help="path to JSONL event log")
    p.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE,
                   help=f"decay half-life in seconds (default {DEFAULT_HALF_LIFE})")
    p.add_argument("--min-heat", type=float, default=DEFAULT_MIN_HEAT,
                   help=f"heat floor for output (default {DEFAULT_MIN_HEAT})")
    p.add_argument("--hop-cap", type=int, default=0, help="blast radius hop cap (default 0)")
    p.add_argument("--falloff", type=float, default=0.5, help="blast radius falloff (default 0.5)")
    p.add_argument("--now", type=float, default=None,
                   help="unix ts for snapshot/inspect (default: last event ts)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--scrub", action="store_true", help="trajectory table mode")
    mode.add_argument("--inspect", metavar="SLUG",
                      help="concept detail mode for SLUG")
    p.add_argument("--slice", default=DEFAULT_SLICE,
                   help=f"scrub slice spec (default {DEFAULT_SLICE!r})")
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                   help=f"scrub fixed-column count (default {DEFAULT_TOP_N})")
    p.add_argument("--format", default="bars", choices=("bars", "numeric"),
                   help="scrub cell format (default 'bars')")
    p.add_argument("--window", type=float, default=DEFAULT_WINDOW,
                   help=f"inspect backward window seconds (default {DEFAULT_WINDOW})")
    return p


def _resolve_now(log: EventLog, explicit: float | None) -> float | None:
    if explicit is not None:
        return explicit
    events = list(log.read_all())
    if not events:
        return None
    return events[-1].ts


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"dct.replay: log not found: {args.log}", file=sys.stderr)
        return 1
    log = EventLog(log_path)
    config = DecayConfig(
        half_life_seconds=args.half_life,
        radius_hop_cap=args.hop_cap,
        radius_falloff=args.falloff,
    )
    if args.scrub:
        slice_spec = parse_slice_spec(args.slice)
        return run_scrub(log, config, slice_spec, args.top_n, args.min_heat, args.format)
    now = _resolve_now(log, args.now)
    if args.inspect is not None:
        if now is None:
            print(f"{args.inspect}: never ignited")
            return 0
        return run_inspect(log, config, args.inspect, now, args.window)
    # default: snapshot
    if now is None:
        return 0  # empty log → nothing to print
    return run_snapshot(log, config, now, args.min_heat)


if __name__ == "__main__":
    raise SystemExit(main())
