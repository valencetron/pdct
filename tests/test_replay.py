import pytest

from dct.events import Event, EventOp, EventSource
from dct.replay import compute_slice_cuts, parse_slice_spec


def test_parse_slice_spec_days():
    assert parse_slice_spec("1d") == ("time", 86400.0)
    assert parse_slice_spec("7d") == ("time", 604800.0)


def test_parse_slice_spec_hours():
    assert parse_slice_spec("6h") == ("time", 21600.0)


def test_parse_slice_spec_minutes():
    assert parse_slice_spec("30m") == ("time", 1800.0)


def test_parse_slice_spec_events():
    assert parse_slice_spec("500e") == ("events", 500.0)
    assert parse_slice_spec("1e") == ("events", 1.0)


def test_parse_slice_spec_rejects_unknown_unit():
    with pytest.raises(ValueError):
        parse_slice_spec("1w")


def test_parse_slice_spec_rejects_empty():
    with pytest.raises(ValueError):
        parse_slice_spec("")


def test_parse_slice_spec_rejects_garbage():
    with pytest.raises(ValueError):
        parse_slice_spec("foo")


def test_parse_slice_spec_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        parse_slice_spec("0d")
    with pytest.raises(ValueError):
        parse_slice_spec("-1h")


def _ev(ts: float, *concepts: str, source_file: str = "f", turn_index: int = 0) -> Event:
    return Event(
        ts=ts,
        source=EventSource.CLAUDE_CODE,
        op=EventOp.WRITE,
        concepts=list(concepts) if concepts else ["x"],
        metadata={"source_file": source_file, "turn_index": turn_index},
    )


def test_compute_slice_cuts_empty_events():
    assert compute_slice_cuts([], ("time", 86400.0)) == []
    assert compute_slice_cuts([], ("events", 10.0)) == []


def test_compute_slice_cuts_time_exact_multiple():
    # Events span exactly 3 days starting at ts=0.
    evs = [_ev(0.0), _ev(86400.0), _ev(172800.0), _ev(259200.0)]
    # With 1-day slice: cuts at day 1, day 2, day 3. Last cut = last ts.
    assert compute_slice_cuts(evs, ("time", 86400.0)) == [86400.0, 172800.0, 259200.0]


def test_compute_slice_cuts_time_partial_final_slice():
    # Events span 2.5 days; final cut clamps to last ts.
    evs = [_ev(0.0), _ev(86400.0), _ev(216000.0)]  # 0, +1d, +2.5d
    cuts = compute_slice_cuts(evs, ("time", 86400.0))
    assert cuts == [86400.0, 172800.0, 216000.0]


def test_compute_slice_cuts_time_single_event():
    # A single event gives exactly one cut at its ts.
    assert compute_slice_cuts([_ev(1000.0)], ("time", 86400.0)) == [1000.0]


def test_compute_slice_cuts_events_even_division():
    evs = [_ev(float(i)) for i in range(10)]
    # 3-event slice → cuts at events[2], [5], [8], and final at last event.
    assert compute_slice_cuts(evs, ("events", 3.0)) == [2.0, 5.0, 8.0, 9.0]


def test_compute_slice_cuts_events_exact_division():
    evs = [_ev(float(i)) for i in range(9)]
    # 3-event slice, 9 events: cuts at events[2], [5], [8]; last cut already = last ts.
    assert compute_slice_cuts(evs, ("events", 3.0)) == [2.0, 5.0, 8.0]


def test_compute_slice_cuts_events_slice_larger_than_log():
    evs = [_ev(0.0), _ev(1.0), _ev(2.0)]
    assert compute_slice_cuts(evs, ("events", 10.0)) == [2.0]


from dct.replay import render_bar


def test_render_bar_below_min_heat_is_spaces():
    assert render_bar(0.0, min_heat=0.01) == "    "
    assert render_bar(0.005, min_heat=0.01) == "    "


def test_render_bar_low_bucket():
    assert render_bar(0.01, min_heat=0.01) == "▁▁▁▁"
    assert render_bar(0.24, min_heat=0.01) == "▁▁▁▁"


def test_render_bar_quarter_bucket():
    assert render_bar(0.25, min_heat=0.01) == "▇▁▁▁"
    assert render_bar(0.49, min_heat=0.01) == "▇▁▁▁"


def test_render_bar_half_bucket():
    assert render_bar(0.5, min_heat=0.01) == "▇▇▁▁"
    assert render_bar(0.74, min_heat=0.01) == "▇▇▁▁"


def test_render_bar_top_bucket():
    assert render_bar(0.75, min_heat=0.01) == "▇▇▇▇"
    assert render_bar(1.0, min_heat=0.01) == "▇▇▇▇"


def test_render_bar_always_four_chars_wide():
    for h in (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0):
        assert len(render_bar(h, min_heat=0.01)) == 4


from dct.replay import select_fixed_concepts


def test_select_fixed_concepts_uses_max_across_slices():
    # `beta` is briefly hot (0.9) in one slice; `alpha` is mid (0.6, 0.6) throughout.
    # With n=1, beta should win on max.
    snaps = [
        (1.0, {"alpha": 0.6, "beta": 0.9}),
        (2.0, {"alpha": 0.6, "beta": 0.1}),
    ]
    assert select_fixed_concepts(snaps, n=1) == ["beta"]


def test_select_fixed_concepts_tie_break_alpha():
    snaps = [(1.0, {"alpha": 0.5, "beta": 0.5})]
    assert select_fixed_concepts(snaps, n=2) == ["alpha", "beta"]


def test_select_fixed_concepts_honors_n():
    snaps = [(1.0, {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6})]
    assert select_fixed_concepts(snaps, n=2) == ["a", "b"]


def test_select_fixed_concepts_empty_snapshots():
    assert select_fixed_concepts([], n=10) == []


def test_select_fixed_concepts_fewer_concepts_than_n():
    snaps = [(1.0, {"only": 0.5})]
    assert select_fixed_concepts(snaps, n=10) == ["only"]


from dct.replay import detect_new_ignitions


def test_detect_new_ignitions_first_slice_all_non_fixed():
    # No prior. Everything hot and outside fixed is "new".
    fixed = ["alpha"]
    snap = {"alpha": 0.9, "beta": 0.5, "gamma": 0.3}
    prior: dict[str, float] = {}
    assert detect_new_ignitions(snap, prior, fixed, min_heat=0.01) == ["beta", "gamma"]


def test_detect_new_ignitions_cold_concepts_ignored():
    fixed = ["alpha"]
    snap = {"alpha": 0.9, "beta": 0.005}  # beta below min_heat
    assert detect_new_ignitions(snap, {}, fixed, min_heat=0.01) == []


def test_detect_new_ignitions_prior_hot_suppresses():
    fixed: list[str] = []
    # beta was already hot in the prior slice → not a new ignition.
    assert detect_new_ignitions({"beta": 0.5}, {"beta": 0.5}, fixed, min_heat=0.01) == []


def test_detect_new_ignitions_prior_cold_permits():
    fixed: list[str] = []
    # beta was below min_heat in the prior slice → new ignition this slice.
    assert detect_new_ignitions({"beta": 0.5}, {"beta": 0.005}, fixed, min_heat=0.01) == ["beta"]


def test_detect_new_ignitions_sorted_alphabetically():
    fixed: list[str] = []
    snap = {"zeta": 0.5, "alpha": 0.5, "mu": 0.5}
    assert detect_new_ignitions(snap, {}, fixed, min_heat=0.01) == ["alpha", "mu", "zeta"]


from dct.replay import format_concept_header, format_scrub_label, format_scrub_row


def test_format_scrub_label_time_uses_iso_date():
    # 1744300800 = 2025-04-10T16:00:00Z → date = 2025-04-10 UTC
    label = format_scrub_label(1744300800.0, slice_kind="time")
    assert label == "2025-04-10"


def test_format_scrub_label_events_uses_ts_prefix():
    label = format_scrub_label(1744300800.0, slice_kind="events")
    assert label == "t=1744300800"


def test_format_concept_header_min_width():
    # Short slugs padded to MIN_CONCEPT_WIDTH (12) + 2-space gutter each.
    header = format_concept_header(["a", "bb"])
    # "a" + 11 spaces + 2-space gap + "bb" + 10 spaces
    assert header == "a           " + "  " + "bb          "


def test_format_concept_header_truncates_long_slugs():
    # Slug longer than MAX_CONCEPT_WIDTH (20) gets "…" suffix.
    long = "a" * 25
    header = format_concept_header([long])
    assert header == "a" * 19 + "…"
    assert len(header) == 20


def test_format_scrub_row_time_bars_format():
    fixed = ["alpha", "beta"]
    snap = {"alpha": 0.9, "beta": 0.1}
    # 1744300800 = 2025-04-10
    row = format_scrub_row(1744300800.0, snap, fixed, min_heat=0.01,
                           slice_kind="time", format_mode="bars")
    # label (10 chars: "2025-04-10") + 2-space gutter + alpha bar (▇▇▇▇ in 12-wide field) + 2-space gap + beta bar (▁▁▁▁ in 12-wide)
    # Bar glyph in a 12-wide column is left-aligned, padded with spaces.
    expected = "2025-04-10" + "  " + "▇▇▇▇        " + "  " + "▁▁▁▁        "
    assert row == expected


def test_format_scrub_row_numeric_format():
    fixed = ["alpha"]
    snap = {"alpha": 0.734}
    row = format_scrub_row(1744300800.0, snap, fixed, min_heat=0.01,
                           slice_kind="time", format_mode="numeric")
    # Numeric cell: 2-decimal, left-aligned in 12-wide column.
    expected = "2025-04-10" + "  " + "0.73        "
    assert row == expected


def test_format_scrub_row_missing_concept_is_empty_cell():
    fixed = ["alpha", "beta"]
    snap = {"alpha": 0.9}  # beta absent
    row = format_scrub_row(1744300800.0, snap, fixed, min_heat=0.01,
                           slice_kind="time", format_mode="bars")
    # beta cell is "    " bar padded to 12 wide = 12 spaces.
    expected = "2025-04-10" + "  " + "▇▇▇▇        " + "  " + "            "
    assert row == expected


from pathlib import Path

from dct.activation import DecayConfig
from dct.event_log import EventLog
from dct.replay import run_snapshot


def _seed_log(path: Path, events: list) -> None:
    log = EventLog(path)
    for e in events:
        log.append(e)


def test_run_snapshot_prints_concept_heat(tmp_path, capsys):
    log_path = tmp_path / "events.jsonl"
    _seed_log(log_path, [
        _ev(100.0, "alpha"),
        _ev(200.0, "beta", turn_index=1),
    ])
    log = EventLog(log_path)
    config = DecayConfig(half_life_seconds=3600.0)
    rc = run_snapshot(log, config, now=200.0, min_heat=0.01)
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    # beta is fresh (1.0), alpha decayed ~1.8% over 100s at 3600s half-life → ~0.98
    assert out[0].startswith("beta\t1.0000")
    assert out[1].startswith("alpha\t0.98")


def test_run_snapshot_empty_log_prints_nothing(tmp_path, capsys):
    log_path = tmp_path / "empty.jsonl"
    log_path.touch()
    log = EventLog(log_path)
    config = DecayConfig(half_life_seconds=3600.0)
    rc = run_snapshot(log, config, now=100.0, min_heat=0.01)
    assert rc == 0
    assert capsys.readouterr().out == ""


from dct.replay import run_scrub


def test_run_scrub_emits_header_and_rows(tmp_path, capsys):
    # Three 1-day-apart events, 1-day slice → 3 rows.
    base = 1744300800.0  # 2025-04-10T16:00:00Z
    log_path = tmp_path / "events.jsonl"
    _seed_log(log_path, [
        _ev(base, "alpha"),
        _ev(base + 86400.0, "beta", turn_index=1),
        _ev(base + 172800.0, "gamma", turn_index=2),
    ])
    log = EventLog(log_path)
    config = DecayConfig(half_life_seconds=86400.0)  # 1-day half-life
    rc = run_scrub(log, config, slice_spec=("time", 86400.0),
                   top_n=3, min_heat=0.01, format_mode="bars")
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    # First line is header ("slice" + fixed concepts).
    assert lines[0].startswith("slice")
    # Three date rows follow.
    date_rows = [l for l in lines if l.startswith("2025-04-")]
    assert len(date_rows) == 3


def test_run_scrub_empty_log(tmp_path, capsys):
    log_path = tmp_path / "empty.jsonl"
    log_path.touch()
    log = EventLog(log_path)
    config = DecayConfig(half_life_seconds=3600.0)
    rc = run_scrub(log, config, slice_spec=("time", 86400.0),
                   top_n=10, min_heat=0.01, format_mode="bars")
    assert rc == 0
    err = capsys.readouterr().err
    assert "empty log" in err


def test_run_scrub_new_ignitions_line_when_present(tmp_path, capsys):
    # Two slices, top_n=1, a second concept ignites only in slice 2.
    base = 1744300800.0
    log_path = tmp_path / "events.jsonl"
    _seed_log(log_path, [
        _ev(base, "alpha"),
        _ev(base + 86400.0, "alpha", "beta", turn_index=1),
    ])
    log = EventLog(log_path)
    config = DecayConfig(half_life_seconds=86400.0)
    rc = run_scrub(log, config, slice_spec=("time", 86400.0),
                   top_n=1, min_heat=0.01, format_mode="bars")
    assert rc == 0
    out = capsys.readouterr().out
    assert "+ new: [beta]" in out


from dct.replay import run_inspect


def test_run_inspect_never_ignited(tmp_path, capsys):
    log_path = tmp_path / "events.jsonl"
    _seed_log(log_path, [
        _ev(100.0, "alpha"),
    ])
    log = EventLog(log_path)
    config = DecayConfig(half_life_seconds=3600.0)
    rc = run_inspect(log, config, concept="zzz", now=100.0, window=86400.0)
    assert rc == 0
    assert "zzz: never ignited" in capsys.readouterr().out


def test_run_inspect_lists_recent_ignitions_in_window(tmp_path, capsys):
    base = 1000.0
    log_path = tmp_path / "events.jsonl"
    _seed_log(log_path, [
        _ev(base,          "voice", source_file="a.jsonl", turn_index=3),
        _ev(base + 1000.0, "voice", source_file="b.json",  turn_index=7),
        _ev(base + 2000.0, "voice", source_file="c.jsonl", turn_index=11),
    ])
    log = EventLog(log_path)
    config = DecayConfig(half_life_seconds=3600.0)
    rc = run_inspect(log, config, concept="voice", now=base + 2000.0, window=86400.0)
    assert rc == 0
    out = capsys.readouterr().out
    assert "concept: voice" in out
    assert "current heat:" in out
    assert "ignitions in last" in out
    # All 3 ignitions visible within 24h window (span = 2000s).
    assert "a.jsonl#3" in out
    assert "b.json#7" in out
    assert "c.jsonl#11" in out


def test_run_inspect_window_excludes_old_ignitions(tmp_path, capsys):
    base = 1000.0
    log_path = tmp_path / "events.jsonl"
    _seed_log(log_path, [
        _ev(base,          "voice", source_file="old.json", turn_index=0),
        _ev(base + 5000.0, "voice", source_file="new.json", turn_index=1),
    ])
    log = EventLog(log_path)
    config = DecayConfig(half_life_seconds=3600.0)
    # window = 3000s; only the second ignition is within (now - 3000, now].
    rc = run_inspect(log, config, concept="voice", now=base + 5000.0, window=3000.0)
    assert rc == 0
    out = capsys.readouterr().out
    assert "new.json#1" in out
    assert "old.json#0" not in out


from dct.replay import main


def test_main_snapshot_default(tmp_path, capsys):
    log_path = tmp_path / "events.jsonl"
    _seed_log(log_path, [
        _ev(100.0, "alpha"),
    ])
    rc = main(["--log", str(log_path), "--now", "100.0", "--half-life", "3600"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("alpha\t1.0000")


def test_main_now_defaults_to_last_event_ts(tmp_path, capsys):
    log_path = tmp_path / "events.jsonl"
    _seed_log(log_path, [
        _ev(100.0, "alpha"),
        _ev(500.0, "beta", turn_index=1),
    ])
    rc = main(["--log", str(log_path), "--half-life", "3600"])
    assert rc == 0
    out = capsys.readouterr().out
    # beta fresh at now=500, alpha 400s old.
    assert out.splitlines()[0].startswith("beta\t1.0000")


def test_main_scrub_flag(tmp_path, capsys):
    log_path = tmp_path / "events.jsonl"
    _seed_log(log_path, [
        _ev(1000.0, "alpha"),
    ])
    rc = main(["--log", str(log_path), "--scrub", "--slice", "1d"])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("slice")


def test_main_inspect_flag(tmp_path, capsys):
    log_path = tmp_path / "events.jsonl"
    _seed_log(log_path, [
        _ev(100.0, "alpha"),
    ])
    rc = main(["--log", str(log_path), "--inspect", "alpha", "--half-life", "3600"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "concept: alpha" in out


def test_main_missing_log_file_fails(tmp_path, capsys):
    rc = main(["--log", str(tmp_path / "nope.jsonl"), "--now", "100"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "log not found" in err


def test_main_scrub_and_inspect_mutually_exclusive(tmp_path):
    log_path = tmp_path / "events.jsonl"
    log_path.touch()
    with pytest.raises(SystemExit):
        main(["--log", str(log_path), "--scrub", "--inspect", "alpha"])
