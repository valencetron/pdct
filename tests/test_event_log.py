from pathlib import Path

from dct.event_log import EventLog
from dct.events import Event, EventOp, EventSource


def _make_event(ts: float, concept: str = "x") -> Event:
    return Event(
        ts=ts,
        source=EventSource.TELEGRAM,
        op=EventOp.WRITE,
        concepts=[concept],
    )


def test_append_one_event_then_read_all(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)

    e = _make_event(1.0, "alpha")
    log.append(e)

    events = log.read_all()
    assert events == [e]


def test_append_multiple_events_preserves_order_on_read(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)

    e1 = _make_event(1.0, "alpha")
    e2 = _make_event(2.0, "beta")
    e3 = _make_event(3.0, "gamma")

    log.append(e1)
    log.append(e2)
    log.append(e3)

    assert log.read_all() == [e1, e2, e3]


def test_append_creates_file_if_absent(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "dir" / "events.jsonl"
    log = EventLog(log_path)

    log.append(_make_event(1.0))

    assert log_path.exists()


def test_read_all_on_missing_file_returns_empty(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "does-not-exist.jsonl")
    assert log.read_all() == []


def test_read_all_tolerates_partial_trailing_line(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)

    e1 = _make_event(1.0, "alpha")
    log.append(e1)

    # Simulate a crash mid-write: append a garbage partial line.
    with log_path.open("a", encoding="utf-8") as f:
        f.write('{"ts": 2.0, "source": "telegram", "op": "wr')  # no newline, truncated

    events = log.read_all()
    assert events == [e1]


def test_read_all_skips_blank_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)

    log.append(_make_event(1.0, "a"))
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n\n")
    log.append(_make_event(2.0, "b"))

    events = log.read_all()
    assert [e.concepts[0] for e in events] == ["a", "b"]


def test_read_all_reorders_by_timestamp(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)

    # Append out of order.
    log.append(_make_event(3.0, "c"))
    log.append(_make_event(1.0, "a"))
    log.append(_make_event(2.0, "b"))

    events = log.read_all()
    assert [e.ts for e in events] == [1.0, 2.0, 3.0]
    assert [e.concepts[0] for e in events] == ["a", "b", "c"]


def test_read_all_is_stable_for_equal_timestamps(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)

    log.append(_make_event(1.0, "first"))
    log.append(_make_event(1.0, "second"))
    log.append(_make_event(1.0, "third"))

    events = log.read_all()
    assert [e.concepts[0] for e in events] == ["first", "second", "third"]
