import json
from dct.events import Event, EventOp, EventSource


def test_event_construction_minimal():
    e = Event(
        ts=1713456789.123,
        source=EventSource.TELEGRAM,
        op=EventOp.WRITE,
        concepts=["consciousness", "AI"],
    )
    assert e.ts == 1713456789.123
    assert e.source is EventSource.TELEGRAM
    assert e.op is EventOp.WRITE
    assert e.concepts == ["consciousness", "AI"]
    assert e.metadata == {}


def test_event_to_dict_and_back():
    e = Event(
        ts=1713456789.123,
        source=EventSource.VOICE,
        op=EventOp.READ,
        concepts=["quantum"],
        metadata={"call_id": "abc123"},
    )
    d = e.to_dict()
    assert d == {
        "ts": 1713456789.123,
        "source": "voice",
        "op": "read",
        "concepts": ["quantum"],
        "metadata": {"call_id": "abc123"},
    }
    restored = Event.from_dict(d)
    assert restored == e


def test_event_json_round_trip():
    e = Event(
        ts=1713456789.123,
        source=EventSource.CLAUDE_CODE,
        op=EventOp.WRITE,
        concepts=["path-dependence"],
    )
    line = json.dumps(e.to_dict())
    restored = Event.from_dict(json.loads(line))
    assert restored == e


def test_event_rejects_empty_concepts():
    import pytest
    with pytest.raises(ValueError, match="concepts"):
        Event(
            ts=1.0,
            source=EventSource.TELEGRAM,
            op=EventOp.WRITE,
            concepts=[],
        )


def test_event_source_vault_roundtrip():
    ev = Event(
        ts=1234.0,
        source=EventSource.VAULT,
        op=EventOp.WRITE,
        concepts=["voice-pipeline"],
        metadata={"extraction_source": "vault", "source_file": "/tmp/x.md"},
    )
    d = ev.to_dict()
    assert d["source"] == "vault"
    back = Event.from_dict(d)
    assert back.source == EventSource.VAULT


def test_event_op_feedback_member_exists():
    assert EventOp.FEEDBACK.value == "feedback"


def test_event_with_feedback_op_roundtrip():
    ev = Event(
        ts=1700000000.0,
        source=EventSource.TELEGRAM,
        op=EventOp.FEEDBACK,
        concepts=["a", "b"],
        metadata={
            "thread_id": "92",
            "useful_concept": "phenomenology",
            "path": ["consciousness", "memory", "phenomenology"],
            "multipliers": [2, 5],
            "anti_leak_applied": ["seed_filter"],
        },
    )
    d = ev.to_dict()
    assert d["op"] == "feedback"
    assert d["metadata"]["multipliers"] == [2, 5]
    rt = Event.from_dict(d)
    assert rt == ev
