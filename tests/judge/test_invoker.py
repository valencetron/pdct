"""Tests for the real Haiku judge invoker (P1.3b).

All tests mock the Anthropic client — no live API calls in CI.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from dct.judge.invoker import invoke_judge, JUDGE_SYSTEM_PROMPT
from dct.judge.worker import JudgeInvocationResult


def _mock_client(content_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=content_text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    return mock_client


def test_invoke_judge_ok():
    payload = json.dumps({"score": 4, "rationale": "cascade context was directly used", "era_assessment": "helpful"})
    client = _mock_client(payload)
    with patch("dct.judge.invoker._get_client", return_value=client):
        result = invoke_judge("USER:\nhi\n\nCASCADE:\nsome context\n\nREPLY:\nsome reply\n")
    assert result.status == "ok"
    assert result.score == 4
    assert result.rationale == "cascade context was directly used"
    assert result.era_assessment == "helpful"
    assert result.latency_ms is not None


def test_invoke_judge_bad_json():
    client = _mock_client("not json at all")
    with patch("dct.judge.invoker._get_client", return_value=client):
        result = invoke_judge("prompt")
    assert result.status == "parse_error"
    assert result.score is None


def test_invoke_judge_schema_violation():
    payload = json.dumps({"rationale": "missing score field"})
    client = _mock_client(payload)
    with patch("dct.judge.invoker._get_client", return_value=client):
        result = invoke_judge("prompt")
    assert result.status == "schema_violation"
    assert result.score is None


def test_invoke_judge_score_out_of_range():
    payload = json.dumps({"score": 99, "rationale": "oops", "era_assessment": "none"})
    client = _mock_client(payload)
    with patch("dct.judge.invoker._get_client", return_value=client):
        result = invoke_judge("prompt")
    assert result.status == "schema_violation"


def test_invoke_judge_api_exception():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("network fail")
    with patch("dct.judge.invoker._get_client", return_value=mock_client):
        result = invoke_judge("prompt")
    assert result.status == "unexpected_error"
    assert result.score is None


def test_system_prompt_has_json_instruction():
    assert "JSON" in JUDGE_SYSTEM_PROMPT
    assert "score" in JUDGE_SYSTEM_PROMPT


# --- JSON salvage on prose-wrapped output (live failure mode, 2026-06-09) ---

def test_invoke_judge_salvages_object_after_prose():
    """Thin-input failure mode: judge emits a sentence THEN a JSON object.
    Strict json.loads() raises 'Extra data'; salvage must recover the object."""
    payload = (
        'that is a pretty minimal signal. '
        '{"score": 2, "rationale": "marginal", "era_assessment": "neutral"}'
    )
    client = _mock_client(payload)
    with patch("dct.judge.invoker._get_client", return_value=client):
        from dct.judge.invoker import invoke_judge
        r = invoke_judge("p")
    assert r.status == "ok"
    assert r.score == 2
    assert r.era_assessment == "neutral"


def test_invoke_judge_salvages_object_before_prose():
    """Object first, then trailing commentary — also 'Extra data' under strict."""
    payload = (
        '{"score": 4, "rationale": "useful", "era_assessment": "helpful"} '
        'Hope that helps!'
    )
    client = _mock_client(payload)
    with patch("dct.judge.invoker._get_client", return_value=client):
        from dct.judge.invoker import invoke_judge
        r = invoke_judge("p")
    assert r.status == "ok"
    assert r.score == 4
    assert r.era_assessment == "helpful"


def test_invoke_judge_salvage_respects_braces_in_strings():
    """A '}' inside a string value must not prematurely close the object."""
    payload = (
        'note: {"score": 3, "rationale": "ref to {curly} token", '
        '"era_assessment": "neutral"}'
    )
    client = _mock_client(payload)
    with patch("dct.judge.invoker._get_client", return_value=client):
        from dct.judge.invoker import invoke_judge
        r = invoke_judge("p")
    assert r.status == "ok"
    assert r.score == 3
    assert "curly" in (r.rationale or "")


def test_invoke_judge_no_salvageable_object_still_parse_error():
    """Pure prose with no JSON object must still fail cleanly as parse_error."""
    client = _mock_client("there is no json here at all, just words.")
    with patch("dct.judge.invoker._get_client", return_value=client):
        from dct.judge.invoker import invoke_judge
        r = invoke_judge("p")
    assert r.status == "parse_error"
    assert r.score is None


def test_salvage_helper_unbalanced_returns_none():
    """Unit-level: an unterminated object yields None, not a crash."""
    from dct.judge.invoker import _salvage_json_object
    assert _salvage_json_object('prefix {"score": 2, "rationale": "x"') is None
    assert _salvage_json_object("no brace here") is None
