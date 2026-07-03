"""Tests for the PDCT turn classifier — taxonomy v2 (cognitive-mode-v2)."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, patch
from dct.retrieval.classifier import (
    classify_turn,
    ClassifierResult,
    InputMode,
    TurnMode,
    SessionMode,
    EvidenceBundle,
    _compute_evidence,
    to_companion_row,
    detect_input_mode,
)


# ── ClassifierResult ──────────────────────────────────────────────────────────

class TestClassifierResult:
    def test_defaults_to_unclassified(self):
        r = ClassifierResult()
        assert r.turn_mode == TurnMode.UNCLASSIFIED
        assert r.session_mode == SessionMode.UNCLASSIFIED
        assert r.transition_flag is False
        assert r.confidence == 0.0
        assert r.classifier_reason == ""
        assert r.classifier_latency_ms == 0
        assert r.taxonomy_version == "cognitive-mode-v2"

    def test_to_dict_keys(self):
        r = ClassifierResult(
            input_mode=InputMode.VOICE,
            turn_mode=TurnMode.CONCEPTUAL,
            session_mode=SessionMode.CONCEPTUAL,
            transition_flag=False,
            confidence=0.9,
            classifier_reason="Alex is exploring ideas verbally.",
            classifier_latency_ms=450,
        )
        d = r.to_dict()
        assert d["input_mode"] == "voice"
        assert d["turn_mode"] == "conceptual"
        assert d["session_mode"] == "conceptual"
        assert d["transition_flag"] is False
        assert d["confidence"] == 0.9
        assert d["classifier_reason"] == "Alex is exploring ideas verbally."
        assert d["classifier_latency_ms"] == 450
        assert d["taxonomy_version"] == "cognitive-mode-v2"

    def test_all_new_modes_are_valid_enum_values(self):
        for mode in ("conceptual", "build", "tool_heavy", "lookup", "creative", "debug", "transition", "unclassified"):
            assert TurnMode(mode) is not None

    def test_session_mode_mixed_is_valid(self):
        """Previously 'mixed' fell through to UNCLASSIFIED — now it's a valid SessionMode."""
        assert SessionMode("mixed") == SessionMode.MIXED

    def test_session_mode_mixed_does_not_bleed_into_turn_mode(self):
        """TurnMode does not have 'mixed' — it should still be UNCLASSIFIED for turn_mode."""
        from dct.retrieval.classifier import _safe_turn_mode
        assert _safe_turn_mode("mixed") == TurnMode.UNCLASSIFIED

    def test_safe_session_mode_mixed(self):
        from dct.retrieval.classifier import _safe_session_mode
        assert _safe_session_mode("mixed") == SessionMode.MIXED

    def test_safe_session_mode_unknown_falls_back(self):
        from dct.retrieval.classifier import _safe_session_mode
        assert _safe_session_mode("BOGUS_VALUE") == SessionMode.UNCLASSIFIED


# ── InputMode detection ───────────────────────────────────────────────────────

class TestInputModeDetection:
    def test_detects_voice_marker(self):
        text = "some message\u200b[voice:abc12345]"
        assert detect_input_mode(text, source="telegram") == InputMode.VOICE

    def test_voice_marker_priority_over_source(self):
        text = "transcript\u200b[voice:abc12345]"
        assert detect_input_mode(text, source="claude-code") == InputMode.VOICE

    def test_detects_code_source(self):
        assert detect_input_mode("any text", source="claude-code") == InputMode.CODE

    def test_defaults_to_chat(self):
        assert detect_input_mode("plain message", source="telegram") == InputMode.CHAT

    def test_empty_text_no_marker_is_chat(self):
        assert detect_input_mode("", source="") == InputMode.CHAT


# ── _compute_evidence (Stage 1, pure function) ────────────────────────────────

class TestComputeEvidence:
    def test_no_tools_is_conceptual_weak(self):
        ev = _compute_evidence([], "what's your take on this?", "Great question...")
        assert ev.heuristic_mode == TurnMode.CONCEPTUAL
        assert ev.heuristic_strength == "weak"
        assert ev.tool_count == 0

    def test_build_tools_yield_build_strong(self):
        ev = _compute_evidence(["Edit", "Bash"], "fix the bug", "patched it")
        assert ev.heuristic_mode == TurnMode.BUILD
        assert ev.heuristic_strength == "strong"
        assert ev.has_build is True

    def test_write_tool_is_build(self):
        ev = _compute_evidence(["Write"], "create this file", "done")
        assert ev.heuristic_mode == TurnMode.BUILD

    def test_debug_tool_plus_error_signal_yields_debug(self):
        ev = _compute_evidence(
            ["Bash", "Grep"],
            "fix this traceback",
            "TypeError: expected str got int"
        )
        assert ev.heuristic_mode == TurnMode.DEBUG
        assert ev.heuristic_strength == "strong"
        assert ev.has_error_signal is True

    def test_debug_precedence_over_build(self):
        """debug beats build even when Edit is also present."""
        ev = _compute_evidence(
            ["Edit", "Bash", "Grep"],
            "this is failing with an exception",
            "AssertionError: test failed",
        )
        assert ev.heuristic_mode == TurnMode.DEBUG

    def test_lookup_tools_yield_lookup(self):
        ev = _compute_evidence(["query_memory", "CheckCalendar"], "what's on my calendar?", "You have 2 events")
        assert ev.heuristic_mode == TurnMode.LOOKUP
        assert ev.heuristic_strength == "moderate"
        assert ev.has_lookup is True

    def test_build_beats_lookup(self):
        """build takes precedence over lookup when both present."""
        ev = _compute_evidence(["Edit", "query_memory"], "update the file and check memory", "done")
        assert ev.heuristic_mode == TurnMode.BUILD

    def test_creative_tool_yields_creative(self):
        ev = _compute_evidence(["GenerateImage"], "make an image of a robot", "generated image")
        assert ev.heuristic_mode == TurnMode.CREATIVE
        assert ev.has_creative is True

    def test_creative_intent_no_tool_yields_creative(self):
        ev = _compute_evidence([], "write me a story about Alex", "Once upon a time...")
        assert ev.heuristic_mode == TurnMode.CREATIVE
        assert ev.has_creative_intent is True

    def test_tool_heavy_four_mixed_tools(self):
        ev = _compute_evidence(
            ["Read", "Glob", "ObsidianSearch", "CheckEmail"],
            "look through everything",
            "found stuff"
        )
        # ObsidianSearch/CheckEmail = lookup, Read/Glob = debug-ish, 4 tools total
        # lookup wins over tool_heavy because has_lookup=True and no build
        assert ev.tool_count == 4
        assert ev.heuristic_mode == TurnMode.LOOKUP  # lookup wins

    def test_tool_heavy_four_debug_ish_no_error(self):
        """4 debug-class tools but no error signal → not debug, falls to tool_heavy."""
        ev = _compute_evidence(
            ["Read", "Glob", "CtxRead", "CtxSearch"],
            "explore the codebase",
            "here are the files"
        )
        assert ev.heuristic_mode == TurnMode.TOOL_HEAVY
        assert ev.heuristic_strength == "moderate"

    def test_error_signal_without_debug_tools_not_debug(self):
        """Error text alone without debug tools is not classified as debug."""
        ev = _compute_evidence([], "we got a TypeError earlier", "I see, let's think about it")
        assert ev.heuristic_mode == TurnMode.CONCEPTUAL  # no tools at all


# ── classify_turn (full async flow, Haiku mocked) ─────────────────────────────

class TestClassifyTurn:
    @pytest.mark.anyio
    async def test_returns_unclassified_on_api_failure(self):
        with patch("dct.retrieval.classifier._call_haiku", side_effect=Exception("API down")):
            result = await classify_turn(
                user_text="hello", reply_text="hi",
                tools_invoked=[], pdct_concepts=[],
                input_mode=InputMode.CHAT, turn_index=0,
            )
        assert result.turn_mode == TurnMode.UNCLASSIFIED
        assert result.confidence == 0.0

    @pytest.mark.anyio
    async def test_returns_unclassified_on_timeout(self):
        import asyncio
        async def slow(*a, **kw):
            await asyncio.sleep(10)
            return ""
        with patch("dct.retrieval.classifier._call_haiku", new=slow):
            result = await classify_turn(
                user_text="hello", reply_text="hi",
                tools_invoked=[], pdct_concepts=[],
                input_mode=InputMode.CHAT, turn_index=0,
            )
        assert result.turn_mode == TurnMode.UNCLASSIFIED

    @pytest.mark.anyio
    async def test_conceptual_mode(self):
        mock_response = json.dumps({
            "turn_mode": "conceptual", "session_mode": "conceptual",
            "transition_flag": False, "confidence": 0.92,
            "reason": "Alex is brainstorming about design.",
        })
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value=mock_response)):
            result = await classify_turn(
                user_text="I want to talk through how the classifier should work",
                reply_text="Great idea, let's explore the design",
                tools_invoked=[], pdct_concepts=["PDCT", "classifier"],
                input_mode=InputMode.VOICE, turn_index=2,
            )
        assert result.turn_mode == TurnMode.CONCEPTUAL
        assert result.session_mode == SessionMode.CONCEPTUAL
        assert result.transition_flag is False
        assert result.confidence == pytest.approx(0.92)

    @pytest.mark.anyio
    async def test_build_mode(self):
        mock_response = json.dumps({
            "turn_mode": "build", "session_mode": "build",
            "transition_flag": False, "confidence": 0.88,
            "reason": "Editing source files and running tests.",
        })
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value=mock_response)):
            result = await classify_turn(
                user_text="fix the failing test",
                reply_text="Done, patched measurement.py and all tests pass",
                tools_invoked=["Edit", "Bash"], pdct_concepts=[],
                input_mode=InputMode.CODE, turn_index=12,
            )
        assert result.turn_mode == TurnMode.BUILD

    @pytest.mark.anyio
    async def test_tool_heavy_mode(self):
        mock_response = json.dumps({
            "turn_mode": "tool_heavy", "session_mode": "mixed",
            "transition_flag": False, "confidence": 0.78,
            "reason": "Many tool calls of mixed types.",
        })
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value=mock_response)):
            result = await classify_turn(
                user_text="look through everything",
                reply_text="ran 6 tools",
                tools_invoked=["Read", "Glob", "CtxSearch", "CtxRead", "Grep", "Bash"],
                pdct_concepts=[],
                input_mode=InputMode.CHAT, turn_index=5,
            )
        assert result.turn_mode == TurnMode.TOOL_HEAVY

    @pytest.mark.anyio
    async def test_lookup_mode(self):
        mock_response = json.dumps({
            "turn_mode": "lookup", "session_mode": "conceptual",
            "transition_flag": False, "confidence": 0.85,
            "reason": "Checking calendar and email.",
        })
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value=mock_response)):
            result = await classify_turn(
                user_text="what's on my calendar today?",
                reply_text="You have 2 events",
                tools_invoked=["CheckCalendar", "CheckEmail"],
                pdct_concepts=[],
                input_mode=InputMode.CHAT, turn_index=1,
            )
        assert result.turn_mode == TurnMode.LOOKUP

    @pytest.mark.anyio
    async def test_creative_mode(self):
        mock_response = json.dumps({
            "turn_mode": "creative", "session_mode": "conceptual",
            "transition_flag": False, "confidence": 0.82,
            "reason": "Generating an image.",
        })
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value=mock_response)):
            result = await classify_turn(
                user_text="generate an image of a futuristic lab",
                reply_text="Here's the image",
                tools_invoked=["GenerateImage"],
                pdct_concepts=[],
                input_mode=InputMode.CHAT, turn_index=3,
            )
        assert result.turn_mode == TurnMode.CREATIVE

    @pytest.mark.anyio
    async def test_debug_mode(self):
        mock_response = json.dumps({
            "turn_mode": "debug", "session_mode": "build",
            "transition_flag": False, "confidence": 0.91,
            "reason": "Tracing a TypeError in the stack.",
        })
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value=mock_response)):
            result = await classify_turn(
                user_text="fix this traceback",
                reply_text="TypeError: expected str at line 42",
                tools_invoked=["Bash", "Grep"],
                pdct_concepts=[],
                input_mode=InputMode.CHAT, turn_index=9,
            )
        assert result.turn_mode == TurnMode.DEBUG

    @pytest.mark.anyio
    async def test_transition_flag_propagated(self):
        mock_response = json.dumps({
            "turn_mode": "transition", "session_mode": "mixed",
            "transition_flag": True, "confidence": 0.75,
            "reason": "Session shifted from design to implementation.",
        })
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value=mock_response)):
            result = await classify_turn(
                user_text="Okay let's start coding", reply_text="Starting now",
                tools_invoked=[], pdct_concepts=[],
                input_mode=InputMode.CHAT, turn_index=8,
            )
        assert result.turn_mode == TurnMode.TRANSITION
        assert result.transition_flag is True

    @pytest.mark.anyio
    async def test_session_mode_mixed_accepted(self):
        """session_mode=mixed is valid in v2 (was UNCLASSIFIED in v1 due to missing SessionMode enum)."""
        mock_response = json.dumps({
            "turn_mode": "build", "session_mode": "mixed",
            "transition_flag": False, "confidence": 0.8,
            "reason": "Mixed session.",
        })
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value=mock_response)):
            result = await classify_turn(
                user_text="let's keep going", reply_text="ok",
                tools_invoked=["Edit"], pdct_concepts=[],
                input_mode=InputMode.CHAT, turn_index=15,
            )
        assert result.session_mode == SessionMode.MIXED
        assert result.turn_mode == TurnMode.BUILD

    @pytest.mark.anyio
    async def test_malformed_haiku_response_returns_unclassified(self):
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value="not json {")):
            result = await classify_turn(
                user_text="test", reply_text="test",
                tools_invoked=[], pdct_concepts=[],
                input_mode=InputMode.CHAT, turn_index=0,
            )
        assert result.turn_mode == TurnMode.UNCLASSIFIED

    @pytest.mark.anyio
    async def test_unknown_turn_mode_value_becomes_unclassified(self):
        mock_response = json.dumps({
            "turn_mode": "BOGUS_VALUE", "session_mode": "conceptual",
            "transition_flag": False, "confidence": 0.5, "reason": "test",
        })
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value=mock_response)):
            result = await classify_turn(
                user_text="test", reply_text="test",
                tools_invoked=[], pdct_concepts=[],
                input_mode=InputMode.CHAT, turn_index=0,
            )
        assert result.turn_mode == TurnMode.UNCLASSIFIED

    @pytest.mark.anyio
    async def test_taxonomy_version_in_result(self):
        mock_response = json.dumps({
            "turn_mode": "conceptual", "session_mode": "conceptual",
            "transition_flag": False, "confidence": 0.7, "reason": "ok",
        })
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value=mock_response)):
            result = await classify_turn(
                user_text="hi", reply_text="hi",
                tools_invoked=[], pdct_concepts=[],
                input_mode=InputMode.CHAT, turn_index=0,
            )
        assert result.taxonomy_version == "cognitive-mode-v2"

    @pytest.mark.anyio
    async def test_markdown_fence_stripped(self):
        """Haiku sometimes wraps JSON in ```json fences despite instructions."""
        mock_response = '```json\n{"turn_mode": "build", "session_mode": "build", "transition_flag": false, "confidence": 0.8, "reason": "test"}\n```'
        with patch("dct.retrieval.classifier._call_haiku", new=AsyncMock(return_value=mock_response)):
            result = await classify_turn(
                user_text="fix", reply_text="done",
                tools_invoked=["Edit"], pdct_concepts=[],
                input_mode=InputMode.CODE, turn_index=5,
            )
        assert result.turn_mode == TurnMode.BUILD


# ── to_companion_row ──────────────────────────────────────────────────────────

class TestCompanionRow:
    def test_companion_row_shape_v2(self):
        r = ClassifierResult(
            input_mode=InputMode.VOICE,
            turn_mode=TurnMode.CONCEPTUAL,
            session_mode=SessionMode.CONCEPTUAL,
            transition_flag=False,
            confidence=0.9,
            classifier_reason="Exploring ideas.",
            classifier_latency_ms=320,
        )
        row = to_companion_row(r, turn_id="chat|topic|5|1234567890")
        assert row["kind"] == "turn_classification"
        assert row["schema_version"] == 1
        assert row["turn_id"] == "chat|topic|5|1234567890"
        assert row["turn_mode"] == "conceptual"
        assert row["input_mode"] == "voice"
        assert row["confidence"] == 0.9
        assert row["taxonomy_version"] == "cognitive-mode-v2"
        assert "ts" in row

    def test_all_seven_modes_round_trip_through_companion_row(self):
        for mode in TurnMode:
            r = ClassifierResult(turn_mode=mode)
            row = to_companion_row(r, turn_id="t1")
            assert row["turn_mode"] == mode.value


# ── pdct_report Track D smoke test ───────────────────────────────────────────

class TestReportTrackD:
    def test_all_seven_modes_appear_in_track_d(self, tmp_path):
        """Fixture rows covering all 7 modes → pdct_report renders all in Track D."""
        import json as _json
        import subprocess
        import sys
        from pathlib import Path

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mfile = logs_dir / "measurement.jsonl"
        ufile = logs_dir / "utility.jsonl"

        modes = ["conceptual", "build", "tool_heavy", "lookup", "creative", "debug", "transition"]
        rows_m = []
        rows_c = []

        for i, mode in enumerate(modes):
            turn_id = f"t{i}"
            rows_m.append({
                "kind": "turn_measurement",
                "schema_version": 1,
                "ts": "2026-05-21T10:00:00Z",
                "turn_id": turn_id,
                "thread_id": "0",
                "topic_id": "0",
                "injected_tokens": 100,
                "concepts_matched": 2,
                "skip_reason": "none",
            })
            rows_c.append({
                "kind": "turn_classification",
                "schema_version": 1,
                "ts": "2026-05-21T10:00:01Z",
                "turn_id": turn_id,
                "turn_mode": mode,
                "session_mode": "conceptual",
                "transition_flag": False,
                "confidence": 0.8,
                "classifier_reason": f"test {mode}",
                "classifier_latency_ms": 100,
                "taxonomy_version": "cognitive-mode-v2",
            })

        with mfile.open("w") as f:
            for r in rows_m:
                f.write(_json.dumps(r) + "\n")
            for r in rows_c:
                f.write(_json.dumps(r) + "\n")

        # utility.jsonl can be empty (Track D doesn't require it)
        ufile.write_text("")

        result = subprocess.run(
            [sys.executable, "scripts/pdct_report.py"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
            env={**__import__("os").environ, "PDCT_LOGS_DIR": str(logs_dir)},
        )
        output = result.stdout + result.stderr
        for mode in modes:
            assert mode in output, f"Track D missing mode: {mode}\nOutput:\n{output}"
