"""Tests for ccbot.codex_transcript_parser — pure logic, no I/O."""

import json
import pytest

from ccbot.codex_transcript_parser import CodexTranscriptParser


# ── parse_line ───────────────────────────────────────────────────────────


class TestParseLine:
    @pytest.mark.parametrize(
        "line, expected",
        [
            ('{"type": "response_item"}', {"type": "response_item"}),
            (
                '{"type": "session_meta", "cwd": "/home/user"}',
                {"type": "session_meta", "cwd": "/home/user"},
            ),
            ("not-json", None),
            ("", None),
            ("   \t  ", None),
        ],
        ids=["response_item", "session_meta", "invalid_json", "empty", "whitespace"],
    )
    def test_parse_line(self, line: str, expected: dict | None) -> None:
        assert CodexTranscriptParser.parse_line(line) == expected


# ── extract_cwd_from_session_meta ────────────────────────────────────────


class TestExtractCwdFromSessionMeta:
    def test_session_meta_with_cwd(self) -> None:
        data = {"type": "session_meta", "cwd": "/home/user/project"}
        assert (
            CodexTranscriptParser.extract_cwd_from_session_meta(data)
            == "/home/user/project"
        )

    def test_non_session_meta(self) -> None:
        data = {"type": "response_item", "cwd": "/home/user/project"}
        assert CodexTranscriptParser.extract_cwd_from_session_meta(data) is None

    def test_session_meta_missing_cwd(self) -> None:
        data = {"type": "session_meta"}
        assert CodexTranscriptParser.extract_cwd_from_session_meta(data) is None


# ── parse_entries: assistant text ────────────────────────────────────────


class TestParseEntriesAssistantText:
    def _make_message_entry(self, text: str, role: str = "assistant") -> dict:
        return {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": role,
                "content": [{"type": "output_text", "text": text}],
            },
        }

    def test_assistant_message_produces_text_entry(self) -> None:
        entry = self._make_message_entry("Hello, world!")
        result, remaining = CodexTranscriptParser.parse_entries([entry])
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content_type == "text"
        assert result[0].text == "Hello, world!"
        assert remaining == {}

    def test_empty_assistant_message_skipped(self) -> None:
        entry = self._make_message_entry("   ")
        result, _ = CodexTranscriptParser.parse_entries([entry])
        assert result == []

    def test_user_message_produces_user_entry(self) -> None:
        entry = self._make_message_entry("What is 2+2?", role="user")
        result, _ = CodexTranscriptParser.parse_entries([entry])
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content_type == "text"


# ── parse_entries: function_call (tool_use) ───────────────────────────────


class TestParseEntriesFunctionCall:
    def _make_function_call(
        self, name: str, arguments: dict, call_id: str = "call_abc"
    ) -> dict:
        return {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": name,
                "arguments": json.dumps(arguments),
                "call_id": call_id,
            },
        }

    def test_function_call_produces_tool_use_entry(self) -> None:
        entry = self._make_function_call("Read", {"file_path": "src/main.py"})
        # Use carry-over mode to check pending_tools behaviour
        result, remaining = CodexTranscriptParser.parse_entries(
            [entry], pending_tools={}
        )
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content_type == "tool_use"
        assert result[0].tool_name == "Read"
        assert result[0].tool_use_id == "call_abc"
        assert "Read" in result[0].text
        # Tool stored in pending for carry-over mode
        assert "call_abc" in remaining

    def test_function_call_no_call_id(self) -> None:
        entry = {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "Bash",
                "arguments": json.dumps({"command": "ls"}),
            },
        }
        result, remaining = CodexTranscriptParser.parse_entries([entry])
        assert len(result) == 1
        assert result[0].content_type == "tool_use"
        assert result[0].tool_use_id is None
        assert remaining == {}

    def test_function_call_invalid_arguments_json(self) -> None:
        entry = {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "CustomTool",
                "arguments": "not valid json",
                "call_id": "call_xyz",
            },
        }
        result, _ = CodexTranscriptParser.parse_entries([entry])
        assert len(result) == 1
        assert result[0].content_type == "tool_use"
        assert result[0].tool_name == "CustomTool"


# ── parse_entries: function_call_output (tool_result) ────────────────────


class TestParseEntriesFunctionCallOutput:
    def _make_tool_pair(
        self, name: str, arguments: dict, output: str, call_id: str = "call_123"
    ) -> list[dict]:
        return [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": name,
                    "arguments": json.dumps(arguments),
                    "call_id": call_id,
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            },
        ]

    def test_tool_pair_produces_tool_result(self) -> None:
        entries = self._make_tool_pair(
            "Read", {"file_path": "src/main.py"}, "line1\nline2"
        )
        result, remaining = CodexTranscriptParser.parse_entries(entries)
        # Should have tool_use + tool_result
        assert len(result) == 2
        assert result[0].content_type == "tool_use"
        assert result[1].content_type == "tool_result"
        assert remaining == {}

    def test_unmatched_tool_result_skipped(self) -> None:
        entry = {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "unknown_id",
                "output": "some output",
            },
        }
        result, _ = CodexTranscriptParser.parse_entries([entry])
        # No pending tool, output is non-empty → still produces a result entry
        assert len(result) == 1
        assert result[0].content_type == "tool_result"


# ── parse_entries: reasoning (thinking) ──────────────────────────────────


class TestParseEntriesReasoning:
    def test_reasoning_produces_thinking_entry(self) -> None:
        entry = {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Let me think..."}],
            },
        }
        result, _ = CodexTranscriptParser.parse_entries([entry])
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content_type == "thinking"
        assert "Let me think..." in result[0].text

    def test_empty_reasoning_skipped(self) -> None:
        entry = {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [],
            },
        }
        result, _ = CodexTranscriptParser.parse_entries([entry])
        assert result == []


# ── parse_entries: ignored entry types ───────────────────────────────────


class TestParseEntriesIgnored:
    @pytest.mark.parametrize(
        "entry_type",
        ["session_meta", "event_msg", "unknown_type"],
    )
    def test_ignored_entry_types(self, entry_type: str) -> None:
        entry = {"type": entry_type, "data": "whatever"}
        result, _ = CodexTranscriptParser.parse_entries([entry])
        assert result == []


# ── parse_entries: carry-over mode (monitor) ─────────────────────────────


class TestParseEntriesCarryOver:
    def test_pending_tools_carried_across_calls(self) -> None:
        """Tool call in one batch, result in next — simulates monitor mode."""
        call_entry = {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "Bash",
                "arguments": json.dumps({"command": "ls"}),
                "call_id": "call_carry",
            },
        }
        result1, remaining = CodexTranscriptParser.parse_entries(
            [call_entry], pending_tools={}
        )
        assert len(result1) == 1
        assert result1[0].content_type == "tool_use"
        assert "call_carry" in remaining

        # Second batch with the tool_result
        output_entry = {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_carry",
                "output": "file.txt",
            },
        }
        result2, remaining2 = CodexTranscriptParser.parse_entries(
            [output_entry], pending_tools=remaining
        )
        assert len(result2) == 1
        assert result2[0].content_type == "tool_result"
        assert remaining2 == {}

    def test_one_shot_mode_emits_tool_use_immediately(self) -> None:
        """In one-shot mode, tool_use is emitted immediately (not accumulated)."""
        call_entry = {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "Read",
                "arguments": json.dumps({"file_path": "README.md"}),
                "call_id": "call_flush",
            },
        }
        # No pending_tools argument → one-shot mode
        result, remaining = CodexTranscriptParser.parse_entries([call_entry])
        # Tool is emitted immediately, remaining is empty in one-shot mode
        tool_use_entries = [e for e in result if e.content_type == "tool_use"]
        assert len(tool_use_entries) == 1
        assert tool_use_entries[0].tool_name == "Read"
        # In one-shot mode, pending tools are discarded (already emitted above)
        assert remaining == {}
