"""Codex CLI JSONL transcript parser.

Parses OpenAI Codex CLI session JSONL files and extracts structured messages.
Codex uses the same JSONL streaming approach as Claude Code, but with a
different schema based on OpenAI's Responses API format.

Supported Codex JSONL entry types:
- session_meta: Contains cwd and session metadata (extracted, not displayed)
- response_item(type=message, role=assistant): Assistant text → text entry
- response_item(type=function_call): Tool invocation → tool_use entry
- response_item(type=function_call_output): Tool result → tool_result entry
- response_item(type=reasoning): Thinking/reasoning → thinking entry
- event_msg: Lifecycle events (ignored)

Session file path pattern: ~/.codex/sessions/YYYY/MM/DD/rollup-{ts}-{uuid}.jsonl

Key classes: CodexTranscriptParser, CodexSessionInfo.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transcript_parser import ParsedEntry, PendingToolInfo, TranscriptParser

logger = logging.getLogger(__name__)


@dataclass
class CodexSessionInfo:
    """Information about a Codex CLI session file."""

    session_id: str  # Derived from filename stem
    file_path: Path
    cwd: str  # From session_meta line


class CodexTranscriptParser:
    """Parser for Codex CLI JSONL session files.

    Implements the same interface as TranscriptParser so that
    session_monitor.py can use either parser without code changes.

    Codex JSONL entry structure:
    - type: "session_meta" | "response_item" | "event_msg"
    - For response_item: payload.type = "message" | "function_call"
      | "function_call_output" | "reasoning"
    """

    @staticmethod
    def parse_line(line: str) -> dict | None:
        """Parse a single JSONL line.

        Returns:
            Parsed dict or None if line is empty/invalid
        """
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def extract_cwd_from_session_meta(data: dict) -> str | None:
        """Extract cwd from a session_meta entry.

        Returns:
            cwd string or None if not a session_meta entry
        """
        if data.get("type") != "session_meta":
            return None
        return data.get("cwd") or data.get("workdir") or None

    @staticmethod
    def _extract_text_from_content(content: list[Any]) -> str:
        """Extract text from Codex message content blocks.

        Codex uses output_text blocks inside message content.
        """
        if not isinstance(content, list):
            if isinstance(content, str):
                return content
            return ""
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            # Codex uses "output_text" for assistant message content
            if btype in ("output_text", "text"):
                t = block.get("text", "")
                if t:
                    parts.append(t)
        return "\n".join(parts)

    @staticmethod
    def _extract_reasoning_text(summary: list[Any]) -> str:
        """Extract text from Codex reasoning summary blocks."""
        if not isinstance(summary, list):
            if isinstance(summary, str):
                return summary
            return ""
        parts: list[str] = []
        for block in summary:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("summary_text", "text"):
                t = block.get("text", "")
                if t:
                    parts.append(t)
        return "\n".join(parts)

    @classmethod
    def parse_entries(
        cls,
        entries: list[dict],
        pending_tools: dict[str, PendingToolInfo] | None = None,
    ) -> tuple[list[ParsedEntry], dict[str, PendingToolInfo]]:
        """Parse a list of Codex JSONL entries into display-ready messages.

        Implements the same interface as TranscriptParser.parse_entries so
        session_monitor.py works without changes.

        Codex tool pairing model:
        - function_call emits tool_use with call_id as the tool_use_id
        - function_call_output emits tool_result, matched by call_id

        Args:
            entries: List of parsed JSONL dicts
            pending_tools: Carry-over tool state from prior poll cycle (monitor mode)

        Returns:
            Tuple of (parsed entries, remaining pending_tools)
        """
        result: list[ParsedEntry] = []
        _carry_over = pending_tools is not None
        if pending_tools is None:
            pending_tools = {}
        else:
            pending_tools = dict(pending_tools)  # don't mutate caller's dict

        for data in entries:
            entry_type = data.get("type")

            # Skip non-response entries
            if entry_type not in ("response_item",):
                continue

            item = data.get("payload")
            if not isinstance(item, dict):
                continue

            item_type = item.get("type", "")

            if item_type == "message":
                # Assistant text message
                role = item.get("role", "")
                if role != "assistant":
                    # User messages in Codex are typically just prompts; skip them
                    # unless we want to surface them
                    content = item.get("content", [])
                    text = cls._extract_text_from_content(content)
                    if text.strip():
                        result.append(
                            ParsedEntry(
                                role="user",
                                text=text.strip(),
                                content_type="text",
                            )
                        )
                    continue

                content = item.get("content", [])
                text = cls._extract_text_from_content(content)
                if text.strip():
                    result.append(
                        ParsedEntry(
                            role="assistant",
                            text=text.strip(),
                            content_type="text",
                        )
                    )

            elif item_type == "function_call":
                # Tool invocation → tool_use entry
                call_id = item.get("call_id", item.get("id", ""))
                name = item.get("name", "unknown")
                arguments_raw = item.get("arguments", "{}")

                # Parse arguments JSON string into dict
                try:
                    inp: dict[str, Any] = (
                        json.loads(arguments_raw)
                        if isinstance(arguments_raw, str)
                        else arguments_raw
                    )
                except (json.JSONDecodeError, TypeError):
                    inp = {}

                summary = TranscriptParser.format_tool_use_summary(name, inp)

                # Store for later pairing with function_call_output
                input_data = inp if name in ("Edit", "NotebookEdit") else None
                if call_id:
                    pending_tools[call_id] = PendingToolInfo(
                        summary=summary,
                        tool_name=name,
                        input_data=input_data,
                    )

                result.append(
                    ParsedEntry(
                        role="assistant",
                        text=summary,
                        content_type="tool_use",
                        tool_use_id=call_id or None,
                        tool_name=name,
                    )
                )

            elif item_type == "function_call_output":
                # Tool result → tool_result entry
                call_id = item.get("call_id", "")
                output = item.get("output", "")
                result_text = output if isinstance(output, str) else str(output)

                tool_info = pending_tools.pop(call_id, None) if call_id else None
                tool_summary = tool_info.summary if tool_info else None
                tool_name = tool_info.tool_name if tool_info else None
                tool_input_data = tool_info.input_data if tool_info else None

                if tool_summary:
                    entry_text = tool_summary
                    # For Edit tool, generate diff stats
                    if tool_name == "Edit" and tool_input_data and result_text:
                        old_s = tool_input_data.get("old_string", "")
                        new_s = tool_input_data.get("new_string", "")
                        if old_s and new_s:
                            diff_text = TranscriptParser._format_edit_diff(old_s, new_s)
                            if diff_text:
                                added = sum(
                                    1
                                    for line in diff_text.split("\n")
                                    if line.startswith("+")
                                    and not line.startswith("+++")
                                )
                                removed = sum(
                                    1
                                    for line in diff_text.split("\n")
                                    if line.startswith("-")
                                    and not line.startswith("---")
                                )
                                stats = (
                                    f"  ⎿  Added {added} lines, removed {removed} lines"
                                )
                                entry_text += (
                                    "\n"
                                    + stats
                                    + "\n"
                                    + TranscriptParser._format_expandable_quote(
                                        diff_text
                                    )
                                )
                    elif (
                        result_text
                        and TranscriptParser.EXPANDABLE_QUOTE_START not in tool_summary
                    ):
                        entry_text += "\n" + TranscriptParser._format_tool_result_text(
                            result_text, tool_name
                        )
                elif result_text:
                    entry_text = TranscriptParser._format_tool_result_text(
                        result_text, tool_name
                    )
                else:
                    continue

                result.append(
                    ParsedEntry(
                        role="assistant",
                        text=entry_text,
                        content_type="tool_result",
                        tool_use_id=call_id or None,
                    )
                )

            elif item_type == "reasoning":
                # Thinking/reasoning entry
                summary_blocks = item.get("summary", [])
                thinking_text = cls._extract_reasoning_text(summary_blocks)
                if thinking_text.strip():
                    quoted = TranscriptParser._format_expandable_quote(
                        thinking_text.strip()
                    )
                    result.append(
                        ParsedEntry(
                            role="assistant",
                            text=quoted,
                            content_type="thinking",
                        )
                    )

        # Unlike TranscriptParser, we do NOT flush pending tools at the end
        # because tool_use entries are emitted immediately when function_call
        # is processed. Pending tools here are just carry-over for matching
        # with future function_call_output entries. In carry-over mode (monitor)
        # they persist; in one-shot mode we discard them (they were emitted).
        remaining_pending = dict(pending_tools) if _carry_over else {}

        # Strip whitespace from all entries
        for entry in result:
            entry.text = entry.text.strip()

        return result, remaining_pending

    @staticmethod
    def get_codex_sessions_path() -> Path:
        """Return the default Codex sessions root directory."""
        return Path.home() / ".codex" / "sessions"

    @classmethod
    def scan_codex_sessions(cls, cwd: str) -> list[CodexSessionInfo]:
        """Scan Codex session files matching the given cwd.

        Searches ~/.codex/sessions/YYYY/MM/DD/ directories in reverse
        chronological order and matches files whose session_meta.cwd equals cwd.

        Args:
            cwd: Working directory to match against session_meta.cwd

        Returns:
            List of CodexSessionInfo sorted newest-first
        """
        sessions_root = cls.get_codex_sessions_path()
        if not sessions_root.exists():
            return []

        try:
            resolved_cwd = str(Path(cwd).resolve())
        except (OSError, ValueError):
            resolved_cwd = cwd

        results: list[CodexSessionInfo] = []

        # Walk year/month/day dirs in reverse chronological order
        year_dirs = (
            sorted(sessions_root.iterdir(), reverse=True)
            if sessions_root.exists()
            else []
        )
        for year_dir in year_dirs:
            if not year_dir.is_dir():
                continue
            month_dirs = sorted(year_dir.iterdir(), reverse=True)
            for month_dir in month_dirs:
                if not month_dir.is_dir():
                    continue
                day_dirs = sorted(month_dir.iterdir(), reverse=True)
                for day_dir in day_dirs:
                    if not day_dir.is_dir():
                        continue
                    # Scan jsonl files in this day dir
                    jsonl_files = sorted(
                        day_dir.glob("*.jsonl"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for f in jsonl_files:
                        session_cwd = cls._read_cwd_from_codex_file(f)
                        if not session_cwd:
                            continue
                        try:
                            norm_session_cwd = str(Path(session_cwd).resolve())
                        except (OSError, ValueError):
                            norm_session_cwd = session_cwd
                        if norm_session_cwd == resolved_cwd:
                            results.append(
                                CodexSessionInfo(
                                    session_id=f.stem,
                                    file_path=f,
                                    cwd=session_cwd,
                                )
                            )

        return results

    @staticmethod
    def _read_cwd_from_codex_file(file_path: Path) -> str:
        """Read cwd from the first session_meta line of a Codex JSONL file.

        Returns empty string if not found.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "session_meta":
                        return data.get("cwd", "") or data.get("workdir", "")
                    # Stop after first few lines if no session_meta found
                    break
        except OSError:
            pass
        return ""
