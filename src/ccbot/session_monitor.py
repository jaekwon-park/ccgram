"""Session monitoring service — watches JSONL files for new messages.

Runs an async polling loop that:
  1. Loads the current session_map to know which sessions to watch.
  2. Detects session_map changes (new/changed/deleted windows) and cleans up.
  3. Reads new JSONL lines from each session file using byte-offset tracking.
  4. Parses entries via TranscriptParser and emits NewMessage objects to a callback.

Optimizations: mtime cache skips unchanged files; byte offset avoids re-reading.

Key classes: SessionMonitor, NewMessage, SessionInfo.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

import aiofiles

from .codex_transcript_parser import CodexTranscriptParser, CodexSessionInfo
from .config import config
from .monitor_state import MonitorState, TrackedSession
from .tmux_manager import tmux_manager
from .transcript_parser import TranscriptParser
from .utils import read_cwd_from_jsonl

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about a Claude Code session."""

    session_id: str
    file_path: Path
    cwd: str = ""  # populated by scan_codex_sessions for window matching


@dataclass
class NewMessage:
    """A new message detected by the monitor."""

    session_id: str
    text: str
    is_complete: bool  # True when stop_reason is set (final message)
    content_type: str = "text"  # "text" or "thinking"
    tool_use_id: str | None = None
    role: str = "assistant"  # "user" or "assistant"
    tool_name: str | None = None  # For tool_use messages, the tool name
    image_data: list[tuple[str, bytes]] | None = None  # From tool_result images


class SessionMonitor:
    """Monitors Claude Code sessions for new assistant messages.

    Uses simple async polling with aiofiles for non-blocking I/O.
    Emits both intermediate and complete assistant messages.
    """

    def __init__(
        self,
        projects_path: Path | None = None,
        poll_interval: float | None = None,
        state_file: Path | None = None,
    ):
        self.projects_path = (
            projects_path if projects_path is not None else config.claude_projects_path
        )
        self.poll_interval = (
            poll_interval if poll_interval is not None else config.monitor_poll_interval
        )

        self.state = MonitorState(state_file=state_file or config.monitor_state_file)
        self.state.load()

        self._running = False
        self._task: asyncio.Task | None = None
        self._message_callback: Callable[[NewMessage], Awaitable[None]] | None = None
        # Per-session pending tool_use state carried across poll cycles
        self._pending_tools: dict[str, dict[str, Any]] = {}  # session_id -> pending
        # Track last known session_map for detecting changes
        # Keys may be window_id (@12) or window_name (old format) during transition
        self._last_session_map: dict[str, str] = {}  # window_key -> session_id
        # In-memory mtime cache for quick file change detection (not persisted)
        self._file_mtimes: dict[str, float] = {}  # session_id -> last_seen_mtime

    def set_message_callback(
        self, callback: Callable[[NewMessage], Awaitable[None]]
    ) -> None:
        self._message_callback = callback

    async def _get_active_cwds(
        self,
        *,
        window_ids: set[str] | None = None,
        exclude_window_ids: set[str] | None = None,
    ) -> set[str]:
        """Get normalized cwds of active tmux windows.

        window_ids: if set, only include these windows.
        exclude_window_ids: if set, exclude these windows.
        """
        cwds = set()
        windows = await tmux_manager.list_windows()
        for w in windows:
            if window_ids is not None and w.window_id not in window_ids:
                continue
            if exclude_window_ids is not None and w.window_id in exclude_window_ids:
                continue
            try:
                cwds.add(str(Path(w.cwd).resolve()))
            except (OSError, ValueError):
                cwds.add(w.cwd)
        return cwds

    async def scan_projects(self) -> list[SessionInfo]:
        """Scan projects that have active tmux windows."""
        active_cwds = await self._get_active_cwds(
            exclude_window_ids=config.codex_windows or None
        )
        if not active_cwds:
            return []

        sessions = []

        if not self.projects_path.exists():
            return sessions

        for project_dir in self.projects_path.iterdir():
            if not project_dir.is_dir():
                continue

            index_file = project_dir / "sessions-index.json"
            original_path = ""
            indexed_ids: set[str] = set()

            if index_file.exists():
                try:
                    async with aiofiles.open(index_file, "r") as f:
                        content = await f.read()
                    index_data = json.loads(content)
                    entries = index_data.get("entries", [])
                    original_path = index_data.get("originalPath", "")

                    for entry in entries:
                        session_id = entry.get("sessionId", "")
                        full_path = entry.get("fullPath", "")
                        project_path = entry.get("projectPath", original_path)

                        if not session_id or not full_path:
                            continue

                        try:
                            norm_pp = str(Path(project_path).resolve())
                        except (OSError, ValueError):
                            norm_pp = project_path
                        if norm_pp not in active_cwds:
                            continue

                        indexed_ids.add(session_id)
                        file_path = Path(full_path)
                        if file_path.exists():
                            sessions.append(
                                SessionInfo(
                                    session_id=session_id,
                                    file_path=file_path,
                                )
                            )

                except (json.JSONDecodeError, OSError) as e:
                    logger.debug(f"Error reading index {index_file}: {e}")

            # Pick up un-indexed .jsonl files
            try:
                for jsonl_file in project_dir.glob("*.jsonl"):
                    session_id = jsonl_file.stem
                    if session_id in indexed_ids:
                        continue

                    # Determine project_path for this file
                    file_project_path = original_path
                    if not file_project_path:
                        file_project_path = await asyncio.to_thread(
                            read_cwd_from_jsonl, jsonl_file
                        )
                    if not file_project_path:
                        dir_name = project_dir.name
                        if dir_name.startswith("-"):
                            file_project_path = dir_name.replace("-", "/")

                    try:
                        norm_fp = str(Path(file_project_path).resolve())
                    except (OSError, ValueError):
                        norm_fp = file_project_path

                    if norm_fp not in active_cwds:
                        continue

                    sessions.append(
                        SessionInfo(
                            session_id=session_id,
                            file_path=jsonl_file,
                        )
                    )
            except OSError as e:
                logger.debug(f"Error scanning jsonl files in {project_dir}: {e}")

        return sessions

    async def _read_new_lines(
        self, session: TrackedSession, file_path: Path
    ) -> list[dict]:
        """Read new lines from a session file using byte offset for efficiency.

        Detects file truncation (e.g. after /clear) and resets offset.
        Recovers from corrupted offsets (mid-line) by scanning to next line.
        """
        new_entries = []
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                # Get file size to detect truncation
                await f.seek(0, 2)  # Seek to end
                file_size = await f.tell()

                # Detect file truncation: if offset is beyond file size, reset
                if session.last_byte_offset > file_size:
                    logger.info(
                        "File truncated for session %s "
                        "(offset %d > size %d). Resetting.",
                        session.session_id,
                        session.last_byte_offset,
                        file_size,
                    )
                    session.last_byte_offset = 0

                # Seek to last read position for incremental reading
                await f.seek(session.last_byte_offset)

                # Detect corrupted offset: if we're mid-line (not at '{'),
                # scan forward to the next line start. This can happen if
                # the state file was manually edited or corrupted.
                if session.last_byte_offset > 0:
                    first_char = await f.read(1)
                    if first_char and first_char != "{":
                        logger.warning(
                            "Corrupted offset %d in session %s (mid-line), "
                            "scanning to next line",
                            session.last_byte_offset,
                            session.session_id,
                        )
                        await f.readline()  # Skip rest of partial line
                        session.last_byte_offset = await f.tell()
                        return []
                    await f.seek(session.last_byte_offset)  # Reset for normal read

                # Read only new lines from the offset.
                # Track safe_offset: only advance past lines that parsed
                # successfully. A non-empty line that fails JSON parsing is
                # likely a partial write; stop and retry next cycle.
                # Select parser by file path: codex sessions live under .codex/
                is_codex_file = ".codex" in str(file_path)
                parser = CodexTranscriptParser if is_codex_file else TranscriptParser
                safe_offset = session.last_byte_offset
                async for line in f:
                    data = parser.parse_line(line)
                    if data:
                        new_entries.append(data)
                        safe_offset = await f.tell()
                    elif line.strip():
                        # Partial JSONL line — don't advance offset past it
                        logger.warning(
                            "Partial JSONL line in session %s, will retry next cycle",
                            session.session_id,
                        )
                        break
                    else:
                        # Empty line — safe to skip
                        safe_offset = await f.tell()

                session.last_byte_offset = safe_offset

        except OSError as e:
            logger.error("Error reading session file %s: %s", file_path, e)
        return new_entries

    async def scan_codex_sessions(self) -> list[SessionInfo]:
        """Scan Codex session files for active tmux windows.

        Used when config.backend == 'codex' or config.codex_windows is set.
        Searches ~/.codex/sessions/ for files whose cwd matches an active window.
        When codex_windows is set, only scans cwds for those windows.
        """
        # When codex_windows is configured, only scan those specific windows
        window_filter = config.codex_windows if config.codex_windows else None
        active_cwds = await self._get_active_cwds(window_ids=window_filter)
        if not active_cwds:
            return []

        sessions: list[SessionInfo] = []
        codex_root = CodexTranscriptParser.get_codex_sessions_path()
        if not codex_root.exists():
            return sessions

        # Collect all JSONL files sorted by mtime descending
        all_files: list[tuple[float, CodexSessionInfo]] = []
        year_dirs = (
            sorted(codex_root.iterdir(), reverse=True) if codex_root.exists() else []
        )
        for year_dir in year_dirs:
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir(), reverse=True):
                if not month_dir.is_dir():
                    continue
                for day_dir in sorted(month_dir.iterdir(), reverse=True):
                    if not day_dir.is_dir():
                        continue
                    for jsonl_file in day_dir.glob("*.jsonl"):
                        try:
                            mtime = jsonl_file.stat().st_mtime
                        except OSError:
                            continue
                        session_cwd = await asyncio.to_thread(
                            CodexTranscriptParser._read_cwd_from_codex_file, jsonl_file
                        )
                        if not session_cwd:
                            continue
                        try:
                            norm_cwd = str(
                                __import__("pathlib").Path(session_cwd).resolve()
                            )
                        except (OSError, ValueError):
                            norm_cwd = session_cwd
                        if norm_cwd in active_cwds:
                            codex_info = CodexSessionInfo(
                                session_id=jsonl_file.stem,
                                file_path=jsonl_file,
                                cwd=session_cwd,
                            )
                            all_files.append((mtime, codex_info))

        # Sort newest first
        all_files.sort(key=lambda x: x[0], reverse=True)
        for _, info in all_files:
            sessions.append(
                SessionInfo(
                    session_id=info.session_id, file_path=info.file_path, cwd=info.cwd
                )
            )

        return sessions

    async def check_for_updates(self, active_session_ids: set[str]) -> list[NewMessage]:
        """Check all sessions for new assistant messages.

        Reads from last byte offset. Emits both intermediate
        (stop_reason=null) and complete messages.

        Args:
            active_session_ids: Set of session IDs currently in session_map
        """
        new_messages = []

        # Scan sessions: always run claude scan; also run codex scan when needed
        if config.codex_windows:
            # Per-window codex: run both scans and merge
            claude_sessions = await self.scan_projects()
            codex_sessions = await self.scan_codex_sessions()
            sessions = claude_sessions + codex_sessions
        elif config.backend == "codex":
            sessions = await self.scan_codex_sessions()
        else:
            sessions = await self.scan_projects()

        # Only process sessions that are in session_map
        for session_info in sessions:
            if session_info.session_id not in active_session_ids:
                continue
            try:
                tracked = self.state.get_session(session_info.session_id)

                if tracked is None:
                    # For new sessions, initialize offset to end of file
                    # to avoid re-processing old messages
                    try:
                        file_size = session_info.file_path.stat().st_size
                        current_mtime = session_info.file_path.stat().st_mtime
                    except OSError:
                        file_size = 0
                        current_mtime = 0.0
                    tracked = TrackedSession(
                        session_id=session_info.session_id,
                        file_path=str(session_info.file_path),
                        last_byte_offset=file_size,
                    )
                    self.state.update_session(tracked)
                    self._file_mtimes[session_info.session_id] = current_mtime
                    logger.info(f"Started tracking session: {session_info.session_id}")
                    continue

                # Check mtime + file size to see if file has changed
                try:
                    st = session_info.file_path.stat()
                    current_mtime = st.st_mtime
                    current_size = st.st_size
                except OSError:
                    continue

                last_mtime = self._file_mtimes.get(session_info.session_id, 0.0)
                if (
                    current_mtime <= last_mtime
                    and current_size <= tracked.last_byte_offset
                ):
                    # File hasn't changed, skip reading
                    continue

                # File changed, read new content from last offset
                new_entries = await self._read_new_lines(
                    tracked, session_info.file_path
                )
                self._file_mtimes[session_info.session_id] = current_mtime

                if new_entries:
                    logger.debug(
                        f"Read {len(new_entries)} new entries for "
                        f"session {session_info.session_id}"
                    )

                # Parse new entries using the configured backend parser
                carry = self._pending_tools.get(session_info.session_id, {})
                parser = (
                    CodexTranscriptParser
                    if config.backend == "codex"
                    else TranscriptParser
                )
                parsed_entries, remaining = parser.parse_entries(
                    new_entries,
                    pending_tools=carry,
                )
                if remaining:
                    self._pending_tools[session_info.session_id] = remaining
                else:
                    self._pending_tools.pop(session_info.session_id, None)

                for entry in parsed_entries:
                    if not entry.text and not entry.image_data:
                        continue
                    # Skip user messages unless show_user_messages is enabled
                    if entry.role == "user" and not config.show_user_messages:
                        continue
                    new_messages.append(
                        NewMessage(
                            session_id=session_info.session_id,
                            text=entry.text,
                            is_complete=True,
                            content_type=entry.content_type,
                            tool_use_id=entry.tool_use_id,
                            role=entry.role,
                            tool_name=entry.tool_name,
                            image_data=entry.image_data,
                        )
                    )

                self.state.update_session(tracked)

            except OSError as e:
                logger.debug(f"Error processing session {session_info.session_id}: {e}")

        self.state.save_if_dirty()
        return new_messages

    async def _load_current_session_map(self) -> dict[str, str]:
        """Load current session_map and return window_key -> session_id mapping.

        Keys in session_map are formatted as "tmux_session:window_id"
        (e.g. "ccbot:@12"). Old-format keys ("ccbot:window_name") are also
        accepted so that sessions running before a code upgrade continue
        to be monitored until the hook re-fires with new format.
        Only entries matching our tmux_session_name are processed.
        """
        window_to_session: dict[str, str] = {}
        if config.session_map_file.exists():
            try:
                async with aiofiles.open(config.session_map_file, "r") as f:
                    content = await f.read()
                session_map = json.loads(content)
                prefix = f"{config.tmux_session_name}:"
                for key, info in session_map.items():
                    # Only process entries for our tmux session
                    if not key.startswith(prefix):
                        continue
                    window_key = key[len(prefix) :]
                    session_id = info.get("session_id", "")
                    if session_id:
                        window_to_session[window_key] = session_id
            except (json.JSONDecodeError, OSError):
                pass
        return window_to_session

    async def _cleanup_all_stale_sessions(self) -> None:
        """Clean up all tracked sessions not in current session_map (used on startup)."""
        current_map = await self._load_current_session_map()
        active_session_ids = set(current_map.values())

        stale_sessions = []
        for session_id in self.state.tracked_sessions.keys():
            if session_id not in active_session_ids:
                stale_sessions.append(session_id)

        if stale_sessions:
            logger.info(
                f"[Startup cleanup] Removing {len(stale_sessions)} stale sessions"
            )
            for session_id in stale_sessions:
                self.state.remove_session(session_id)
                self._file_mtimes.pop(session_id, None)
            self.state.save_if_dirty()

    async def _detect_and_cleanup_changes(self) -> dict[str, str]:
        """Detect session_map changes and cleanup replaced/removed sessions.

        Returns current session_map for further processing.
        """
        current_map = await self._load_current_session_map()

        sessions_to_remove: set[str] = set()

        # Check for window session changes (window exists in both, but session_id changed)
        for window_id, old_session_id in self._last_session_map.items():
            new_session_id = current_map.get(window_id)
            if new_session_id and new_session_id != old_session_id:
                logger.info(
                    "Window '%s' session changed: %s -> %s",
                    window_id,
                    old_session_id,
                    new_session_id,
                )
                sessions_to_remove.add(old_session_id)

        # Check for deleted windows (window in old map but not in current)
        old_windows = set(self._last_session_map.keys())
        current_windows = set(current_map.keys())
        deleted_windows = old_windows - current_windows

        for window_id in deleted_windows:
            old_session_id = self._last_session_map[window_id]
            logger.info(
                "Window '%s' deleted, removing session %s",
                window_id,
                old_session_id,
            )
            sessions_to_remove.add(old_session_id)

        # Perform cleanup
        if sessions_to_remove:
            for session_id in sessions_to_remove:
                self.state.remove_session(session_id)
                self._file_mtimes.pop(session_id, None)
            self.state.save_if_dirty()

        # Update last known map
        self._last_session_map = current_map

        return current_map

    async def _monitor_loop(self) -> None:
        """Background loop for checking session updates.

        Uses simple async polling with aiofiles for non-blocking I/O.
        """
        logger.info("Session monitor started, polling every %ss", self.poll_interval)

        # Deferred import to avoid circular dependency (cached once)
        from .session import session_manager

        # Clean up all stale sessions on startup
        await self._cleanup_all_stale_sessions()
        # Initialize last known session_map
        self._last_session_map = await self._load_current_session_map()

        while self._running:
            try:
                # Load hook-based session map updates
                await session_manager.load_session_map()

                # Detect session_map changes and cleanup replaced/removed sessions
                current_map = await self._detect_and_cleanup_changes()
                active_session_ids = set(current_map.values())

                # Register codex sessions for per-window codex windows
                if config.codex_windows:
                    codex_scanned = await self.scan_codex_sessions()
                    windows = await tmux_manager.list_windows()
                    # Build cwd → window_id map for codex windows
                    cwd_to_wid: dict[str, str] = {}
                    for w in windows:
                        if w.window_id not in config.codex_windows:
                            continue
                        try:
                            cwd_to_wid[str(Path(w.cwd).resolve())] = w.window_id
                        except (OSError, ValueError):
                            cwd_to_wid[w.cwd] = w.window_id
                    for s in codex_scanned:
                        active_session_ids.add(s.session_id)
                        # Match session to its window via cwd
                        try:
                            norm_cwd = str(Path(s.cwd).resolve())
                        except (OSError, ValueError):
                            norm_cwd = s.cwd
                        wid = cwd_to_wid.get(norm_cwd)
                        if wid:
                            state = session_manager.get_window_state(wid)
                            if state.session_id != s.session_id:
                                state.session_id = s.session_id
                                state.cwd = s.cwd
                                logger.info(
                                    "Codex session registered: window=%s sid=%s",
                                    wid,
                                    s.session_id,
                                )

                # Check for new messages (all I/O is async)
                new_messages = await self.check_for_updates(active_session_ids)

                for msg in new_messages:
                    status = "complete" if msg.is_complete else "streaming"
                    preview = msg.text[:80] + ("..." if len(msg.text) > 80 else "")
                    logger.info("[%s] session=%s: %s", status, msg.session_id, preview)
                    if self._message_callback:
                        try:
                            await self._message_callback(msg)
                        except Exception as e:
                            logger.error(f"Message callback error: {e}")

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            await asyncio.sleep(self.poll_interval)

        logger.info("Session monitor stopped")

    def start(self) -> None:
        if self._running:
            logger.warning("Monitor already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self.state.save()
        logger.info("Session monitor stopped and state saved")
