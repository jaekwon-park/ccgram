"""Outbox watcher: scan ~/.ccgram/outbox/@{window_id}/ for files and send to Telegram.

Claude Code can write files to the outbox directory and ccgram will deliver
them to the correct Telegram thread. Files are deleted after CCBOT_OUTBOX_TTL
seconds (default 60).

Directory layout:
    ~/.ccgram/outbox/@1/report.txt   → sent to thread bound to window @1
    ~/.ccgram/outbox/@7/data.csv     → sent to thread bound to window @7
"""

import asyncio
import logging
import time
from pathlib import Path

from telegram import Bot
from telegram.error import TelegramError

from ..config import config
from ..session import session_manager

logger = logging.getLogger(__name__)

POLL_INTERVAL = 2.0

# Track files that have been sent: path -> sent_at timestamp
_sent_files: dict[Path, float] = {}

_watcher_task: asyncio.Task | None = None


def start_outbox_watcher(bot: Bot) -> None:
    global _watcher_task
    config.outbox_dir.mkdir(parents=True, exist_ok=True)
    _watcher_task = asyncio.create_task(_outbox_watcher_loop(bot))
    logger.info(
        "Outbox watcher started (dir=%s, ttl=%ds)", config.outbox_dir, config.outbox_ttl
    )


def stop_outbox_watcher() -> None:
    if _watcher_task:
        _watcher_task.cancel()


async def _outbox_watcher_loop(bot: Bot) -> None:
    try:
        while True:
            await _scan_outbox(bot)
            await _cleanup_sent_files()
            await asyncio.sleep(POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Outbox watcher stopped")


async def _scan_outbox(bot: Bot) -> None:
    try:
        for window_dir in config.outbox_dir.iterdir():
            if not window_dir.is_dir():
                continue
            window_id = window_dir.name  # e.g. "@1"
            if not window_id.startswith("@"):
                continue
            for file_path in window_dir.iterdir():
                if not file_path.is_file():
                    continue
                if file_path in _sent_files:
                    continue
                await _send_file(bot, window_id, file_path)
    except Exception:
        logger.exception("Error scanning outbox")


async def _send_file(bot: Bot, window_id: str, file_path: Path) -> None:
    targets = session_manager.find_users_for_window(window_id)
    if not targets:
        logger.warning(
            "Outbox: no users bound to window %s, skipping %s",
            window_id,
            file_path.name,
        )
        _sent_files[file_path] = time.monotonic()
        return

    for user_id, chat_id, thread_id in targets:
        try:
            kwargs: dict = {}
            if thread_id is not None:
                kwargs["message_thread_id"] = thread_id
            with file_path.open("rb") as f:
                await bot.send_document(
                    chat_id=chat_id,
                    document=f,
                    filename=file_path.name,
                    **kwargs,
                )
            logger.info(
                "Outbox: sent %s to user=%d chat=%d thread=%s",
                file_path.name,
                user_id,
                chat_id,
                thread_id,
            )
        except TelegramError as e:
            logger.error(
                "Outbox: failed to send %s to user=%d: %s", file_path.name, user_id, e
            )

    _sent_files[file_path] = time.monotonic()


async def _cleanup_sent_files() -> None:
    now = time.monotonic()
    to_delete = [
        p for p, sent_at in _sent_files.items() if now - sent_at >= config.outbox_ttl
    ]
    for path in to_delete:
        try:
            path.unlink(missing_ok=True)
            logger.debug("Outbox: deleted %s (ttl expired)", path)
        except OSError as e:
            logger.warning("Outbox: could not delete %s: %s", path, e)
        del _sent_files[path]
