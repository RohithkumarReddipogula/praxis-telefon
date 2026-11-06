"""Stop LiveKit console-mode audio before the job's event loop closes.

Bug: stopping ``console`` with Ctrl+C printed "RuntimeError: Event loop is closed" from
the sounddevice callbacks. LiveKit's ``_audio_mode`` (``livekit/agents/cli/_legacy.py``,
still so in 1.8.11) stops the mic and speaker streams only after its loop ends normally,
not in a ``finally``. On Ctrl+C they keep running on PortAudio's thread and post audio
frames to the job's event loop after it was closed.

Fix: a job shutdown callback. LiveKit awaits those inside the job's loop, so the streams
are stopped while the loop is still open. In ``dev`` mode the console is not enabled
and this does nothing.

``_legacy`` is private and "being phased out", so all use of it is here, and a missing
API only logs at debug level. tests/test_console_audio.py checks the API still exists.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("praxis_telefon.console_audio")


def _livekit_console() -> Any | None:
    try:
        from livekit.agents.cli._legacy import AgentsConsole
    except ImportError:
        return None
    get_instance = getattr(AgentsConsole, "get_instance", None)
    return get_instance() if callable(get_instance) else None


def stop_console_audio(console: object) -> None:
    """Stop mic and speaker of an enabled LiveKit console; otherwise do nothing."""
    if console is None or not getattr(console, "enabled", False):
        return
    for name in ("set_microphone_enabled", "set_speaker_enabled"):
        stop = getattr(console, name, None)
        if not callable(stop):
            logger.debug("console audio: %s not available", name)
            continue
        try:
            # LiveKit's own method: closes the stream and clears the reference, so its
            # later cleanup does not close it a second time.
            stop(False)
        except Exception:
            # One failing device must not keep the other stream running.
            logger.debug("console audio: %s(False) failed", name, exc_info=True)


async def release_console_audio(_reason: str = "") -> None:
    """Job shutdown callback (LiveKit passes the shutdown reason).

    Closing a PortAudio stream waits for the device (~0.3 s), so it runs in a worker
    thread; awaiting it keeps the loop open until the streams are stopped.
    """
    await asyncio.to_thread(stop_console_audio, _livekit_console())
