"""Make the reply currently being generated uninterruptible (crisis messages).

``llm_node`` runs inside the reply's speech task, where LiveKit keeps the reply's
SpeechHandle in a private context variable (``agent_activity._SpeechHandleContextVar``).
Setting ``allow_interruptions`` on it is public API. The private lookup is isolated here;
tests/test_speech_lock.py fails if a LiveKit upgrade removes it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("praxis_telefon.speech_lock")


def lock_current_speech() -> bool:
    """True if the current reply is now uninterruptible."""
    try:
        from livekit.agents.voice.agent_activity import _SpeechHandleContextVar
    except ImportError:
        logger.debug("speech lock: SpeechHandle context not available")
        return False
    handle = _SpeechHandleContextVar.get(None)
    if handle is None:
        return False
    try:
        handle.allow_interruptions = False
    except RuntimeError:  # already interrupted
        return False
    return True
