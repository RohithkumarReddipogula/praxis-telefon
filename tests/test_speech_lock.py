"""Crisis messages are made uninterruptible via the reply's SpeechHandle."""

from __future__ import annotations

import asyncio

from livekit.agents.voice.speech_handle import SpeechHandle

from praxis_telefon.agent.speech_lock import lock_current_speech


def test_no_reply_in_progress_returns_false() -> None:
    assert lock_current_speech() is False


def test_locks_the_current_speech_handle() -> None:
    from livekit.agents.voice.agent_activity import _SpeechHandleContextVar

    async def scenario() -> tuple[bool, bool]:
        handle = SpeechHandle.create(allow_interruptions=True)
        token = _SpeechHandleContextVar.set(handle)
        try:
            return lock_current_speech(), handle.allow_interruptions
        finally:
            _SpeechHandleContextVar.reset(token)

    assert asyncio.run(scenario()) == (True, False)


def test_livekit_internal_still_exists() -> None:
    """Fails loudly after a LiveKit upgrade that removes what the lock relies on."""
    from livekit.agents.voice import agent_activity

    assert hasattr(agent_activity, "_SpeechHandleContextVar")
    assert isinstance(SpeechHandle.allow_interruptions, property)
    assert SpeechHandle.allow_interruptions.fset is not None
