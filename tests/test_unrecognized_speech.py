"""Caller speech without a transcript becomes an "empty transcript" strike."""

from __future__ import annotations

import asyncio

from livekit.agents import (
    AgentSession,
    AgentStateChangedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)

from praxis_telefon.agent.main import entrypoint
from praxis_telefon.agent.unrecognized_speech import (
    MIN_SPEECH_S,
    UnrecognizedSpeechDetector,
    attach_unrecognized_speech,
)


def test_speech_without_transcript_is_a_candidate() -> None:
    d = UnrecognizedSpeechDetector()
    d.caller_started(10.0, agent_speaking=False)
    assert d.caller_stopped(10.0 + MIN_SPEECH_S + 0.1)
    assert d.still_empty()


def test_speech_with_transcript_is_not() -> None:
    d = UnrecognizedSpeechDetector()
    d.caller_started(10.0, agent_speaking=False)
    d.final_transcript("Ja")
    assert not d.caller_stopped(12.0)


def test_late_transcript_cancels_the_strike() -> None:
    d = UnrecognizedSpeechDetector()
    d.caller_started(10.0, agent_speaking=False)
    assert d.caller_stopped(12.0)
    d.final_transcript("Guten Tag")
    assert not d.still_empty()


def test_short_noise_is_ignored() -> None:
    d = UnrecognizedSpeechDetector()
    d.caller_started(10.0, agent_speaking=False)
    assert not d.caller_stopped(10.0 + MIN_SPEECH_S / 2)


def test_speech_over_the_agent_is_ignored() -> None:
    d = UnrecognizedSpeechDetector()
    d.caller_started(10.0, agent_speaking=True)
    assert not d.caller_stopped(13.0)


def test_empty_final_transcript_does_not_count_as_text() -> None:
    d = UnrecognizedSpeechDetector()
    d.caller_started(10.0, agent_speaking=False)
    d.final_transcript("   ")
    assert d.caller_stopped(12.0)


def _user(old: str, new: str) -> UserStateChangedEvent:
    return UserStateChangedEvent(old_state=old, new_state=new)  # type: ignore[arg-type]


def test_session_wiring_reports_unrecognized_speech() -> None:
    async def scenario(with_transcript: bool, agent_speaking: bool) -> int:
        loop = asyncio.get_running_loop()
        session: AgentSession[None] = AgentSession(loop=loop)
        hits: list[int] = []
        times = iter([0.0, 1.5])
        attach_unrecognized_speech(
            session, lambda: hits.append(1), clock=lambda: next(times), grace_s=0.01
        )
        if agent_speaking:
            session.emit(
                "agent_state_changed",
                AgentStateChangedEvent(old_state="listening", new_state="speaking"),
            )
        session.emit("user_state_changed", _user("listening", "speaking"))
        if with_transcript:
            session.emit(
                "user_input_transcribed",
                UserInputTranscribedEvent(transcript="Hallo", is_final=True),
            )
        session.emit("user_state_changed", _user("speaking", "listening"))
        await asyncio.sleep(0.05)
        return len(hits)

    assert asyncio.run(scenario(with_transcript=False, agent_speaking=False)) == 1
    assert asyncio.run(scenario(with_transcript=True, agent_speaking=False)) == 0
    assert asyncio.run(scenario(with_transcript=False, agent_speaking=True)) == 0


def test_entrypoint_attaches_the_detector() -> None:
    assert "attach_unrecognized_speech" in entrypoint.__code__.co_names
