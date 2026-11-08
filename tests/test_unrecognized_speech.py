"""Caller speech without a transcript becomes an "empty transcript" strike."""

from __future__ import annotations

import asyncio

import pytest
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


# --- regression: cloud browser test 2026-11-08 (log 13:47:14) ----------------------------
# "Der Erste bitte" got the repeat message: no final transcript within GRACE_S after the
# caller stopped (it came about 10 s later). Interim transcripts now count as recognised
# speech, and every utterance logs its timing (times and counts only) to show the cause.


def test_interim_transcript_means_the_speech_was_recognised() -> None:
    d = UnrecognizedSpeechDetector()
    d.caller_started(10.0, agent_speaking=False)
    d.interim_transcript(10.4, "Der Erste")
    assert not d.caller_stopped(11.0)
    assert not d.still_empty()


def test_interim_after_the_stop_cancels_the_strike() -> None:
    d = UnrecognizedSpeechDetector()
    d.caller_started(10.0, agent_speaking=False)
    assert d.caller_stopped(11.0)
    d.interim_transcript(12.0, "Der Erste")
    assert not d.still_empty()


def test_empty_interim_does_not_count() -> None:
    d = UnrecognizedSpeechDetector()
    d.caller_started(10.0, agent_speaking=False)
    d.interim_transcript(10.4, "  ")
    assert d.caller_stopped(11.0)
    assert d.still_empty()


def test_timing_record_holds_times_and_counts_only() -> None:
    d = UnrecognizedSpeechDetector()
    d.caller_started(10.0, agent_speaking=False)
    d.interim_transcript(10.5, "Der Erste")
    d.interim_transcript(10.9, "Der Erste bitte")
    d.caller_stopped(11.2)
    d.final_transcript("Der Erste bitte", now=21.2)
    record = d.timing(strike=False)
    assert record == {
        "event": "utterance_timing",
        "speech_s": 1.2,
        "first_interim_after_start_s": 0.5,
        "final_after_stop_s": 10.0,
        "interims": 2,
        "agent_was_speaking": False,
        "strike": False,
    }
    assert "Erste" not in str(record)


def test_session_wiring_does_not_strike_on_an_interim(caplog: pytest.LogCaptureFixture) -> None:
    import json
    import logging

    async def scenario() -> list[int]:
        loop = asyncio.get_running_loop()
        session: AgentSession[None] = AgentSession(loop=loop)
        hits: list[int] = []
        times = iter([0.0, 0.4, 1.5, 9.0, 9.1])
        attach_unrecognized_speech(
            session, lambda: hits.append(1), clock=lambda: next(times), grace_s=0.01
        )
        session.emit("user_state_changed", _user("listening", "speaking"))
        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="Der Erste", is_final=False),
        )
        session.emit("user_state_changed", _user("speaking", "listening"))
        await asyncio.sleep(0.05)
        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="Der Erste bitte", is_final=True),
        )
        return hits

    with caplog.at_level(logging.INFO, logger="praxis_telefon.turns"):
        assert asyncio.run(scenario()) == []
    records = [json.loads(r.getMessage()) for r in caplog.records]
    assert [r["event"] for r in records] == ["utterance_timing"]
    assert records[0]["interims"] == 1 and records[0]["strike"] is False
    assert "Erste" not in caplog.text


def test_timing_is_complete_when_the_final_comes_before_the_stop() -> None:
    # Cloud browser test 2026-11-08: most finals came before VAD reported the end of
    # speech, so the record (written at the final) had no speech_s / final_after_stop_s.
    # It is written at whichever comes later; a final before the stop is negative.
    d = UnrecognizedSpeechDetector()
    d.caller_started(10.0, agent_speaking=False)
    d.interim_transcript(10.4, "Ja")
    d.final_transcript("Ja, tus.", now=11.0)
    assert d.timing_ready() is False  # the stop is still to come
    d.caller_stopped(11.3)
    assert d.timing_ready() is True
    record = d.timing(strike=False)
    assert record["speech_s"] == 1.3 and record["final_after_stop_s"] == -0.3
