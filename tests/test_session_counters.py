"""M3 Part C counters (offline): barge-in time to quiet, and late transcripts.

Both only record times, counts and lengths, never the caller's words.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
from livekit.agents import (
    AgentSession,
    AgentStateChangedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)

from praxis_telefon.agent.barge_in import BargeInDetector, attach_barge_in
from praxis_telefon.agent.late_transcript import LateTranscriptDetector, attach_late_transcript
from praxis_telefon.agent.main import entrypoint

WORDS = "Nein lieber Donnerstag"


# --- barge-in --------------------------------------------------------------------------


def test_interruption_records_time_to_quiet() -> None:
    d = BargeInDetector()
    d.agent_speaking(True)
    d.caller_started(10.0)
    record = d.playback_finished(10.42, interrupted=True)
    assert record == {"event": "barge_in", "outcome": "stopped", "time_to_quiet_s": 0.42}


def test_caller_speech_without_a_stop_is_recorded_as_not_stopped() -> None:
    # A backchannel ("mhm") the agent talked through: no interrupted playback.
    d = BargeInDetector()
    d.agent_speaking(True)
    d.caller_started(10.0)
    # The caller stopped; LiveKit may still stop the agent, so nothing is recorded yet.
    record = d.playback_finished(14.0, interrupted=False)
    assert record == {"event": "barge_in", "outcome": "not_stopped", "time_to_quiet_s": None}


def test_caller_speech_while_the_agent_is_silent_is_no_barge_in() -> None:
    d = BargeInDetector()
    d.caller_started(10.0)
    assert d.playback_finished(11.0, interrupted=True) is None


def test_quiet_is_when_the_agent_pauses_not_when_the_turn_commits() -> None:
    # LiveKit pauses the audio on an interruption (agent state "listening") and clears it
    # only when the caller's turn commits; the caller hears silence from the pause.
    d = BargeInDetector()
    d.agent_speaking(True)
    d.caller_started(10.0)
    d.agent_quiet(10.4)
    record = d.playback_finished(11.5, interrupted=True)
    assert record == {"event": "barge_in", "outcome": "stopped", "time_to_quiet_s": 0.4}


def test_false_interruption_is_one_barge_in_with_its_pause() -> None:
    d = BargeInDetector()
    d.agent_speaking(True)
    d.caller_started(10.0)
    d.agent_quiet(10.5)
    assert d.false_interruption(13.0, resumed=True) == {
        "event": "barge_in",
        "outcome": "false_interruption",
        "time_to_quiet_s": 0.5,
        "paused_s": 2.5,
        "resumed": True,
    }


def test_start_time_is_reset_after_a_false_interruption() -> None:
    # Bug, voice test 2026-11-07 17:56:48: time_to_quiet_s 10.863 for a quick stop. The
    # "mhm" start was kept after LiveKit resumed (no playback_finished), so the next real
    # interruption was measured from the "mhm".
    d = BargeInDetector()
    d.agent_speaking(True)
    d.caller_started(10.0)  # "mhm"
    d.agent_quiet(10.5)
    d.false_interruption(13.0, resumed=True)
    d.agent_speaking(True)  # resumed
    d.caller_started(20.0)  # "Nein, lieber Donnerstag"
    d.agent_quiet(20.4)
    record = d.playback_finished(21.2, interrupted=True)
    assert record == {"event": "barge_in", "outcome": "stopped", "time_to_quiet_s": 0.4}


def test_speech_during_uninterruptible_speech_is_its_own_outcome() -> None:
    # Voice test 2026-11-07: talking during the uninterruptible greeting was logged as
    # not_stopped; it is no missed interruption.
    d = BargeInDetector()
    d.agent_speaking(True)
    d.caller_started(10.0, interruptible=False)
    record = d.playback_finished(14.0, interrupted=False)
    assert record == {"event": "barge_in", "outcome": "uninterruptible", "time_to_quiet_s": None}


def _user(old: str, new: str) -> UserStateChangedEvent:
    return UserStateChangedEvent(old_state=old, new_state=new)  # type: ignore[arg-type]


def _agent(old: str, new: str) -> AgentStateChangedEvent:
    return AgentStateChangedEvent(old_state=old, new_state=new)  # type: ignore[arg-type]


class FakeAudioOutput:
    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}

    def on(self, name: str, handler: Any) -> None:
        self.handlers[name] = handler


def test_session_wiring_measures_to_the_pause(caplog: pytest.LogCaptureFixture) -> None:
    from livekit.agents.voice.io import PlaybackFinishedEvent

    async def scenario() -> None:
        session: AgentSession[None] = AgentSession(loop=asyncio.get_running_loop())
        output = FakeAudioOutput()
        times = iter([10.0, 10.3, 11.4])
        attach_barge_in(session, clock=lambda: next(times), audio_output=lambda: output)
        session.emit("agent_state_changed", _agent("thinking", "speaking"))
        session.emit("user_state_changed", _user("listening", "speaking"))
        session.emit("agent_state_changed", _agent("speaking", "listening"))  # paused
        output.handlers["playback_finished"](
            PlaybackFinishedEvent(playback_position=2.0, interrupted=True)
        )

    with caplog.at_level(logging.INFO, logger="praxis_telefon.turns"):
        asyncio.run(scenario())
    records = [json.loads(r.getMessage()) for r in caplog.records]
    assert records == [{"event": "barge_in", "outcome": "stopped", "time_to_quiet_s": 0.3}]


def test_session_wiring_logs_barge_in(caplog: pytest.LogCaptureFixture) -> None:
    from livekit.agents.voice.io import PlaybackFinishedEvent

    async def scenario() -> None:
        session: AgentSession[None] = AgentSession(loop=asyncio.get_running_loop())
        output = FakeAudioOutput()
        times = iter([10.0, 10.35])
        attach_barge_in(session, clock=lambda: next(times), audio_output=lambda: output)
        session.emit("agent_state_changed", _agent("thinking", "speaking"))
        session.emit("user_state_changed", _user("listening", "speaking"))
        output.handlers["playback_finished"](
            PlaybackFinishedEvent(playback_position=2.0, interrupted=True)
        )

    with caplog.at_level(logging.INFO, logger="praxis_telefon.turns"):
        asyncio.run(scenario())
    records = [json.loads(r.getMessage()) for r in caplog.records]
    assert records == [{"event": "barge_in", "outcome": "stopped", "time_to_quiet_s": 0.35}]


# --- late transcripts ------------------------------------------------------------------


def test_final_after_the_turn_was_committed_is_late() -> None:
    d = LateTranscriptDetector()
    d.caller_started()
    assert d.final_transcript(10.0, WORDS) is None
    d.turn_committed(10.4)
    record = d.final_transcript(10.7, "und um neun")
    assert record == {
        "event": "late_transcript",
        "turn": 1,
        "after_commit_s": 0.3,
        "late_chars": len("und um neun"),
    }


def test_final_of_a_new_utterance_is_not_late() -> None:
    d = LateTranscriptDetector()
    d.caller_started()
    d.final_transcript(10.0, WORDS)
    d.turn_committed(10.4)
    d.caller_started()  # the caller talks again
    assert d.final_transcript(12.0, "Ja") is None


def test_empty_final_is_not_late() -> None:
    d = LateTranscriptDetector()
    d.caller_started()
    d.turn_committed(10.0)
    assert d.final_transcript(10.5, "  ") is None


def test_summary_gives_the_rate_per_turn() -> None:
    d = LateTranscriptDetector()
    for i in range(4):
        d.caller_started()
        d.final_transcript(10.0 + i, WORDS)
        d.turn_committed(10.2 + i)
    d.final_transcript(13.5, "noch was")
    assert d.summary() == {
        "event": "late_transcript_summary",
        "caller_turns": 4,
        "late": 1,
        "late_share": 0.25,
    }


def test_session_wiring_logs_late_transcripts_without_words(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from livekit.agents import ConversationItemAddedEvent, llm

    async def scenario() -> None:
        session: AgentSession[None] = AgentSession(loop=asyncio.get_running_loop())
        times = iter([10.0, 10.4, 10.9])
        attach_late_transcript(session, clock=lambda: next(times))
        session.emit("user_state_changed", _user("listening", "speaking"))
        session.emit(
            "user_input_transcribed", UserInputTranscribedEvent(transcript=WORDS, is_final=True)
        )
        msg = llm.ChatMessage(role="user", content=[WORDS])
        session.emit("conversation_item_added", ConversationItemAddedEvent(item=msg))
        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="und um neun", is_final=True),
        )

    with caplog.at_level(logging.INFO, logger="praxis_telefon.turns"):
        asyncio.run(scenario())
    records = [json.loads(r.getMessage()) for r in caplog.records]
    assert records == [
        {"event": "late_transcript", "turn": 1, "after_commit_s": 0.5, "late_chars": 11}
    ]
    assert "Donnerstag" not in caplog.text and "neun" not in caplog.text


def test_livekit_warning_is_counted(caplog: pytest.LogCaptureFixture) -> None:
    # LiveKit warns once per session, then logs at debug level; the counter notes the
    # warning so the two can be compared.
    from praxis_telefon.agent.late_transcript import LiveKitLateWarning

    counter = LiveKitLateWarning()
    lk = logging.getLogger("livekit.agents")
    lk.addFilter(counter)
    try:
        lk.warning("transcript arrives after turn has been committed. consider raising ...")
        lk.warning("something else")
    finally:
        lk.removeFilter(counter)
    assert counter.count == 1


def test_entrypoint_attaches_both_counters() -> None:
    names = entrypoint.__code__.co_names
    assert "attach_barge_in" in names and "attach_late_transcript" in names
