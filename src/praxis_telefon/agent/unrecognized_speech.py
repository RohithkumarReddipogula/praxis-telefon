"""Detect caller speech that produced no transcript (an "empty transcript" strike).

LiveKit never ends a user turn without a transcript, so ``llm_node`` never sees these.
Instead, session events are watched: the caller spoke (VAD) for at least
MIN_SPEECH_S while the agent was not speaking, and no transcript arrived within
GRACE_S after the caller stopped. Speech over the agent's own voice is ignored, so
echo and backchannels ("mhm") are not strikes.

Bug, cloud browser test 2026-11-08: "Der Erste bitte" got the repeat message because its
final transcript came about 10 s after the caller stopped. Interim transcripts now count
as recognised speech too, and each utterance logs its timing (``utterance_timing``: speech
length, first interim after the start, final after the stop, interim count; times and
counts only) so the cause of a late final shows in the logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from livekit.agents import (
    AgentSession,
    AgentStateChangedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)

logger = logging.getLogger("praxis_telefon.turns")
MIN_SPEECH_S = 0.6
GRACE_S = 2.0


class UnrecognizedSpeechDetector:
    """Pure bookkeeping; ``main.attach_unrecognized_speech`` wires it to the session."""

    def __init__(self) -> None:
        self._started_at: float | None = None
        self._agent_was_speaking = False
        self._got_transcript = False
        self._began: float | None = None  # start of the utterance being timed
        self._stopped: float | None = None
        self._first_interim: float | None = None
        self._final: float | None = None
        self._interims = 0
        self._agent_at_start = False
        self._timed = False

    def caller_started(self, now: float, *, agent_speaking: bool) -> None:
        self._started_at = now
        self._agent_was_speaking = agent_speaking
        self._got_transcript = False
        self._began, self._stopped, self._first_interim, self._final = now, None, None, None
        self._interims = 0
        self._agent_at_start = agent_speaking
        self._timed = False

    def agent_started_speaking(self) -> None:
        self._agent_was_speaking = True

    def interim_transcript(self, now: float, text: str) -> None:
        if text.strip():
            self._got_transcript = True
            self._interims += 1
            if self._first_interim is None:
                self._first_interim = now

    def final_transcript(self, text: str, now: float | None = None) -> None:
        if text.strip():
            self._got_transcript = True
            if now is not None and self._final is None:
                self._final = now

    def has_final(self) -> bool:
        return self._final is not None

    def timing_ready(self) -> bool:
        """Final and end of speech both known (cloud test 2026-11-08: finals often come
        before VAD reports the end, so the record waits for whichever is later)."""
        return self._final is not None and self._stopped is not None and not self._timed

    def take_timing(self) -> dict[str, Any] | None:
        """The utterance's timing record once, when it is complete."""
        if not self.timing_ready():
            return None
        self._timed = True
        return self.timing(strike=False)

    def caller_stopped(self, now: float) -> bool:
        """True if this speech is a candidate; confirm with ``still_empty`` after GRACE_S."""
        started, self._started_at = self._started_at, None
        self._stopped = now
        if started is None or self._agent_was_speaking:
            return False
        return now - started >= MIN_SPEECH_S and not self._got_transcript

    def still_empty(self) -> bool:
        # A new utterance started meanwhile resets the flag; a late transcript sets it.
        return not self._got_transcript and self._started_at is None

    def timing(self, *, strike: bool) -> dict[str, Any]:
        """Times and counts of the last utterance, never its words."""

        def since(a: float | None, b: float | None) -> float | None:
            return round(b - a, 3) if a is not None and b is not None else None

        return {
            "event": "utterance_timing",
            "speech_s": since(self._began, self._stopped),
            "first_interim_after_start_s": since(self._began, self._first_interim),
            "final_after_stop_s": since(self._stopped, self._final),
            "interims": self._interims,
            "agent_was_speaking": self._agent_at_start,
            "strike": strike,
        }


def attach_unrecognized_speech(
    session: AgentSession[Any],
    on_unrecognized: Callable[[], object],
    *,
    clock: Callable[[], float] | None = None,
    grace_s: float = GRACE_S,
) -> UnrecognizedSpeechDetector:
    detector = UnrecognizedSpeechDetector()
    now = clock or (lambda: asyncio.get_running_loop().time())
    agent_state = {"speaking": False}

    @session.on("agent_state_changed")
    def _on_agent(ev: AgentStateChangedEvent) -> None:
        agent_state["speaking"] = ev.new_state == "speaking"
        if agent_state["speaking"]:
            detector.agent_started_speaking()

    @session.on("user_input_transcribed")
    def _on_transcript(ev: UserInputTranscribedEvent) -> None:
        if not ev.is_final:
            detector.interim_transcript(now(), ev.transcript)
            return
        detector.final_transcript(ev.transcript, now=now())
        if (record := detector.take_timing()) is not None:
            _log(record)

    @session.on("user_state_changed")
    def _on_user(ev: UserStateChangedEvent) -> None:
        if ev.new_state == "speaking":
            detector.caller_started(now(), agent_speaking=agent_state["speaking"])
        elif ev.old_state == "speaking":
            if detector.caller_stopped(now()):
                asyncio.get_running_loop().call_later(grace_s, _confirm)
            elif (record := detector.take_timing()) is not None:
                _log(record)

    def _confirm() -> None:
        if detector.still_empty():
            _log(detector.timing(strike=True))
            on_unrecognized()

    return detector


def _log(record: dict[str, Any]) -> None:
    logger.info(json.dumps(record, separators=(",", ":")))
