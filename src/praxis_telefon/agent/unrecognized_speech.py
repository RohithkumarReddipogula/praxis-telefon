"""Detect caller speech that produced no transcript (an "empty transcript" strike).

LiveKit never ends a user turn without a transcript, so ``llm_node`` never sees these.
Instead, session events are watched: the caller spoke (VAD) for at least
MIN_SPEECH_S while the agent was not speaking, and no final transcript arrived within
GRACE_S after the caller stopped. Speech over the agent's own voice is ignored, so
echo and backchannels ("mhm") are not strikes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from livekit.agents import (
    AgentSession,
    AgentStateChangedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)

MIN_SPEECH_S = 0.6
GRACE_S = 2.0


class UnrecognizedSpeechDetector:
    """Pure bookkeeping; ``main.attach_unrecognized_speech`` wires it to the session."""

    def __init__(self) -> None:
        self._started_at: float | None = None
        self._agent_was_speaking = False
        self._got_transcript = False

    def caller_started(self, now: float, *, agent_speaking: bool) -> None:
        self._started_at = now
        self._agent_was_speaking = agent_speaking
        self._got_transcript = False

    def agent_started_speaking(self) -> None:
        self._agent_was_speaking = True

    def final_transcript(self, text: str) -> None:
        if text.strip():
            self._got_transcript = True

    def caller_stopped(self, now: float) -> bool:
        """True if this speech is a candidate; confirm with ``still_empty`` after GRACE_S."""
        started, self._started_at = self._started_at, None
        if started is None or self._agent_was_speaking:
            return False
        return now - started >= MIN_SPEECH_S and not self._got_transcript

    def still_empty(self) -> bool:
        # A new utterance started meanwhile resets the flag; a late final sets it.
        return not self._got_transcript and self._started_at is None


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
        if ev.is_final:
            detector.final_transcript(ev.transcript)

    @session.on("user_state_changed")
    def _on_user(ev: UserStateChangedEvent) -> None:
        if ev.new_state == "speaking":
            detector.caller_started(now(), agent_speaking=agent_state["speaking"])
        elif ev.old_state == "speaking" and detector.caller_stopped(now()):
            loop = asyncio.get_running_loop()
            loop.call_later(grace_s, _confirm)

    def _confirm() -> None:
        if detector.still_empty():
            on_unrecognized()

    return detector
