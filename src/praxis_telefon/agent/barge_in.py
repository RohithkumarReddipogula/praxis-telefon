"""Barge-in record: how long the agent keeps talking after the caller starts (M3 Part C).

When the caller starts speaking while the agent speaks, the time until the agent's audio
output reports an interrupted playback is the time to quiet. If the agent's playback ends
without an interruption, the caller's speech was talked through (a backchannel, or speech
LiveKit did not count as an interruption). LiveKit's false-interruption event (the agent
resumed after the caller went quiet) is logged as well. Times only, never words.

The console's speaker buffer is not included; it adds a constant few tens of ms.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from livekit.agents import (
    AgentFalseInterruptionEvent,
    AgentSession,
    AgentStateChangedEvent,
    UserStateChangedEvent,
)

logger = logging.getLogger("praxis_telefon.turns")


class BargeInDetector:
    """Pure bookkeeping; ``attach_barge_in`` wires it to the session."""

    def __init__(self) -> None:
        self._agent_speaking = False
        self._caller_started_at: float | None = None

    def agent_speaking(self, speaking: bool) -> None:
        self._agent_speaking = speaking

    def caller_started(self, now: float) -> None:
        if self._agent_speaking and self._caller_started_at is None:
            self._caller_started_at = now

    def playback_finished(self, now: float, *, interrupted: bool) -> dict[str, Any] | None:
        started, self._caller_started_at = self._caller_started_at, None
        if started is None:
            return None
        return {
            "event": "barge_in",
            "outcome": "stopped" if interrupted else "not_stopped",
            "time_to_quiet_s": round(now - started, 3) if interrupted else None,
        }

    def false_interruption(self, *, resumed: bool) -> dict[str, Any]:
        return {"event": "false_interruption", "resumed": resumed}


def _log(record: dict[str, Any] | None) -> None:
    if record is not None:
        logger.info(json.dumps(record, separators=(",", ":")))


def attach_barge_in(
    session: AgentSession[Any],
    *,
    clock: Callable[[], float] | None = None,
    audio_output: Callable[[], Any] | None = None,
) -> BargeInDetector:
    """The audio output exists only once the session runs, so it is subscribed to the
    first time the agent speaks."""
    detector = BargeInDetector()
    now = clock or time.perf_counter
    output_of = audio_output or (lambda: session.output.audio)
    watched: list[Any] = []

    def _on_playback_finished(ev: Any) -> None:
        _log(detector.playback_finished(now(), interrupted=bool(ev.interrupted)))

    @session.on("agent_state_changed")
    def _on_agent(ev: AgentStateChangedEvent) -> None:
        speaking = ev.new_state == "speaking"
        detector.agent_speaking(speaking)
        output = output_of() if speaking else None
        if output is not None and output not in watched:
            watched.append(output)
            output.on("playback_finished", _on_playback_finished)

    @session.on("user_state_changed")
    def _on_user(ev: UserStateChangedEvent) -> None:
        # Only the start matters: when the caller stops, LiveKit may still stop the agent.
        if ev.new_state == "speaking":
            detector.caller_started(now())

    @session.on("agent_false_interruption")
    def _on_false(ev: AgentFalseInterruptionEvent) -> None:
        _log(detector.false_interruption(resumed=ev.resumed))

    return detector
