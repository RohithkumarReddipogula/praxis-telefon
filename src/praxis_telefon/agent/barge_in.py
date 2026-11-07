"""Barge-in record: how long the agent keeps talking after the caller starts (M3 Part C).

When the caller starts speaking while the agent speaks, LiveKit pauses the agent's audio
(its state goes to "listening") once the speech is long enough, and clears the audio
when the caller's turn commits; without a turn it resumes (a false interruption). The
caller hears silence from the pause, so time to quiet is caller start -> pause (or ->
interrupted playback when there was no pause). Outcomes, one record per episode:
- ``stopped``: the turn committed and the audio was cleared;
- ``false_interruption``: paused, then resumed (``paused_s``: how long it was quiet);
- ``not_stopped``: the agent talked through (a backchannel LiveKit did not count);
- ``uninterruptible``: the caller spoke during speech that cannot be interrupted
  (greeting, crisis messages); not a miss.
Times only, never words. The console's speaker buffer (a few tens of ms) is not included.

Bugs (voice test 2026-11-07): the start of a false interruption ("mhm") was kept after
the resume, so the next real interruption read 10.863 s; and talk over the greeting was
logged as not_stopped.
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
        self._started_at: float | None = None
        self._quiet_at: float | None = None
        self._interruptible = True

    def _reset(self) -> None:
        self._started_at = self._quiet_at = None
        self._interruptible = True

    def agent_speaking(self, speaking: bool) -> None:
        self._agent_speaking = speaking

    def caller_started(self, now: float, *, interruptible: bool = True) -> None:
        if self._agent_speaking and self._started_at is None:
            self._started_at = now
            self._interruptible = interruptible

    def agent_quiet(self, now: float) -> None:
        """The agent's audio paused (state speaking -> listening) during the episode."""
        if self._started_at is not None and self._quiet_at is None:
            self._quiet_at = now

    def _time_to_quiet(self, now: float) -> float | None:
        if self._started_at is None:
            return None
        return round((self._quiet_at if self._quiet_at is not None else now) - self._started_at, 3)

    def playback_finished(self, now: float, *, interrupted: bool) -> dict[str, Any] | None:
        if self._started_at is None:
            return None
        if interrupted:
            record = {"event": "barge_in", "outcome": "stopped",
                      "time_to_quiet_s": self._time_to_quiet(now)}  # fmt: skip
        else:
            outcome = "not_stopped" if self._interruptible else "uninterruptible"
            record = {"event": "barge_in", "outcome": outcome, "time_to_quiet_s": None}
        self._reset()
        return record

    def false_interruption(self, now: float, *, resumed: bool) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": "barge_in",
            "outcome": "false_interruption",
            "time_to_quiet_s": self._time_to_quiet(now) if self._quiet_at is not None else None,
            "paused_s": round(now - self._quiet_at, 3) if self._quiet_at is not None else None,
            "resumed": resumed,
        }
        self._reset()
        return record


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
        if ev.old_state == "speaking" and not speaking:
            detector.agent_quiet(now())  # paused for an interruption, or finished
        detector.agent_speaking(speaking)
        output = output_of() if speaking else None
        if output is not None and output not in watched:
            watched.append(output)
            output.on("playback_finished", _on_playback_finished)

    @session.on("user_state_changed")
    def _on_user(ev: UserStateChangedEvent) -> None:
        # Only the start matters: when the caller stops, LiveKit may still stop the agent.
        if ev.new_state == "speaking":
            speech = session.current_speech
            interruptible = speech is None or bool(speech.allow_interruptions)
            detector.caller_started(now(), interruptible=interruptible)

    @session.on("agent_false_interruption")
    def _on_false(ev: AgentFalseInterruptionEvent) -> None:
        _log(detector.false_interruption(now(), resumed=ev.resumed))

    return detector
