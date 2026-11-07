"""Late transcripts: a final arriving after the caller's turn was committed (M3 Part C).

LiveKit warns "transcript arrives after turn has been committed, consider raising
min_delay" only once per session and logs later ones at debug level, so the warning says
nothing about the rate. This counts every final that arrives after the caller's turn was
added to the conversation, with no new speech in between, and logs a summary per call.
Decision rule (user, 2026-11-07): raise ``min_delay`` if more than 5 % of caller turns are
late, or any late turn caused a wrong or doubled answer. Lengths only, never words.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from livekit.agents import (
    AgentSession,
    CloseEvent,
    ConversationItemAddedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    llm,
)

logger = logging.getLogger("praxis_telefon.turns")
LIVEKIT_WARNING = "transcript arrives after turn has been committed"


class LateTranscriptDetector:
    """Pure bookkeeping; ``attach_late_transcript`` wires it to the session."""

    def __init__(self) -> None:
        self.turns = 0
        self.late = 0
        self._committed_at: float | None = None

    def caller_started(self) -> None:
        self._committed_at = None  # finals from now on belong to the new utterance

    def turn_committed(self, now: float) -> None:
        self.turns += 1
        self._committed_at = now

    def final_transcript(self, now: float, text: str) -> dict[str, Any] | None:
        if not text.strip() or self._committed_at is None:
            return None
        self.late += 1
        return {
            "event": "late_transcript",
            "turn": self.turns,
            "after_commit_s": round(now - self._committed_at, 3),
            "late_chars": len(text.strip()),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "event": "late_transcript_summary",
            "caller_turns": self.turns,
            "late": self.late,
            "late_share": round(self.late / self.turns, 3) if self.turns else 0.0,
        }


class LiveKitLateWarning(logging.Filter):
    """Counts LiveKit's own warning (first occurrence per session), to compare."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if LIVEKIT_WARNING in record.getMessage():
            self.count += 1
        return True


def _log(record: dict[str, Any] | None) -> None:
    if record is not None:
        logger.info(json.dumps(record, separators=(",", ":")))


def attach_late_transcript(
    session: AgentSession[Any], *, clock: Callable[[], float] | None = None
) -> LateTranscriptDetector:
    detector = LateTranscriptDetector()
    now = clock or time.perf_counter
    warnings = LiveKitLateWarning()
    livekit_logger = logging.getLogger("livekit.agents")
    livekit_logger.addFilter(warnings)

    @session.on("user_state_changed")
    def _on_user(ev: UserStateChangedEvent) -> None:
        if ev.new_state == "speaking":
            detector.caller_started()

    @session.on("user_input_transcribed")
    def _on_transcript(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final:
            _log(detector.final_transcript(now(), ev.transcript))

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        if isinstance(ev.item, llm.ChatMessage) and ev.item.role == "user":
            detector.turn_committed(now())

    @session.on("close")
    def _on_close(ev: CloseEvent) -> None:
        livekit_logger.removeFilter(warnings)
        _log({**detector.summary(), "livekit_warnings": warnings.count})

    return detector
