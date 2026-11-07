"""Per-call safety state, derived purely from the call's events.

The state is never stored and mutated by the LLM path. It is replayed from the caller's
turns (and unrecognised-speech events) every time it is needed. That matters because
LiveKit runs ``llm_node`` preemptively on interim transcripts and may discard that
run: a pure replay cannot be left in a wrong state by a discarded run.

Rules, in order, for each caller turn:
1. Emergency or suicide phrase: crisis message, booking off for the rest of the call.
2. Asks for a human (normal mode only): offer a message for the team, message-only mode.
3. Strike: an empty transcript or low STT confidence, unless it is a short valid answer
   ("Ja", "Nein", a number, a name). Two strikes in a row: message-only mode.
   Any understood turn resets the counter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal

from praxis_telefon.safety.phrases import (
    Language,
    asks_for_human,
    detect_crisis,
    is_short_answer,
)

LOW_CONFIDENCE = 0.5
STRIKES_FOR_HANDOVER = 2


class Mode(StrEnum):
    NORMAL = "normal"
    EMERGENCY = "emergency"  # booking off; a message for the team is still possible
    HANDOVER = "handover"  # booking off; offer and take a message for the team


ActionKind = Literal[
    "none", "emergency", "suicide", "human_request", "handover_strikes", "repeat", "switch_back"
]  # switch_back: set by Receptionist (agent/call_language.py), never by replay


@dataclass(frozen=True)
class CallerTurn:
    text: str
    confidence: float | None = None  # None: typed input (tests, console text mode)


@dataclass(frozen=True)
class UnrecognizedSpeech:
    """The caller spoke (VAD) but STT produced no transcript."""


Event = CallerTurn | UnrecognizedSpeech


@dataclass(frozen=True)
class State:
    mode: Mode = Mode.NORMAL
    strikes: int = 0
    language: Language = "de"
    suicide_mentioned: bool = False


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    language: Language


def is_strike(turn: CallerTurn) -> bool:
    text = turn.text.strip()
    if not text:
        return True
    if is_short_answer(text):
        return False
    return turn.confidence is not None and turn.confidence < LOW_CONFIDENCE


def step(state: State, event: Event) -> tuple[State, Action]:
    if isinstance(event, CallerTurn):
        crisis = detect_crisis(event.text)
        if crisis is not None:
            new = State(
                mode=Mode.EMERGENCY,
                language=crisis.language,
                suicide_mentioned=state.suicide_mentioned or crisis.kind == "suicide",
            )
            return new, Action(crisis.kind, crisis.language)

        human = asks_for_human(event.text)
        if human is not None and state.mode is Mode.NORMAL:
            return State(mode=Mode.HANDOVER, language=human), Action("human_request", human)

        if not is_strike(event):
            return replace(state, strikes=0), Action("none", state.language)

    strikes = state.strikes + 1
    if state.mode is Mode.NORMAL and strikes >= STRIKES_FOR_HANDOVER:
        new = replace(state, mode=Mode.HANDOVER, strikes=0)
        return new, Action("handover_strikes", state.language)
    return replace(state, strikes=strikes), Action("repeat", state.language)


def replay(events: Iterable[Event]) -> tuple[State, Action]:
    """State after all events, and the action for the last one."""
    state, action = State(), Action("none", "de")
    for event in events:
        state, action = step(state, event)
    return state, action
