"""Rule-based callers: what the scripted caller says next, from the agent's last text.

``cooperative`` answers like the live tests' caller (tests/live/harness.py): the first
slot when slots are offered, "yes" to a confirmation, its name and date of birth when
asked, "any" for time of day or doctor. Two personas change one answer:
- ``change_mind``: the first time slots are offered, asks for another day instead;
- ``confused``: the first question gets "Pardon?" before the normal answer;
- ``correct_readback``: the first read-back is corrected (another time and doctor).
The caller keeps its own language (a caller who follows the agent hides wrong-language
answers). Deterministic: the same agent text and state give the same reply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from praxis_telefon.agent.speech_filters import is_new_or_existing_question

REPLIES: dict[str, dict[str, str]] = {
    "yes": {"de": "Ja, das stimmt so.", "en": "Yes, that's right."},
    "new": {"de": "Einen neuen Termin, bitte.", "en": "A new appointment, please."},
    "first_slot": {"de": "Der erste Termin passt mir.", "en": "The first one works for me."},
    "any_time": {"de": "Egal, jede Uhrzeit passt.", "en": "Any time is fine."},
    "any_doctor": {"de": "Egal, bei wem.", "en": "Any doctor is fine."},
    "any_both": {
        "de": "Egal, jede Uhrzeit und jeder Arzt passt.",
        "en": "Any time and any doctor is fine.",
    },
    "identity": {"de": "Mein Name ist {identity}.", "en": "My name is {identity}."},
    "other_day": {
        "de": "Ach nein, Donnerstag wäre mir doch lieber.",
        "en": "Actually, Thursday would suit me better.",
    },
    "correct": {
        "de": "Nein, nicht ganz: Ich hätte gern den Termin um 9 Uhr bei Dr. Weber.",
        "en": "No, not quite: I'd like the nine o'clock appointment with Dr. Weber.",
    },
    "pardon": {
        "de": "Wie bitte? Das habe ich nicht ganz verstanden.",
        "en": "Sorry, could you say that again?",
    },
}

_CONFIRM = re.compile(
    r"stimmt das|ist das (so )?(richtig|korrekt)|passt das|soll ich|einverstanden|"
    r"richtig so|darf ich|is that (right|correct)|shall i|should i|would you like",
    re.I,
)
_CHOICE = re.compile(
    r"\boder\b|\bwelche[rnms]?\b|passt (Ihnen|einer)|\bor\b|\bwhich\b|works? (best )?for you|"
    r"suits? you",
    re.I,
)
_ASKS_IDENTITY = re.compile(
    r"\bnamen?\b|geburtsdatum|geboren|date of birth|birthday|your name", re.I
)
_ASKS_SPELLING = re.compile(r"buchstabier|spell", re.I)
_ASKS_TIME = re.compile(
    r"vormittag|nachmittag|tageszeit|uhrzeit|morning|afternoon|time of day|what time", re.I
)
_ASKS_DOCTOR = re.compile(
    r"welche[nrm]? (ärztin|arzt|doktor)|bestimmte[nrm]? (ärztin|arzt)|bei wem|"
    r"which doctor|doctor would you prefer|preferred doctor|particular doctor",
    re.I,
)


@dataclass
class CallerState:
    changed_mind: bool = False
    asked_pardon: bool = False
    corrected: bool = False
    intent_asked: int = 0
    identity_asked: int = 0
    answered_slots: int = -1  # the find_slots result the caller last chose from


def reply(
    behaviour: str,
    agent_text: str,
    state: CallerState,
    *,
    identity: str,
    language: str,
    slots: bool = False,
    spelling: str = "",
) -> str:
    """``slots``: the agent's last reply came after new find_slots results; ``spelling``:
    what the caller says when asked to spell the surname (roadmap item 4)."""
    lang = language if language in ("de", "en") else "de"

    def say(key: str) -> str:
        return REPLIES[key][lang].format(identity=identity)

    asks = agent_text.rstrip().endswith("?")
    if spelling and _ASKS_SPELLING.search(agent_text):
        return spelling
    if behaviour == "confused" and asks and not state.asked_pardon:
        state.asked_pardon = True
        return say("pardon")
    if is_new_or_existing_question(agent_text):
        state.intent_asked += 1
        return say("new")
    if slots and _CHOICE.search(agent_text):
        if behaviour == "change_mind" and not state.changed_mind:
            state.changed_mind = True
            return say("other_day")
        return say("first_slot")
    # A read-back names a time and a doctor; it is a confirmation, not a time question
    # (harness run 2026-11-08, date_tomorrow_afternoon_en[3]).
    if _CONFIRM.search(agent_text):
        if behaviour == "correct_readback" and not state.corrected:
            state.corrected = True
            return say("correct")
        return say("yes")
    time_q, doctor_q = bool(_ASKS_TIME.search(agent_text)), bool(_ASKS_DOCTOR.search(agent_text))
    if asks and (time_q or doctor_q):
        return say("any_both" if time_q and doctor_q else "any_time" if time_q else "any_doctor")
    if asks and _ASKS_IDENTITY.search(agent_text):
        state.identity_asked += 1
        return say("identity")
    return say("yes")


def reply_in_call(
    behaviour: str,
    agent_text: str,
    state: CallerState,
    *,
    identity: str,
    language: str,
    slots_index: int,
    spelling: str = "",
) -> str:
    """``reply`` with the slot offer tracked: an offer counts as answered only once the
    caller chose a slot (M4 rerun 2026-11-08, confused_caller_books: a "Wie bitte?" used
    up the offer, and later choice questions got "yes")."""
    line = reply(
        behaviour,
        agent_text,
        state,
        identity=identity,
        language=language,
        slots=slots_index > state.answered_slots,
        spelling=spelling,
    )
    lang = language if language in ("de", "en") else "de"
    if line == REPLIES["first_slot"][lang]:
        state.answered_slots = slots_index
    return line
