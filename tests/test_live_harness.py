"""Offline checks of the scripted caller used by the live tests (no LLM calls)."""

from __future__ import annotations

import pytest
from livekit.agents import llm
from livekit.agents.voice.run_result import ChatMessageEvent

from tests.live.harness import Call, caller_reply


def _call_after(agent_text: str) -> Call:
    call = Call()
    call.events.append(
        ChatMessageEvent(item=llm.ChatMessage(role="assistant", content=[agent_text]))
    )
    return call


@pytest.mark.parametrize(
    "question",
    [
        "Möchten Sie einen neuen Termin, oder geht es um einen bestehenden?",
        "Geht es um einen neuen Termin oder um einen bestehenden Termin?",
        "Geht es um einen neuen oder einen bestehenden Termin?",
    ],
)
def test_caller_answers_new_to_new_or_existing_question(question: str) -> None:
    # Regression (live test friday_tomorrow_closed-0): the caller answered "Ja, das stimmt
    # so" and the agent asked again six times.
    call = _call_after(question)
    assert caller_reply(call, "Max Mustermann") == "Einen neuen Termin, bitte."
    assert call.intent_asked == 1


def test_caller_still_says_yes_to_a_read_back() -> None:
    call = _call_after("Dienstag um 8 Uhr bei Dr. Weber, für Max Mustermann. Stimmt das so?")
    assert caller_reply(call, "Max Mustermann") == "Ja, das stimmt so."
    assert call.intent_asked == 0


# --- C: time of day / doctor questions ---------------------------------------------------


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("Möchten Sie lieber vormittags oder nachmittags?", "Egal, jede Uhrzeit passt."),
        (
            "Hätten Sie lieber einen Termin am Vormittag oder am Nachmittag?",
            "Egal, jede Uhrzeit passt.",
        ),
        (
            "Would you prefer a morning or an afternoon appointment, or does any time work?",
            "Any time is fine.",
        ),
        (
            "Which doctor would you prefer, Dr. Weber or another doctor?",
            "Any doctor is fine.",
        ),
        ("Ist Ihnen ein bestimmter Arzt wichtig?", "Egal, bei wem."),
    ],
)
def test_caller_answers_time_of_day_and_doctor_questions(question: str, answer: str) -> None:
    assert caller_reply(_call_after(question), "Max Mustermann") == answer


def test_english_agent_gets_english_answers() -> None:
    assert caller_reply(_call_after("Is that correct?"), "Max Mustermann") == "Yes, that's right."
    call = _call_after("Is this about a new appointment or an existing one?")
    assert caller_reply(call, "Max Mustermann") == "A new appointment, please."


# --- B: a day named in either language counts as named -------------------------------


def test_translated_day_counts_as_named_by_the_caller() -> None:
    from tests.live.harness import self_computed_days

    call = _call_after("Gern, ich schaue für nächsten Donnerstag im November nach.")
    assert self_computed_days(call, "Do you have anything next Thursday in November?") == set()
    assert self_computed_days(call, "Haben Sie morgen etwas frei?") == {"thursday", "november"}


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        (
            "Could you please tell me your date of birth so I can look up your details?",
            "My name is Max Mustermann.",
        ),
        ("May I have your name, please?", "My name is Max Mustermann."),
        ("Wie ist Ihr Geburtsdatum?", "Mein Name ist Max Mustermann."),
    ],
)
def test_caller_answers_identity_questions_in_both_languages(question: str, answer: str) -> None:
    # Regression (live run 2, english_next_thursday-2): "date of birth" got "Yes, that's
    # right." and the agent kept asking.
    call = _call_after(question)
    assert caller_reply(call, "Max Mustermann") == answer
    assert call.identity_asked == 1
