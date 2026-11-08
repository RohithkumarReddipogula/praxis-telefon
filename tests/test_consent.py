"""book, move_booking and cancel_booking need a read-back question and a clear yes.

Bug, cloud browser test 2026-11-08 (log 13:47:38): after the caller chose a slot and gave
name and date of birth, the agent called book without reading back and without a yes,
then said "Ihr Termin ist am Donnerstag, 12. November ...". The prompt rule "Bestätige
genau einmal, direkt vor book" was ignored, so it is now checked in code: the agent's last
message before the caller's line ends with a confirmation question, and the caller's line
is a clear yes (polite forms included; "Ja, aber ...", "lieber" refuse).
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

import pytest
from livekit.agents import llm

from praxis_telefon.agent import consent
from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_service import ToolService
from tests.conftest import NOW

READ_BACK = (
    "Donnerstag, 12. November, um 8 Uhr bei Dr. Weber, für Max Mustermann, geboren am "
    "ersten Februar neunzehnhundertneunundsiebzig. Stimmt das so?"
)


def chat(*turns: tuple[str, str]) -> llm.ChatContext:
    ctx = llm.ChatContext()
    for role, text in turns:
        ctx.add_message(role=role, content=text)  # type: ignore[arg-type]
    return ctx


def test_the_browser_test_sequence_is_not_consent() -> None:
    # Slot choice, then name and date of birth: no read-back question, no yes.
    ctx = chat(
        ("assistant", "Am Donnerstag habe ich um 8 Uhr bei Dr. Weber oder um 8 Uhr 30 frei."),
        ("user", "Der Erste bitte."),
        ("assistant", "Gerne. Wie lautet Ihr Vor- und Nachname und Ihr Geburtsdatum?"),
        ("user", "Max Mustermann, erster Februar neunzehnhundertneunundsiebzig."),
    )
    assert consent.confirmed(ctx) is False


@pytest.mark.parametrize(
    "yes",
    [
        "Ja, das stimmt so.",
        "Ja.",
        "Ja, gerne.",
        "Ja, bitte.",
        "Jawohl.",
        "Genau so.",
        "Ja, passt.",
        "Genau.",
        "Stimmt.",
        "Richtig.",
        "Korrekt.",
        "Einverstanden.",
        "Yes, please.",
        "Sure.",
        "Yes, that's right.",
        "Correct.",
        "Okay, ja.",
    ],
)
def test_read_back_question_and_a_clear_yes_is_consent(yes: str) -> None:
    assert consent.confirmed(chat(("assistant", READ_BACK), ("user", yes))) is True


@pytest.mark.parametrize(
    "not_yes",
    [
        "Ja, aber lieber um 9 Uhr.",
        "Lieber am Freitag.",
        "Nein.",
        "Nein, das stimmt nicht.",
        "Ja, nein, warten Sie.",
        "Doch nicht.",
        "Yes, but rather on Friday.",
        "No.",
        "Hmm, vielleicht.",
        "Max Mustermann.",
    ],
)
def test_anything_but_a_clear_yes_is_no_consent(not_yes: str) -> None:
    assert consent.confirmed(chat(("assistant", READ_BACK), ("user", not_yes))) is False


def test_a_yes_to_a_question_that_is_no_read_back_is_no_consent() -> None:
    ctx = chat(("assistant", "Möchten Sie einen Termin am Vormittag?"), ("user", "Ja, gerne."))
    assert consent.confirmed(ctx) is False


def test_the_read_back_must_be_the_last_agent_message_before_the_yes() -> None:
    ctx = chat(
        ("assistant", READ_BACK),
        ("user", "Ja, gerne."),
        ("assistant", "Und noch eine Frage: Haben Sie Ihre Versichertenkarte?"),
        ("user", "Ja."),
    )
    assert consent.confirmed(ctx) is False


def test_no_caller_line_is_no_consent() -> None:
    assert consent.confirmed(chat(("assistant", READ_BACK))) is False


# --- the tools refuse without consent ---------------------------------------------------


@pytest.fixture
def agent(conn: sqlite3.Connection) -> Receptionist:
    return Receptionist(ToolService(conn, lambda: NOW))


def _call(agent: Receptionist, ctx: llm.ChatContext, tool: str, **args: Any) -> str:
    from livekit.agents import ToolError

    agent._chat_ctx = ctx  # the conversation as the tool sees it
    try:
        return str(asyncio.run(getattr(agent, tool)(None, **args)))
    except ToolError as e:
        return f"ToolError: {e}"


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("book", {"slot_id": 1, "patient_name": "Max Mustermann", "date_of_birth": "1979-02-01"}),
        ("move_booking", {"booking_id": 1, "new_slot_id": 2}),
        ("cancel_booking", {"booking_id": 1}),
    ],
)
def test_changing_tools_refuse_without_consent(
    agent: Receptionist, tool: str, args: dict[str, Any]
) -> None:
    ctx = chat(("assistant", "Wie ist Ihr Name?"), ("user", "Max Mustermann, erster Februar 1979."))
    text = _call(agent, ctx, tool, **args)
    assert text.startswith("ToolError: consent_missing")
    assert "vor" in text and "Zustimmung" in text  # read back, then act on agreement


def test_book_goes_on_to_the_normal_checks_with_consent(agent: Receptionist) -> None:
    # With consent the tool runs: here the slot was not offered, so the slot guard answers.
    ctx = chat(("assistant", READ_BACK), ("user", "Ja, gerne."))
    text = _call(
        agent, ctx, "book", slot_id=1, patient_name="Max Mustermann", date_of_birth="1979-02-01"
    )
    assert "consent_missing" not in text and "slot_not_offered" in text


# --- regression: cloud browser test 2026-11-08, call 1 ------------------------------------
# "Ja, tus." (meaning "Ja, tu's") was refused, so the caller got no booking, and the agent
# then said "Bitte sagen Sie Ja oder Nein". New rule (user, 2026-11-08): a line starting
# with "Ja" or "Yes" is a yes whatever follows, unless it contains nein, aber, lieber,
# doch nicht, nicht, no (or the English but, rather); also "okay", "in Ordnung", "machen
# Sie das", "tun Sie das", "go ahead".


@pytest.mark.parametrize(
    "yes",
    [
        "Ja, tus.",
        "Ja, tu's.",
        "Ja, machen Sie.",
        "Ja klar, gerne doch so.",
        "Yes, do it.",
        "Okay.",
        "Okay, machen Sie das.",
        "In Ordnung.",
        "Machen Sie das.",
        "Tun Sie das.",
        "Go ahead.",
        "Ja, prima.",
    ],
)
def test_lines_starting_with_ja_or_the_new_phrases_are_yes(yes: str) -> None:
    assert consent.confirmed(chat(("assistant", READ_BACK), ("user", yes))) is True


@pytest.mark.parametrize(
    "not_yes",
    [
        "Ja, aber lieber um 9 Uhr.",
        "Ja, nein, warten Sie.",
        "Ja, doch nicht.",
        "Ja, das stimmt nicht.",
        "Yes, but rather Friday.",
        "Yes, no, wait.",
    ],
)
def test_ja_with_a_refusal_word_is_no_consent(not_yes: str) -> None:
    assert consent.confirmed(chat(("assistant", READ_BACK), ("user", not_yes))) is False


def test_a_goodbye_with_okay_is_no_consent() -> None:
    # Call 1 ended with "Alles okay, danke. Tschüs." Decision: not a yes. It closes the
    # call and does not answer whether to book; a booking nobody clearly agreed to blocks a
    # slot and leaves the caller with an appointment they may not know about, which is
    # worse than no booking. "Okay" counts only at the start of the line.
    assert (
        consent.confirmed(chat(("assistant", READ_BACK), ("user", "Alles okay, danke. Tschüs.")))
        is False
    )


def test_refusal_note_never_asks_the_caller_to_say_ja_or_nein() -> None:
    # "Bitte sagen Sie Ja oder Nein" sounded robotic (call 1).
    from praxis_telefon.tools.errors import ConsentMissing

    note = ConsentMissing.message
    assert "Ja oder Nein" not in note
    assert "nie" in note and "natürlich" in note


def test_the_last_question_counts_even_with_a_sentence_after_it() -> None:
    # Harness run 2026-11-08 (date_tomorrow_afternoon_en[3]): "Should I book Tuesday ...?
    # Please answer yes or no." then "Yes, that's right." was refused: only the last
    # sentence was checked. The last question in the agent's message decides.
    read_back = (
        "I want to be sure I have it right. Should I book Tuesday, November 10th, at 3 "
        "o'clock with Dr. Brandt, for Max Mustermann? Please answer yes or no."
    )
    assert consent.confirmed(chat(("assistant", read_back), ("user", "Yes, that's right.")))


def test_a_later_other_question_still_decides() -> None:
    text = READ_BACK + " Und haben Sie Ihre Versichertenkarte dabei?"
    assert consent.confirmed(chat(("assistant", text), ("user", "Ja."))) is False
