"""FAQ answers and the sentence filters (M3 Part C, offline).

Practice facts (opening hours, closed days, address, parking, holidays, insurance card,
prescriptions, referrals, sick notes, new patients) may only be said when practice_info
backed them in this call; a closed day is also backed by find_slots ("Samstag, 14.
November: Praxis geschlossen", user 2026-11-07). Weekdays from the FAQ text count as
sourced for the weekday filter, in German and English calls.
"""

from __future__ import annotations

import pytest

from praxis_telefon.agent.claim_gate import FilterContext, SentenceFilter, check_sentence
from praxis_telefon.faq import answer_text

HOURS_DE = "Auskunft der Praxis (nur diese Angaben verwenden): " + answer_text(
    "opening_hours", "de"
)
HOURS_EN = "Auskunft der Praxis (nur diese Angaben verwenden): " + answer_text(
    "opening_hours", "en"
)
PARKING_DE = "Auskunft der Praxis: " + answer_text("parking", "de")
SATURDAY_CLOSED = (
    "Samstag, 14. November: Praxis geschlossen. Stattdessen gesucht: Montag, 16. November."
)


def ctx(*, tools: dict[str, str] | None = None, caller: str = "Hallo.") -> FilterContext:
    tools = tools or {}
    return FilterContext(
        succeeded=frozenset(tools), caller_texts=(caller,), tool_texts=tuple(tools.values())
    )


@pytest.mark.parametrize(
    ("sentence", "family"),
    [
        ("Wir haben mittwochs nur vormittags geöffnet.", "hours"),
        ("Die Praxis ist Montag bis Freitag ab 8 Uhr offen.", "hours"),
        ("We are open on Wednesday from 8 to 12.", "hours"),
        ("Samstags ist die Praxis geschlossen.", "closed"),
        ("Hinter dem Haus gibt es vier Parkplätze.", "parking"),
        ("Die Praxis ist in der Musterstraße 12.", "address"),
        ("An Feiertagen übernimmt eine Vertretungspraxis.", "holidays"),
        ("Bitte bringen Sie Ihre Gesundheitskarte mit.", "insurance_card"),
        ("Folgerezepte liegen nach zwei Werktagen zur Abholung bereit.", "procedure"),
        ("Für eine Krankschreibung brauchen Sie einen Termin.", "procedure"),
        ("Die Praxis nimmt neue Patienten auf.", "procedure"),
    ],
)
def test_practice_fact_without_practice_info_is_blocked(sentence: str, family: str) -> None:
    assert f"fact:{family}" in check_sentence(sentence, ctx())


def test_opening_hours_after_practice_info_pass_with_their_weekdays() -> None:
    c = ctx(tools={"practice_info": HOURS_DE})
    assert check_sentence("Mittwochs haben wir nur vormittags geöffnet, von 8 bis 12 Uhr.", c) == []


def test_english_call_opening_hours_pass() -> None:
    c = ctx(tools={"practice_info": HOURS_EN}, caller="Are you open on Wednesday afternoon?")
    assert check_sentence("On Wednesday we are open from 8 to 12 only.", c) == []


def test_german_faq_text_backs_english_weekdays() -> None:
    # The weekday filter compares across languages ("Mittwoch" backs "Wednesday").
    c = ctx(tools={"practice_info": HOURS_DE}, caller="Are you open on Wednesday afternoon?")
    assert check_sentence("On Wednesday we are open from 8 to 12 only.", c) == []


def test_closed_day_after_find_slots_passes() -> None:
    # User, 2026-11-07: find_slots says a day is closed; the agent may say so.
    c = ctx(tools={"find_slots": SATURDAY_CLOSED}, caller="Haben Sie am Samstag etwas frei?")
    assert check_sentence("Am Samstag ist die Praxis geschlossen.", c) == []


def test_find_slots_does_not_back_opening_hours() -> None:
    c = ctx(tools={"find_slots": SATURDAY_CLOSED}, caller="Haben Sie am Samstag etwas frei?")
    assert "fact:hours" in check_sentence("Am Montag haben wir ab 8 Uhr geöffnet.", c)


def test_parking_from_another_entry_is_not_backed_by_hours() -> None:
    # Any practice_info answer backs the facts of the call; the LLM answers only from
    # the text it got, which the prompt requires. Parking after the parking entry passes.
    c = ctx(tools={"practice_info": PARKING_DE})
    assert check_sentence("Hinter dem Haus gibt es vier Parkplätze.", c) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "Möchten Sie unsere Öffnungszeiten wissen?",
        "Wenn die Praxis geschlossen ist, erreichen Sie den Bereitschaftsdienst unter eins "
        "eins sechs eins eins sieben.",
        "Ich nehme gern eine Nachricht wegen Ihrer Überweisung für das Praxisteam auf.",
        "Dazu weiß ich leider nichts, ich kann aber eine Nachricht für das Praxisteam aufnehmen.",
        "Ihr Termin ist am Dienstag um 9 Uhr bei Dr. Weber.",
    ],
)
def test_questions_conditionals_and_messages_are_not_practice_facts(sentence: str) -> None:
    assert not [r for r in check_sentence(sentence, ctx()) if r.startswith("fact:")]


def test_spoken_tool_name_is_meta() -> None:
    assert "meta:tool" in check_sentence("Ich rufe jetzt practice_info auf.", ctx())


def test_blocked_fact_without_the_tool_needs_a_retry() -> None:
    sf = SentenceFilter(ctx())
    assert sf.push("Samstags haben wir geschlossen. ") == []
    sf.finish()
    assert sf.needs_retry()


def test_blocked_fact_while_practice_info_is_called_needs_no_retry() -> None:
    sf = SentenceFilter(ctx())
    sf.tool_calls.add("practice_info")
    sf.push("Samstags haben wir geschlossen. ")
    sf.finish()
    assert not sf.needs_retry()


def test_retry_note_points_to_practice_info() -> None:
    from praxis_telefon.safety.messages import retry_note_for

    assert "practice_info" in retry_note_for(["fact:hours"])


def test_auf_in_a_crisis_sentence_is_not_opening_hours() -> None:
    # Bug, live suite 2026-11-07 (emergency_de-1): "ist ... auf" in this crisis sentence
    # was taken for "is open" and the sentence was blocked (fact:hours).
    sentence = (
        "Bei Brustschmerzen ist es aber wichtig, dass Sie nicht auf einen Termin warten: "
        "Bitte rufen Sie jetzt die eins eins zwei an."
    )
    assert check_sentence(sentence, ctx()) == []


@pytest.mark.parametrize(
    "sentence",
    ["Wir haben samstags nicht auf.", "Die Praxis hat heute bis 18 Uhr auf.", "Wir haben offen."],
)
def test_opening_with_auf_or_offen_at_the_end_is_still_a_fact(sentence: str) -> None:
    assert "fact:hours" in check_sentence(sentence, ctx())
