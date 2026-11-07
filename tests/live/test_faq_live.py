"""Live FAQ tests (M3 Part C). Skipped by default; run with:

    uv run pytest -m live tests/live/test_faq_live.py -v

The agent answers practice questions only from practice_info's text, in the call's
language, and says it does not know (offering a message) when there is no entry.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.live.harness import (
    MAX,
    RUNS,
    Call,
    Step,
    agent_texts,
    converse,
    new_db,
    ok,
    tool_arguments,
    transcript_of,
)

pytestmark = pytest.mark.live

# (caller line, caller language, expected topic, a fact the answer must contain)
ANSWERED: dict[str, tuple[str, str, str, str]] = {
    "hours_wednesday_de": (
        "Guten Tag, haben Sie am Mittwochnachmittag geöffnet?", "de", "opening_hours", r"12",
    ),
    "hours_saturday_de": (
        "Hallo, haben Sie samstags Sprechstunde?", "de", "opening_hours", r"geschlossen",
    ),
    "hours_wednesday_en": (
        "Hello, are you open on Wednesday afternoon?", "en", "opening_hours", r"12",
    ),
    "parking_en": (
        "Hi, is there somewhere to park near the practice?", "en", "parking", r"four|4",
    ),
    "sick_note_de": (
        "Ich brauche eine Krankschreibung, wie läuft das bei Ihnen?", "de", "sick_note",
        r"Termin",
    ),
    "insurance_card_en": (
        "I forgot my insurance card, can I still come?", "en", "insurance_card", r"ten|10",
    ),
    "new_patients_de": (
        "Nehmen Sie noch neue Patienten auf?", "de", "new_patients", r"Gesundheitskarte",
    ),
    "holidays_de": (
        "Haben Sie an Feiertagen geöffnet?", "de", "holidays", r"geschlossen",
    ),
}  # fmt: skip

NOT_ANSWERED: dict[str, tuple[str, str]] = {
    "home_visits_de": ("Machen Sie auch Hausbesuche?", "de"),
    "flu_vaccine_de": ("Kann ich bei Ihnen eine Grippeimpfung bekommen?", "de"),
    "pharmacy_en": ("Is there a pharmacy close to your practice?", "en"),
}

UNKNOWN = re.compile(
    r"\b(nicht|leider|keine|don't|do not|not|unfortunately|no information)\b", re.I
)
OFFER = re.compile(r"\b(nachricht|praxisteam|rückruf|message|practice team|call you back)\b", re.I)
GERMAN_WORDS = re.compile(r"\b(und|nicht|Sie|ist|die|der|das)\b")


def _topics(call: Call) -> list[str]:
    return [str(json.loads(a).get("topic")) for a in tool_arguments(call, "practice_info")]


def _blocked_facts(call: Call) -> list[str]:
    return [note for _, note in call.filter_notes if ":fact:" in note]


@pytest.mark.parametrize("run", range(RUNS))
@pytest.mark.parametrize("case", ANSWERED)
def test_practice_question_is_answered_from_the_faq(case: str, run: int) -> None:
    line, lang, topic, fact = ANSWERED[case]
    conn, _ = new_db(erika_booked=False)
    calls = converse(conn, MAX, [Step((line,))], caller_language=lang)

    info = transcript_of(calls) + f"\nblocked: {calls[0].filter_notes}"
    assert _topics(calls[0])[:1] == [topic], info
    said = " ".join(agent_texts(calls[0]))
    assert re.search(fact, said, re.I), info
    if lang == "en":
        assert not GERMAN_WORDS.search(said.replace("Musterstraße", "")), info


@pytest.mark.parametrize("run", range(RUNS))
@pytest.mark.parametrize("case", NOT_ANSWERED)
def test_question_without_an_entry_gets_no_invented_fact(case: str, run: int) -> None:
    line, lang = NOT_ANSWERED[case]
    conn, _ = new_db(erika_booked=False)
    calls = converse(conn, MAX, [Step((line,))], caller_language=lang)

    info = transcript_of(calls) + f"\nblocked: {calls[0].filter_notes}"
    assert all(t == "other" for t in _topics(calls[0])), info
    said = " ".join(agent_texts(calls[0]))
    assert UNKNOWN.search(said) and OFFER.search(said), info
    assert not re.search(r"\b(ja|yes),? (wir|we)\b", said, re.I), info  # "Ja, wir machen ..."


@pytest.mark.parametrize("run", range(RUNS))
def test_faq_question_then_booking(run: int) -> None:
    conn, _ = new_db(erika_booked=False)
    calls = converse(
        conn,
        MAX,
        [
            Step(("Guten Tag, wann haben Sie am Mittwoch geöffnet?",)),
            Step(
                (f"Gut, dann hätte ich gern einen Termin am Mittwoch vormittags. {MAX}.",),
                ok("book"),
            ),
        ],
    )

    info = transcript_of(calls)
    assert _topics(calls[0])[:1] == ["opening_hours"], info
    assert calls[1].succeeded("book") == 1, info
    assert _blocked_facts(calls[0]) == [], info
