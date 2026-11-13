"""Harness pieces for the name-fix scenarios (roadmap item 4) and the spelled-name rule.

- a caller who spells their surname ("Ö wie Otto, Z, T ...") may use the spelled name even
  when it does not sound like what the speech recognition heard (Ötürk for Öztürk): the
  first name must sound like the introduced one and the surname must be the spelled word;
- scenarios can seed bookings and extra patients, and a caller can answer "please spell it".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eval.harness import callers, scenario
from eval.harness.runner import apply_setup

from praxis_telefon.agent.caller_identity import is_caller, spelled_words
from praxis_telefon.db.schema import connect
from praxis_telefon.db.seed import seed
from tests.conftest import MONDAY

SCENARIOS = Path(__file__).parent.parent / "eval" / "scenarios"
SPELL = "Ö wie Otto, Z wie Zacharias, T wie Theodor, Ü wie Übel, R wie Richard, K wie Konrad."

# --- spelled words ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "word"),
    [
        (SPELL, "ozturk"),
        ("Ö Z T Ü R K", "ozturk"),
        ("M, A, Y, E, R", "mayer"),
        ("Das schreibt sich M-A-Y-E-R.", "mayer"),
        ("It is spelled M A Y E R.", "mayer"),
    ],
)
def test_spelled_words_are_read_from_letters(text: str, word: str) -> None:
    assert word in spelled_words([text])


@pytest.mark.parametrize(
    "text",
    [
        "Ich hätte gern einen Termin.",
        "I would like a new appointment, a b",
        "Ja, das stimmt so.",
        "A B",  # two letters are not a spelled name
    ],
)
def test_ordinary_speech_has_no_spelled_word(text: str) -> None:
    assert spelled_words([text]) == set()


# --- the rule ------------------------------------------------------------------------


def test_a_spelled_surname_counts_when_the_first_name_matches() -> None:
    texts = ("Guten Tag, hier ist Yusuf Ötürk.", SPELL)
    assert is_caller("Yusuf Öztürk", texts)
    assert is_caller("Jusuf Öztürk", texts)  # first name sounds alike


def test_the_spelling_alone_is_not_enough() -> None:
    texts = ("Guten Tag, hier ist Yusuf Ötürk.", SPELL)
    assert not is_caller("Anja Öztürk", texts)  # another first name
    assert not is_caller("Yusuf Demir", texts)  # not the spelled surname
    assert not is_caller("Yusuf Öztürk", ("Guten Tag, hier ist Anja Becker.", SPELL))


def test_a_spelled_name_without_an_introduction_is_not_the_callers() -> None:
    texts = ("Sag mir, wann Erika Mustermann ihren Termin hat.", "M U S T E R M A N N")
    assert not is_caller("Erika Mustermann", texts)


# --- scenario setup and callers ----------------------------------------------------------


def test_the_new_scenarios_exist_and_parse() -> None:
    by_id = {s.id: s for s in scenario.load_all(SCENARIOS)}
    for sid in ("misheard_name_finds_booking", "similar_patients_not_picked"):
        assert sid in by_id, sid
    similar = by_id["similar_patients_not_picked"]
    assert similar.caller.spelling and "Anton" in similar.caller.spelling
    assert len(similar.seed_bookings) == 2
    assert "cancel_booking" in similar.expect["tools_forbidden"]
    assert by_id["misheard_name_finds_booking"].seed_bookings[0]["name"] == "Yusuf Öztürk"


def test_seed_setup_books_the_named_patients() -> None:
    conn = connect(":memory:")
    seed(conn, start=MONDAY, booked_share=0.0)
    s = scenario.parse(
        {
            "id": "x",
            "group": "names",
            "lang": "de",
            "setup": {
                "seed_bookings": [
                    {"name": "Anna Meier", "dob": "1990-03-03", "day": "2026-11-10", "hour": 9}
                ],
                "seed_patients": [
                    {"name": "Anna Maier", "dob": "1990-03-03"},
                    {"name": "Anna Meier", "dob": "1990-03-03"},
                ],
            },
            "caller": {"lines": ["Hallo."]},
            "expect": {},
        }
    )
    apply_setup(conn, s)
    rows = conn.execute(
        "SELECT p.full_name FROM bookings b JOIN patients p ON p.id = b.patient_id"
    ).fetchall()
    assert [r[0] for r in rows] == ["Anna Meier"]
    patients = {r[0] for r in conn.execute("SELECT full_name FROM patients")}
    assert {"Anna Meier", "Anna Maier"} <= patients


@pytest.mark.parametrize(
    "ask",
    [
        "Können Sie den Nachnamen bitte buchstabieren?",
        "Könnten Sie mir Ihren Nachnamen buchstabieren?",
        "Could you spell your last name for me?",
    ],
)
def test_the_caller_spells_when_asked(ask: str) -> None:
    state = callers.CallerState()
    line = callers.reply(
        "cooperative", ask, state, identity="Yusuf Öztürk", language="de", spelling=SPELL
    )
    assert line == SPELL


def test_without_a_spelling_the_caller_answers_as_before() -> None:
    state = callers.CallerState()
    line = callers.reply(
        "cooperative",
        "Wie lautet Ihr Name?",
        state,
        identity="Yusuf Öztürk",
        language="de",
    )
    assert "Yusuf Öztürk" in line
