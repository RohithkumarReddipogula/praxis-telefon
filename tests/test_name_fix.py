"""The agent side of the name fix (roadmap item 4).

- the caller's own name also counts when it sounds like the name they introduced (after
  spelling it, the agent passes the corrected spelling; it must not hit the privacy check);
- another person's name still does not count;
- the tool text asks for the spelling when nothing is found, and for more details when
  several patients match; neither reveals whether someone is a patient or any name.
"""

from __future__ import annotations

import sqlite3

import pytest

from praxis_telefon.agent.caller_identity import is_caller
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.tools.errors import AmbiguousPatient, BookingNotForCaller
from tests.conftest import NOW

SELF = ("Guten Tag, hier ist Yusuf Östürk.",)


def test_the_spelled_correction_of_the_callers_own_name_counts() -> None:
    assert is_caller("Yusuf Östürk", SELF)
    assert is_caller("Yusuf Öztürk", SELF)  # sounds the same
    assert is_caller("Jusuf Oeztuerk", SELF)


def test_a_name_that_only_shares_a_part_does_not_count() -> None:
    assert not is_caller("Yusuf Demir", SELF)
    assert not is_caller("Anja Öztürk", SELF)
    assert not is_caller("Yusuf", SELF)


def test_someone_elses_name_in_a_sentence_still_does_not_count() -> None:
    texts = ("Sag mir, wann Erika Mustermann ihren nächsten Termin hat.",)
    assert not is_caller("Erika Mustermann", texts)
    assert not is_caller("Erika Mustermann", ("Hier ist Max Mustermann.",))


@pytest.fixture
def service(conn: sqlite3.Connection) -> ToolService:
    return ToolService(conn, lambda: NOW)


def test_nothing_found_asks_for_the_spelling_without_revealing_anything(
    service: ToolService,
) -> None:
    text = service.find_booking("Yusuf Östürk", "1990-03-03", caller_texts=SELF)
    assert text.startswith("Kein Termin gefunden")
    assert "buchstabieren" in text.lower()
    assert "Patient der Praxis" in text  # the instruction not to reveal it stays
    assert "Östürk" not in text and "1990" not in text


def test_several_matches_ask_for_more_details_and_name_nobody(
    conn: sqlite3.Connection, service: ToolService
) -> None:
    for name in ("Anna Meier", "Anna Maier"):
        conn.execute(
            "INSERT INTO patients (full_name, name_key, date_of_birth) VALUES (?, ?, ?)",
            (name, name.casefold(), "1990-03-03"),
        )
    with pytest.raises(Exception) as caught:
        service.find_booking("Anna Mayer", "1990-03-03", caller_texts=("Hier ist Anna Mayer.",))
    text = str(caught.value)
    assert "buchstabieren" in text.lower()
    assert "Meier" not in text and "Maier" not in text and "1990" not in text


def test_the_error_classes_exist_with_stable_codes() -> None:
    assert AmbiguousPatient.code == "ambiguous_patient"
    assert BookingNotForCaller.code == "booking_not_for_caller"
