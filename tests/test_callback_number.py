"""The saved callback number is the number the caller said, and the read-back is it.

Bug, M4 harness rerun 2026-11-07 (two_requests_one_line[0]): the caller said "null drei
null, zwei drei eins zwei fünf, vier fünf sechs"; the LLM passed "0321 2345 456" to
take_message, which saved 03212345456, and the agent read back the caller's words, so
the caller heard the right number while the team got a wrong one. Now:
- take_message refuses a number that matches none the caller said in this call (the
  LLM is told to pass the caller's own words, which code converts);
- its result gives the saved number digit by digit, for the read-back.
"""

from __future__ import annotations

import sqlite3

import pytest

from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.tools.errors import CallbackNumberMismatch
from praxis_telefon.tools.spoken_numbers import phone_numbers_in
from tests.conftest import NOW

NAME = "Wilhelmina Quellenberg"
SAID = (
    "Guten Tag, Max Mustermann, geboren am ersten Februar neunzehnhundertneunundsiebzig. "
    "Ich hätte gern einen Termin am Donnerstag vormittags, und außerdem brauche ich ein "
    "Folgerezept für meine Blutdrucktabletten. Meine Nummer ist null drei null, zwei drei "
    "eins zwei fünf, vier fünf sechs.",
)


@pytest.fixture
def service(conn: sqlite3.Connection) -> ToolService:
    return ToolService(conn, lambda: NOW)


def _saved(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT callback_number FROM staff_messages")]


def test_phone_numbers_in_the_callers_words() -> None:
    # The year in the date of birth and "ein Folgerezept" are not phone numbers.
    assert phone_numbers_in(SAID[0]) == {"03023125456"}
    assert phone_numbers_in("Sie erreichen mich unter 030 23125456.") == {"03023125456"}
    assert phone_numbers_in("You can reach me on oh three oh, double two three one two five.") == {
        "030223125"
    }
    assert phone_numbers_in("Ich brauche ein Rezept.") == set()


def test_saved_callback_number_must_match_what_the_caller_said(
    conn: sqlite3.Connection, service: ToolService
) -> None:
    with pytest.raises(CallbackNumberMismatch):
        service.take_message(NAME, "0321 2345 456", "prescription", caller_texts=SAID)
    assert _saved(conn) == []


@pytest.mark.parametrize(
    "number", ["030 23125 456", "null drei null, zwei drei eins zwei fünf, vier fünf sechs"]
)
def test_the_callers_number_is_saved(
    conn: sqlite3.Connection, service: ToolService, number: str
) -> None:
    service.take_message(NAME, number, "prescription", caller_texts=SAID)
    assert _saved(conn) == ["03023125456"]


def test_result_gives_the_saved_number_for_the_read_back(service: ToolService) -> None:
    text = service.take_message(NAME, "030 23125 456", "prescription", caller_texts=SAID)
    assert "0 3 0 2 3 1 2 5 4 5 6" in text and "vorlesen" in text.lower()


def test_without_a_number_from_the_caller_the_llm_number_is_used(
    conn: sqlite3.Connection, service: ToolService
) -> None:
    service.take_message(
        NAME, "030 23125 456", "callback", caller_texts=("Rufen Sie mich zurück.",)
    )
    assert _saved(conn) == ["03023125456"]


def test_mismatch_message_tells_the_llm_what_to_do(service: ToolService) -> None:
    with pytest.raises(CallbackNumberMismatch) as e:
        service.take_message(NAME, "0321 2345 456", "prescription", caller_texts=SAID)
    assert "Worten des Anrufers" in e.value.message


def test_number_said_in_two_parts_is_accepted(
    conn: sqlite3.Connection, service: ToolService
) -> None:
    # A caller may give the area code first and the rest in the next line.
    said = (
        "Meine Vorwahl ist null drei null.",
        "Und dann zwei drei eins zwei fünf, vier fünf sechs.",
    )
    service.take_message(NAME, "030 23125 456", "callback", caller_texts=said)
    assert _saved(conn) == ["03023125456"]


def test_parts_must_be_consecutive_and_complete(service: ToolService) -> None:
    said = (
        "Meine Vorwahl ist null drei null.",
        "Und dann zwei drei eins zwei fünf, vier fünf sechs.",
    )
    with pytest.raises(CallbackNumberMismatch):
        service.take_message(NAME, "030 23125 457", "callback", caller_texts=said)
