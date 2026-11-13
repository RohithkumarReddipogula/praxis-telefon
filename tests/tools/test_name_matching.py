"""Name matching in find_booking and book: the date of birth exactly, the name by sound.

Roadmap item 4 (user, 2026-11-09): speech recognition mishears about 1 in 3 German names,
so a lookup by exact name failed although the date of birth was right.
- the date of birth must match exactly;
- the name matches exactly, or by Kölner Phonetik of first and last name;
- several patients with that birth date and a similar-sounding name: never pick one, raise
  AmbiguousPatient (the agent asks for more details);
- a wrong date of birth or an unknown name still looks the same (empty list, no probing).
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from praxis_telefon.tools.errors import AmbiguousPatient
from praxis_telefon.tools.scheduling import book, find_booking
from tests.conftest import MONDAY, NOW, at, slot_at

DOB = date(1990, 3, 3)


def _add_patient(conn: sqlite3.Connection, name: str, dob: date = DOB) -> int:
    from praxis_telefon.db.schema import name_key

    return int(
        conn.execute(
            "INSERT INTO patients (full_name, name_key, date_of_birth) VALUES (?, ?, ?)",
            (name, name_key(name), dob.isoformat()),
        ).lastrowid
        or 0
    )


def _book_for(conn: sqlite3.Connection, name: str, hour: int, dob: date = DOB) -> int:
    slot_id = slot_at(conn, at(MONDAY, hour))
    return book(conn, now=NOW, slot_id=slot_id, patient_name=name, date_of_birth=dob).booking_id


def _find(conn: sqlite3.Connection, name: str, dob: date = DOB) -> list[int]:
    return [b.booking_id for b in find_booking(conn, now=NOW, patient_name=name, date_of_birth=dob)]


def test_a_misheard_name_finds_the_booking_by_sound(conn: sqlite3.Connection) -> None:
    booking = _book_for(conn, "Yusuf Öztürk", 9)
    assert _find(conn, "Yusuf Östürk") == [booking]  # the STT mistake from the A/B
    assert _find(conn, "Jusuf Oeztuerk") == [booking]


def test_the_exact_name_still_wins_and_needs_no_phonetics(conn: sqlite3.Connection) -> None:
    booking = _book_for(conn, "Erika Mustermann", 9)
    assert _find(conn, "erika  MUSTERMANN") == [booking]


def test_the_date_of_birth_must_match_exactly(conn: sqlite3.Connection) -> None:
    _book_for(conn, "Yusuf Öztürk", 9)
    assert _find(conn, "Yusuf Östürk", date(1990, 3, 4)) == []
    assert _find(conn, "Yusuf Östürk", date(1991, 3, 3)) == []


def test_a_different_name_with_the_same_birth_date_is_not_a_match(conn: sqlite3.Connection) -> None:
    _book_for(conn, "Yusuf Öztürk", 9)
    assert _find(conn, "Anja Hartmann") == []
    assert _find(conn, "Yusuf Demir") == []  # only the first name sounds the same


def test_wrong_birth_date_and_unknown_name_still_look_the_same(conn: sqlite3.Connection) -> None:
    _book_for(conn, "Yusuf Öztürk", 9)
    assert _find(conn, "Yusuf Öztürk", date(1990, 3, 4)) == _find(conn, "Hans Unbekannt") == []


def test_several_similar_patients_are_never_resolved_by_picking_one(
    conn: sqlite3.Connection,
) -> None:
    first = _add_patient(conn, "Anna Meier")
    second = _add_patient(conn, "Anna Maier")
    assert first != second
    with pytest.raises(AmbiguousPatient):
        find_booking(conn, now=NOW, patient_name="Anna Mayer", date_of_birth=DOB)


def test_an_exact_name_is_not_ambiguous_even_if_others_sound_alike(
    conn: sqlite3.Connection,
) -> None:
    _add_patient(conn, "Anna Maier")
    booking = _book_for(conn, "Anna Meier", 9)
    assert _find(conn, "Anna Meier") == [booking]  # exact beats the sound-alike Maier


def test_the_ambiguity_message_gives_no_names_and_no_dates() -> None:
    message = AmbiguousPatient().message
    assert "buchstabieren" in message.lower()
    assert "Meier" not in message and "1990" not in message


def test_booking_reuses_the_patient_found_by_sound(conn: sqlite3.Connection) -> None:
    first = _book_for(conn, "Yusuf Öztürk", 9)
    second = _book_for(conn, "Yusuf Östürk", 11)  # misheard again on the second call
    count = conn.execute(
        "SELECT count(*) FROM patients WHERE date_of_birth = ?", (DOB.isoformat(),)
    ).fetchone()[0]
    assert count == 1  # no duplicate patient
    assert sorted(_find(conn, "Yusuf Öztürk")) == sorted([first, second])


def test_booking_with_several_similar_patients_is_ambiguous_not_a_guess(
    conn: sqlite3.Connection,
) -> None:
    _add_patient(conn, "Anna Meier")
    _add_patient(conn, "Anna Maier")
    slot_id = slot_at(conn, at(MONDAY, 9))
    with pytest.raises(AmbiguousPatient):
        book(conn, now=NOW, slot_id=slot_id, patient_name="Anna Mayer", date_of_birth=DOB)
    assert conn.execute("SELECT count(*) FROM bookings").fetchone()[0] == 0


def test_a_new_patient_is_still_created_when_nothing_sounds_alike(
    conn: sqlite3.Connection,
) -> None:
    _book_for(conn, "Yusuf Öztürk", 9)
    _book_for(conn, "Anja Hartmann", 11)
    assert (
        conn.execute(
            "SELECT count(*) FROM patients WHERE date_of_birth = ?", (DOB.isoformat(),)
        ).fetchone()[0]
        == 2
    )
