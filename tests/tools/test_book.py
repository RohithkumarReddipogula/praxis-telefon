import sqlite3
from datetime import date, timedelta

import pytest

from praxis_telefon.tools import errors
from praxis_telefon.tools.scheduling import book, find_booking
from tests.conftest import DOB, MONDAY, NAME, NOW, OTHER_DOB, OTHER_NAME, at, slot_at


def _count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def test_book_success(conn: sqlite3.Connection) -> None:
    slot_id = slot_at(conn, at(MONDAY, 9))
    result = book(conn, now=NOW, slot_id=slot_id, patient_name=NAME, date_of_birth=DOB)
    assert not result.already_booked
    assert result.slot.slot_id == slot_id
    status = conn.execute("SELECT status FROM slots WHERE id = ?", (slot_id,)).fetchone()[0]
    assert status == "booked"
    assert [
        b.booking_id for b in find_booking(conn, now=NOW, patient_name=NAME, date_of_birth=DOB)
    ] == [result.booking_id]


def test_double_book_call_creates_one_booking(conn: sqlite3.Connection) -> None:
    slot_id = slot_at(conn, at(MONDAY, 9))
    first = book(conn, now=NOW, slot_id=slot_id, patient_name=NAME, date_of_birth=DOB)
    second = book(conn, now=NOW, slot_id=slot_id, patient_name=NAME, date_of_birth=DOB)
    assert second.already_booked
    assert second.booking_id == first.booking_id
    assert _count(conn, "SELECT COUNT(*) FROM bookings") == 1
    assert _count(conn, "SELECT COUNT(*) FROM patients WHERE name_key = 'erika mustermann'") == 1


def test_double_book_is_idempotent_despite_name_formatting(conn: sqlite3.Connection) -> None:
    slot_id = slot_at(conn, at(MONDAY, 9))
    book(conn, now=NOW, slot_id=slot_id, patient_name=NAME, date_of_birth=DOB)
    again = book(
        conn, now=NOW, slot_id=slot_id, patient_name="  erika   MUSTERMANN ", date_of_birth=DOB
    )
    assert again.already_booked
    assert _count(conn, "SELECT COUNT(*) FROM bookings") == 1


def test_slot_taken_by_other_patient(conn: sqlite3.Connection) -> None:
    slot_id = slot_at(conn, at(MONDAY, 9))
    book(conn, now=NOW, slot_id=slot_id, patient_name=NAME, date_of_birth=DOB)
    with pytest.raises(errors.SlotTaken):
        book(conn, now=NOW, slot_id=slot_id, patient_name=OTHER_NAME, date_of_birth=OTHER_DOB)
    assert _count(conn, "SELECT COUNT(*) FROM bookings") == 1


def test_same_name_other_birthday_is_a_different_patient(conn: sqlite3.Connection) -> None:
    slot_id = slot_at(conn, at(MONDAY, 9))
    book(conn, now=NOW, slot_id=slot_id, patient_name=NAME, date_of_birth=DOB)
    with pytest.raises(errors.SlotTaken):
        book(conn, now=NOW, slot_id=slot_id, patient_name=NAME, date_of_birth=OTHER_DOB)


def test_slot_in_past(conn: sqlite3.Connection) -> None:
    slot_id = slot_at(conn, at(MONDAY, 9))
    later = at(MONDAY, 9, 1)
    with pytest.raises(errors.SlotInPast):
        book(conn, now=later, slot_id=slot_id, patient_name=NAME, date_of_birth=DOB)
    assert _count(conn, "SELECT COUNT(*) FROM bookings") == 0


def test_slot_starting_now_counts_as_past(conn: sqlite3.Connection) -> None:
    slot_id = slot_at(conn, at(MONDAY, 9))
    with pytest.raises(errors.SlotInPast):
        book(conn, now=at(MONDAY, 9), slot_id=slot_id, patient_name=NAME, date_of_birth=DOB)


def test_unknown_slot(conn: sqlite3.Connection) -> None:
    with pytest.raises(errors.SlotNotFound):
        book(conn, now=NOW, slot_id=999_999, patient_name=NAME, date_of_birth=DOB)


@pytest.mark.parametrize(
    "dob",
    [MONDAY + timedelta(days=1), date(1890, 1, 1)],
    ids=["future", "older_than_120"],
)
def test_invalid_date_of_birth(conn: sqlite3.Connection, dob: date) -> None:
    with pytest.raises(errors.InvalidDateOfBirth):
        book(
            conn,
            now=NOW,
            slot_id=slot_at(conn, at(MONDAY, 9)),
            patient_name=NAME,
            date_of_birth=dob,
        )


@pytest.mark.parametrize("name", ["Erika", "   ", "A" * 60 + " " + "B" * 60])
def test_invalid_name(conn: sqlite3.Connection, name: str) -> None:
    with pytest.raises(errors.InvalidName):
        book(
            conn,
            now=NOW,
            slot_id=slot_at(conn, at(MONDAY, 9)),
            patient_name=name,
            date_of_birth=DOB,
        )


def test_new_patient_is_created_once(conn: sqlite3.Connection) -> None:
    before = _count(conn, "SELECT COUNT(*) FROM patients")
    book(conn, now=NOW, slot_id=slot_at(conn, at(MONDAY, 9)), patient_name=NAME, date_of_birth=DOB)
    book(conn, now=NOW, slot_id=slot_at(conn, at(MONDAY, 10)), patient_name=NAME, date_of_birth=DOB)
    assert _count(conn, "SELECT COUNT(*) FROM patients") == before + 1
