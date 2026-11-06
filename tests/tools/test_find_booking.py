import sqlite3
from datetime import date

from praxis_telefon.tools.scheduling import book, cancel, find_booking
from tests.conftest import DOB, MONDAY, NAME, NOW, at, slot_at


def _book(conn: sqlite3.Connection, hour: int) -> int:
    slot_id = slot_at(conn, at(MONDAY, hour))
    return book(conn, now=NOW, slot_id=slot_id, patient_name=NAME, date_of_birth=DOB).booking_id


def test_finds_active_future_bookings_in_order(conn: sqlite3.Connection) -> None:
    later, earlier = _book(conn, 11), _book(conn, 9)
    found = find_booking(conn, now=NOW, patient_name=NAME, date_of_birth=DOB)
    assert [b.booking_id for b in found] == [earlier, later]


def test_wrong_date_of_birth_returns_nothing(conn: sqlite3.Connection) -> None:
    _book(conn, 9)
    assert find_booking(conn, now=NOW, patient_name=NAME, date_of_birth=date(1985, 6, 16)) == []


def test_unknown_name_looks_the_same_as_wrong_birthday(conn: sqlite3.Connection) -> None:
    _book(conn, 9)
    wrong_dob = find_booking(conn, now=NOW, patient_name=NAME, date_of_birth=date(1985, 6, 16))
    unknown = find_booking(conn, now=NOW, patient_name="Hans Unbekannt", date_of_birth=DOB)
    assert wrong_dob == unknown == []


def test_name_match_ignores_case_and_spacing(conn: sqlite3.Connection) -> None:
    booking_id = _book(conn, 9)
    found = find_booking(conn, now=NOW, patient_name=" ERIKA  mustermann", date_of_birth=DOB)
    assert [b.booking_id for b in found] == [booking_id]


def test_cancelled_bookings_not_returned(conn: sqlite3.Connection) -> None:
    booking_id = _book(conn, 9)
    cancel(conn, now=NOW, booking_id=booking_id)
    assert find_booking(conn, now=NOW, patient_name=NAME, date_of_birth=DOB) == []


def test_past_bookings_not_returned(conn: sqlite3.Connection) -> None:
    _book(conn, 9)
    assert find_booking(conn, now=at(MONDAY, 12), patient_name=NAME, date_of_birth=DOB) == []
