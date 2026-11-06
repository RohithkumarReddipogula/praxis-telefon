import sqlite3

import pytest

from praxis_telefon.tools import errors
from praxis_telefon.tools.scheduling import book, cancel, move
from tests.conftest import DOB, MONDAY, NAME, NOW, OTHER_DOB, OTHER_NAME, at, slot_at


def _status(conn: sqlite3.Connection, slot_id: int) -> str:
    return str(conn.execute("SELECT status FROM slots WHERE id = ?", (slot_id,)).fetchone()[0])


def _booking_slot(conn: sqlite3.Connection, booking_id: int) -> int:
    return int(
        conn.execute("SELECT slot_id FROM bookings WHERE id = ?", (booking_id,)).fetchone()[0]
    )


@pytest.fixture
def booked(conn: sqlite3.Connection) -> tuple[int, int]:
    """(booking_id, slot_id) for Erika at Monday 09:00."""
    slot_id = slot_at(conn, at(MONDAY, 9))
    result = book(conn, now=NOW, slot_id=slot_id, patient_name=NAME, date_of_birth=DOB)
    return result.booking_id, slot_id


def test_move_success_frees_old_slot(conn: sqlite3.Connection, booked: tuple[int, int]) -> None:
    booking_id, old_slot = booked
    new_slot = slot_at(conn, at(MONDAY, 10))
    result = move(conn, now=NOW, booking_id=booking_id, new_slot_id=new_slot)
    assert result.slot.slot_id == new_slot
    assert (_status(conn, old_slot), _status(conn, new_slot)) == ("free", "booked")
    assert _booking_slot(conn, booking_id) == new_slot


def test_move_to_taken_slot_changes_nothing(
    conn: sqlite3.Connection, booked: tuple[int, int]
) -> None:
    booking_id, old_slot = booked
    taken = slot_at(conn, at(MONDAY, 10))
    book(conn, now=NOW, slot_id=taken, patient_name=OTHER_NAME, date_of_birth=OTHER_DOB)
    with pytest.raises(errors.SlotTaken):
        move(conn, now=NOW, booking_id=booking_id, new_slot_id=taken)
    assert _booking_slot(conn, booking_id) == old_slot
    assert (_status(conn, old_slot), _status(conn, taken)) == ("booked", "booked")


def test_move_to_past_slot(conn: sqlite3.Connection) -> None:
    slot_id = slot_at(conn, at(MONDAY, 11))
    booking_id = book(
        conn, now=NOW, slot_id=slot_id, patient_name=NAME, date_of_birth=DOB
    ).booking_id
    past_slot = slot_at(conn, at(MONDAY, 8))
    with pytest.raises(errors.SlotInPast):
        move(conn, now=at(MONDAY, 9), booking_id=booking_id, new_slot_id=past_slot)
    assert _booking_slot(conn, booking_id) == slot_id


def test_move_to_same_slot_is_noop(conn: sqlite3.Connection, booked: tuple[int, int]) -> None:
    booking_id, slot_id = booked
    result = move(conn, now=NOW, booking_id=booking_id, new_slot_id=slot_id)
    assert result.already_booked and _status(conn, slot_id) == "booked"


def test_move_cancelled_booking(conn: sqlite3.Connection, booked: tuple[int, int]) -> None:
    booking_id, _ = booked
    cancel(conn, now=NOW, booking_id=booking_id)
    with pytest.raises(errors.BookingNotActive):
        move(conn, now=NOW, booking_id=booking_id, new_slot_id=slot_at(conn, at(MONDAY, 10)))


def test_move_past_booking(conn: sqlite3.Connection, booked: tuple[int, int]) -> None:
    booking_id, _ = booked
    new_slot = slot_at(conn, at(MONDAY, 11))
    with pytest.raises(errors.BookingNotActive):
        move(conn, now=at(MONDAY, 9, 30), booking_id=booking_id, new_slot_id=new_slot)


def test_move_unknown_booking_or_slot(conn: sqlite3.Connection, booked: tuple[int, int]) -> None:
    booking_id, _ = booked
    with pytest.raises(errors.BookingNotFound):
        move(conn, now=NOW, booking_id=999_999, new_slot_id=slot_at(conn, at(MONDAY, 10)))
    with pytest.raises(errors.SlotNotFound):
        move(conn, now=NOW, booking_id=booking_id, new_slot_id=999_999)


def test_cancel_frees_slot(conn: sqlite3.Connection, booked: tuple[int, int]) -> None:
    booking_id, slot_id = booked
    result = cancel(conn, now=NOW, booking_id=booking_id)
    assert not result.already_cancelled
    assert _status(conn, slot_id) == "free"


def test_cancel_twice_is_idempotent(conn: sqlite3.Connection, booked: tuple[int, int]) -> None:
    booking_id, slot_id = booked
    cancel(conn, now=NOW, booking_id=booking_id)
    second = cancel(conn, now=NOW, booking_id=booking_id)
    assert second.already_cancelled
    assert _status(conn, slot_id) == "free"


def test_cancelled_slot_can_be_booked_again(
    conn: sqlite3.Connection, booked: tuple[int, int]
) -> None:
    booking_id, slot_id = booked
    cancel(conn, now=NOW, booking_id=booking_id)
    book(conn, now=NOW, slot_id=slot_id, patient_name=OTHER_NAME, date_of_birth=OTHER_DOB)
    assert _status(conn, slot_id) == "booked"


def test_cancel_past_booking_rejected(conn: sqlite3.Connection, booked: tuple[int, int]) -> None:
    booking_id, slot_id = booked
    with pytest.raises(errors.BookingNotActive):
        cancel(conn, now=at(MONDAY, 10), booking_id=booking_id)
    assert _status(conn, slot_id) == "booked"


def test_cancel_unknown_booking(conn: sqlite3.Connection) -> None:
    with pytest.raises(errors.BookingNotFound):
        cancel(conn, now=NOW, booking_id=999_999)
