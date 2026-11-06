import sqlite3
from datetime import date, timedelta

import pytest

from praxis_telefon.db.schema import PRACTICE_TZ, to_utc_iso
from praxis_telefon.tools import errors
from praxis_telefon.tools.scheduling import book, find_slots
from tests.conftest import DOB, MONDAY, NAME, NOW, at


def _book_whole_day(conn: sqlite3.Connection, day: date) -> None:
    conn.execute(
        "UPDATE slots SET status = 'booked' WHERE start_utc >= ? AND start_utc < ?",
        (to_utc_iso(at(day, 0)), to_utc_iso(at(day + timedelta(days=1), 0))),
    )


def test_returns_at_most_three_earliest_free(conn: sqlite3.Connection) -> None:
    slots = find_slots(conn, now=NOW, date_from=MONDAY, date_to=MONDAY + timedelta(days=6))
    assert len(slots) == 3
    starts = [s.start for s in slots]
    assert starts == sorted(starts)
    assert starts[0] == at(MONDAY, 8)


def test_limit_cannot_exceed_three(conn: sqlite3.Connection) -> None:
    assert len(find_slots(conn, now=NOW, date_from=MONDAY, date_to=MONDAY, limit=10)) == 3


def test_only_free_slots(conn: sqlite3.Connection) -> None:
    first = find_slots(conn, now=NOW, date_from=MONDAY, date_to=MONDAY)[0]
    book(conn, now=NOW, slot_id=first.slot_id, patient_name=NAME, date_of_birth=DOB)
    again = find_slots(conn, now=NOW, date_from=MONDAY, date_to=MONDAY)
    assert first.slot_id not in {s.slot_id for s in again}


def test_fully_booked_day_returns_nothing(conn: sqlite3.Connection) -> None:
    wednesday = MONDAY + timedelta(days=2)
    _book_whole_day(conn, wednesday)
    assert find_slots(conn, now=NOW, date_from=wednesday, date_to=wednesday) == []


def test_weekend_returns_nothing(conn: sqlite3.Connection) -> None:
    saturday = MONDAY + timedelta(days=5)
    assert find_slots(conn, now=NOW, date_from=saturday, date_to=saturday) == []


def test_past_slots_today_excluded(conn: sqlite3.Connection) -> None:
    late_morning = at(MONDAY, 10, 15)
    slots = find_slots(conn, now=late_morning, date_from=MONDAY, date_to=MONDAY)
    assert slots and all(s.start > late_morning for s in slots)
    assert slots[0].start == at(MONDAY, 10, 30)


def test_past_date_range_rejected(conn: sqlite3.Connection) -> None:
    yesterday = MONDAY - timedelta(days=1)
    with pytest.raises(errors.InvalidDateRange):
        find_slots(conn, now=NOW, date_from=yesterday - timedelta(days=3), date_to=yesterday)


def test_range_starting_in_past_is_clamped_to_today(conn: sqlite3.Connection) -> None:
    slots = find_slots(conn, now=NOW, date_from=MONDAY - timedelta(days=3), date_to=MONDAY)
    assert slots and all(s.start.astimezone(PRACTICE_TZ).date() == MONDAY for s in slots)


@pytest.mark.parametrize(
    ("offset_from", "offset_to"),
    [(3, 1), (0, 28), (0, 30)],
)
def test_invalid_ranges_rejected(
    conn: sqlite3.Connection, offset_from: int, offset_to: int
) -> None:
    with pytest.raises(errors.InvalidDateRange):
        find_slots(
            conn,
            now=NOW,
            date_from=MONDAY + timedelta(days=offset_from),
            date_to=MONDAY + timedelta(days=offset_to),
        )


def test_afternoon_filter(conn: sqlite3.Connection) -> None:
    slots = find_slots(conn, now=NOW, date_from=MONDAY, date_to=MONDAY, part_of_day="afternoon")
    assert slots and all(s.start.astimezone(PRACTICE_TZ).hour >= 15 for s in slots)


def test_morning_filter(conn: sqlite3.Connection) -> None:
    slots = find_slots(conn, now=NOW, date_from=MONDAY, date_to=MONDAY, part_of_day="morning")
    assert slots and all(s.start.astimezone(PRACTICE_TZ).hour < 12 for s in slots)


def test_no_afternoon_on_wednesday(conn: sqlite3.Connection) -> None:
    wednesday = MONDAY + timedelta(days=2)
    assert (
        find_slots(conn, now=NOW, date_from=wednesday, date_to=wednesday, part_of_day="afternoon")
        == []
    )


def test_invalid_part_of_day(conn: sqlite3.Connection) -> None:
    with pytest.raises(errors.InvalidPartOfDay):
        find_slots(conn, now=NOW, date_from=MONDAY, date_to=MONDAY, part_of_day="evening")


@pytest.mark.parametrize("doctor", ["Brandt", "Dr. Brandt", "dr. jonas brandt", " BRANDT "])
def test_doctor_filter(conn: sqlite3.Connection, doctor: str) -> None:
    slots = find_slots(conn, now=NOW, date_from=MONDAY, date_to=MONDAY, doctor=doctor)
    assert slots and all(s.doctor == "Dr. Jonas Brandt" for s in slots)


def test_unknown_doctor(conn: sqlite3.Connection) -> None:
    with pytest.raises(errors.DoctorNotFound):
        find_slots(conn, now=NOW, date_from=MONDAY, date_to=MONDAY, doctor="Dr. Schmidt")


def test_spoken_text_is_german(conn: sqlite3.Connection) -> None:
    first = find_slots(conn, now=NOW, date_from=MONDAY, date_to=MONDAY)[0]
    assert first.spoken == "Montag, 9. November, um 8 Uhr, bei Dr. Brandt"


def test_naive_now_rejected(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        find_slots(conn, now=NOW.replace(tzinfo=None), date_from=MONDAY, date_to=MONDAY)
