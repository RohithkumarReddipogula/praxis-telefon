"""Appointment tools: find_slots, book, find_booking, move, cancel.

Plain functions over a ``sqlite3.Connection`` with an injected ``now`` (timezone-aware).
No LiveKit code here. Writes run inside ``BEGIN IMMEDIATE`` transactions so a failed
check leaves the database unchanged.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal

from praxis_telefon.db.schema import PRACTICE_TZ, name_key, to_utc_iso
from praxis_telefon.tools.errors import (
    BookingNotActive,
    BookingNotFound,
    DoctorNotFound,
    InvalidDateRange,
    InvalidPartOfDay,
    SlotInPast,
    SlotNotFound,
    SlotTaken,
)
from praxis_telefon.tools.formatting import spoken_slot
from praxis_telefon.tools.validation import check_date_of_birth, clean_name, require_aware

PartOfDay = Literal["morning", "afternoon", "any"]
PARTS_OF_DAY: tuple[PartOfDay, ...] = ("morning", "afternoon", "any")
MAX_RESULTS = 3
MAX_RANGE_DAYS = 14
NOON = time(12, 0)


@dataclass(frozen=True)
class SlotOption:
    slot_id: int
    start: datetime
    doctor: str
    spoken: str


@dataclass(frozen=True)
class BookingResult:
    booking_id: int
    slot: SlotOption
    already_booked: bool = False


@dataclass(frozen=True)
class BookingInfo:
    booking_id: int
    slot: SlotOption


@dataclass(frozen=True)
class CancelResult:
    booking_id: int
    slot: SlotOption
    already_cancelled: bool = False


# --- helpers -------------------------------------------------------------------------


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _slot_option(row: sqlite3.Row) -> SlotOption:
    start = _parse_utc(row["start_utc"])
    return SlotOption(
        slot_id=int(row["slot_id"]),
        start=start,
        doctor=str(row["doctor"]),
        spoken=spoken_slot(start, str(row["doctor"])),
    )


_SLOT_SELECT = """
    SELECT s.id AS slot_id, s.start_utc, s.status, d.name AS doctor
    FROM slots s JOIN doctors d ON d.id = s.doctor_id
"""


def _get_slot(conn: sqlite3.Connection, slot_id: int) -> sqlite3.Row:
    row: sqlite3.Row | None = conn.execute(_SLOT_SELECT + " WHERE s.id = ?", (slot_id,)).fetchone()
    if row is None:
        raise SlotNotFound()
    return row


def _find_patient(conn: sqlite3.Connection, name: str, date_of_birth: date) -> int | None:
    row = conn.execute(
        "SELECT id FROM patients WHERE name_key = ? AND date_of_birth = ?",
        (name_key(name), date_of_birth.isoformat()),
    ).fetchone()
    return None if row is None else int(row["id"])


def _local_day_bounds_utc(day: date) -> tuple[str, str]:
    start = datetime.combine(day, time(0), tzinfo=PRACTICE_TZ)
    end = datetime.combine(day + timedelta(days=1), time(0), tzinfo=PRACTICE_TZ)
    return to_utc_iso(start), to_utc_iso(end)


def _resolve_doctor(conn: sqlite3.Connection, doctor: str) -> int:
    wanted = name_key(doctor).removeprefix("dr. ").removeprefix("dr ")
    for row in conn.execute("SELECT id, name FROM doctors"):
        full = name_key(str(row["name"])).removeprefix("dr. ")
        if wanted and (wanted == full or wanted == full.split()[-1]):
            return int(row["id"])
    raise DoctorNotFound()


# --- tools ---------------------------------------------------------------------------


def find_slots(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    date_from: date,
    date_to: date,
    part_of_day: str = "any",
    doctor: str | None = None,
    limit: int = MAX_RESULTS,
) -> list[SlotOption]:
    """Return up to ``limit`` (max 3) free future slots, earliest first."""
    require_aware(now)
    if part_of_day not in PARTS_OF_DAY:
        raise InvalidPartOfDay()
    today = now.astimezone(PRACTICE_TZ).date()
    if date_to < date_from or date_to < today or (date_to - date_from).days >= MAX_RANGE_DAYS:
        raise InvalidDateRange()
    date_from = max(date_from, today)

    lower, _ = _local_day_bounds_utc(date_from)
    _, upper = _local_day_bounds_utc(date_to)
    lower = max(lower, to_utc_iso(now))

    sql = _SLOT_SELECT + " WHERE s.status = 'free' AND s.start_utc > ? AND s.start_utc < ?"
    params: list[object] = [lower, upper]
    if doctor:
        sql += " AND s.doctor_id = ?"
        params.append(_resolve_doctor(conn, doctor))
    sql += " ORDER BY s.start_utc, d.name"

    results: list[SlotOption] = []
    for row in conn.execute(sql, params):
        option = _slot_option(row)
        is_morning = option.start.astimezone(PRACTICE_TZ).time() < NOON
        if part_of_day == "morning" and not is_morning:
            continue
        if part_of_day == "afternoon" and is_morning:
            continue
        results.append(option)
        if len(results) >= min(limit, MAX_RESULTS):
            break
    return results


def book(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    slot_id: int,
    patient_name: str,
    date_of_birth: date,
) -> BookingResult:
    """Book a slot. Idempotent: the same patient booking the same slot again is a no-op."""
    require_aware(now)
    name = clean_name(patient_name)
    check_date_of_birth(date_of_birth, now)

    with _transaction(conn):
        slot = _get_slot(conn, slot_id)
        option = _slot_option(slot)
        if option.start <= now:
            raise SlotInPast()
        patient_id = _find_patient(conn, name, date_of_birth)

        if slot["status"] == "booked":
            existing = conn.execute(
                "SELECT id, patient_id FROM bookings WHERE slot_id = ? AND status = 'active'",
                (slot_id,),
            ).fetchone()
            if existing is not None and patient_id == int(existing["patient_id"]):
                return BookingResult(int(existing["id"]), option, already_booked=True)
            raise SlotTaken()

        if patient_id is None:
            patient_id = int(
                conn.execute(
                    "INSERT INTO patients (full_name, name_key, date_of_birth) VALUES (?, ?, ?)",
                    (name, name_key(name), date_of_birth.isoformat()),
                ).lastrowid
                or 0
            )
        booking_id = int(
            conn.execute(
                "INSERT INTO bookings (slot_id, patient_id, status, created_at) "
                "VALUES (?, ?, 'active', ?)",
                (slot_id, patient_id, to_utc_iso(now)),
            ).lastrowid
            or 0
        )
        conn.execute("UPDATE slots SET status = 'booked' WHERE id = ?", (slot_id,))
    return BookingResult(booking_id, option)


def find_booking(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    patient_name: str,
    date_of_birth: date,
) -> list[BookingInfo]:
    """Active future bookings for this patient.

    A wrong date of birth returns the same empty list as an unknown name, so callers
    cannot probe who is a patient.
    """
    require_aware(now)
    name = clean_name(patient_name)
    check_date_of_birth(date_of_birth, now)
    patient_id = _find_patient(conn, name, date_of_birth)
    if patient_id is None:
        return []
    rows = conn.execute(
        _SLOT_SELECT.replace("SELECT ", "SELECT b.id AS booking_id, ", 1)
        + " JOIN bookings b ON b.slot_id = s.id"
        " WHERE b.patient_id = ? AND b.status = 'active' AND s.start_utc > ?"
        " ORDER BY s.start_utc",
        (patient_id, to_utc_iso(now)),
    ).fetchall()
    return [BookingInfo(int(r["booking_id"]), _slot_option(r)) for r in rows]


def _get_booking(conn: sqlite3.Connection, booking_id: int) -> tuple[sqlite3.Row, sqlite3.Row]:
    booking = conn.execute(
        "SELECT id, slot_id, status FROM bookings WHERE id = ?", (booking_id,)
    ).fetchone()
    if booking is None:
        raise BookingNotFound()
    slot = _get_slot(conn, int(booking["slot_id"]))
    return booking, slot


def move(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    booking_id: int,
    new_slot_id: int,
) -> BookingResult:
    """Move an active future booking to a free future slot, atomically."""
    require_aware(now)
    with _transaction(conn):
        booking, old_slot = _get_booking(conn, booking_id)
        if booking["status"] != "active" or _slot_option(old_slot).start <= now:
            raise BookingNotActive()
        if int(booking["slot_id"]) == new_slot_id:
            return BookingResult(booking_id, _slot_option(old_slot), already_booked=True)

        new_slot = _get_slot(conn, new_slot_id)
        new_option = _slot_option(new_slot)
        if new_option.start <= now:
            raise SlotInPast()
        if new_slot["status"] != "free":
            raise SlotTaken()

        conn.execute("UPDATE bookings SET slot_id = ? WHERE id = ?", (new_slot_id, booking_id))
        conn.execute("UPDATE slots SET status = 'free' WHERE id = ?", (int(old_slot["slot_id"]),))
        conn.execute("UPDATE slots SET status = 'booked' WHERE id = ?", (new_slot_id,))
    return BookingResult(booking_id, new_option)


def cancel(conn: sqlite3.Connection, *, now: datetime, booking_id: int) -> CancelResult:
    """Cancel a future booking and free its slot. Cancelling twice is a no-op."""
    require_aware(now)
    with _transaction(conn):
        booking, slot = _get_booking(conn, booking_id)
        option = _slot_option(slot)
        if booking["status"] == "cancelled":
            return CancelResult(booking_id, option, already_cancelled=True)
        if option.start <= now:
            raise BookingNotActive()
        conn.execute("UPDATE bookings SET status = 'cancelled' WHERE id = ?", (booking_id,))
        conn.execute("UPDATE slots SET status = 'free' WHERE id = ?", (int(slot["slot_id"]),))
    return CancelResult(booking_id, option)
