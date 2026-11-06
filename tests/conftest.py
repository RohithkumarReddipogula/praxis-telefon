"""Shared fixtures: an in-memory database with synthetic data and a fixed clock."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import date, datetime

import pytest

from praxis_telefon.db.schema import PRACTICE_TZ, connect, to_utc_iso
from praxis_telefon.db.seed import seed

# Monday 9 November 2026, 07:00 Berlin: before opening, so every seeded slot is in the future.
MONDAY = date(2026, 11, 9)
NOW = datetime(2026, 11, 9, 7, 0, tzinfo=PRACTICE_TZ)

# Synthetic test identities. Phone numbers are from the Bundesnetzagentur block reserved
# for media use (030 23125 000-999).
NAME = "Erika Mustermann"
DOB = date(1985, 6, 15)
OTHER_NAME = "Max Mustermann"
OTHER_DOB = date(1979, 2, 1)
PHONE = "030 23125 456"


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """All slots free (no pre-booked synthetic bookings)."""
    c = connect(":memory:")
    seed(c, start=MONDAY, booked_share=0.0)
    yield c
    c.close()


def slot_at(conn: sqlite3.Connection, local: datetime, doctor: str = "Weber") -> int:
    row = conn.execute(
        "SELECT s.id FROM slots s JOIN doctors d ON d.id = s.doctor_id "
        "WHERE s.start_utc = ? AND d.name LIKE ?",
        (to_utc_iso(local), f"%{doctor}"),
    ).fetchone()
    assert row is not None, "no such slot in seed data"
    return int(row["id"])


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=PRACTICE_TZ)
