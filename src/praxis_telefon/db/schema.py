"""SQLite schema and connection helper.

All timestamps are stored as UTC ISO-8601 strings (``YYYY-MM-DDTHH:MM:SS+00:00``) so
string comparison equals time comparison. Convert to ``Europe/Berlin`` only for display.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PRACTICE_TZ = ZoneInfo("Europe/Berlin")


def to_utc_iso(dt: datetime) -> str:
    """Aware datetime -> UTC ISO string used in all timestamp columns."""
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def name_key(name: str) -> str:
    """Case- and whitespace-insensitive key for matching patient names."""
    return " ".join(name.split()).casefold()


SCHEMA = """
CREATE TABLE IF NOT EXISTS doctors (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS slots (
    id           INTEGER PRIMARY KEY,
    doctor_id    INTEGER NOT NULL REFERENCES doctors(id),
    start_utc    TEXT NOT NULL,
    duration_min INTEGER NOT NULL CHECK (duration_min > 0),
    status       TEXT NOT NULL DEFAULT 'free' CHECK (status IN ('free', 'booked')),
    UNIQUE (doctor_id, start_utc)
);
CREATE INDEX IF NOT EXISTS idx_slots_start ON slots(start_utc, status);

CREATE TABLE IF NOT EXISTS patients (
    id            INTEGER PRIMARY KEY,
    full_name     TEXT NOT NULL,
    name_key      TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,
    UNIQUE (name_key, date_of_birth)
);

CREATE TABLE IF NOT EXISTS bookings (
    id         INTEGER PRIMARY KEY,
    slot_id    INTEGER NOT NULL REFERENCES slots(id),
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    status     TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'cancelled')),
    created_at TEXT NOT NULL
);
-- Safety net: a slot can never have two active bookings, even if application code is wrong.
CREATE UNIQUE INDEX IF NOT EXISTS uq_bookings_active_slot
    ON bookings(slot_id) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS staff_messages (
    id              INTEGER PRIMARY KEY,
    caller_name     TEXT NOT NULL,
    callback_number TEXT NOT NULL,
    reason_category TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done'))
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection with foreign keys on. ``":memory:"`` works for tests.

    ``isolation_level=None`` disables Python's implicit transactions; tools open their
    own ``BEGIN IMMEDIATE`` transactions where they write.
    """
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
