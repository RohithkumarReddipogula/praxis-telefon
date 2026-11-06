"""Create synthetic seed data: 2 fictional doctors, 30-minute slots, some bookings.

Usage: ``uv run python -m praxis_telefon.db.seed [--reset] [--db PATH]``

All names are invented. Opening hours (Europe/Berlin):
Mon-Fri 08:00-12:00, plus Mon, Tue and Thu 15:00-18:00. Public holidays are ignored.
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path

from praxis_telefon.db.schema import PRACTICE_TZ, connect, init_db, name_key, to_utc_iso

DOCTORS = ("Dr. Katharina Weber", "Dr. Jonas Brandt")
SLOT_MINUTES = 30
DAYS = 28

MORNING = (time(8, 0), time(12, 0))
AFTERNOON = (time(15, 0), time(18, 0))
# Monday=0 ... Friday=4
OPENING_HOURS: dict[int, tuple[tuple[time, time], ...]] = {
    0: (MORNING, AFTERNOON),
    1: (MORNING, AFTERNOON),
    2: (MORNING,),
    3: (MORNING, AFTERNOON),
    4: (MORNING,),
}

SYNTHETIC_PATIENTS = (
    ("Lena Hoffmann", "1988-03-14"),
    ("Markus Becker", "1975-11-02"),
    ("Sabine Krüger", "1962-07-21"),
    ("Tobias Schulte", "1994-01-30"),
    ("Ingrid Vogel", "1948-09-08"),
    ("Felix Neumann", "2001-05-17"),
    ("Petra Lorenz", "1970-12-24"),
    ("Yusuf Demir", "1983-04-11"),
    ("Anja Hartmann", "1991-08-03"),
    ("Klaus Richter", "1956-02-27"),
)

BOOKED_SHARE = 0.4


def slot_starts(day: date) -> list[datetime]:
    """Local slot start times for one day (empty on weekends)."""
    starts: list[datetime] = []
    for open_t, close_t in OPENING_HOURS.get(day.weekday(), ()):
        current = datetime.combine(day, open_t, tzinfo=PRACTICE_TZ)
        close = datetime.combine(day, close_t, tzinfo=PRACTICE_TZ)
        while current + timedelta(minutes=SLOT_MINUTES) <= close:
            starts.append(current)
            current += timedelta(minutes=SLOT_MINUTES)
    return starts


def seed(
    conn: sqlite3.Connection,
    *,
    start: date,
    days: int = DAYS,
    rng_seed: int = 42,
    booked_share: float = BOOKED_SHARE,
) -> None:
    """Insert doctors, slots for ``days`` days from ``start``, and synthetic bookings."""
    init_db(conn)
    rng = random.Random(rng_seed)
    now_iso = to_utc_iso(datetime.combine(start, time(0), tzinfo=PRACTICE_TZ))

    conn.execute("BEGIN IMMEDIATE")
    try:
        doctor_ids = [
            int(conn.execute("INSERT INTO doctors (name) VALUES (?)", (n,)).lastrowid or 0)
            for n in DOCTORS
        ]
        patient_ids = [
            int(
                conn.execute(
                    "INSERT INTO patients (full_name, name_key, date_of_birth) VALUES (?, ?, ?)",
                    (n, name_key(n), dob),
                ).lastrowid
                or 0
            )
            for n, dob in SYNTHETIC_PATIENTS
        ]
        for offset in range(days):
            day = start + timedelta(days=offset)
            for local_start in slot_starts(day):
                for doctor_id in doctor_ids:
                    booked = rng.random() < booked_share
                    slot_id = conn.execute(
                        "INSERT INTO slots (doctor_id, start_utc, duration_min, status) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            doctor_id,
                            to_utc_iso(local_start),
                            SLOT_MINUTES,
                            "booked" if booked else "free",
                        ),
                    ).lastrowid
                    if booked:
                        conn.execute(
                            "INSERT INTO bookings (slot_id, patient_id, status, created_at) "
                            "VALUES (?, ?, 'active', ?)",
                            (slot_id, rng.choice(patient_ids), now_iso),
                        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed the Praxis-Telefon SQLite database.")
    parser.add_argument("--db", type=Path, default=None, help="default: DATABASE_PATH")
    parser.add_argument("--reset", action="store_true", help="delete the database file first")
    args = parser.parse_args(argv)

    path: Path = args.db or Path(os.environ.get("DATABASE_PATH") or "./data/praxis.db")
    if path.exists():
        if not args.reset:
            parser.error(f"{path} already exists; use --reset to recreate it")
        path.unlink()
        for suffix in ("-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(path)
    try:
        seed(conn, start=datetime.now(PRACTICE_TZ).date())
        free = conn.execute("SELECT COUNT(*) FROM slots WHERE status = 'free'").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM slots").fetchone()[0]
    finally:
        conn.close()
    print(f"Seeded {path}: {len(DOCTORS)} doctors, {total} slots ({free} free).")


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Same DATABASE_PATH as the agent, which also loads .env at startup.
    load_dotenv()
    main()
