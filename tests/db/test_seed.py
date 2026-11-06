import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from praxis_telefon.db.schema import PRACTICE_TZ, connect
from praxis_telefon.db.seed import DOCTORS, SYNTHETIC_PATIENTS, seed, slot_starts
from tests.conftest import MONDAY


def _local_starts(conn: sqlite3.Connection) -> list[datetime]:
    rows = conn.execute("SELECT start_utc FROM slots").fetchall()
    return [datetime.fromisoformat(r["start_utc"]).astimezone(PRACTICE_TZ) for r in rows]


def test_two_doctors(conn: sqlite3.Connection) -> None:
    names = [r["name"] for r in conn.execute("SELECT name FROM doctors ORDER BY id")]
    assert names == list(DOCTORS) and len(names) == 2


def test_weekdays_only_and_within_14_days(conn: sqlite3.Connection) -> None:
    starts = _local_starts(conn)
    assert starts
    assert all(s.weekday() < 5 for s in starts)
    assert min(s.date() for s in starts) >= MONDAY
    assert max(s.date() for s in starts) < MONDAY + timedelta(days=14)


def test_slots_inside_opening_hours(conn: sqlite3.Connection) -> None:
    for s in _local_starts(conn):
        minutes = s.hour * 60 + s.minute
        morning = 8 * 60 <= minutes <= 11 * 60 + 30
        afternoon = 15 * 60 <= minutes <= 17 * 60 + 30
        assert morning or (afternoon and s.weekday() in (0, 1, 3)), s


def test_all_slots_30_minutes(conn: sqlite3.Connection) -> None:
    durations = {r[0] for r in conn.execute("SELECT DISTINCT duration_min FROM slots")}
    assert durations == {30}


def test_slot_count_per_week() -> None:
    # Mon/Tue/Thu: 8 + 6 slots, Wed/Fri: 8 slots -> 58 per doctor per week.
    week = [MONDAY + timedelta(days=i) for i in range(7)]
    assert sum(len(slot_starts(d)) for d in week) == 58


def test_dst_change_keeps_local_opening_time() -> None:
    # Germany switches to summer time on Sunday 28 March 2027.
    before, after = slot_starts(date(2027, 3, 26)), slot_starts(date(2027, 3, 29))
    assert (before[0].hour, after[0].hour) == (8, 8)
    assert before[0].utcoffset() != after[0].utcoffset()


def test_seed_is_reproducible() -> None:
    def snapshot(rng_seed: int) -> list[tuple[str, str]]:
        c = connect(":memory:")
        seed(c, start=MONDAY, rng_seed=rng_seed)
        return [tuple(r) for r in c.execute("SELECT start_utc, status FROM slots ORDER BY id")]

    assert snapshot(1) == snapshot(1)
    assert snapshot(1) != snapshot(2)


def test_default_seed_prebooks_some_slots_consistently() -> None:
    c = connect(":memory:")
    seed(c, start=MONDAY)
    booked = c.execute("SELECT COUNT(*) FROM slots WHERE status = 'booked'").fetchone()[0]
    active = c.execute("SELECT COUNT(*) FROM bookings WHERE status = 'active'").fetchone()[0]
    total = c.execute("SELECT COUNT(*) FROM slots").fetchone()[0]
    assert 0.2 * total < booked < 0.6 * total
    assert booked == active
    patients = c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    assert patients == len(SYNTHETIC_PATIENTS)


def test_two_active_bookings_for_one_slot_are_impossible(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO bookings (slot_id, patient_id, created_at) VALUES (1, 1, 'x')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO bookings (slot_id, patient_id, created_at) VALUES (1, 2, 'x')")


def test_seed_cli_creates_db_and_refuses_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from praxis_telefon.db.seed import main

    db = tmp_path / "praxis.db"
    main(["--db", str(db)])
    assert db.exists() and "2 doctors" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--db", str(db)])
    main(["--db", str(db), "--reset"])
    c = connect(db)
    assert c.execute("SELECT COUNT(*) FROM doctors").fetchone()[0] == 2
    c.close()
