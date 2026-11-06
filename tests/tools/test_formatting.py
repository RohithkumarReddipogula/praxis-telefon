from datetime import date, datetime

import pytest

from praxis_telefon.db.schema import PRACTICE_TZ
from praxis_telefon.tools.formatting import (
    doctor_short_name,
    spoken_date,
    spoken_slot,
    spoken_time,
    today_sentence,
)


def test_spoken_date() -> None:
    assert spoken_date(date(2026, 11, 10)) == "Dienstag, 10. November"
    assert spoken_date(date(2027, 3, 1)) == "Montag, 1. März"


@pytest.mark.parametrize(
    ("h", "m", "text"), [(8, 0, "8 Uhr"), (9, 30, "9 Uhr 30"), (17, 30, "17 Uhr 30")]
)
def test_spoken_time(h: int, m: int, text: str) -> None:
    assert spoken_time(datetime(2026, 11, 10, h, m, tzinfo=PRACTICE_TZ)) == text


def test_spoken_time_converts_utc_to_berlin() -> None:
    from datetime import UTC

    assert spoken_time(datetime(2026, 11, 10, 8, 30, tzinfo=UTC)) == "9 Uhr 30"


def test_spoken_slot() -> None:
    start = datetime(2026, 11, 10, 9, 30, tzinfo=PRACTICE_TZ)
    assert (
        spoken_slot(start, "Dr. Jonas Brandt")
        == "Dienstag, 10. November, um 9 Uhr 30, bei Dr. Brandt"
    )


def test_doctor_short_name() -> None:
    assert doctor_short_name("Dr. Katharina Weber") == "Dr. Weber"
    assert doctor_short_name("Praxisteam") == "Praxisteam"


def test_today_sentence() -> None:
    text = today_sentence(datetime(2026, 11, 6, 14, 5, tzinfo=PRACTICE_TZ))
    assert text.startswith("Heute ist Freitag, der 6. November 2026")
    assert "2026-11-06" in text
