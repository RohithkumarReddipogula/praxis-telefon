"""Live relative-date tests (M3 Part B). Skipped by default; run with:

    uv run pytest -m live tests/live/test_dates_live.py -v

The LLM must pass the caller's own words to find_slots (``when``); code turns them into
dates. Each scenario checks the date find_slots resolved ("Gesucht: ...") and that the
LLM did not send an ISO date it computed itself. Clock: Monday 9 November 2026, 07:00
Berlin, unless stated.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import pytest

from praxis_telefon.db.schema import PRACTICE_TZ
from tests.live.harness import (
    ERIKA,
    MAX,
    RUNS,
    Call,
    Step,
    agent_texts,
    converse,
    count,
    new_db,
    ok,
    self_computed_days,
    tool_arguments,
    tool_outputs,
    transcript_of,
)

pytestmark = pytest.mark.live

FRIDAY_10 = datetime(2026, 11, 13, 10, 0, tzinfo=PRACTICE_TZ)
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
# (caller line, expected start of the find_slots result, clock)
CASES: dict[str, tuple[str, str, datetime | None]] = {
    "morgen_vormittag": (
        f"Guten Tag, hier ist {MAX}. Ich hätte gern morgen Vormittag einen Termin.",
        "Gesucht: Dienstag, 10. November.",
        None,
    ),
    "naechsten_dienstag": (
        f"Hallo, {MAX}. Haben Sie nächsten Dienstag etwas frei?",
        "Gesucht: Dienstag, 10. November.",
        None,
    ),
    "uebermorgen": (
        f"Guten Tag, {MAX}. Geht es übermorgen?",
        "Gesucht: Mittwoch, 11. November.",
        None,
    ),
    "ende_der_woche": (
        f"Hallo, hier ist {MAX}. Ich bräuchte einen Termin Ende der Woche.",
        "Gesucht: Donnerstag, 12. November bis Freitag, 13. November.",
        None,
    ),
    "in_zwei_wochen": (
        f"Guten Tag, {MAX}. Ich hätte gern einen Termin in zwei Wochen.",
        "Gesucht: Montag, 23. November bis Freitag, 27. November.",
        None,
    ),
    "explicit_date": (
        f"Hallo, {MAX}. Haben Sie am zwölften November einen Termin?",
        "Gesucht: Donnerstag, 12. November.",
        None,
    ),
    "friday_tomorrow_closed": (
        f"Guten Tag, {MAX}. Haben Sie morgen einen Termin?",
        "Samstag, 14. November: Praxis geschlossen. Stattdessen gesucht: Montag, 16. November.",
        FRIDAY_10,
    ),
    "english_tomorrow_afternoon": (
        "Hi, my name is Max Mustermann. Could I get an appointment tomorrow afternoon?",
        "Gesucht: Dienstag, 10. November.",
        None,
    ),
    "english_next_thursday": (
        "Hello, this is Max Mustermann. Do you have anything next Thursday?",
        "Gesucht: Donnerstag, 12. November.",
        None,
    ),
}


def _when_args(call: Call) -> list[str]:
    return [str(json.loads(a).get("when", "")) for a in tool_arguments(call, "find_slots")]


@pytest.mark.parametrize("run", range(RUNS))
@pytest.mark.parametrize("case", CASES)
def test_relative_date_resolved_in_code(case: str, run: int) -> None:
    line, expected, clock = CASES[case]
    conn, _ = new_db(erika_booked=False)

    fixed: datetime | None = clock
    calls = converse(
        conn,
        MAX,
        [Step((line,), until=ok("find_slots"))],
        clock=None if fixed is None else (lambda: fixed),
    )

    info = transcript_of(calls)
    outputs = tool_outputs(calls[0], "find_slots")
    assert outputs, info
    assert outputs[0].startswith(expected), info
    assert not any(ISO_DATE.search(w) for w in _when_args(calls[0])), info
    # Regression (friday_tomorrow_closed-0): a wanted day means a new appointment.
    assert calls[0].intent_asked == 0, "asked new or existing\n" + info
    assert self_computed_days(calls[0], line) == set(), info


@pytest.mark.parametrize("run", range(RUNS))
def test_booking_lands_on_the_resolved_day(run: int) -> None:
    conn, _ = new_db(erika_booked=False)

    calls = converse(
        conn,
        MAX,
        [Step((f"Guten Tag, hier ist {MAX}. Nächsten Dienstag vormittags bitte.",), ok("book"))],
    )

    info = transcript_of(calls)
    assert calls[0].succeeded("book") == 1, info
    booked = conn.execute(
        "SELECT s.start_utc FROM bookings b JOIN slots s ON s.id = b.slot_id"
    ).fetchone()
    day = datetime.fromisoformat(booked["start_utc"]).astimezone(PRACTICE_TZ)
    assert (day.date().isoformat(), day.hour < 12) == ("2026-11-10", True), info
    assert count(conn, "SELECT count(*) FROM bookings") == 1, info


# --- new vs existing appointment (regression: explicit_date[1] called find_booking) -----


@pytest.mark.parametrize("run", range(RUNS))
def test_availability_question_searches_slots(run: int) -> None:
    conn, _ = new_db(erika_booked=False)

    calls = converse(
        conn,
        MAX,
        [
            Step(
                (f"Hallo, {MAX}. Haben Sie am zwölften November noch einen Termin frei?",),
                until=ok("find_slots"),
            )
        ],
    )

    info = transcript_of(calls)
    assert calls[0].calls("find_booking") == 0, info
    assert calls[0].intent_asked == 0, "asked new or existing\n" + info
    outputs = tool_outputs(calls[0], "find_slots")
    assert outputs and outputs[0].startswith("Gesucht: Donnerstag, 12. November."), info


@pytest.mark.parametrize("run", range(RUNS))
def test_own_appointment_question_looks_up_the_booking(run: int) -> None:
    conn, _ = new_db(erika_booked=True)  # Erika is booked on Tuesday 10 November, 9:00

    calls = converse(
        conn,
        ERIKA,
        [
            Step(
                (f"Guten Tag, hier ist {ERIKA}. Habe ich am Dienstag einen Termin?",),
                until=ok("find_booking"),
            )
        ],
    )

    info = transcript_of(calls)
    assert calls[0].succeeded("find_booking") >= 1, info
    assert calls[0].calls("find_slots") == 0, info
    assert any("Dienstag" in t for t in agent_texts(calls[0])), info
