"""Relative dates are resolved in code with a fixed clock (Europe/Berlin)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from praxis_telefon.db.schema import PRACTICE_TZ
from praxis_telefon.tools import errors
from praxis_telefon.tools.relative_dates import (
    HORIZON_DAYS,
    DateRange,
    resolve_for_search,
    resolve_when,
)

MON = date(2026, 11, 9)
TUE, WED, THU, FRI, SAT, SUN = (MON + timedelta(days=i) for i in range(1, 7))
WEEK = [MON, TUE, WED, THU, FRI, SAT, SUN]
D = timedelta(days=1)


def _range(phrase: str, today: date) -> tuple[date, date]:
    r = resolve_when(phrase, today)
    return r.start, r.end


def _day(phrase: str, today: date) -> date:
    start, end = _range(phrase, today)
    assert start == end, (phrase, start, end)
    return start


# --- hand-written table, Monday 9 November 2026 ----------------------------------------


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("heute", MON),
        ("heute Morgen", MON),
        ("morgen", TUE),
        ("morgen Vormittag", TUE),
        ("morgen früh", TUE),
        ("übermorgen", WED),
        ("übermorgen nachmittags", WED),
        ("Dienstag", TUE),
        ("am Dienstag", TUE),
        ("nächsten Dienstag", TUE),  # approved: next occurrence, i.e. tomorrow
        ("kommenden Dienstag", TUE),
        ("diesen Freitag", FRI),
        ("übernächsten Dienstag", TUE + 7 * D),
        ("Dienstag in einer Woche", TUE + 7 * D),
        ("Dienstag nächste Woche", TUE + 7 * D),
        ("nächste Woche Mittwoch", WED + 7 * D),
        ("Dienstag in zwei Wochen", TUE + 14 * D),
        ("nächsten Montag", MON + 7 * D),
        ("Montag", MON + 7 * D),  # today is Monday: the next one
        ("dienstags", TUE),
        ("Dienstagvormittag", TUE),
        ("Dienstag morgen", TUE),  # Tuesday morning, not tomorrow
        ("in drei Tagen", THU),
        ("in 3 Tagen", THU),
        ("in 14 Tagen", MON + 14 * D),
        ("12. November", THU),
        ("am zwölften November", THU),
        ("den 12. November 2026", THU),
        ("12.11.", THU),
        ("12.11.2026", THU),
        ("2026-11-12", THU),
        ("am 12.", THU),
        ("am zwölften", THU),
        ("am ersten Dezember", date(2026, 12, 1)),
        ("einunddreißigsten Dezember", date(2026, 12, 31)),
        # English
        ("today", MON),
        ("tomorrow", TUE),
        ("tomorrow afternoon", TUE),
        ("the day after tomorrow", WED),
        ("Tuesday", TUE),
        ("next Thursday", THU),
        ("this Friday", FRI),
        ("Tuesday after next", TUE + 7 * D),
        ("Tuesday next week", TUE + 7 * D),
        ("in three days", THU),
        ("November 12", THU),
        ("the 12th of November", THU),
        ("12th November", THU),
        ("on the 12th", THU),
        ("December the first", date(2026, 12, 1)),
    ],
)
def test_single_days_from_monday(phrase: str, expected: date) -> None:
    assert _day(phrase, MON) == expected


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("diese Woche", (MON, FRI)),
        ("im Laufe der Woche", (MON, FRI)),
        ("nächste Woche", (MON + 7 * D, FRI + 7 * D)),
        ("kommende Woche", (MON + 7 * D, FRI + 7 * D)),
        ("übernächste Woche", (MON + 14 * D, FRI + 14 * D)),
        ("Ende der Woche", (THU, FRI)),
        ("Ende dieser Woche", (THU, FRI)),
        ("Ende nächster Woche", (THU + 7 * D, FRI + 7 * D)),
        ("Anfang der Woche", (MON, TUE)),
        ("Anfang nächster Woche", (MON + 7 * D, TUE + 7 * D)),
        ("in zwei Wochen", (MON + 14 * D, FRI + 14 * D)),
        ("in einer Woche", (MON + 7 * D, FRI + 7 * D)),
        ("Dienstag bis Donnerstag", (TUE, THU)),
        ("zwischen Dienstag und Donnerstag", (TUE, THU)),
        ("von morgen bis Freitag", (TUE, FRI)),
        ("12.11. bis 13.11.", (THU, FRI)),
        ("2026-11-09 bis 2026-11-13", (MON, FRI)),
        ("so bald wie möglich", (MON, MON + (HORIZON_DAYS - 1) * D)),
        ("egal wann", (MON, MON + (HORIZON_DAYS - 1) * D)),
        ("nächster freier Termin", (MON, MON + (HORIZON_DAYS - 1) * D)),
        ("nächsten Monat", (date(2026, 12, 1), date(2026, 12, 31))),
        # English
        ("this week", (MON, FRI)),
        ("next week", (MON + 7 * D, FRI + 7 * D)),
        ("the week after next", (MON + 14 * D, FRI + 14 * D)),
        ("end of the week", (THU, FRI)),
        ("end of next week", (THU + 7 * D, FRI + 7 * D)),
        ("early next week", (MON + 7 * D, TUE + 7 * D)),
        ("in two weeks", (MON + 14 * D, FRI + 14 * D)),
        ("from Tuesday to Thursday", (TUE, THU)),
        ("as soon as possible", (MON, MON + (HORIZON_DAYS - 1) * D)),
    ],
)
def test_ranges_from_monday(phrase: str, expected: tuple[date, date]) -> None:
    assert _range(phrase, MON) == expected


@pytest.mark.parametrize(
    "phrase", ["", "   ", "irgendwann mal", "wenn es passt", "blue", "31.11.", "30. Februar"]
)
def test_not_understood(phrase: str) -> None:
    with pytest.raises(errors.DateNotUnderstood):
        resolve_when(phrase, MON)


# --- every weekday as "today" ------------------------------------------------------------


@pytest.mark.parametrize("today", WEEK)
def test_morgen_and_uebermorgen_on_every_weekday(today: date) -> None:
    assert _day("morgen", today) == today + D
    assert _day("übermorgen", today) == today + 2 * D
    assert _day("heute", today) == today


@pytest.mark.parametrize("today", WEEK)
@pytest.mark.parametrize(
    ("name", "weekday"),
    [("Montag", 0), ("Dienstag", 1), ("Mittwoch", 2), ("Donnerstag", 3), ("Freitag", 4)],
)
def test_next_weekday_is_strictly_after_today(today: date, name: str, weekday: int) -> None:
    got = _day(f"nächsten {name}", today)
    assert got.weekday() == weekday
    assert today < got <= today + 7 * D
    assert _day(f"übernächsten {name}", today) == got + 7 * D


@pytest.mark.parametrize("today", WEEK)
def test_next_week_is_monday_to_friday_of_the_next_calendar_week(today: date) -> None:
    assert _range("nächste Woche", today) == (MON + 7 * D, FRI + 7 * D)


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (MON, (THU, FRI)),
        (WED, (THU, FRI)),
        (THU, (THU, FRI)),
        (FRI, (FRI, FRI)),
        (SAT, (THU + 7 * D, FRI + 7 * D)),
        (SUN, (THU + 7 * D, FRI + 7 * D)),
    ],
)
def test_end_of_week(today: date, expected: tuple[date, date]) -> None:
    assert _range("Ende der Woche", today) == expected


def test_this_week_on_a_weekend_runs_to_sunday() -> None:
    assert _range("diese Woche", SAT) == (SAT, SUN)


def test_this_week_weekday_in_the_past() -> None:
    with pytest.raises(errors.DateInPast):
        resolve_when("Montag diese Woche", WED)


# --- month and year boundaries ---------------------------------------------------------


def test_month_end() -> None:
    nov_30 = date(2026, 11, 30)  # Monday
    assert _day("morgen", nov_30) == date(2026, 12, 1)
    assert _day("übermorgen", nov_30) == date(2026, 12, 2)
    assert _range("Ende der Woche", nov_30) == (date(2026, 12, 3), date(2026, 12, 4))
    assert _day("am 2.", nov_30) == date(2026, 12, 2)  # day already past this month


def test_year_end() -> None:
    dec_31 = date(2026, 12, 31)  # Thursday
    assert _day("morgen", dec_31) == date(2027, 1, 1)
    assert _range("nächste Woche", dec_31) == (date(2027, 1, 4), date(2027, 1, 8))
    assert _day("5. Januar", dec_31) == date(2027, 1, 5)
    assert _day("nächsten Montag", dec_31) == date(2027, 1, 4)


def test_date_without_year_that_has_passed_is_next_year() -> None:
    assert _day("12.11.", date(2026, 11, 20)) == date(2027, 11, 12)


# --- resolve_for_search: Berlin clock, closed days, horizon ------------------------------


def _at(d: date, hour: int, minute: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=PRACTICE_TZ)


def test_tomorrow_on_friday_is_closed_and_falls_back_to_monday() -> None:
    found = resolve_for_search("morgen", _at(FRI, 10))
    assert found.closed
    assert found.requested == DateRange(SAT, SAT)
    assert found.search == DateRange(MON + 7 * D, MON + 7 * D)


def test_weekend_range_is_trimmed_to_open_days() -> None:
    found = resolve_for_search("von Freitag bis Montag", _at(THU, 10))
    assert not found.closed
    assert found.search == DateRange(FRI, MON + 7 * D)


def test_range_starting_in_the_past_starts_today() -> None:
    found = resolve_for_search("diese Woche", _at(WED, 10))
    assert found.search == DateRange(WED, FRI)


def test_too_far_ahead() -> None:
    with pytest.raises(errors.DateTooFar):
        resolve_for_search("12.11.", _at(date(2026, 11, 20), 10))  # next year
    with pytest.raises(errors.DateTooFar):
        resolve_for_search("in sechs Wochen", _at(MON, 10))


def test_long_range_is_clipped_to_the_horizon() -> None:
    found = resolve_for_search("so bald wie möglich", _at(MON, 10))
    assert found.search.start == MON
    assert found.search.end <= MON + (HORIZON_DAYS - 1) * D


def test_berlin_date_is_used_not_utc() -> None:
    # 23:30 UTC on Monday is already Tuesday 00:30 in Berlin (winter time, UTC+1).
    now = datetime(2026, 11, 9, 23, 30, tzinfo=UTC)
    assert resolve_for_search("morgen", now).search == DateRange(WED, WED)
    assert resolve_for_search("heute", now).search == DateRange(TUE, TUE)


def test_dst_end_october_2026() -> None:
    # Sunday 25 October 2026, clocks go back at 03:00. 22:30 UTC Saturday = 00:30 Sunday.
    now = datetime(2026, 10, 24, 22, 30, tzinfo=UTC)
    found = resolve_for_search("morgen", now)
    assert found.search == DateRange(date(2026, 10, 26), date(2026, 10, 26))  # Monday
    assert not found.closed


def test_dst_start_march_2027() -> None:
    # Sunday 28 March 2027, clocks go forward at 02:00. 23:30 UTC Saturday = 00:30 Sunday.
    now = datetime(2027, 3, 27, 23, 30, tzinfo=UTC)
    assert resolve_for_search("heute", now).requested == DateRange(
        date(2027, 3, 28), date(2027, 3, 28)
    )
    found = resolve_for_search("heute", now)
    assert found.closed and found.search == DateRange(date(2027, 3, 29), date(2027, 3, 29))


def test_naive_clock_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_for_search("morgen", datetime(2026, 11, 9, 10, 0))


def test_spoken_labels() -> None:
    assert DateRange(TUE, TUE).spoken() == "Dienstag, 10. November"
    assert DateRange(THU, FRI).spoken() == "Donnerstag, 12. November bis Freitag, 13. November"
