"""Spoken German formatting for times the agent reads out."""

from __future__ import annotations

from datetime import date, datetime

from praxis_telefon.db.schema import PRACTICE_TZ

WEEKDAYS = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")
MONTHS = (
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
)


def spoken_date(d: date) -> str:
    """'Dienstag, 10. November'"""
    return f"{WEEKDAYS[d.weekday()]}, {d.day}. {MONTHS[d.month - 1]}"


def spoken_time(dt: datetime) -> str:
    """'9 Uhr 30' / '8 Uhr' in practice local time."""
    local = dt.astimezone(PRACTICE_TZ)
    return f"{local.hour} Uhr" if local.minute == 0 else f"{local.hour} Uhr {local.minute}"


def spoken_slot(start: datetime, doctor: str) -> str:
    """'Dienstag, 10. November, um 9 Uhr 30, bei Dr. Brandt'"""
    local = start.astimezone(PRACTICE_TZ)
    short_doctor = doctor_short_name(doctor)
    return f"{spoken_date(local.date())}, um {spoken_time(local)}, bei {short_doctor}"


def doctor_short_name(doctor: str) -> str:
    """'Dr. Katharina Weber' -> 'Dr. Weber'"""
    parts = doctor.split()
    if len(parts) >= 3 and parts[0] == "Dr.":
        return f"Dr. {parts[-1]}"
    return doctor


def today_sentence(now: datetime) -> str:
    """'Heute ist Freitag, der 6. November 2026.'"""
    local = now.astimezone(PRACTICE_TZ)
    d = local.date()
    return (
        f"Heute ist {WEEKDAYS[d.weekday()]}, der {d.day}. {MONTHS[d.month - 1]} {d.year}, "
        f"{spoken_time(local)}. Heutiges Datum (ISO): {d.isoformat()}."
    )
