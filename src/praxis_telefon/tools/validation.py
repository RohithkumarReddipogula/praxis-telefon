"""Shared argument checks for tools. Errors never echo the rejected value."""

from __future__ import annotations

from datetime import date, datetime

from praxis_telefon.db.schema import PRACTICE_TZ
from praxis_telefon.tools.errors import InvalidDateOfBirth, InvalidName

MAX_AGE_YEARS = 120
MAX_NAME_LENGTH = 100


def require_aware(now: datetime) -> None:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")


def clean_name(name: str) -> str:
    """Collapse whitespace; require at least first and last name."""
    cleaned = " ".join(name.split())
    if len(cleaned.split()) < 2 or len(cleaned) > MAX_NAME_LENGTH:
        raise InvalidName()
    return cleaned


def check_date_of_birth(date_of_birth: date, now: datetime) -> None:
    today = now.astimezone(PRACTICE_TZ).date()
    if date_of_birth > today or date_of_birth.year < today.year - MAX_AGE_YEARS:
        raise InvalidDateOfBirth()
