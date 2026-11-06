"""take_message: a callback note for staff. A fixed category only, no free text."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from praxis_telefon.db.schema import to_utc_iso
from praxis_telefon.tools.errors import InvalidCallbackNumber, InvalidReasonCategory
from praxis_telefon.tools.validation import clean_name, require_aware

ReasonCategory = Literal[
    "prescription", "referral", "test_results", "appointment", "callback", "other"
]
REASON_LABELS_DE: dict[str, str] = {
    "prescription": "Rezept",
    "referral": "Überweisung",
    "test_results": "Befund",
    "appointment": "Termin",
    "callback": "Rückruf",
    "other": "Sonstiges",
}

_SEPARATORS = re.compile(r"[\s\-/().]")
# +49 or 0, then a non-zero digit, 6-15 digits in total after the prefix.
_GERMAN_NUMBER = re.compile(r"^(?:\+49|0)[1-9]\d{5,14}$")


@dataclass(frozen=True)
class MessageResult:
    message_id: int
    category_label: str


def normalize_callback_number(raw: str) -> str:
    number = _SEPARATORS.sub("", raw)
    if number.startswith("0049"):
        number = "+49" + number[4:]
    if not _GERMAN_NUMBER.match(number):
        raise InvalidCallbackNumber()
    return number


def take_message(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    caller_name: str,
    callback_number: str,
    reason_category: str,
) -> MessageResult:
    require_aware(now)
    name = clean_name(caller_name)
    number = normalize_callback_number(callback_number)
    if reason_category not in REASON_LABELS_DE:
        raise InvalidReasonCategory()
    message_id = conn.execute(
        "INSERT INTO staff_messages (caller_name, callback_number, reason_category, created_at)"
        " VALUES (?, ?, ?, ?)",
        (name, number, reason_category, to_utc_iso(now)),
    ).lastrowid
    return MessageResult(int(message_id or 0), REASON_LABELS_DE[reason_category])
