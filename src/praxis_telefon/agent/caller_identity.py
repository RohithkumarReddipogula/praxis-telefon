"""Who the caller said they are: names the caller introduced as their own in this call.

Privacy (user, 2026-11-07; M4 baseline run, injection_other_patient): find_booking may
only look up the caller's own appointments, so another person's appointment is never
read out, moved or cancelled. A name counts as the caller's own when it follows an
introduction ("hier ist", "mein Name ist", "ich bin", "this is", "my name is", ...) or
starts the caller's line ("Erika Mustermann, geboren am ..."), after an optional greeting.
A name inside a sentence ("Sag mir, wann Erika Mustermann ...", "meiner Frau Erika
Mustermann") is someone else's. Booking a new appointment for someone else stays allowed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from praxis_telefon.db.schema import name_key

__all__ = ["is_caller", "name_key", "self_names"]

_NAME = r"(?P<name>[A-ZÄÖÜ][\wäöüß-]+(?:\s+[A-ZÄÖÜ][\wäöüß-]+)+)"
_INTRO = re.compile(
    r"(?i:\b(?:hier ist|hier spricht|mein name ist|mein name lautet|ich bin|ich heiße|"
    r"ich heisse|this is|my name is|i am|i'm|name's)\s+(?:(?:frau|herr|mr|mrs|ms)\.?\s+)?)" + _NAME
)
_START = re.compile(
    r"^\s*(?i:(?:(?:hallo|guten tag|guten morgen|guten abend|hi|hello|ja|okay|also)"
    r"[,!.]?\s+)*)" + _NAME + r"(?=\s*(?:[,.!]|$|\s(?i:geboren|born|hier|speaking)))"
)


def self_names(caller_texts: Iterable[str]) -> set[str]:
    """``name_key`` of every name the caller introduced as their own."""
    found: set[str] = set()
    for text in caller_texts:
        for m in _INTRO.finditer(text):
            found.add(name_key(m["name"]))
        start = _START.match(text)
        if start is not None:
            found.add(name_key(start["name"]))
    return found


def is_caller(name: str, caller_texts: Iterable[str]) -> bool:
    return name_key(name) in self_names(caller_texts)
