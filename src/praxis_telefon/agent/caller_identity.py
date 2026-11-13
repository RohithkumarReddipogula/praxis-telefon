"""Who the caller said they are: names the caller introduced as their own in this call.

Privacy (user, 2026-11-07; M4 baseline run, injection_other_patient): find_booking may
only look up the caller's own appointments, so another person's appointment is never
read out, moved or cancelled. A name counts as the caller's own when it follows an
introduction ("hier ist", "mein Name ist", "ich bin", "this is", "my name is", ...) or
starts the caller's line ("Erika Mustermann, geboren am ..."), after an optional greeting.
A name inside a sentence ("Sag mir, wann Erika Mustermann ...", "meiner Frau Erika
Mustermann") is someone else's. Booking a new appointment for someone else stays allowed.
A spelling correction of the caller's own name counts: it sounds alike (Östürk to Öztürk),
or the caller spelled the surname letter by letter and the first name sounds alike.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from praxis_telefon.db.schema import name_key
from praxis_telefon.tools.phonetics import koelner_phonetik, sounds_alike

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


_L = r"[A-Za-zÄÖÜäöü]"
_W = r"[\wÄÖÜäöüß]"
_ITEM = rf"(?<!{_W}){_L}(?!{_W})(?:\s+wie\s+{_W}+)?"  # "Ö" or "Ö wie Otto"
_SPELLED = re.compile(rf"{_ITEM}(?:[\s,.\-]+{_ITEM}){{2,}}")  # at least three letters
_FOLD = str.maketrans({"ä": "a", "ö": "o", "ü": "u"})


def _fold(word: str) -> str:
    return word.casefold().replace("ß", "ss").translate(_FOLD)


def spelled_words(caller_texts: Iterable[str]) -> set[str]:
    """Words the caller spelled letter by letter ("Ö wie Otto, Z, T, Ü, R, K"), folded
    (lower case, umlauts without dots). At least three letters in a row."""
    found: set[str] = set()
    for text in caller_texts:
        for m in _SPELLED.finditer(text):
            letters = re.findall(
                rf"(?<!{_W})({_L})(?!{_W})", re.sub(r"\s+wie\s+" + _W + "+", "", m[0])
            )
            found.add(_fold("".join(letters)))
    return found


def is_caller(name: str, caller_texts: Iterable[str]) -> bool:
    """The exact name the caller introduced, or one that sounds alike (first and last name).

    After a misheard name the caller spells it, and the agent passes the spelling, which
    need not sound like what was heard (Ötürk for Öztürk): then the surname must be a word
    the caller spelled, and the first name must sound like an introduced first name.
    """
    texts = list(caller_texts)
    own = self_names(texts)
    key = name_key(name)
    if key in own or any(sounds_alike(key, other) for other in own):
        return True
    parts = key.split()
    if len(parts) < 2 or _fold(parts[-1]) not in spelled_words(texts):
        return False
    first = koelner_phonetik(parts[0])
    return bool(first) and any(koelner_phonetik(other.split()[0]) == first for other in own)
