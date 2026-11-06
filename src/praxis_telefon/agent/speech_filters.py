"""Checks for agent sentences besides claims (see claim_check for those).

Regressions from the live run on 2026-11-06 (test_dates_live):
- meta: "This is a new appointment request, so I'll search for slots for next Tuesday."
  The agent spoke its own reasoning, in English, before the find_slots call.
- intent: "Möchten Sie einen neuen Termin für morgen, oder geht es um einen bestehenden
  Termin?" although the caller had named a day.
- weekday: "Morgen ist Samstag" before find_slots; dates are worked out in code.
- meta (narration, arm A run): "Das ist ein Terminwunsch für einen neuen Termin." before
  find_slots, German, so the language check let it through.

All checks are plain functions over one sentence plus what is known about the call.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date

from praxis_telefon.safety.phrases import Language
from praxis_telefon.text import normalize
from praxis_telefon.tools.errors import ToolFailure
from praxis_telefon.tools.relative_dates import resolve_when

_I = re.IGNORECASE

# --- language ------------------------------------------------------------------------------

_EN_WORDS = frozenset(
    [
        "the",
        "is",
        "a",
        "and",
        "i",
        "ill",
        "im",
        "ive",
        "for",
        "you",
        "your",
        "this",
        "that",
        "thats",
        "to",
        "of",
        "with",
        "can",
        "would",
        "it",
        "have",
        "are",
        "be",
        "please",
        "let",
        "me",
        "my",
        "we",
        "yes",
        "any",
        "what",
        "which",
        "do",
        "does",
        "hello",
        "hi",
        "thank",
        "thanks",
        "sorry",
        "search",
        "next",
        "appointment",
        "slots",
        "request",
        "there",
        "here",
        "go",
        "ahead",
        "check",
    ]
)
_DE_WORDS = frozenset(
    [
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "und",
        "ist",
        "ich",
        "sie",
        "ihnen",
        "ihr",
        "ihre",
        "ihren",
        "ein",
        "eine",
        "einen",
        "einem",
        "fuer",
        "mit",
        "zu",
        "nicht",
        "ja",
        "nein",
        "bitte",
        "gern",
        "gerne",
        "habe",
        "haben",
        "hat",
        "am",
        "um",
        "im",
        "auf",
        "es",
        "wir",
        "sind",
        "kann",
        "koennen",
        "moechten",
        "wie",
        "was",
        "wann",
        "noch",
        "auch",
        "oder",
        "aber",
        "dann",
        "schon",
        "guten",
        "tag",
        "termin",
        "danke",
        "herr",
        "frau",
        "bei",
        "vom",
        "zum",
        "zur",
        "einmal",
        "sehr",
        "gut",
        "egal",
    ]
)


def detect_language(text: str) -> Language | None:
    """German or English by common words; None if there are too few to tell."""
    words = normalize(text).split()
    en = sum(w in _EN_WORDS for w in words)
    de = sum(w in _DE_WORDS for w in words)
    if en >= 2 and en > 2 * de:
        return "en"
    if de >= 2 and de > 2 * en:
        return "de"
    return None


def call_language(caller_texts: Iterable[str]) -> Language:
    """Language of the caller's latest turn that shows one; German by default."""
    for text in reversed(list(caller_texts)):
        if (lang := detect_language(text)) is not None:
            return lang
    return "de"


# --- meta talk -----------------------------------------------------------------------------

_META_TOOL = re.compile(
    r"\b(find_slots|find_booking|move_booking|cancel_booking|take_message|slot_id|"
    r"booking_id|werkzeug\w*|tool|tools|function call|funktionsaufruf)\b",
    _I,
)
_META_CALLER = re.compile(
    r"\b(der anrufer|die anruferin|des anrufers|dem anrufer|den anrufer|der nutzer|"
    r"der benutzer|the caller|the user)\b",
    _I,
)
_META_NOTE = re.compile(
    r"^\s*(?:(?:hinweis|note|anmerkung|notiz|intern\w*|internal|reasoning|begründung|"
    r"überlegung)\s*:|#|[-*•]\s|\*\*)",
    _I,
)


# German narration about the request (live run arm A, explicit_date-0, before find_slots:
# "Das ist ein Terminwunsch für einen neuen Termin."). The "language" check only sees
# English in a German call, so German narration needs its own patterns: classifying the
# request, "also suche ich" (the German "so I'll search"), the caller in third person.
_REQUEST = r"(?:\w+\s+){0,2}?(?:terminwunsch\w*|terminanfrage\w*|anfrage\w*|anliegen)\b"
_META_NARRATION = re.compile(
    rf"^\s*(?:das|dies) ist (?:ein|eine|einen)\s+{_REQUEST}"
    rf"|^\s*es handelt sich um (?:ein|eine|einen)\s+(?:{_REQUEST}|neuen termin\b)"
    r"|\b(?:also|daher|deshalb|deswegen|darum) (?:suche|prüfe|rufe|schaue|buche|frage) ich\b"
    r"|^\s*(?:der patient|die patientin) (?:möchte|will|wünscht|braucht|fragt)\b",
    _I,
)


def meta_reasons(sentence: str, language: Language) -> list[str]:
    """Why the sentence is the agent talking about itself rather than to the caller."""
    reasons = []
    if _META_TOOL.search(sentence):
        reasons.append("tool")
    if _META_CALLER.search(sentence):
        reasons.append("caller")
    if _META_NOTE.search(sentence):
        reasons.append("note")
    if _META_NARRATION.search(sentence):
        reasons.append("narration")
    if language == "de" and detect_language(sentence) == "en":
        reasons.append("language")
    return reasons


# --- new or existing appointment ---------------------------------------------------------

# "einen neuen Termin, oder geht es um einen bestehenden", "neuen oder bestehenden Termin"
_NEW_OR_EXISTING = re.compile(
    r"\bneue[nmrs]?\b.{0,60}\bbestehende[nmrs]?\b|\bbestehende[nmrs]?\b.{0,60}\bneue[nmrs]?\b|"
    r"\bnew\b.{0,60}\bexisting\b|\bexisting\b.{0,60}\bnew\b",
    _I,
)
# "geboren am fünfzehnten Juni neunzehnhundertfünfundachtzig" is not a wanted day.
_BIRTH = re.compile(r" geboren (?:am |den )?(?:\w+ ){0,5}?(?:\d{4}|\w*hundert\w*|zweitausend\w*) ")
_TIME_WORDS = re.compile(
    r" (?:vormittag\w*|nachmittag\w*|morgens|abends|mittags|uhr|morning|afternoon|evening) "
)
_ANY_DAY = date(2026, 1, 5)  # only "does the text name a day" matters, not which day


def is_new_or_existing_question(sentence: str) -> bool:
    return bool(_NEW_OR_EXISTING.search(sentence))


def mentions_wanted_day(text: str) -> bool:
    """The caller named a day or a time of day (a date of birth does not count)."""
    n = _BIRTH.sub(" ", normalize(text))
    if _TIME_WORDS.search(n):
        return True
    try:
        resolve_when(n, _ANY_DAY)
    except ToolFailure:
        return False
    return True


# --- weekdays and months -------------------------------------------------------------------

_WEEKDAY_NAMES = {
    "montag": "monday", "dienstag": "tuesday", "mittwoch": "wednesday",
    "donnerstag": "thursday", "freitag": "friday", "samstag": "saturday",
    "sonnabend": "saturday", "sonntag": "sunday",
    **{
        d: d
        for d in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    },
}  # fmt: skip
# English "may" and "march" are left out: too often not a month.
_MONTH_NAMES = {
    "januar": "january", "februar": "february", "märz": "march", "maerz": "march",
    "april": "april", "mai": "may", "juni": "june", "juli": "july", "august": "august",
    "september": "september", "oktober": "october", "november": "november",
    "dezember": "december",
    **{m: m for m in ("january", "february", "june", "july", "october", "december")},
}  # fmt: skip
# Weekdays also as "montags", "Dienstagvormittag"; months as whole words.
_DAY_RE = re.compile(
    r"\b(" + "|".join(sorted(_WEEKDAY_NAMES, key=len, reverse=True)) + r")"
    r"|\b(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")\b",
    _I,
)


def day_names(text: str) -> set[str]:
    """Weekdays and months named in the text, in English, whatever the language."""
    found = set()
    for weekday, month in _DAY_RE.findall(text):
        found.add(_WEEKDAY_NAMES[weekday.lower()] if weekday else _MONTH_NAMES[month.lower()])
    return found


# A date of birth is not an appointment day. Only the date expression itself is skipped:
# a month followed by a year, or the date words right after "Geburtsdatum" / "geboren" /
# "date of birth". "Danke für Ihr Geburtsdatum, morgen ist Samstag" is still checked.
_ALL_MONTHS = "|".join(sorted({*_MONTH_NAMES, "may", "march"}, key=len, reverse=True))
_YEAR = r"(?:\d{4}|neunzehnhundert\w*|zweitausend\w*|nineteen\b|two thousand\b)"
_MONTH_THEN_YEAR = re.compile(rf"\b(?:{_ALL_MONTHS})\b[\s,.\"]*{_YEAR}", _I)
# Checked word by word; "date of birth" ends with the word "birth".
_BIRTH_TRIGGER = re.compile(r"^(?:geburtsdatum\w*|geboren|birth|birthday|born)$", _I)
_DATE_WORD = re.compile(
    rf"^(?:{_ALL_MONTHS}|\d{{1,4}}\.?|\d{{1,2}}(?:st|nd|rd|th)|\w+(?:te|ten|ter|tes|tem)|"
    r"first|second|third|\w+(?:th|st)|of|the|am|den|der|ist|is|was|on|zum|beispiel|"
    r"example|for|z|b|neunzehnhundert\w*|zweitausend\w*|nineteen|eighty\w*|ninety\w*|"
    r"seventy\w*|sixty\w*|fifty\w*|two|thousand)$",
    _I,
)


def _without_birth_dates(sentence: str) -> str:
    text = _MONTH_THEN_YEAR.sub(" ", sentence)
    out: list[str] = []
    skipping = False
    for token in re.split(r"(\s+)", text):
        word = token.strip(" ,.:;!?\"'„“")
        if not token.strip():
            out.append(token)
            continue
        if _BIRTH_TRIGGER.match(word):
            skipping = True
            out.append(token)
            continue
        if skipping and word and _DATE_WORD.match(word):
            continue  # part of the date of birth
        skipping = False
        out.append(token)
    return "".join(out)


def unsourced_days(sentence: str, sources: Iterable[str]) -> set[str]:
    """Days in the sentence that neither the caller, a tool result nor today's date named."""
    named = set().union(*(day_names(s) for s in sources))
    return day_names(_without_birth_dates(sentence)) - named
