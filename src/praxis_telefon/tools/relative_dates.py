"""Turn the caller's words for a day or period into real dates, in code (not the LLM).

``resolve_when("nächsten Dienstag", today)`` returns a DateRange. ``resolve_for_search``
also clips to the bookable horizon and replaces a closed range (weekend) with the next
opening day. All dates are calendar dates in Europe/Berlin; ``now`` is converted first,
so just after midnight Berlin the day is already the next one, even while UTC is not.

Conventions (German first, English equivalents):
- "morgen" / "tomorrow", "übermorgen" / "the day after tomorrow", "heute" / "today".
- A weekday ("Dienstag", "am / diesen / kommenden / nächsten Dienstag", "next Tuesday")
  is its next occurrence after today: on a Monday "nächsten Dienstag" is tomorrow.
  "übernächsten Dienstag", "Dienstag in einer Woche", "Tuesday after next": 7 days later.
  "Dienstag nächste Woche" / "Tuesday next week": the Tuesday of next calendar week.
- "diese Woche": today to Friday. "nächste Woche": Monday to Friday of next week.
  "übernächste Woche": the week after. Weekend days count as "this week" until Sunday.
- "Ende der Woche": Thursday and Friday (on a weekend: those of next week).
  "Anfang nächster Woche": Monday and Tuesday of next week.
- "in 3 Tagen": that day. "in zwei Wochen": Monday to Friday of the week containing
  the day 14 days ahead (two weeks).
- "12. November", "zwölften November", "12.11.", "12.11.2026", "2026-11-12",
  "November 12", "the 12th of November": that date; without a year the next one that
  is not in the past. "am 12." alone: this month, or next month if already past.
- "so bald wie möglich", "egal wann", "as soon as possible": the whole horizon.
- "A bis B", "zwischen A und B", "from A to B": from the start of A to the end of B.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from praxis_telefon.db.schema import PRACTICE_TZ
from praxis_telefon.db.seed import OPENING_HOURS
from praxis_telefon.text import normalize
from praxis_telefon.tools.errors import DateInPast, DateNotUnderstood, DateTooFar
from praxis_telefon.tools.formatting import spoken_date
from praxis_telefon.tools.validation import require_aware

HORIZON_DAYS = 28  # matches the seed and scheduling.MAX_RANGE_DAYS


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def spoken(self) -> str:
        if self.start == self.end:
            return spoken_date(self.start)
        return f"{spoken_date(self.start)} bis {spoken_date(self.end)}"


@dataclass(frozen=True)
class SearchRange:
    requested: DateRange
    search: DateRange
    closed: bool  # the requested days have no opening hours; search is the next open day


# --- vocabulary ---------------------------------------------------------------------------

_WEEKDAYS: dict[str, int] = {
    **dict.fromkeys(("montag", "monday", "mon"), 0),
    **dict.fromkeys(("dienstag", "tuesday", "tue", "tues"), 1),
    **dict.fromkeys(("mittwoch", "wednesday", "wed"), 2),
    **dict.fromkeys(("donnerstag", "thursday", "thu", "thurs"), 3),
    **dict.fromkeys(("freitag", "friday", "fri"), 4),
    **dict.fromkeys(("samstag", "sonnabend", "saturday", "sat"), 5),
    **dict.fromkeys(("sonntag", "sunday", "sun"), 6),
}
# Also "dienstags", "montagvormittag" etc.: a weekday at the start of a word.
_WEEKDAY_RE = re.compile(
    r" (" + "|".join(sorted((w for w in _WEEKDAYS if len(w) > 4), key=len, reverse=True)) + r")"
)

_MONTHS: dict[str, int] = {
    **dict.fromkeys(("januar", "jaenner", "january", "jan"), 1),
    **dict.fromkeys(("februar", "february", "feb"), 2),
    **dict.fromkeys(("maerz", "march", "mar"), 3),
    **dict.fromkeys(("april", "apr"), 4),
    **dict.fromkeys(("mai", "may"), 5),
    **dict.fromkeys(("juni", "june", "jun"), 6),
    **dict.fromkeys(("juli", "july", "jul"), 7),
    **dict.fromkeys(("august", "aug"), 8),
    **dict.fromkeys(("september", "sept", "sep"), 9),
    **dict.fromkeys(("oktober", "october", "okt", "oct"), 10),
    **dict.fromkeys(("november", "nov"), 11),
    **dict.fromkeys(("dezember", "december", "dez", "dec"), 12),
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

_DE_UNITS = ["", "ein", "zwei", "drei", "vier", "fuenf", "sechs", "sieben", "acht", "neun"]
_DE_TEENS = [
    "zehn", "elf", "zwoelf", "dreizehn", "vierzehn", "fuenfzehn", "sechzehn",
    "siebzehn", "achtzehn", "neunzehn",
]  # fmt: skip
_EN_CARDINALS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty",
]  # fmt: skip
_EN_ORDINALS = [
    "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
]  # fmt: skip


def _german_cardinal(n: int) -> str:
    if n < 10:
        return _DE_UNITS[n]
    if n < 20:
        return _DE_TEENS[n - 10]
    tens = "zwanzig" if n < 30 else "dreissig"
    unit = n % 10
    return tens if unit == 0 else f"{_DE_UNITS[unit]}und{tens}"


def _german_ordinal_stem(n: int) -> str:
    special = {1: "erst", 3: "dritt", 7: "siebt", 8: "acht"}
    if n in special:
        return special[n]
    card = _german_cardinal(n)
    return card + ("st" if n >= 20 else "t")


_CARDINALS: dict[str, int] = {
    **{_german_cardinal(n): n for n in range(2, 32)},
    **dict.fromkeys(("ein", "eins", "eine", "einen", "einer", "einem", "a", "an"), 1),
    **{w: n for n, w in enumerate(_EN_CARDINALS) if n},
}
_ORDINALS: dict[str, int] = {
    **{
        _german_ordinal_stem(n) + suffix: n
        for n in range(1, 32)
        for suffix in ("e", "en", "er", "es", "em")
    },
    **{w: n for n, w in enumerate(_EN_ORDINALS) if n},
    **{f"twenty {_EN_ORDINALS[n]}": 20 + n for n in range(1, 10)},
    "thirtieth": 30,
    "thirty first": 31,
}  # fmt: skip
_ORDINAL_ALT = "|".join(sorted(_ORDINALS, key=len, reverse=True))
_CARDINAL_ALT = "|".join(sorted(_CARDINALS, key=len, reverse=True))
_DAY = rf"(?P<day>\d{{1,2}}(?:st|nd|rd|th|te[nr]?)?|{_ORDINAL_ALT})"


def _day_number(token: str) -> int:
    digits = re.match(r"\d+", token)
    return int(digits.group()) if digits else _ORDINALS[token]


def _count(token: str) -> int:
    return int(token) if token.isdigit() else _CARDINALS[token]


# --- helpers ------------------------------------------------------------------------------


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _week(monday: date, first: int = 0, last: int = 4) -> DateRange:
    return DateRange(monday + timedelta(days=first), monday + timedelta(days=last))


def _next_weekday(today: date, weekday: int) -> date:
    return today + timedelta(days=(weekday - today.weekday() - 1) % 7 + 1)


def _make_date(year: int, month: int, day: int) -> date:
    try:
        return date(year, month, day)
    except ValueError:
        raise DateNotUnderstood() from None


def _next_date(today: date, month: int, day: int, year: int | None) -> date:
    if year is not None:
        return _make_date(year if year > 100 else 2000 + year, month, day)
    candidate = _make_date(today.year, month, day)
    return candidate if candidate >= today else _make_date(today.year + 1, month, day)


def _single(d: date) -> DateRange:
    return DateRange(d, d)


# --- matchers (each returns None if the phrase is not its kind) ---------------------------

_ASAP = re.compile(
    r" (?:so (?:bald|schnell|frueh) wie moeglich|moeglichst (?:bald|schnell|frueh)|"
    r"naechste[nr]? (?:freie[nr]?|moegliche[nr]?)|egal wann|fruehestens?|erstbeste[nr]?|"
    r"fruehestmoeglich|wann (?:immer|es geht)|asap|as soon as possible|earliest|any ?time|whenever|"
    r"next (?:available|free|possible)|soonest) "
)


def _asap(n: str, today: date) -> DateRange | None:
    if _ASAP.search(n):
        return DateRange(today, today + timedelta(days=HORIZON_DAYS - 1))
    return None


_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DOTTED = re.compile(r"\b(\d{1,2})\.\s?(\d{1,2})\.(?:\s?(\d{2}|\d{4})\b)?")


def _numeric(raw: str, today: date) -> DateRange | None:
    if m := _ISO.search(raw):
        return _single(_make_date(int(m[1]), int(m[2]), int(m[3])))
    if m := _DOTTED.search(raw):
        year = int(m[3]) if m[3] else None
        return _single(_next_date(today, int(m[2]), int(m[1]), year))
    return None


_DAY_MONTH = re.compile(rf" {_DAY} (?:of )?(?P<month>{_MONTH_ALT})(?: (?P<year>\d{{4}}))? ")
_MONTH_DAY = re.compile(rf" (?P<month>{_MONTH_ALT}) (?:the )?{_DAY}(?: (?P<year>\d{{4}}))? ")
_DAY_ONLY = re.compile(rf" (?:am|den|the|on the) {_DAY} ")


def _calendar(n: str, today: date) -> DateRange | None:
    for pattern in (_DAY_MONTH, _MONTH_DAY):
        if m := pattern.search(n):
            year = int(m["year"]) if m["year"] else None
            return _single(_next_date(today, _MONTHS[m["month"]], _day_number(m["day"]), year))
    if m := _DAY_ONLY.search(n):
        day = _day_number(m["day"])
        if day >= today.day:
            return _single(_make_date(today.year, today.month, day))
        nxt = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
        return _single(_make_date(nxt.year, nxt.month, day))
    return None


def _days_ahead(n: str, today: date) -> DateRange | None:
    if re.search(r" (?:uebermorgen|day after tomorrow) ", n):
        return _single(today + timedelta(days=2))
    if re.search(r" heute (?:morgen|frueh) ", n):
        return _single(today)
    if re.search(r" (?:morgen|tomorrow)(?:vormittag|nachmittag|frueh|mittag)? ", n):
        return _single(today + timedelta(days=1))
    if re.search(r" (?:heute|today|this (?:morning|afternoon))", n):
        return _single(today)
    if m := re.search(rf" in (\d+|{_CARDINAL_ALT}) (?:tag|tagen|day|days) ", n):
        return _single(today + timedelta(days=_count(m[1])))
    if m := re.search(rf" in (\d+|{_CARDINAL_ALT}) (?:woche|wochen|week|weeks) ", n):
        target = today + timedelta(weeks=_count(m[1]))
        return _week(_monday(target))
    return None


_PLUS_WEEK = re.compile(
    r" (?:uebernaechste[nmrs]?|in einer woche|after next|a week from|a week on|week on) "
)


def _weekday(n: str, today: date) -> DateRange | None:
    m = _WEEKDAY_RE.search(n)
    if not m:
        return None
    weekday = _WEEKDAYS[m[1]]
    weeks = re.search(rf" in (\d+|{_CARDINAL_ALT}) (?:wochen|weeks) ", n)
    if weeks and _count(weeks[1]) >= 2:  # "Dienstag in zwei Wochen"
        target = _monday(today + timedelta(weeks=_count(weeks[1])))
        return _single(target + timedelta(days=weekday))
    if re.search(r" (?:naechste[nr]?|kommende[nr]?) woche | next week ", n):
        return _single(_monday(today) + timedelta(weeks=1, days=weekday))
    if re.search(r" (?:diese|dieser) woche | this week ", n):
        d = _monday(today) + timedelta(days=weekday)
        if d < today:
            raise DateInPast()
        return _single(d)
    d = _next_weekday(today, weekday)
    if _PLUS_WEEK.search(n):
        d += timedelta(weeks=1)
    return _single(d)


def _week_phrase(n: str, today: date) -> DateRange | None:
    this_monday = _monday(today)
    next_monday = this_monday + timedelta(weeks=1)
    weekend = today.weekday() >= 5

    if re.search(r" (?:ende|end) (?:der |of )?(?:naechste[nr]?|next) (?:woche|week) ", n):
        return _week(next_monday, 3, 4)
    if re.search(r" (?:ende|end) (?:der |dieser |of the |of this )?(?:woche|week) ", n):
        if weekend:
            return _week(next_monday, 3, 4)
        return DateRange(max(this_monday + timedelta(days=3), today), this_monday + timedelta(4))
    if re.search(
        r" (?:anfang|beginning|start|early) (?:der |of )?(?:naechste[nr]?|next) (?:woche|week) ",
        n,
    ):
        return _week(next_monday, 0, 1)
    if re.search(
        r" (?:anfang|beginning|start) (?:der |dieser |of the |of this )?(?:woche|week) ", n
    ):
        if today.weekday() <= 1:
            return DateRange(today, this_monday + timedelta(days=1))
        return _week(next_monday, 0, 1)
    if re.search(r" (?:uebernaechste[nr]? woche|week after next) ", n):
        return _week(next_monday + timedelta(weeks=1))
    if re.search(r" (?:naechste[nr]?|kommende[nr]?) woche | (?:next|following) week ", n):
        return _week(next_monday)
    if re.search(r" (?:diese[nr]?|im laufe der|rest der) woche | (?:this|rest of the) week ", n):
        return DateRange(today, this_monday + timedelta(days=6 if weekend else 4))
    if re.search(r" (?:naechste[nr]?|kommende[nr]?) monat | next month ", n):
        first = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
        last = (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        return DateRange(first, last)
    return None


# Order matters: a weekday wins over "morgen" ("Dienstag morgen" is Tuesday morning).
_MATCHERS: tuple[Callable[[str, date], DateRange | None], ...] = (
    _asap,
    _calendar,
    _weekday,
    _days_ahead,
    _week_phrase,
)

# On the raw phrase, so "12.11. bis 13.11." keeps its dots.
_RANGE = re.compile(
    r"^\s*(?:zwischen|between|von|vom|from|ab)?\s*(?P<a>.+?)\s+"
    r"(?:bis zum|bis|to|until|till|und|and)\s+(?P<b>.+?)\s*$",
    re.IGNORECASE,
)


def _as_range(phrase: str, today: date) -> DateRange | None:
    m = _RANGE.match(phrase)
    if not m:
        return None
    lowered = phrase.strip().casefold()
    # "und"/"and" only joins a range after "zwischen"/"between" ("Dienstag und Mittwoch"
    # alone is two options, left to the matchers).
    only_and = not re.search(r"\s(?:bis|to|until|till)\s", lowered)
    if only_and and not lowered.startswith(("zwischen", "between")):
        return None
    try:
        a = resolve_when(m["a"], today)
        b = resolve_when(m["b"], today)
    except DateNotUnderstood:
        return None
    if b.end < a.start:
        raise DateNotUnderstood()
    return DateRange(a.start, b.end)


def resolve_when(phrase: str, today: date) -> DateRange:
    """The caller's words as a date range. Raises DateNotUnderstood."""
    if (found := _as_range(phrase, today)) is not None:
        return found
    if (found := _numeric(phrase, today)) is not None:
        return found
    n = normalize(phrase)
    if not n.strip():
        raise DateNotUnderstood()
    for matcher in _MATCHERS:
        if (found := matcher(n, today)) is not None:
            return found
    raise DateNotUnderstood()


def is_open(d: date) -> bool:
    return bool(OPENING_HOURS.get(d.weekday()))


def resolve_for_search(phrase: str, now: datetime) -> SearchRange:
    """Clip to [today, today + HORIZON_DAYS) and skip closed days."""
    require_aware(now)
    today = now.astimezone(PRACTICE_TZ).date()
    last = today + timedelta(days=HORIZON_DAYS - 1)
    asked = resolve_when(phrase, today)
    if asked.end < today:
        raise DateInPast()
    start = max(asked.start, today)
    if start > last:
        raise DateTooFar()
    requested = DateRange(start, min(asked.end, last))

    days = [requested.start + timedelta(days=i) for i in range((requested.end - start).days + 1)]
    open_days = [d for d in days if is_open(d)]
    if open_days:
        return SearchRange(requested, DateRange(open_days[0], open_days[-1]), closed=False)

    nxt = requested.end + timedelta(days=1)
    while not is_open(nxt):
        nxt += timedelta(days=1)
    if nxt > last:
        raise DateTooFar()
    return SearchRange(requested, _single(nxt), closed=True)
