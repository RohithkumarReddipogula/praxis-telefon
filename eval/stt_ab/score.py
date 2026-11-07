"""Score STT transcripts against the utterances' items, and compare settings.

Each item is right or wrong after normalising like our tools do:
- phone: number words to digits with ``spoken_to_digits`` (the longest run of number
  words in the transcript), compared with the expected digits;
- date of birth: day, month and year read from words or digits (German or English),
  compared as a date;
- name: the first and last name as consecutive words; "strict" keeps umlauts and ß,
  "folded" counts ü/ue and ß/ss as equal (text.normalize).

Settings are compared with ``de`` on the same clips (McNemar, exact). Decision rule for
``multi`` (user, 2026-11-07): rejected if German accuracy on the three items combined
drops by more than 2 percentage points against ``de``; a single item type dropping by
more than 2 points is flagged.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from datetime import date

from praxis_telefon.text import normalize
from praxis_telefon.tools.relative_dates import _ORDINALS
from praxis_telefon.tools.spoken_numbers import spoken_to_digits

# --- phone ------------------------------------------------------------------------------


def _number_run(tokens: Sequence[str]) -> str | None:
    try:
        digits = spoken_to_digits(" ".join(tokens))
    except ValueError:
        return None
    digits = re.sub(r"\D", "", digits)
    return digits or None


def _de_int(word: str) -> int | None:
    """German cardinal up to 999,999 in one word: "neunzigtausendneun" -> 90009."""
    if word in ("", "ein", "eins"):
        return 1 if word else 0
    for unit, factor in (("tausend", 1000), ("hundert", 100)):
        if unit in word:
            head, tail = word.split(unit, 1)
            high = _de_int(head) if head else 1
            low = _de_int(tail) if tail else 0
            return None if high is None or low is None else high * factor + low
    try:
        digits = spoken_to_digits(word)
    except ValueError:
        return None
    return int(digits) if digits.isdigit() else None


def _merged_numbers(tokens: list[str]) -> list[str]:
    """Deepgram `de` sometimes merges digit groups into one number word
    ("sechstausendsechshundertsechsundneunzig", "sechs millionen ..."); write those
    as their digits so the run of numbers still reads as the phone number."""
    out: list[str] = []
    for token in tokens:
        if token in ("million", "millionen") and out and out[-1].isdigit():
            out[-1] = str(int(out[-1]) * 1_000_000)  # the next number adds onto it
            continue
        if ("tausend" in token or "hundert" in token) and (n := _de_int(token)) is not None:
            if out and out[-1].isdigit() and len(out[-1]) > 6 and out[-1].endswith("000000"):
                out[-1] = str(int(out[-1]) + n)
            else:
                out.append(str(n))
            continue
        out.append(token)
    return out


def phone_digits(transcript: str) -> str | None:
    """Digits of the longest run of number words or digits in the transcript."""
    tokens = _merged_numbers(normalize(transcript).split())
    best: str | None = None
    best_len = 0
    for start in range(len(tokens)):
        for end in range(len(tokens), start, -1):
            if end - start <= best_len:
                break
            if (digits := _number_run(tokens[start:end])) is not None:
                best, best_len = digits, end - start
                break
    return best


# --- date of birth ----------------------------------------------------------------------

_MONTHS = {
    **{m: i for i, m in enumerate(
        ["januar", "februar", "maerz", "april", "mai", "juni", "juli", "august",
         "september", "oktober", "november", "dezember"], 1)},
    **{m: i for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august",
         "september", "october", "november", "december"], 1)},
    "jaenner": 1,
}  # fmt: skip
_EN_SMALL = {
    "zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
            "seventy": 70, "eighty": 80, "ninety": 90}  # fmt: skip
_SKIP = frozenset({"the", "of", "am", "den", "der", "dem", "im", "and", "und"})


def _de_below_100(word: str) -> int | None:
    try:
        digits = spoken_to_digits(word)
    except ValueError:
        return None
    return int(digits) if digits.isdigit() and int(digits) < 100 else None


def _two_digit_year(n: int) -> int:
    return 1900 + n if n > 26 else 2000 + n


def _year(tokens: Sequence[str]) -> int | None:
    """Year from the tokens after day and month."""
    toks = [t for t in tokens if t not in _SKIP]
    if not toks:
        return None
    first = toks[0]
    if first.isdigit():
        if len(first) == 4:
            return int(first)
        if len(first) == 2 and len(toks) > 1 and toks[1].isdigit() and len(toks[1]) == 2:
            return int(first + toks[1])  # "19 85"
        if len(first) <= 2:
            return _two_digit_year(int(first))
        return None
    # German, one or two words: "neunzehnhundertfuenfundachtzig", "neunzehnhundert fuenfundachtzig"
    if m := re.fullmatch(r"(neunzehn|achtzehn)hundert(\w*)", first):
        century = 1900 if m[1] == "neunzehn" else 1800
        rest = m[2] or (toks[1] if len(toks) > 1 else "")
        n = 0 if rest == "" else _de_below_100(rest)
        return None if n is None else century + n
    if m := re.fullmatch(r"zweitausend(\w*)", first):
        rest = m[1] or (toks[1] if len(toks) > 1 else "")
        n = 0 if rest == "" else _de_below_100(rest)
        return None if n is None else 2000 + n
    # English: "nineteen eighty five", "nineteen oh five", "two thousand five"
    if first in _EN_SMALL and _EN_SMALL[first] in (18, 19) and len(toks) > 1:
        century = _EN_SMALL[first] * 100
        rest = toks[1:]
        if rest[0] == "hundred":
            return century
        if rest[0] in ("oh", "o") and len(rest) > 1 and rest[1] in _EN_SMALL:
            return century + _EN_SMALL[rest[1]]
        if rest[0] in _EN_TENS:
            unit = _EN_SMALL.get(rest[1], 0) if len(rest) > 1 else 0
            return century + _EN_TENS[rest[0]] + (unit if unit < 10 else 0)
        if rest[0] in _EN_SMALL and _EN_SMALL[rest[0]] >= 10:
            return century + _EN_SMALL[rest[0]]
        return None
    if first == "two" and len(toks) > 1 and toks[1] == "thousand":
        rest = [t for t in toks[2:] if t != "and"]
        if not rest:
            return 2000
        if rest[0] in _EN_SMALL:
            return 2000 + _EN_SMALL[rest[0]]
        if rest[0] in _EN_TENS:
            unit = _EN_SMALL.get(rest[1], 0) if len(rest) > 1 else 0
            return 2000 + _EN_TENS[rest[0]] + (unit if unit < 10 else 0)
        return None
    if (n := _de_below_100(first)) is not None:
        return _two_digit_year(n)  # "fuenfundachtzig"
    return None


def _day(token: str, nxt: str | None) -> tuple[int, int] | None:
    """(day, tokens used) from an ordinal word, "15", "15th", or "twenty fifth"."""
    if nxt is not None and f"{token} {nxt}" in _ORDINALS:
        return _ORDINALS[f"{token} {nxt}"], 2
    if token in _ORDINALS:
        return _ORDINALS[token], 1
    if m := re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th|te[nr]?)?", token):
        return int(m[1]), 1
    return None


def _month(token: str) -> int | None:
    if token in _MONTHS:
        return _MONTHS[token]
    if token in _ORDINALS and _ORDINALS[token] <= 12:
        return _ORDINALS[token]  # "sechste"
    if token.isdigit() and 1 <= int(token) <= 12:
        return int(token)
    return None


def _valid(year: int | None, month: int, day: int) -> date | None:
    if year is None:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _dotted(tokens: list[str]) -> list[str]:
    """Drop "the"/"of"; "zweiundzwanzig punkt fuenf" (Deepgram `de` for "22.5.") -> "22 5"."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if (
            i + 2 < len(tokens)
            and tokens[i + 1] == "punkt"
            and (day := _de_below_100(t)) is not None
            and (month := _de_below_100(tokens[i + 2])) is not None
            and 1 <= day <= 31
            and 1 <= month <= 12
        ):
            out += [str(day), str(month)]
            i += 3
            continue
        if t not in ("the", "of"):
            out.append(t)
        i += 1
    return out


def dob(transcript: str) -> date | None:
    """The date in the transcript: day month year (German, British) or month day year."""
    tokens = _dotted(normalize(transcript).split())
    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        # Month name first: "June fifteenth nineteen eighty five"
        if token in _MONTHS and nxt is not None:
            after = tokens[i + 2] if i + 2 < len(tokens) else None
            if (day := _day(nxt, after)) is not None:
                found = _valid(_year(tokens[i + 1 + day[1] :]), _MONTHS[token], day[0])
                if found:
                    return found
        # Day first: "fuenfzehnten juni ...", "fuenfzehnte sechste ...", "15 06 1985"
        if (day := _day(token, nxt)) is not None:
            j = i + day[1]
            if j < len(tokens) and (month := _month(tokens[j])) is not None:
                found = _valid(_year(tokens[j + 1 :]), month, day[0])
                if found:
                    return found
    return None


# --- name -------------------------------------------------------------------------------


def _strict(text: str) -> list[str]:
    # lower(), not casefold(): casefold turns "ß" into "ss", which strict must not do.
    return re.sub(r"[^\wäöüß]+", " ", text.lower()).split()


def name_found(transcript: str, name: str, *, folded: bool) -> bool:
    """First and last name as consecutive words in the transcript."""
    words = normalize(transcript).split() if folded else _strict(transcript)
    target = normalize(name).split() if folded else _strict(name)
    n = len(target)
    return any(words[i : i + n] == target for i in range(len(words) - n + 1))


# --- item check, WER, statistics --------------------------------------------------------


def item_correct(kind: str, expected: str, transcript: str, *, folded: bool = True) -> bool:
    if kind == "phone":
        return phone_digits(transcript) == expected
    if kind == "dob":
        found = dob(transcript)
        return found is not None and found.isoformat() == expected
    return name_found(transcript, expected, folded=folded)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate on normalised words."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / max(1, len(ref))


def mcnemar_p(base: Iterable[bool], other: Iterable[bool]) -> float:
    """Exact two-sided McNemar test on paired right/wrong results."""
    pairs = list(zip(base, other, strict=True))
    b = sum(1 for x, y in pairs if x and not y)  # base right, other wrong
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def multi_decision(de: dict[str, list[bool]], multi: dict[str, list[bool]]) -> dict[str, object]:
    """User's rule: combined German accuracy drop > 2 points rejects; per-kind > 2 is flagged.

    ``de`` and ``multi`` map item kind to paired right/wrong results on German clips.
    """

    def acc(results: Sequence[bool]) -> float:
        return 100 * sum(results) / len(results) if results else 0.0

    combined_de = [r for k in sorted(de) for r in de[k]]
    combined_multi = [r for k in sorted(de) for r in multi[k]]
    drop = acc(combined_de) - acc(combined_multi)
    flags = {k: round(acc(de[k]) - acc(multi[k]), 1) for k in de if acc(de[k]) - acc(multi[k]) > 2}
    return {
        "combined_drop_points": round(drop, 1),
        "rejected": drop > 2,
        "flagged_kinds": flags,
        "p_mcnemar": mcnemar_p(combined_de, combined_multi),
    }
