"""Turn a phone number in spoken words into digits, before it is validated.

Bug (live test human_request[1], 2026-11-06): the LLM passed the callback number as
"null drei null, zwei drei eins zwei fünf, vier fünf sechs", take_message rejected it,
and the agent kept asking for the number without saving.

German: single digits (null, eins, zwei/zwo, ...), pairs ("dreiundzwanzig"), hundreds
("vierhundertsechsundfünfzig"), "doppel"/"doppelt" before a digit, "plus" for +.
English: zero/oh, one, ..., "twenty three", "four fifty six" (four, 56), "double"/
"triple", "plus". Digits already present are kept. Any other word makes it invalid, so
nothing is guessed.
"""

from __future__ import annotations

import re

from praxis_telefon.text import normalize

_DE_UNITS = {
    "null": 0, "eins": 1, "ein": 1, "eine": 1, "zwei": 2, "zwo": 2, "drei": 3, "vier": 4,
    "fuenf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9,
}  # fmt: skip
_DE_TEENS = {
    "zehn": 10, "elf": 11, "zwoelf": 12, "dreizehn": 13, "vierzehn": 14, "fuenfzehn": 15,
    "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
}  # fmt: skip
_DE_TENS = {
    "zwanzig": 20, "dreissig": 30, "vierzig": 40, "fuenfzig": 50, "sechzig": 60,
    "siebzig": 70, "achtzig": 80, "neunzig": 90,
}  # fmt: skip

_EN_UNITS = {
    "zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}  # fmt: skip
_EN_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip

_REPEAT = {"doppel": 2, "doppelt": 2, "double": 2, "triple": 3, "dreifach": 3}
_FILLERS = frozenset({"und", "and", "komma", "dann", "then"})


def _german_below_100(word: str) -> int | None:
    if word in _DE_UNITS:
        return _DE_UNITS[word]
    if word in _DE_TEENS:
        return _DE_TEENS[word]
    if word in _DE_TENS:
        return _DE_TENS[word]
    if m := re.fullmatch(r"(\w+?)und(\w+)", word):
        unit, tens = _DE_UNITS.get(m[1]), _DE_TENS.get(m[2])
        if unit is not None and unit > 0 and tens is not None:
            return tens + unit
    return None


def _german_word(word: str) -> str | None:
    """Digits for one German number word, keeping the number of digits it stands for."""
    if (n := _german_below_100(word)) is not None:
        return str(n)
    if "hundert" in word:
        left, _, right = word.partition("hundert")
        hundreds = 1 if left in ("", "ein", "eins") else _DE_UNITS.get(left)
        rest = 0 if right == "" else _german_below_100(right)
        if hundreds and rest is not None:
            return f"{hundreds}{rest:02d}"
    return None


def spoken_to_digits(raw: str) -> str:
    """Replace spoken number words with digits; return the input unchanged if it has none.

    Raises ValueError for any word that is not part of a number.
    """
    words = normalize(raw).split()
    if not any(w.isalpha() for w in words):
        return raw
    plus = raw.lstrip().startswith("+")
    out: list[str] = []
    repeat = 1
    i = 0
    while i < len(words):
        w = words[i]
        i += 1
        if w in _FILLERS:
            continue
        if w == "plus" and not out:
            plus = True
            continue
        if w in _REPEAT:
            repeat = _REPEAT[w]
            continue
        if w.isdigit():
            digits = w
        elif w in _EN_TENS:
            value = _EN_TENS[w]
            # "twenty three" is 23; "twenty" alone is 20.
            if i < len(words) and words[i] in _EN_UNITS and _EN_UNITS[words[i]] > 0:
                value += _EN_UNITS[words[i]]
                i += 1
            digits = str(value)
        elif w in _EN_TEENS:
            digits = str(_EN_TEENS[w])
        elif w in _EN_UNITS:
            digits = str(_EN_UNITS[w])
        elif (german := _german_word(w)) is not None:
            digits = german
        else:
            raise ValueError("not a number word")
        out.append(digits * repeat)
        repeat = 1
    if not out:
        raise ValueError("no digits")
    return ("+" if plus else "") + "".join(out)
