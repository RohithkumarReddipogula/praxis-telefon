"""Synthetic caller utterances for the STT A/B, each with the one item it carries.

Every German and English utterance carries exactly one item: a name, a date of birth or
a phone number, spoken the way callers say them. Switch calls start in German (no item)
and switch to English for the item. All names are made up from small pools, phone
numbers are in the Bundesnetzagentur ranges reserved for film and TV (as in
tests/conftest.py), dates are random. Built deterministically; ``python -m
eval.stt_ab.utterances`` writes utterances.json for review.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

Lang = Literal["de", "en"]
Kind = Literal["name", "dob", "phone"]

JSON_PATH = Path(__file__).with_name("utterances.json")
SEED = 20261107

# Two standard ElevenLabs voices, checked against the account on 2026-11-07 (HTTP 200);
# the agent speaks with Roger, so neither is the agent's voice.
VOICES = {"sarah": "EXAVITQu4vr4xnSDxMaL", "george": "JBFqnCBsd6RMkjVDRZzb"}

# Fictional numbers (Bundesnetzagentur, reserved for film and TV): prefix, block.
PHONE_BLOCKS = [("030", "23125"), ("040", "66969"), ("069", "90009"), ("089", "99998"),
                ("0221", "4710")]  # fmt: skip

FIRST = [
    "Jürgen", "Hildegard", "Björn", "Käthe", "Günther", "Mareike", "Sören", "Dörte",
    "Lutz", "Annegret", "Torben", "Wiebke", "Malte", "Henrike", "Jörg", "Gesine",
    "Rüdiger", "Swantje", "Hauke", "Ingeborg", "Volker", "Margit", "Detlef", "Heike",
    "Ömer", "Elif", "Dragan", "Katarzyna", "Mehmet", "Lena", "Till", "Frauke", "Uwe",
]  # fmt: skip
LAST = [
    "Weißmüller", "Quellenberg", "Brückner", "Hoffstätter", "Kühnel", "Großkopf",
    "Pfeiffer", "Schößler", "Dreyer", "Öztürk", "Kowalczyk", "Mühlbauer", "Tiedemann",
    "Wüstenhagen", "Nußbaum", "Haßelbach", "Jäckel", "Rößner", "Ehrenträger",
    "Bäumler", "Stöckl", "Fröhlich", "Grünewald", "Vogelsang", "Schütte", "Lindqvist",
    "Krämer", "Göbel", "Häusler", "Brandstätter", "Möller", "Süßmann", "Wendler",
]  # fmt: skip

# --- German number words (with umlauts, for TTS) ---------------------------------------

_UNITS = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
_TEENS = ["zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn",
          "siebzehn", "achtzehn", "neunzehn"]  # fmt: skip
_TENS = {2: "zwanzig", 3: "dreißig", 4: "vierzig", 5: "fünfzig", 6: "sechzig",
         7: "siebzig", 8: "achtzig", 9: "neunzig"}  # fmt: skip
_MONTHS_DE = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
              "September", "Oktober", "November", "Dezember"]  # fmt: skip
_MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]  # fmt: skip


def de_cardinal(n: int) -> str:
    """0..99 as one German word: 'fünfundachtzig'."""
    if n < 10:
        return _UNITS[n]
    if n < 20:
        return _TEENS[n - 10]
    unit = n % 10
    tens = _TENS[n // 10]
    return tens if unit == 0 else f"{'ein' if unit == 1 else _UNITS[unit]}und{tens}"


def de_ordinal(n: int, ending: str) -> str:
    """'fünfzehnte' + ending: 1 -> 'erst', 3 -> 'dritt', 7 -> 'siebt', 8 -> 'acht'."""
    special = {1: "erst", 3: "dritt", 7: "siebt", 8: "acht"}
    stem = special.get(n) or de_cardinal(n) + ("st" if n >= 20 else "t")
    return stem + ending


def de_year(y: int) -> str:
    """1985 -> 'neunzehnhundertfünfundachtzig', 2005 -> 'zweitausendfünf'."""
    if y >= 2000:
        rest = y - 2000
        return "zweitausend" + ("" if rest == 0 else de_cardinal(rest))
    rest = y - 1900
    return "neunzehnhundert" + ("" if rest == 0 else de_cardinal(rest))


# --- English number words ---------------------------------------------------------------

_EN_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_EN_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
             "seventeen", "eighteen", "nineteen"]  # fmt: skip
_EN_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy",
            8: "eighty", 9: "ninety"}  # fmt: skip
_EN_ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
           7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh",
           12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
           16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
           20: "twentieth", 30: "thirtieth"}  # fmt: skip


def en_cardinal(n: int) -> str:
    if n < 10:
        return _EN_UNITS[n]
    if n < 20:
        return _EN_TEENS[n - 10]
    unit = n % 10
    return _EN_TENS[n // 10] + ("" if unit == 0 else f"-{_EN_UNITS[unit]}")


def en_ordinal(n: int) -> str:
    if n in _EN_ORD:
        return _EN_ORD[n]
    return f"{_EN_TENS[n // 10]}-{_EN_ORD[n % 10]}"


def en_year(y: int) -> str:
    """1985 -> 'nineteen eighty-five', 1905 -> 'nineteen oh five', 2005 -> 'two thousand five'."""
    if y >= 2000:
        rest = y - 2000
        return "two thousand" + ("" if rest == 0 else f" {en_cardinal(rest)}")
    rest = y - 1900
    return "nineteen " + (en_cardinal(rest) if rest >= 10 else f"oh {_EN_UNITS[rest]}")


# --- spoken phone numbers ---------------------------------------------------------------


def _digits_de(digits: str, zwo: bool) -> str:
    return " ".join("zwo" if zwo and d == "2" else _UNITS[int(d)] for d in digits)


def _pairs_de(digits: str) -> str:
    """'23125' -> 'dreiundzwanzig, zwölf, fünf': pairs, a leading 0 spoken alone."""
    words = []
    for i in range(0, len(digits), 2):
        pair = digits[i : i + 2]
        if len(pair) == 1 or pair[0] == "0":
            words.append(_digits_de(pair, zwo=False))
        else:
            words.append(de_cardinal(int(pair)))
    return " ".join(words)


def phone_de(prefix: str, block: str, suffix: str, style: int) -> str:
    if style == 0:  # digit by digit, grouped
        return ", ".join(_digits_de(p, zwo=False) for p in (prefix, block, suffix))
    if style == 1:  # "zwo" for 2
        return ", ".join(_digits_de(p, zwo=True) for p in (prefix, block, suffix))
    return f"{_digits_de(prefix, zwo=False)}, {_pairs_de(block)}, {_pairs_de(suffix)}"


def _digits_en(digits: str, oh: bool) -> str:
    out: list[str] = []
    i = 0
    while i < len(digits):
        d = digits[i]
        if (
            i + 1 < len(digits)
            and digits[i + 1] == d
            and not (i + 2 < len(digits) and digits[i + 2] == d)
        ):
            out.append(f"double {('oh' if oh else 'zero') if d == '0' else _EN_UNITS[int(d)]}")
            i += 2
            continue
        out.append(("oh" if oh else "zero") if d == "0" else _EN_UNITS[int(d)])
        i += 1
    return " ".join(out)


def phone_en(prefix: str, block: str, suffix: str, style: int) -> str:
    return ", ".join(_digits_en(p, oh=style == 0) for p in (prefix, block, suffix))


# --- spoken dates of birth --------------------------------------------------------------


def dob_de(d: date, style: int) -> str:
    if style == 0:  # "am fünfzehnten Juni neunzehnhundertfünfundachtzig"
        return f"{de_ordinal(d.day, 'en')} {_MONTHS_DE[d.month - 1]} {de_year(d.year)}"
    if style == 1:  # "der fünfzehnte sechste neunzehnhundertfünfundachtzig"
        return f"{de_ordinal(d.day, 'e')} {de_ordinal(d.month, 'e')} {de_year(d.year)}"
    # two-digit year, only for 19xx: "fünfzehnter Juni fünfundachtzig"
    return f"{de_ordinal(d.day, 'en')} {_MONTHS_DE[d.month - 1]} {de_cardinal(d.year - 1900)}"


def dob_en(d: date, style: int) -> str:
    month = _MONTHS_EN[d.month - 1]
    if style == 0:
        return f"the {en_ordinal(d.day)} of {month}, {en_year(d.year)}"
    return f"{month} {en_ordinal(d.day)}, {en_year(d.year)}"


# --- the list ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    lang: Lang
    text: str


@dataclass(frozen=True)
class Utterance:
    id: str
    group: Literal["de", "en", "switch"]
    kind: Kind
    expected: str  # name as written, ISO date, or digits
    segments: list[Segment] = field(default_factory=list)

    @property
    def lang(self) -> Lang:
        """Language of the segment that carries the item (the last one)."""
        return self.segments[-1].lang

    @property
    def chars(self) -> int:
        return sum(len(s.text) for s in self.segments)


_DE_TEMPLATES: dict[Kind, list[str]] = {
    "name": ["Mein Name ist {x}.", "Hier spricht {x}.", "Ich heiße {x}.",
             "Der Termin wäre für {x}."],
    "dob": ["Ich bin am {x} geboren.", "Mein Geburtsdatum ist der {x}.", "Geboren am {x}."],
    "phone": ["Meine Nummer ist {x}.", "Sie erreichen mich unter {x}.",
              "Rufen Sie mich bitte zurück unter {x}."],
}  # fmt: skip
_EN_TEMPLATES: dict[Kind, list[str]] = {
    "name": ["My name is {x}.", "This is {x} speaking.", "The appointment is for {x}."],
    "dob": ["I was born on {x}.", "My date of birth is {x}.", "Date of birth, {x}."],
    "phone": ["My number is {x}.", "You can call me back on {x}.", "It's {x}."],
}  # fmt: skip
_SWITCH_OPENERS = [
    "Guten Tag, ich hätte gern einen Termin.",
    "Hallo, ich rufe wegen eines Termins an.",
    "Guten Morgen, ich brauche einen Rückruf.",
    "Ja, hallo, ich wollte einen Termin machen.",
    "Grüß Gott, ich hätte eine Frage zu meinem Termin.",
]
_SWITCH_BRIDGES = [
    "Sorry, my German is not so good. ",
    "Can we continue in English? ",
    "I'm sorry, I'll switch to English. ",
    "Is English okay? ",
]


def _items(
    rng: random.Random, kind: Kind, n: int, lang: Lang, offset: int
) -> list[tuple[str, str]]:
    """(spoken, expected) pairs.

    Range and way of saying it follow the position across the list (``offset + i``), so
    groups of one item, such as each switch call, still vary.
    """
    out: list[tuple[str, str]] = []
    for i in range(n):
        j = offset + i
        if kind == "name":
            name = f"{FIRST[(offset + i) % len(FIRST)]} {LAST[(offset + 7 * i) % len(LAST)]}"
            out.append((name, name))
        elif kind == "dob":
            d = date(rng.randint(1938, 2006), rng.randint(1, 12), rng.randint(1, 28))
            style = j % 3 if lang == "de" else j % 2
            if lang == "de" and style == 2 and d.year >= 2000:
                style = 0
            spoken = dob_de(d, style) if lang == "de" else dob_en(d, style)
            out.append((spoken, d.isoformat()))
        else:
            prefix, block = PHONE_BLOCKS[j % len(PHONE_BLOCKS)]
            suffix = f"{rng.randint(0, 999):03d}"
            style = j % 3 if lang == "de" else j % 2
            spoken = (
                phone_de(prefix, block, suffix, style)
                if lang == "de"
                else phone_en(prefix, block, suffix, style)
            )
            out.append((spoken, prefix + block + suffix))
    return out


_GROUPS: list[tuple[Literal["de", "en"], dict[Kind, int], dict[Kind, list[str]], int]] = [
    ("de", {"name": 33, "dob": 33, "phone": 34}, _DE_TEMPLATES, 0),
    ("en", {"name": 13, "dob": 13, "phone": 14}, _EN_TEMPLATES, 11),
]


def build() -> list[Utterance]:
    rng = random.Random(SEED)
    out: list[Utterance] = []
    for lang, counts, templates, offset in _GROUPS:
        for kind, n in counts.items():
            for i, (spoken, expected) in enumerate(_items(rng, kind, n, lang, offset)):
                text = templates[kind][i % len(templates[kind])].format(x=spoken)
                uid = f"{lang}-{kind}-{i:02d}"
                out.append(Utterance(uid, lang, kind, expected, [Segment(lang, text)]))
    kinds: list[Kind] = ["name", "dob", "phone"]
    for i in range(20):
        kind = kinds[i % 3]
        spoken, expected = _items(rng, kind, 1, "en", 23 + i)[0]
        tmpl = _EN_TEMPLATES[kind][i % len(_EN_TEMPLATES[kind])]
        english = _SWITCH_BRIDGES[i % len(_SWITCH_BRIDGES)] + tmpl.format(x=spoken)
        opener = _SWITCH_OPENERS[i % len(_SWITCH_OPENERS)]
        segments = [Segment("de", opener), Segment("en", english)]
        out.append(Utterance(f"switch-{kind}-{i:02d}", "switch", kind, expected, segments))
    return out


def load() -> list[Utterance]:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return [
        Utterance(u["id"], u["group"], u["kind"], u["expected"],
                  [Segment(s["lang"], s["text"]) for s in u["segments"]])
        for u in data["utterances"]
    ]  # fmt: skip


def main() -> int:
    utterances = build()
    payload = {
        "seed": SEED,
        "voices": VOICES,
        "utterances": [asdict(u) for u in utterances],
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    chars = sum(u.chars for u in utterances)
    print(
        f"{len(utterances)} utterances, {chars} characters ({chars * len(VOICES)} for "
        f"{len(VOICES)} voices) -> {JSON_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
