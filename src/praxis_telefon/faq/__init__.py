"""Practice FAQ (M3 Part C): synthetic entries, one per topic, in German and English.

The LLM picks the topic (``practice_info(topic)`` in agent/receptionist.py); code returns
the entry's text in the call's language. Local keyword retrieval (BM25) was tried first
and missed the bar (docs/experiments/faq-retrieval.md). Opening hours are filled from
``db.seed.OPENING_HOURS`` so they cannot drift from the bookable slots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from praxis_telefon.db.seed import OPENING_HOURS

FAQ_PATH = Path(__file__).with_name("faq.json")

_DAYS = {
    "de": ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"),
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
}


@dataclass(frozen=True)
class Entry:
    id: str
    de: str
    en: str


def _join(words: list[str], last: str) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + f" {last} " + words[-1]


def _range(lang: str, start: int, end: int) -> str:
    if lang == "de":
        return f"von {start} bis {end} Uhr"
    pm = end > 12
    s, e = (start - 12 if start > 12 else start), (end - 12 if end > 12 else end)
    return f"from {s} to {e}{' pm' if pm else ''}"


def opening_hours_text(lang: str) -> str:
    """Spoken opening hours from the seed, days with the same hours grouped."""
    groups: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for day, ranges in sorted(OPENING_HOURS.items()):
        key = tuple((a.hour, b.hour) for a, b in ranges)
        groups.setdefault(key, []).append(day)
    parts = []
    for key, days in groups.items():
        names = [_DAYS[lang][d] for d in days]
        times = (" und " if lang == "de" else " and ").join(_range(lang, a, b) for a, b in key)
        parts.append(f"{_join(names, 'und' if lang == 'de' else 'and')} {times}")
    text = ", ".join(parts)
    return f"{text} geöffnet." if lang == "de" else f"{text}."


def _load() -> tuple[Entry, ...]:
    data = json.loads(FAQ_PATH.read_text(encoding="utf-8"))
    return tuple(Entry(e["id"], e["de"], e["en"]) for e in data["entries"])


ENTRIES = _load()
_BY_ID = {e.id: e for e in ENTRIES}
TOPICS = tuple(e.id for e in ENTRIES)


def answer_text(entry_id: str, lang: str) -> str:
    e = _BY_ID[entry_id]
    text = e.de if lang == "de" else e.en
    return text.replace("{hours}", opening_hours_text(lang))
