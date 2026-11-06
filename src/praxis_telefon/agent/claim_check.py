"""Detects agent replies that claim a booking change no tool has confirmed.

Bug (voice test 2026-11-06): after the caller's "Ja", the agent replied as if the
appointment was moved/cancelled, but move_booking / cancel_booking never ran.

This is a keyword heuristic, used by the live tests and logged at runtime as
``{"event":"unbacked_claim","action":...}`` (no transcript text). It cannot stop the
sentence from being spoken.

Not a claim:
- a question ("Soll ich den Termin verschieben?"),
- a read-back that ends with a confirmation question
  ("Dienstag um 8 Uhr bei Dr. Weber, gebucht auf Max Mustermann. Stimmt das so?").
Known gap: a real claim followed by "Stimmt das so?" is not flagged.
"""

from __future__ import annotations

import re

CLAIM_PATTERNS: dict[str, re.Pattern[str]] = {
    "move_booking": re.compile(r"\b(verschoben|verlegt)\b", re.I),
    "cancel_booking": re.compile(r"\b(abgesagt|storniert)\b", re.I),
    "book": re.compile(r"\b(gebucht|eingetragen|reserviert)\b", re.I),
}
# Tools whose success makes the claim legitimate. find_booking backs "Ihr Termin ist
# gebucht am ..." when reading out an existing appointment.
BACKED_BY: dict[str, frozenset[str]] = {
    "move_booking": frozenset({"move_booking"}),
    "cancel_booking": frozenset({"cancel_booking"}),
    "book": frozenset({"book", "find_booking"}),
}

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ\"„])")
# "Dr. Weber", "z. B.", and day ordinals like "am 1. Februar" do not end a sentence.
_NO_BREAK_AFTER = re.compile(r"(?:\b(?:Dr|Nr|ca|bzw|z|B)\.|(?<!\d)\d{1,2}\.)$")
_CONFIRM_QUESTION = re.compile(
    r"\b(stimmt das|ist das (so )?(richtig|korrekt)|passt das|soll ich|einverstanden|"
    r"richtig so|darf ich)\b.*\?$",
    re.I,
)


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for part in _SENTENCE_BREAK.split(text.strip()):
        if sentences and _NO_BREAK_AFTER.search(sentences[-1]):
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return [s for s in sentences if s]


def claimed_actions(text: str) -> set[str]:
    """Actions the text states as done (questions and read-backs excluded)."""
    sentences = split_sentences(text)
    if sentences and _CONFIRM_QUESTION.search(sentences[-1]):
        return set()
    statements = [s for s in sentences if not s.rstrip().endswith("?")]
    return {a for a, p in CLAIM_PATTERNS.items() if any(p.search(s) for s in statements)}


class ClaimTracker:
    """Tracks successful tools; reports claims made before the matching tool succeeded."""

    def __init__(self) -> None:
        self._succeeded: set[str] = set()

    def tool_result(self, name: str, *, ok: bool) -> None:
        if ok:
            self._succeeded.add(name)

    def unbacked(self, text: str) -> list[str]:
        return sorted(a for a in claimed_actions(text) if not BACKED_BY[a] & self._succeeded)
