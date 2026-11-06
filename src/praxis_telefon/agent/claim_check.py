"""Detects agent sentences that claim an action no tool has confirmed.

Bugs:
- voice test 2026-11-06: after the caller's "Ja", the agent said the appointment was
  moved/cancelled, but move_booking / cancel_booking never ran.
- live test human_request[2]: "Ich habe Ihre Nachricht für das Praxisteam aufgenommen",
  but take_message was never called.

Used in two places: ``claim_gate`` blocks such sentences before they are spoken, and
``main.attach_claim_check`` logs any that still get through
(``{"event":"unbacked_claim","action":...}``, no transcript text).

Not a claim:
- a question ("Soll ich den Termin verschieben?"),
- a read-back that ends with a confirmation question
  ("Dienstag um 8 Uhr bei Dr. Weber, gebucht auf Max Mustermann. Stimmt das so?"),
- a conditional ("Wenn Sie möchten, ...") or a negation just before the claim word
  ("noch nicht gespeichert", "leider schon gebucht", "fully booked").
Known gaps: a real claim followed by "Stimmt das so?" is not flagged; promises like
"Das Team ruft Sie zurück" are not treated as a saved message.
"""

from __future__ import annotations

import re

_I = re.IGNORECASE
CLAIM_PATTERNS: dict[str, re.Pattern[str]] = {
    "move_booking": re.compile(r"\b(verschoben|verlegt|rescheduled|moved)\b", _I),
    "cancel_booking": re.compile(r"\b(abgesagt|storniert|cancell?ed)\b", _I),
    "book": re.compile(r"\b(gebucht|eingetragen|reserviert|booked|scheduled)\b", _I),
    "take_message": re.compile(
        r"\b(nachricht|rückrufbitte|rueckrufbitte|notiz|message|note|callback request)\b"
        r".{0,60}?"
        r"\b(aufgenommen|gespeichert|notiert|weitergeleitet|weitergegeben|hinterlegt|"
        r"saved|taken|recorded|passed on|forwarded|noted|logged)\b",
        _I,
    ),
}
# Tools whose success makes the claim legitimate. Only the matching tool: find_booking
# does not back "gebucht", or "Ich habe Ihren neuen Termin gebucht" would pass after a
# lookup without book. Existing appointments are read out without claim words
# ("Ihr Termin ist am Dienstag um 9 Uhr"), as the prompt says.
BACKED_BY: dict[str, frozenset[str]] = {
    "move_booking": frozenset({"move_booking"}),
    "cancel_booking": frozenset({"cancel_booking"}),
    "book": frozenset({"book"}),
    "take_message": frozenset({"take_message"}),
}

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ\"„])")
# "Dr. Weber", "z. B.", and day ordinals like "am 1. Februar" do not end a sentence.
_NO_BREAK_AFTER = re.compile(r"(?:\b(?:Dr|Nr|ca|bzw|z|B)\.|(?<!\d)\d{1,2}\.)$")
_CONFIRM_QUESTION = re.compile(
    r"\b(stimmt das|ist das (so )?(richtig|korrekt)|passt das|soll ich|einverstanden|"
    r"richtig so|darf ich|is that (right|correct)|shall i|should i)\b.*\?$",
    _I,
)
_CONDITIONAL = re.compile(r"\b(wenn|falls|sobald|if|once)\b", _I)
_NEGATION_BEFORE = re.compile(r"\b(nicht|kein\w*|nie|leider|not|never|unfortunately)\b|n't\b", _I)
_NEGATION_WINDOW = 40  # characters before the claim word
_ALWAYS_FINE = re.compile(r"\b(ausgebucht|fully booked)\b", _I)


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for part in _SENTENCE_BREAK.split(text.strip()):
        if sentences and _NO_BREAK_AFTER.search(sentences[-1]):
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return [s for s in sentences if s]


def take_sentences(buffer: str) -> tuple[list[str], str]:
    """Complete sentences at the start of a streamed buffer, and the rest.

    A sentence is complete once the first character of the next one has arrived
    (the same point at which LiveKit's sentence tokenizer releases it to the TTS).
    Each returned sentence keeps its trailing whitespace.
    """
    done: list[str] = []
    start = 0
    for m in _SENTENCE_BREAK.finditer(buffer):
        if _NO_BREAK_AFTER.search(buffer[start : m.start()]):
            continue
        done.append(buffer[start : m.end()])
        start = m.end()
    return done, buffer[start:]


def is_confirmation_question(sentence: str) -> bool:
    return bool(_CONFIRM_QUESTION.search(sentence.strip()))


def sentence_claims(sentence: str) -> set[str]:
    """Actions one sentence states as done."""
    s = sentence.strip()
    if not s or s.endswith("?") or _CONDITIONAL.search(s) or _ALWAYS_FINE.search(s):
        return set()
    found: set[str] = set()
    for action, pattern in CLAIM_PATTERNS.items():
        for m in pattern.finditer(s):
            # For take_message the verb is the claim word; look before it.
            word_start = m.start(m.lastindex or 0)
            window = s[max(0, word_start - _NEGATION_WINDOW) : word_start]
            if not _NEGATION_BEFORE.search(window):
                found.add(action)
                break
    return found


def claimed_actions(text: str) -> set[str]:
    """Actions the text states as done (questions and read-backs excluded)."""
    sentences = split_sentences(text)
    if sentences and is_confirmation_question(sentences[-1]):
        return set()
    return set().union(*(sentence_claims(s) for s in sentences)) if sentences else set()


class ClaimTracker:
    """Tracks successful tools; reports claims made before the matching tool succeeded."""

    def __init__(self, succeeded: set[str] | None = None) -> None:
        self._succeeded: set[str] = set(succeeded or ())

    def tool_result(self, name: str, *, ok: bool) -> None:
        if ok:
            self._succeeded.add(name)

    def is_backed(self, action: str) -> bool:
        return bool(BACKED_BY[action] & self._succeeded)

    def unbacked(self, text: str) -> list[str]:
        return sorted(a for a in claimed_actions(text) if not self.is_backed(a))

    def unbacked_sentence(self, sentence: str) -> list[str]:
        return sorted(a for a in sentence_claims(sentence) if not self.is_backed(a))
