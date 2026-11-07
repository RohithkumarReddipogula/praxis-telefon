"""The call's language, decided by code from the caller's finals, not by the LLM.

Voice test 2026-11-07 (STT_LANGUAGE=multi): 2 of the user's German finals were tagged
``en``; after the first, the LLM answered in English. So one final must not switch the
call. Rule (user, 2026-11-07):

- A final counts as English only if Deepgram's tag and ``detect_language`` both say
  English; as German if either says German. Typed input (no tag): the detector alone.
- The language changes only after 2 consecutive finals in the other language, in both
  directions. Before the caller said anything German, one English final is enough (an
  English caller's first line gets an English answer).
- Finals that show no language ("Ja.", digits, a name) neither count nor break a run.
- An explicit request ("Can we speak English?", "Sprechen Sie Deutsch?") switches at once.

Like the safety state, the language is replayed from the recorded finals every time, so
a discarded preemptive LLM run cannot change it. A final keeps only labels, never the
caller's words.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from praxis_telefon.agent.speech_filters import detect_language
from praxis_telefon.safety.phrases import Language
from praxis_telefon.text import normalize

FINALS_TO_SWITCH = 2

_EN = r"(?:english|englisch)"
_DE = r"(?:german|deutsch)"


def _request(lang: str, other: str) -> re.Pattern[str]:
    # On normalize()d text: lower case, umlauts folded, punctuation removed.
    return re.compile(
        rf" (?:can|could|may|shall) we (?:please )?(?:speak|talk|continue) (?:in )?{lang} "
        rf"| do you (?:also )?speak {lang} "
        rf"| (?:speak|talk|continue) (?:in )?{lang} please "
        rf"| koennen wir (?:bitte )?(?:auf |in )?{lang} (?:sprechen|reden|weitermachen) "
        rf"| sprechen sie (?:auch )?{lang} "
        rf"| (?:bitte )?auf {lang} bitte "
        rf"| i (?:don t|do not|can t|cannot) speak (?:any |much )?{other} "
        rf"| ich spreche (?:kein|nicht|kaum) (?:gut )?{other} "
    )


_REQUESTS: tuple[tuple[Language, re.Pattern[str]], ...] = (
    ("en", _request(_EN, _DE)),
    ("de", _request(_DE, _EN)),
)


def language_request(text: str) -> Language | None:
    """The language the caller explicitly asks for, if any."""
    n = f" {normalize(text)} "
    for language, pattern in _REQUESTS:
        if pattern.search(n):
            return language
    return None


def _tag(stt_language: str | None) -> Language | str | None:
    if not stt_language:
        return None
    tag = stt_language.lower()
    if tag.startswith("en"):
        return "en"
    if tag.startswith("de"):
        return "de"
    return tag


@dataclass(frozen=True)
class CallerFinal:
    """One final transcript, as labels only."""

    stt: Language | str | None  # Deepgram's tag; None for typed input
    detected: Language | None  # detect_language(text)
    request: Language | None = None  # language_request(text)

    @classmethod
    def from_text(cls, text: str, stt_language: str | None = None) -> CallerFinal:
        return cls(_tag(stt_language), detect_language(text), language_request(text))


def vote(final: CallerFinal) -> Language | None:
    """The language this final counts for; None if it shows none."""
    if final.stt is None:
        return final.detected
    if final.stt == "en" and final.detected == "en":
        return "en"
    if final.stt == "de" or final.detected == "de":
        return "de"
    return None


@dataclass(frozen=True)
class LanguageState:
    language: Language = "de"
    run: int = 0  # consecutive finals in the other language
    german_heard: bool = False
    english_heard: bool = False  # any English final or request: the LLM gets a note


def step(state: LanguageState, final: CallerFinal) -> LanguageState:
    if final.request is not None:
        return LanguageState(
            final.request,
            0,
            state.german_heard or final.request == "de",
            state.english_heard or final.request == "en",
        )
    v = vote(final)
    if v is None:
        return state
    german_heard = state.german_heard or v == "de"
    english_heard = state.english_heard or v == "en"
    if v == state.language:
        return LanguageState(v, 0, german_heard, english_heard)
    if v == "en" and not state.german_heard:
        return LanguageState("en", 0, german_heard, english_heard)
    run = state.run + 1
    language = v if run >= FINALS_TO_SWITCH else state.language
    return LanguageState(language, 0 if language == v else run, german_heard, english_heard)


def decide(finals: Iterable[CallerFinal]) -> LanguageState:
    state = LanguageState()
    for final in finals:
        state = step(state, final)
    return state


def from_texts(texts: Iterable[str]) -> LanguageState:
    """Typed caller lines (text mode, replayed live corpus): one final each, no tag."""
    return decide(CallerFinal.from_text(t) for t in texts)
