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
- An explicit request ("Can we speak English?", "Sprechen Sie Deutsch?") switches at once;
  so does "English" / "Englisch" / "Inglisch" (with "please" / "bitte") as the whole
  utterance; "German" / "Deutsch" alone means German, and on the en stream so does
  "Deutsch" or "Deutsche" anywhere in the final.

Like the safety state, the language is replayed from the recorded finals every time, so
a discarded preemptive LLM run cannot change it. A final keeps only labels, never the
caller's words.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

from praxis_telefon.agent.speech_filters import detect_language
from praxis_telefon.safety.phrases import Language
from praxis_telefon.text import normalize

FINALS_TO_SWITCH = 2

_EN = r"(?:english|englisch)"
_DE = r"(?:german|deutsch)"


def _request(lang: str, other: str) -> re.Pattern[str]:
    # On normalize()d text: lower case, umlauts folded, punctuation removed.
    return re.compile(
        rf" (?:can|could|may|shall) we (?:(?:please|now|again) )*(?:speak|talk|continue) "
        rf"(?:in )?{lang} "
        rf"| do you (?:also )?speak {lang} "
        rf"| (?:speak|talk|continue) (?:in )?{lang} please "
        rf"| koennen wir (?:(?:bitte|wieder|jetzt) )*(?:auf |in )?{lang} "
        rf"(?:sprechen|reden|weitermachen) "
        rf"| sprechen sie (?:auch )?{lang} "
        rf"| (?:bitte )?auf {lang} bitte "
        rf"| i (?:don t|do not|can t|cannot) speak (?:any |much )?{other} "
        rf"| ich spreche (?:kein|nicht|kaum) (?:gut )?{other} "
    )


_REQUESTS: tuple[tuple[Language, re.Pattern[str]], ...] = (
    ("en", _request(_EN, _DE)),
    ("de", _request(_DE, _EN)),
)


# Only as the whole utterance (user, 2026-11-07): the bilingual "say that again" message
# tells English callers to "just say 'English'"; "Inglisch" is how de may write it.
_ENGLISH_WORD = re.compile(r"\s*(?:english|englisch|inglisch)(?: please| bitte)?\s*")
# On the en stream only (own recordings 2026-11-07: en wrote "Können wir wieder Deutsch
# sprechen?" as "Deutsche expression." and "Sprechen Sie Deutsch?" as nothing): these
# exact words mean back to German. Exact words, so "Deutschmann" does not switch.
# "German" only alone or in a request ("I have German health insurance" is no request).
# "doish": own recording 11, en's spelling of "Deutsch.".
_GERMAN_WORDS = frozenset({"deutsch", "deutsche", "doish"})
_GERMAN_WORD = re.compile(r"\s*(?:german|deutsch)(?: please| bitte)?\s*")


def language_request(text: str) -> Language | None:
    """The language the caller explicitly asks for, if any."""
    if _ENGLISH_WORD.fullmatch(normalize(text)):
        return "en"
    if _GERMAN_WORD.fullmatch(normalize(text)):
        return "de"
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
        tag = _tag(stt_language)
        request = language_request(text)
        if request is None and tag == "en" and _GERMAN_WORDS & set(normalize(text).split()):
            request = "de"
        return cls(tag, detect_language(text), request)


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
    first_language: Language | None = None  # of the first final that showed one


def step(state: LanguageState, final: CallerFinal) -> LanguageState:
    shown = final.request if final.request is not None else vote(final)
    if shown is None:
        return state
    state = replace(
        state,
        german_heard=state.german_heard or shown == "de",
        english_heard=state.english_heard or shown == "en",
        first_language=state.first_language or shown,
    )
    if final.request is not None:
        return replace(state, language=final.request, run=0)
    if shown == state.language:
        return replace(state, run=0)
    if shown == "en" and not state.german_heard:  # nothing German yet: one final is enough
        return replace(state, language="en", run=0)
    run = state.run + 1
    if run >= FINALS_TO_SWITCH:
        return replace(state, language=shown, run=0)
    return replace(state, run=run)


# Recorded by Receptionist when an empty final on the en stream, in a call that started
# German, switches back to de (the caller most likely spoke German; user, 2026-11-07).
SWITCH_BACK = CallerFinal(stt="en", detected=None, request="de")


def decide(finals: Iterable[CallerFinal]) -> LanguageState:
    state = LanguageState()
    for final in finals:
        state = step(state, final)
    return state


def from_texts(texts: Iterable[str]) -> LanguageState:
    """Typed caller lines (text mode, replayed live corpus): one final each, no tag."""
    return decide(CallerFinal.from_text(t) for t in texts)
