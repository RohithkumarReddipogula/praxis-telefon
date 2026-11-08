"""The agent never tells the caller which word to answer with ("Please answer yes or no").

Bug, cloud browser test 2026-11-08 and the booking harness run: after the consent check
refused a booking, the agent said "Bitte sagen Sie Ja oder Nein", "Please answer yes or
no", "I need a clear yes before I book" or "ich brauche noch Ihr ausdrückliches Ja". That
sounds robotic. The sentence filter blocks such sentences (reason ``demand``, German and
English); the retry note and the fixed fallback ask naturally ("Soll ich den Termin so
eintragen?" / "Shall I book it like that?").
"""

from __future__ import annotations

import sqlite3

import pytest

from praxis_telefon.agent.claim_gate import FilterContext, check_sentence
from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.safety.messages import fallback_for, retry_note_for
from tests.conftest import NOW
from tests.test_claim_gate import ScriptedLLM, _install, _reply, _words
from tests.test_speech_filters import _system_notes

CTX = FilterContext(caller_texts=("Guten Tag, ich möchte einen Termin.",))

DEMANDS = [
    # German
    "Bitte sagen Sie Ja oder Nein.",
    "Antworten Sie bitte mit Ja oder Nein.",
    "Sagen Sie einfach Ja, dann buche ich.",
    "Ich brauche noch Ihr ausdrückliches Ja.",
    "Entschuldigung, ich brauche noch Ihr eindeutiges Ja.",
    "Dafür brauche ich ein klares Ja von Ihnen.",
    # English (from the booking harness run 2026-11-08)
    "Please answer yes or no.",
    "I'm sorry, I need a clear yes before I book.",
    "I'm sorry, I still need a clear answer.",
    "Our system still needs a clear confirmation.",
    "Please say yes if that is correct.",
    "Just reply with yes or no.",
]

NATURAL = [
    "Soll ich den Termin so eintragen?",
    "Stimmt das so?",
    "Shall I book it like that?",
    "Is that correct?",
    "Ich brauche noch Ihr Geburtsdatum.",
    "Ich brauche noch Ihren Namen.",
    "I need your date of birth.",
    "Dafür sage ich Ihnen gern Bescheid.",
    "Sagen Sie mir bitte Ihren Namen.",
    "Please tell me your name.",
    "Ja, das mache ich gern.",
]


@pytest.mark.parametrize("sentence", DEMANDS)
def test_telling_the_caller_which_word_to_say_is_blocked(sentence: str) -> None:
    assert "demand" in check_sentence(sentence, CTX)


@pytest.mark.parametrize("sentence", NATURAL)
def test_natural_questions_and_requests_are_kept(sentence: str) -> None:
    assert "demand" not in check_sentence(sentence, CTX)


@pytest.mark.parametrize("language", ["de", "en"])
def test_retry_note_offers_a_natural_alternative(language: str) -> None:
    note = retry_note_for(["demand"], language)  # type: ignore[arg-type]
    expected = "Ist das so richtig?" if language == "de" else "Is that correct?"
    assert expected in note  # nothing pending: the neutral question (see test_pending_action)


@pytest.mark.parametrize("language", ["de", "en"])
def test_fallback_is_a_natural_question_not_the_generic_one(language: str) -> None:
    text = fallback_for(["demand"], language)  # type: ignore[arg-type]
    assert text.endswith("?")
    assert text != fallback_for(["other"], language)  # type: ignore[arg-type]
    assert "ja oder nein" not in text.lower() and "yes or no" not in text.lower()


# --- in a reply -------------------------------------------------------------------------


@pytest.fixture
def agent(conn: sqlite3.Connection) -> Receptionist:
    return Receptionist(ToolService(conn, lambda: NOW))


def test_demand_after_a_question_is_dropped_and_the_question_is_spoken(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = "Soll ich den Termin um 8 Uhr bei Dr. Weber so eintragen? "
    fake = ScriptedLLM(_words(question + "Bitte sagen Sie Ja oder Nein."))
    _install(monkeypatch, fake)
    text, _ = _reply(agent, "Hmm, ich weiß nicht.")
    assert "Ja oder Nein" not in text
    assert text.strip() == question.strip()
    assert len(fake.contexts) == 1  # something was spoken: no retry needed


def test_demand_alone_is_retried_with_a_natural_question(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    natural = "Ist das so richtig? "
    fake = ScriptedLLM(_words("Bitte sagen Sie Ja oder Nein. "), _words(natural))
    _install(monkeypatch, fake)
    text, _ = _reply(agent, "Hmm, ich weiß nicht.")
    assert text.strip() == natural.strip()
    assert any("Ist das so richtig?" in n for n in _system_notes(fake.contexts[1]))


def test_english_demand_is_retried_in_english_calls(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    natural = "Is that correct? "
    fake = ScriptedLLM(_words("I'm sorry, I need a clear yes before I book. "), _words(natural))
    _install(monkeypatch, fake)
    text, _ = _reply(agent, "Hello, I would like an appointment, please. Hmm, I am not sure.")
    assert text.strip() == natural.strip()
    assert any("Is that correct?" in n for n in _system_notes(fake.contexts[1]))


def test_demand_twice_gets_the_fixed_natural_question(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM(
        _words("Bitte sagen Sie Ja oder Nein. "), _words("Ich brauche noch Ihr klares Ja. ")
    )
    _install(monkeypatch, fake)
    text, _ = _reply(agent, "Hmm, ich weiß nicht.")
    assert text.strip() == fallback_for(["demand"], "de")
