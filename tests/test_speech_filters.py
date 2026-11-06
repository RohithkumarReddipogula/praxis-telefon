"""Sentence filters besides claims, in Receptionist.llm_node (offline, scripted LLM).

Regressions from the live run on 2026-11-06 (test_dates_live):
- naechsten_dienstag: "This is a new appointment request, so I'll search for slots for
  next Tuesday." was spoken (the agent's own reasoning, in English).
- friday_tomorrow_closed: "Möchten Sie einen neuen Termin für morgen, oder geht es um
  einen bestehenden Termin?" although the caller named a day.
- friday_tomorrow_closed (earlier run): "Morgen ist Samstag" before find_slots.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime

import pytest
from livekit.agents import llm

from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.db.schema import PRACTICE_TZ
from tests.conftest import NOW
from tests.test_claim_gate import ScriptedLLM, _install, _reply, _words

FRIDAY_10 = datetime(2026, 11, 13, 10, 0, tzinfo=PRACTICE_TZ)


@pytest.fixture
def friday_agent(conn: sqlite3.Connection) -> Receptionist:
    return Receptionist(ToolService(conn, lambda: FRIDAY_10))


@pytest.fixture
def agent(conn: sqlite3.Connection) -> Receptionist:
    return Receptionist(ToolService(conn, lambda: NOW))


def _ctx_with_tool_output(agent: Receptionist, tool: str, output: str) -> llm.ChatContext:
    ctx = agent.chat_ctx.copy()
    ctx.add_message(role="user", content="Haben Sie morgen einen Termin?")
    ctx.items.append(llm.FunctionCall(call_id="c1", name=tool, arguments="{}"))
    ctx.items.append(llm.FunctionCallOutput(call_id="c1", name=tool, output=output, is_error=False))
    return ctx


# --- A: internal reasoning / meta talk is never spoken ---------------------------------


def test_spoken_reasoning_before_tool_call_is_dropped(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM(
        [
            *_words(
                "This is a new appointment request, so I'll search for slots for next Tuesday. "
            ),
            ("tool", "find_slots"),
        ]
    )
    _install(monkeypatch, fake)
    text, calls = _reply(agent, "Hallo, Max Mustermann. Haben Sie nächsten Dienstag etwas frei?")
    assert text == ""
    assert calls == ["find_slots"]
    assert len(fake.contexts) == 1  # the tool call follows: no retry needed


@pytest.mark.parametrize(
    "meta",
    [
        "Ich rufe jetzt find_slots auf.",
        "Der Anrufer möchte einen neuen Termin.",
        "Hinweis: Das Datum wurde vom Werkzeug berechnet.",
        "I will use the tool to look this up.",
        "The caller wants an appointment.",
    ],
)
def test_meta_sentences_are_dropped(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch, meta: str
) -> None:
    fake = ScriptedLLM([*_words(f"Einen Moment, ich schaue nach. {meta} "), ("tool", "find_slots")])
    _install(monkeypatch, fake)
    text, calls = _reply(agent, "Haben Sie morgen etwas frei?")
    assert text == "Einen Moment, ich schaue nach. "
    assert calls == ["find_slots"]


@pytest.mark.parametrize(
    "narration",
    [
        # Live run arm A (2026-11-06), explicit_date-0, before find_slots; it was spoken.
        "Das ist ein Terminwunsch für einen neuen Termin.",
        "Das ist eine neue Terminanfrage, also suche ich freie Termine.",
        "Es handelt sich um einen neuen Terminwunsch.",
        "Dies ist eine Anfrage für einen neuen Termin.",
        "Der Patient möchte einen Termin, also suche ich direkt.",
    ],
)
def test_german_request_narration_is_dropped(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch, narration: str
) -> None:
    # German narration passes the "English in a German call" check, so it needs its own.
    fake = ScriptedLLM([*_words(f"{narration} "), ("tool", "find_slots")])
    _install(monkeypatch, fake)
    text, calls = _reply(agent, "Geht es am zwölften November?")
    assert text == ""
    assert calls == ["find_slots"]


@pytest.mark.parametrize(
    "sentence",
    [
        "Das ist ein Termin am Dienstag um neun Uhr bei Dr. Weber.",
        "Das ist kein Problem, ich suche Ihnen gern einen Termin.",
        "Ihren Terminwunsch für Dienstag habe ich verstanden, einen Moment bitte.",
        "Sie möchten also einen Termin für übermorgen.",
        "Ich schaue also gleich nach.",
    ],
)
def test_sentences_like_german_narration_are_kept(sentence: str) -> None:
    from praxis_telefon.agent.speech_filters import meta_reasons

    assert meta_reasons(sentence, "de") == []


def test_english_reply_to_an_english_caller_is_spoken(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = "Sure, let me check that for you. "
    fake = ScriptedLLM([*_words(reply), ("tool", "find_slots")])
    _install(monkeypatch, fake)
    text, _ = _reply(agent, "Hello, this is Max Mustermann. Do you have anything next Thursday?")
    assert text == reply


# --- D: no "neuer oder bestehender Termin?" when a day was named, and at most once ------


def test_new_or_existing_question_after_a_wanted_day_is_dropped(
    friday_agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM(
        _words(
            "Guten Tag Herr Mustermann. Möchten Sie einen neuen Termin für morgen, oder geht "
            "es um einen bestehenden Termin?"
        ),
        [("tool", "find_slots")],
    )
    _install(monkeypatch, fake)
    text, calls = _reply(
        friday_agent,
        "Guten Tag, Max Mustermann, geboren am ersten Februar neunzehnhundertneunundsiebzig. "
        "Haben Sie morgen einen Termin?",
    )
    assert text == "Guten Tag Herr Mustermann. "
    assert calls == ["find_slots"]  # from the one retry
    assert len(fake.contexts) == 2


def test_new_or_existing_question_is_asked_at_most_once(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = agent.chat_ctx.copy()
    ctx.add_message(role="user", content="Guten Tag, ich rufe wegen eines Termins an.")
    ctx.add_message(
        role="assistant",
        content="Möchten Sie einen neuen Termin, oder geht es um einen bestehenden?",
    )
    fake = ScriptedLLM(
        _words("Geht es um einen neuen oder einen bestehenden Termin?"),
        _words("Für welchen Tag hätten Sie gern einen Termin?"),
    )
    _install(monkeypatch, fake)
    text, _ = _reply(agent, "Ja.", ctx=ctx)
    assert text == "Für welchen Tag hätten Sie gern einen Termin?"


def test_new_or_existing_question_without_any_hint_is_allowed(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = "Möchten Sie einen neuen Termin, oder geht es um einen bestehenden?"
    _install(monkeypatch, ScriptedLLM(_words(question)))
    text, _ = _reply(
        agent,
        "Guten Tag, hier ist Erika Mustermann, geboren am fünfzehnten Juni "
        "neunzehnhundertfünfundachtzig. Ich rufe wegen eines Termins an.",
    )
    assert text == question  # the date of birth is not a wanted day


# --- E: no weekday or date the agent worked out itself --------------------------------


def test_self_computed_weekday_before_find_slots_is_dropped(
    friday_agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM([*_words("Morgen ist Samstag. "), ("tool", "find_slots")])
    _install(monkeypatch, fake)
    text, calls = _reply(friday_agent, "Haben Sie morgen einen Termin?")
    assert text == ""
    assert calls == ["find_slots"]


def test_weekday_from_the_tool_result_is_spoken(
    friday_agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = "Am Samstag, 14. November, ist die Praxis geschlossen. Am Montag geht es um 8 Uhr."
    _install(monkeypatch, ScriptedLLM(_words(reply)))
    out = "Samstag, 14. November: Praxis geschlossen. Stattdessen gesucht: Montag, 16. November."
    ctx = _ctx_with_tool_output(friday_agent, "find_slots", out)
    text, _ = _reply(friday_agent, "Okay.", ctx=ctx)
    assert text == reply


def test_weekday_named_by_the_caller_in_another_language_is_spoken(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = "Gern, ich schaue für Donnerstag nach. "
    _install(monkeypatch, ScriptedLLM([*_words(reply), ("tool", "find_slots")]))
    text, _ = _reply(agent, "Do you have anything next Thursday?")
    assert text == reply


def test_todays_weekday_is_known(
    friday_agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = "Heute ist Freitag, da habe ich leider nichts mehr frei."
    _install(monkeypatch, ScriptedLLM(_words(reply)))
    text, _ = _reply(friday_agent, "Geht es heute noch?")
    assert text == reply


# --- at most one retry per reply, across all filters ----------------------------------


def test_one_retry_then_fixed_sentence(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM(
        _words("The caller wants an appointment."),
        _words("The caller still wants an appointment."),
        _words("This must never be requested."),
    )
    _install(monkeypatch, fake)
    text, _ = _reply(agent, "Ich hätte gern einen Termin.")
    assert len(fake.contexts) == 2
    assert text == "Entschuldigung, wie kann ich Ihnen weiterhelfen?"


def test_claim_and_meta_in_one_reply_share_the_single_retry(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM(
        _words("Der Anrufer ist zufrieden. Ihre Nachricht ist gespeichert."),
        _words("Ihre Nachricht ist gespeichert."),
    )
    _install(monkeypatch, fake)
    text, _ = _reply(agent)
    assert len(fake.contexts) == 2
    assert text == (
        "Entschuldigung, die Nachricht ist noch nicht gespeichert. Soll ich sie jetzt für "
        "Sie speichern?"
    )
    note = [m for m in fake.contexts[1].items if isinstance(m, llm.ChatMessage)][-1]
    assert note.role == "system"
    assert "take_message" in (note.text_content or "")
    assert "Überlegungen" in (note.text_content or "")


def _run(coro: object) -> None:
    asyncio.run(coro)  # type: ignore[arg-type]


# --- regression (live run 2): example dates of birth were blocked as self-computed days --


@pytest.mark.parametrize(
    "sentence",
    [
        'Could you please say it once more, for example "fifteenth of June, nineteen eighty-five"?',
        "Could you please say it as day, month and year, for example the fifteenth of June, "
        "nineteen eighty-five?",
        "Könnten Sie mir bitte Tag, Monat und Jahr Ihres Geburtsdatums nennen, zum Beispiel "
        "fünfzehnter Juni neunzehnhundertfünfundachtzig?",
        "Bitte nennen Sie mir Tag, Monat und Jahr, zum Beispiel den fünfzehnten Juni "
        "neunzehnhundertfünfundachtzig.",
        'Könnten Sie mir bitte Ihr Geburtsdatum nennen, zum Beispiel "zwölfter März '
        'neunzehnhundertachtzig"?',
        "Sie sind also geboren am ersten Februar?",
        "Your date of birth is the first of February, then.",
    ],
)
def test_date_of_birth_is_not_a_self_computed_day(sentence: str) -> None:
    from praxis_telefon.agent.claim_gate import FilterContext, check_sentence

    ctx = FilterContext(caller_texts=("Ja, das stimmt so.",))
    assert "weekday" not in check_sentence(sentence, ctx)


@pytest.mark.parametrize(
    "sentence",
    [
        "Danke für Ihr Geburtsdatum, morgen ist Samstag.",
        "Ihr Geburtsdatum ist der fünfzehnte Juni neunzehnhundertfünfundachtzig, und morgen "
        "ist Samstag.",
        "Thanks for your date of birth, tomorrow is Saturday.",
    ],
)
def test_weekday_next_to_a_date_of_birth_is_still_blocked(sentence: str) -> None:
    from praxis_telefon.agent.claim_gate import FilterContext, check_sentence

    ctx = FilterContext(caller_texts=("Haben Sie morgen etwas frei?",))
    assert "weekday" in check_sentence(sentence, ctx)
