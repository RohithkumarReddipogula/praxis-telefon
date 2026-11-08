"""What is pending decides the natural question after a blocked "answer yes or no".

User, 2026-11-08: the fallback must fit the situation; the demand also happens at the
callback number read-back ("Please answer yes or tell me the correct number") and can
happen before move and cancel. Pending action, from the conversation:
1. a refused action tool (book, move_booking, cancel_booking, take_message) after the last
   successful one;
2. otherwise the callback number: the agent's last message talks about the number, or the
   caller's last line has one;
3. otherwise the caller's words since the last success: cancel or move;
4. otherwise find_slots has run since then: book;
5. otherwise nothing (neutral question).
"""

from __future__ import annotations

import sqlite3

import pytest
from livekit.agents import llm

from praxis_telefon.agent.pending_action import pending_action
from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.safety.messages import DEMAND_QUESTIONS, fallback_for, retry_note_for
from tests.conftest import NOW
from tests.test_claim_gate import ScriptedLLM, _install, _reply, _words
from tests.test_speech_filters import _system_notes

QUESTIONS = {
    ("book", "de"): "Soll ich den Termin so eintragen?",
    ("book", "en"): "Shall I book it like that?",
    ("move", "de"): "Soll ich den Termin so verschieben?",
    ("move", "en"): "Shall I move the appointment like that?",
    ("cancel", "de"): "Soll ich den Termin absagen?",
    ("cancel", "en"): "Shall I cancel the appointment?",
    ("callback", "de"): "Ist die Nummer so richtig?",
    ("callback", "en"): "Is that number correct?",
    (None, "de"): "Ist das so richtig?",
    (None, "en"): "Is that correct?",
}


def ctx(*items: tuple[str, ...]) -> llm.ChatContext:
    """("user", text) | ("assistant", text) | ("call", tool) | ("ok", tool) | ("err", tool)."""
    c = llm.ChatContext()
    n = 0
    for kind, value in items:
        if kind in ("user", "assistant"):
            c.add_message(role=kind, content=value)  # type: ignore[arg-type]
        elif kind == "call":
            n += 1
            c.items.append(llm.FunctionCall(call_id=f"c{n}", name=value, arguments="{}"))
        else:
            c.items.append(
                llm.FunctionCallOutput(
                    call_id=f"c{n}", name=value, output="x", is_error=(kind == "err")
                )
            )
    return c


@pytest.mark.parametrize(
    ("tool", "pending"),
    [("book", "book"), ("move_booking", "move"), ("cancel_booking", "cancel"),
     ("take_message", "callback")],
)  # fmt: skip
def test_a_refused_action_tool_is_the_pending_action(tool: str, pending: str) -> None:
    c = ctx(("user", "Ja, bitte."), ("call", tool), ("err", tool))
    assert pending_action(c) == pending


def test_a_refusal_before_a_later_success_is_not_pending() -> None:
    c = ctx(
        ("call", "book"), ("err", "book"), ("call", "book"), ("ok", "book"),
        ("user", "Danke."),
    )  # fmt: skip
    assert pending_action(c) is None


def test_callback_number_read_back_is_the_callback() -> None:
    # "Please answer yes or tell me the correct number" came at this read-back.
    c = ctx(
        ("user", "Meine Nummer ist null drei null, zwei drei eins zwei fünf, vier fünf sechs."),
        ("assistant", "Ich notiere Ihre Rückrufnummer: null drei null, zwei drei eins zwei."),
        ("user", "Hmm."),
    )
    assert pending_action(c) == "callback"


def test_a_number_in_the_callers_last_line_is_the_callback() -> None:
    c = ctx(
        ("assistant", "Unter welcher Nummer erreicht Sie das Team?"),
        ("user", "Null drei null, zwei drei eins zwei fünf, vier fünf sechs."),
    )
    assert pending_action(c) == "callback"


def test_the_callers_move_and_cancel_words_decide_without_a_refusal() -> None:
    move = ctx(("call", "find_booking"), ("ok", "find_booking"),
               ("user", "Ich möchte den Termin auf Mittwoch verschieben."))  # fmt: skip
    cancel = ctx(("call", "find_booking"), ("ok", "find_booking"),
                 ("user", "Ich muss den Termin leider absagen."))  # fmt: skip
    english = ctx(("user", "Could I move my appointment to Wednesday?"))
    assert pending_action(move) == "move"
    assert pending_action(cancel) == "cancel"
    assert pending_action(english) == "move"


def test_find_slots_without_a_success_since_means_book() -> None:
    c = ctx(("user", "Dienstag vormittags."), ("call", "find_slots"), ("ok", "find_slots"),
            ("user", "Der Erste."))  # fmt: skip
    assert pending_action(c) == "book"


def test_old_words_before_the_last_success_are_ignored() -> None:
    c = ctx(
        ("user", "Ich möchte den Termin verschieben."),
        ("call", "move_booking"), ("ok", "move_booking"),
        ("user", "Ich hätte noch gern einen weiteren Termin."),
        ("call", "find_slots"), ("ok", "find_slots"),
    )  # fmt: skip
    assert pending_action(c) == "book"


def test_nothing_pending_is_none() -> None:
    assert pending_action(ctx(("user", "Wie sind Ihre Öffnungszeiten?"))) is None
    assert pending_action(ctx(("call", "book"), ("ok", "book"), ("user", "Danke."))) is None


# --- the questions ------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "question"), list(QUESTIONS.items()))
def test_fallback_fits_the_pending_action(key: tuple[str | None, str], question: str) -> None:
    pending, language = key
    assert fallback_for(["demand"], language, pending) == question  # type: ignore[arg-type]
    assert DEMAND_QUESTIONS[key] == question  # type: ignore[index]


@pytest.mark.parametrize(("key", "question"), list(QUESTIONS.items()))
def test_retry_note_example_fits_the_pending_action(
    key: tuple[str | None, str], question: str
) -> None:
    pending, language = key
    note = retry_note_for(["demand"], language, pending)  # type: ignore[arg-type]
    assert question in note
    others = {q for k, q in QUESTIONS.items() if k[1] == language and q != question}
    assert not any(q in note for q in others)  # only the matching example


def test_fallback_questions_count_as_confirmation_questions() -> None:
    from praxis_telefon.agent.claim_check import is_confirmation_question

    for question in QUESTIONS.values():
        assert is_confirmation_question(question), question


# --- in a reply -------------------------------------------------------------------------


@pytest.fixture
def agent(conn: sqlite3.Connection) -> Receptionist:
    return Receptionist(ToolService(conn, lambda: NOW))


def _with(agent: Receptionist, *items: tuple[str, ...]) -> llm.ChatContext:
    c = agent.chat_ctx.copy()
    c.items.extend(ctx(*items).items)
    return c


@pytest.mark.parametrize(
    ("items", "question"),
    [
        ((("user", "Ja, bitte."), ("call", "cancel_booking"), ("err", "cancel_booking")),
         "Soll ich den Termin absagen?"),
        ((("user", "Ja, tu es."), ("call", "move_booking"), ("err", "move_booking")),
         "Soll ich den Termin so verschieben?"),
        ((("user", "Ja, bitte."), ("call", "book"), ("err", "book")),
         "Soll ich den Termin so eintragen?"),
        ((("assistant", "Ich notiere Ihre Rückrufnummer: null drei null."), ("user", "Hmm.")),
         "Ist die Nummer so richtig?"),
        ((("user", "Hmm, ich weiß nicht."),), "Ist das so richtig?"),
    ],
)  # fmt: skip
def test_demand_twice_gets_the_question_for_the_pending_action(
    agent: Receptionist,
    monkeypatch: pytest.MonkeyPatch,
    items: tuple[tuple[str, ...], ...],
    question: str,
) -> None:
    fake = ScriptedLLM(
        _words("Bitte sagen Sie Ja oder Nein. "), _words("Ich brauche noch Ihr klares Ja. ")
    )
    _install(monkeypatch, fake)
    text, _ = _reply(agent, "Hmm.", _with(agent, *items))
    assert text.strip() == question
    # the retry note carried the same example
    assert any(question in note for note in _system_notes(fake.contexts[1]))


def test_english_demand_at_a_cancel_gets_the_english_cancel_question(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM(
        _words("Please answer yes or no. "), _words("I need a clear yes before I cancel. ")
    )
    _install(monkeypatch, fake)
    c = _with(
        agent,
        ("user", "Hello, I need to cancel my appointment, please. Yes, go ahead."),
        ("call", "cancel_booking"),
        ("err", "cancel_booking"),
    )
    text, _ = _reply(agent, "Hello, I need to cancel my appointment, please.", c)
    assert text.strip() == "Shall I cancel the appointment?"
