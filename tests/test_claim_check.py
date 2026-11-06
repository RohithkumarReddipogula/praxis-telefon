import asyncio
import io
import json
import logging
from collections.abc import Iterator

import pytest
from livekit.agents import AgentSession, ConversationItemAddedEvent, FunctionToolsExecutedEvent
from livekit.agents.llm import ChatMessage, FunctionCall, FunctionCallOutput

from praxis_telefon.agent.claim_check import ClaimTracker, claimed_actions, split_sentences
from praxis_telefon.agent.main import attach_claim_check


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("Alles klar, ich habe Ihren Termin auf Mittwoch verschoben.", "move_booking"),
        ("Ihr Termin ist abgesagt. Kann ich sonst noch helfen?", "cancel_booking"),
        ("Gut, der Termin ist storniert.", "cancel_booking"),
        ("Prima, Sie sind für Dienstag um 8 Uhr eingetragen.", "book"),
    ],
)
def test_false_success_claim_is_detected(text: str, action: str) -> None:
    assert claimed_actions(text) == {action}


@pytest.mark.parametrize(
    "text",
    [
        "Soll ich den Termin auf Mittwoch verschieben?",
        "Möchten Sie den Termin absagen?",
        "Ich habe Mittwoch um 8 Uhr oder um 9 Uhr. Welcher passt Ihnen?",
    ],
)
def test_questions_are_not_claims(text: str) -> None:
    assert claimed_actions(text) == set()


@pytest.mark.parametrize(
    "text",
    [
        # Live test false positives on 2026-11-06: read-back questions were flagged.
        "Dienstag, 10. November, um 8 Uhr bei Dr. Weber, gebucht auf Max Mustermann, "
        "geboren am 1. Februar 1979. Stimmt das so?",
        "Also Dienstag um 8 Uhr bei Dr. Weber, gebucht auf Max Mustermann, stimmt das so?",
        "Dann wird der Termin auf Mittwoch um 8 Uhr verschoben. Ist das so richtig?",
    ],
)
def test_read_back_with_confirmation_question_is_not_a_claim(text: str) -> None:
    assert claimed_actions(text) == set()


def test_abbreviations_and_ordinals_do_not_split_sentences() -> None:
    text = "Termin bei Dr. Weber am 1. Februar. Stimmt das?"
    assert split_sentences(text) == ["Termin bei Dr. Weber am 1. Februar.", "Stimmt das?"]


def test_year_ends_a_sentence() -> None:
    assert len(split_sentences("Geboren 1979. Stimmt das?")) == 2


def test_tracker_flags_claim_before_tool_success() -> None:
    tracker = ClaimTracker()
    tracker.tool_result("move_booking", ok=False)
    assert tracker.unbacked("Ihr Termin ist verschoben.") == ["move_booking"]
    tracker.tool_result("move_booking", ok=True)
    assert tracker.unbacked("Ihr Termin ist verschoben.") == []


def test_earlier_book_does_not_back_a_move_claim() -> None:
    tracker = ClaimTracker()
    tracker.tool_result("book", ok=True)
    assert tracker.unbacked("Ihr Termin ist gebucht.") == []
    assert tracker.unbacked("Ich habe ihn auf Mittwoch verschoben.") == ["move_booking"]


@pytest.fixture
def captured_logs() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    log = logging.getLogger("praxis_telefon.turns")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    yield stream
    log.removeHandler(handler)


@pytest.fixture
def session() -> Iterator[AgentSession[None]]:
    loop = asyncio.new_event_loop()
    s: AgentSession[None] = AgentSession(loop=loop)
    attach_claim_check(s)
    yield s
    loop.close()


def _say(session: AgentSession[None], text: str) -> None:
    item = ChatMessage(role="assistant", content=[text])
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=item))


def _tool(session: AgentSession[None], name: str, *, is_error: bool) -> None:
    session.emit(
        "function_tools_executed",
        FunctionToolsExecutedEvent(
            function_calls=[FunctionCall(call_id="c1", name=name, arguments="{}")],
            function_call_outputs=[
                FunctionCallOutput(call_id="c1", name=name, output="x", is_error=is_error)
            ],
        ),
    )


def test_session_logs_unbacked_claim_without_text(
    session: AgentSession[None], captured_logs: io.StringIO
) -> None:
    _tool(session, "cancel_booking", is_error=True)
    _say(session, "Erledigt, Erika Mustermann, Ihr Termin ist abgesagt.")

    lines = captured_logs.getvalue().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"event": "unbacked_claim", "action": "cancel_booking"}
    ]
    assert "Mustermann" not in captured_logs.getvalue()


def test_session_does_not_log_backed_claim(
    session: AgentSession[None], captured_logs: io.StringIO
) -> None:
    _tool(session, "cancel_booking", is_error=False)
    _say(session, "Ihr Termin ist abgesagt.")
    assert captured_logs.getvalue() == ""
