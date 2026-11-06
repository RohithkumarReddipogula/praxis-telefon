"""Safety screening in Receptionist.llm_node and the ToolService booking guard (offline)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sqlite3
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, cast

import pytest
from livekit.agents import Agent, ModelSettings, llm

from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.safety.call_state import LOW_CONFIDENCE
from praxis_telefon.safety.messages import CRISIS_LINE_NUMBER, FIXED_MESSAGES
from praxis_telefon.tools import errors
from tests.conftest import DOB, NAME, NOW, PHONE

LLM_REPLY = "<llm reply>"


class FakeLLM:
    """Stands in for Agent.default.llm_node; records what the LLM would have received."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    async def node(
        self, agent: Agent, chat_ctx: llm.ChatContext, tools: list[llm.Tool], _: Any
    ) -> AsyncIterator[str]:
        names = [str(getattr(getattr(t, "info", None), "name", "")) for t in tools]
        system = [
            m.text_content or ""
            for m in chat_ctx.items
            if isinstance(m, llm.ChatMessage) and m.role == "system"
        ]
        self.calls.append((names, system))
        yield LLM_REPLY


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr(Agent.default, "llm_node", fake.node)
    return fake


@pytest.fixture
def service(conn: sqlite3.Connection) -> ToolService:
    return ToolService(conn, lambda: NOW)


@pytest.fixture
def agent(service: ToolService) -> Receptionist:
    return Receptionist(service)


def _tools(agent: Receptionist) -> list[llm.Tool]:
    return cast(list[llm.Tool], list(agent.tools))


def _say(agent: Receptionist, text: str, confidence: float | None = None) -> str:
    """Add a caller turn, run llm_node like LiveKit does, commit the reply."""
    ctx = agent.chat_ctx.copy()
    msg = ctx.add_message(role="user", content=text)
    msg.transcript_confidence = confidence

    async def run() -> str:
        parts = [c async for c in agent.llm_node(ctx, _tools(agent), ModelSettings())]
        return "".join(p for p in parts if isinstance(p, str))

    reply = asyncio.run(run())
    ctx.add_message(role="assistant", content=reply)
    asyncio.run(agent.update_chat_ctx(ctx))
    return reply


def test_emergency_gets_fixed_message_without_llm(agent: Receptionist, fake_llm: FakeLLM) -> None:
    reply = _say(agent, "Ich habe starke Brustschmerzen.")
    assert reply == FIXED_MESSAGES[("emergency", "de")]
    assert "eins eins zwei" in reply
    assert fake_llm.calls == []


def test_emergency_message_asks_no_questions() -> None:
    for kind in ("emergency", "suicide"):
        for lang in ("de", "en"):
            assert "?" not in FIXED_MESSAGES[(kind, lang)]


def test_suicide_message_names_112_and_crisis_line(agent: Receptionist, fake_llm: FakeLLM) -> None:
    reply = _say(agent, "Ich will nicht mehr leben.")
    assert reply == FIXED_MESSAGES[("suicide", "de")]
    assert "eins eins zwei" in reply
    assert "null acht null null, eins eins eins, null, eins eins eins" in reply
    assert CRISIS_LINE_NUMBER == "0800 111 0 111"
    assert fake_llm.calls == []


def test_english_suicide_message(agent: Receptionist, fake_llm: FakeLLM) -> None:
    reply = _say(agent, "I want to kill myself.")
    assert reply == FIXED_MESSAGES[("suicide", "en")]
    assert "one one two" in reply and "Telefonseelsorge" in reply


def test_after_emergency_llm_only_gets_take_message(agent: Receptionist, fake_llm: FakeLLM) -> None:
    _say(agent, "Mein Mann ist bewusstlos.")
    assert _say(agent, "Kann ich trotzdem morgen einen Termin haben?") == LLM_REPLY
    tools, system = fake_llm.calls[-1]
    assert tools == ["take_message"]
    assert any("keine Termine" in s for s in system)


def test_after_suicide_the_note_names_the_crisis_line(
    agent: Receptionist, fake_llm: FakeLLM
) -> None:
    _say(agent, "Ich will mich umbringen.")
    _say(agent, "Danke.")
    _, system = fake_llm.calls[-1]
    assert any("Telefonseelsorge" in s for s in system)


def test_normal_turn_gets_all_tools(agent: Receptionist, fake_llm: FakeLLM) -> None:
    assert _say(agent, "Ich hätte gern einen Termin.") == LLM_REPLY
    tools, system = fake_llm.calls[-1]
    assert "book" in tools and "find_slots" in tools
    assert not any("WICHTIG" in s for s in system)


def test_human_request_offers_message(agent: Receptionist, fake_llm: FakeLLM) -> None:
    assert (
        _say(agent, "Ich will mit einem Menschen sprechen.")
        == FIXED_MESSAGES[("human_request", "de")]
    )
    _say(agent, "Ja, bitte.")
    assert fake_llm.calls[-1][0] == ["take_message"]


def test_two_mumbles_hand_over(agent: Receptionist, fake_llm: FakeLLM) -> None:
    low = LOW_CONFIDENCE / 2
    assert _say(agent, "der äh wann denn", low) == FIXED_MESSAGES[("repeat", "de")]
    assert _say(agent, "äh also der", low) == FIXED_MESSAGES[("handover_strikes", "de")]
    assert fake_llm.calls == []


def test_ja_twice_with_low_confidence_reaches_llm(agent: Receptionist, fake_llm: FakeLLM) -> None:
    low = LOW_CONFIDENCE / 2
    assert _say(agent, "Ja", low) == LLM_REPLY
    assert _say(agent, "Ja", low) == LLM_REPLY
    assert "book" in fake_llm.calls[-1][0]


def test_mumble_then_ja_resets(agent: Receptionist, fake_llm: FakeLLM) -> None:
    low = LOW_CONFIDENCE / 2
    _say(agent, "der äh wann denn", low)
    assert _say(agent, "Ja", low) == LLM_REPLY
    assert _say(agent, "äh also der", low) == FIXED_MESSAGES[("repeat", "de")]


def test_screening_has_no_side_effects_on_discarded_runs(
    agent: Receptionist, fake_llm: FakeLLM
) -> None:
    # LiveKit runs llm_node preemptively and may discard the run: the agent's own
    # (committed) state must not change.
    ctx = agent.chat_ctx.copy()
    ctx.add_message(role="user", content="Ich bekomme keine Luft.")

    async def run() -> None:
        _ = [c async for c in agent.llm_node(ctx, _tools(agent), ModelSettings())]

    asyncio.run(run())
    assert agent.safety_state().mode.value == "normal"


# --- unrecognized speech (empty transcript) ---------------------------------------------


class _FakeSession:
    def __init__(self) -> None:
        self.replies: list[dict[str, Any]] = []

    def generate_reply(self, **kwargs: Any) -> str:
        self.replies.append(kwargs)
        return "handle"


class _SessionReceptionist(Receptionist):
    fake_session = _FakeSession()

    @property
    def session(self) -> Any:
        return self.fake_session


def test_two_unrecognized_speeches_hand_over(service: ToolService, fake_llm: FakeLLM) -> None:
    agent = _SessionReceptionist(service)
    agent.fake_session = _FakeSession()

    async def reply() -> str:
        ctx = agent.chat_ctx.copy()
        parts = [c async for c in agent.llm_node(ctx, _tools(agent), ModelSettings())]
        return "".join(p for p in parts if isinstance(p, str))

    assert agent.on_unrecognized_speech() == "handle"
    assert asyncio.run(reply()) == FIXED_MESSAGES[("repeat", "de")]
    agent.on_unrecognized_speech()
    assert asyncio.run(reply()) == FIXED_MESSAGES[("handover_strikes", "de")]
    assert agent.safety_state().mode.value == "handover"
    assert service.booking_allowed() is False
    assert len(agent.fake_session.replies) == 2


# --- ToolService guard ------------------------------------------------------------------


def test_booking_tools_refused_in_code_after_emergency(
    agent: Receptionist, service: ToolService, fake_llm: FakeLLM
) -> None:
    _say(agent, "Ich habe Atemnot.")
    calls: list[Callable[[], object]] = [
        lambda: service.find_slots("2026-11-10", "any", None),
        lambda: service.book(1, NAME, DOB.isoformat()),
        lambda: service.find_booking(NAME, DOB.isoformat()),
        lambda: service.move(1, 2),
        lambda: service.cancel(1),
    ]
    for call in calls:
        with pytest.raises(errors.BookingDisabled):
            call()
    assert "Nachricht" in service.take_message(NAME, PHONE, "callback")


def test_booking_allowed_in_normal_mode(agent: Receptionist, service: ToolService) -> None:
    assert "Freie Termine" in service.find_slots("2026-11-10", "any", None)


# --- logging ----------------------------------------------------------------------------


@pytest.fixture
def safety_logs() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    log = logging.getLogger("praxis_telefon.safety")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    yield stream
    log.removeHandler(handler)


def test_safety_log_has_action_only(
    agent: Receptionist, fake_llm: FakeLLM, safety_logs: io.StringIO
) -> None:
    _say(agent, "Ich habe starke Brustschmerzen, Erika Mustermann hier.")
    lines = [json.loads(line) for line in safety_logs.getvalue().splitlines()]
    assert lines == [{"event": "safety", "action": "emergency"}]
    assert "Mustermann" not in safety_logs.getvalue()
