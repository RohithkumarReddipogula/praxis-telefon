"""Option B (docs/experiments/stt-language.md): Deepgram stays on `de` and is switched to
`en` on an explicit request, and back to `de` on a German one. `multi` was rejected after
voice tests with accented German (tagged Telugu, digits lost). Offline, no API calls."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from praxis_telefon.agent import main
from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.safety.messages import FIXED_MESSAGES
from tests.conftest import NOW
from tests.test_agent_options import settings
from tests.test_claim_gate import _reply


@pytest.fixture
def switched() -> list[str]:
    return []


@pytest.fixture
def agent(conn: sqlite3.Connection, switched: list[str]) -> Receptionist:
    return Receptionist(ToolService(conn, lambda: NOW), on_language_change=switched.append)


def test_english_request_through_de_switches_the_stream_to_en(
    agent: Receptionist, switched: list[str]
) -> None:
    agent.record_final("Guten Tag, ich hätte gern einen Termin.", "de")
    agent.record_final("Can we speak English?", "de")
    assert switched == ["en"]


def test_german_request_switches_the_stream_back(agent: Receptionist, switched: list[str]) -> None:
    agent.record_final("Können wir Englisch sprechen?", "de")
    agent.record_final("I would like an appointment, please.", "en")
    agent.record_final("Können wir wieder Deutsch sprechen?", "en")
    assert switched == ["en", "de"]


def test_two_german_finals_on_the_en_stream_switch_back(
    agent: Receptionist, switched: list[str]
) -> None:
    agent.record_final("English.", "de")
    agent.record_final("Ich hätte gern einen Termin bei Dr. Weber.", "en")
    assert switched == ["en"]
    agent.record_final("Mein Name ist Max Mustermann, das ist richtig.", "en")
    assert switched == ["en", "de"]


def test_english_words_on_the_de_stream_do_not_switch(
    agent: Receptionist, switched: list[str]
) -> None:
    # de tags every final de, so only a request switches.
    agent.record_final("Guten Tag.", "de")
    agent.record_final("Yes, that is right, thank you.", "de")
    agent.record_final("I would like an appointment, please.", "de")
    assert switched == []


def test_no_switch_callback_in_text_mode(conn: sqlite3.Connection) -> None:
    agent = Receptionist(ToolService(conn, lambda: NOW))
    agent.record_final("Can we speak English?", None)  # must not fail


def test_repeat_message_in_a_german_call_offers_english(agent: Receptionist) -> None:
    # An English caller on the de stream gets empty transcripts; the first "say that
    # again" tells them how to switch (user decision 2026-11-07).
    text, _ = _reply(agent, "")
    assert text == FIXED_MESSAGES[("repeat", "de")]
    assert "English" in text and "'English'" in text


class FakeSTT:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update_options(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeSession:
    def __init__(self) -> None:
        self.stt = FakeSTT()


def test_stream_switch_is_wired_when_stt_follows_the_call_language() -> None:
    session = FakeSession()
    switch = main.stt_switcher(settings(), session)
    assert switch is not None
    switch("en")
    assert session.stt.updates == [{"language": "en"}]


@pytest.mark.parametrize("fixed", ["de", "en", "multi"])
def test_no_stream_switch_with_a_fixed_stt_language(fixed: str) -> None:
    # STT_LANGUAGE set: an experiment setting; the stream keeps that language.
    assert main.stt_switcher(settings(STT_LANGUAGE=fixed), FakeSession()) is None


def test_repeat_message_in_an_english_call_offers_german(agent: Receptionist) -> None:
    # The en stream hears German poorly; "Deutsch, bitte" switches back.
    from tests.test_speech_filters import _ctx_with_callers

    ctx = _ctx_with_callers(agent, "Hello, I would like an appointment, please.")
    text, _ = _reply(agent, "", ctx)
    assert text == FIXED_MESSAGES[("repeat", "en")]
    assert "'Deutsch, bitte'" in text


def test_english_repeat_message_asks_for_deutsch_bitte() -> None:
    # Own recordings 2026-11-07: "Deutsch." came out of en as "Doish", "Deutsch, bitte."
    # as "Deutsche"; ask for the form that worked.
    assert "'Deutsch, bitte'" in FIXED_MESSAGES[("repeat", "en")]


# --- an empty final on the en stream in a call that started German ---------------------


class FakeReplySession:
    def __init__(self) -> None:
        self.replies: list[dict[str, Any]] = []

    def generate_reply(self, **kwargs: Any) -> str:
        self.replies.append(kwargs)
        return "handle"


class SessionReceptionist(Receptionist):
    fake_session = FakeReplySession()

    @property
    def session(self) -> Any:
        return self.fake_session


@pytest.fixture
def live_agent(conn: sqlite3.Connection, switched: list[str]) -> SessionReceptionist:
    agent = SessionReceptionist(ToolService(conn, lambda: NOW), on_language_change=switched.append)
    agent.fake_session = FakeReplySession()
    return agent


def _pending_reply(agent: Receptionist) -> str:
    import asyncio
    from typing import cast

    from livekit.agents import ModelSettings, llm

    async def run() -> str:
        tools = cast(list[llm.Tool], list(agent.tools))
        parts = [c async for c in agent.llm_node(agent.chat_ctx.copy(), tools, ModelSettings())]
        return "".join(p for p in parts if isinstance(p, str))

    return asyncio.run(run())


def test_empty_en_final_in_a_call_that_started_german_switches_back(
    live_agent: SessionReceptionist, switched: list[str]
) -> None:
    # User decision 2026-11-07: the en stream hears German as nothing ("Sprechen Sie
    # Deutsch?" came back empty), so in a call that started German an empty final on en
    # switches back to de, the agent asks bilingually to repeat, and it is no strike.
    live_agent.record_final("Guten Tag, ich hätte gern einen Termin.", "de")
    live_agent.record_final("Können wir Englisch sprechen?", "de")
    assert switched == ["en"]
    assert live_agent.on_unrecognized_speech() == "handle"
    assert switched == ["en", "de"]
    assert live_agent.language_state(live_agent.chat_ctx).language == "de"
    reply = _pending_reply(live_agent)
    assert reply == FIXED_MESSAGES[("switch_back", "de")]
    assert "Sorry" in reply and "noch einmal" in reply
    assert live_agent.safety_state().strikes == 0


def test_empty_final_after_switching_back_is_a_strike_again(
    live_agent: SessionReceptionist, switched: list[str]
) -> None:
    live_agent.record_final("Guten Tag.", "de")
    live_agent.record_final("Englisch.", "de")
    live_agent.on_unrecognized_speech()  # switch back, no strike
    _pending_reply(live_agent)
    live_agent.on_unrecognized_speech()  # now on de: the usual strike
    assert _pending_reply(live_agent) == FIXED_MESSAGES[("repeat", "de")]
    assert live_agent.safety_state().strikes == 1
    assert switched == ["en", "de"]


def test_empty_en_final_in_an_english_call_is_a_strike(
    live_agent: SessionReceptionist, switched: list[str]
) -> None:
    # Only calls that started German switch back.
    live_agent.record_final("Hello, I would like an appointment, please.", "en")
    live_agent.on_unrecognized_speech()
    assert _pending_reply(live_agent) == FIXED_MESSAGES[("repeat", "en")]
    assert live_agent.safety_state().strikes == 1
    assert switched == ["en"]


def test_empty_final_without_stream_switching_is_a_strike(conn: sqlite3.Connection) -> None:
    # Fixed STT_LANGUAGE (no switcher): unchanged behaviour.
    agent = SessionReceptionist(ToolService(conn, lambda: NOW))
    agent.fake_session = FakeReplySession()
    agent.record_final("Guten Tag.", "de")
    agent.record_final("Englisch.", "de")
    agent.on_unrecognized_speech()
    assert agent.safety_state().strikes == 1
