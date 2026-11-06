"""Code-spoken filler before a tool call (agent/tool_filler.py), offline with a scripted LLM.

Leak experiment (docs/experiments/reasoning-leak.md): with reasoning "low" the agent no
longer says a preamble before a tool, so caller line to first speech rose from 1.0 s to
2.1 s at p50.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import cast

import pytest
from livekit.agents import ModelSettings, llm

from praxis_telefon.agent.claim_gate import FilterContext
from praxis_telefon.agent.prompts import TOOL_FILLER
from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_filler import is_tool_filler, said_since_caller
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.safety.call_state import LOW_CONFIDENCE, Mode, replay
from praxis_telefon.safety.messages import FIXED_MESSAGES
from tests.conftest import NOW
from tests.test_claim_gate import ScriptedLLM, _install, _reply, _words

DE, EN = TOOL_FILLER["de"] + " ", TOOL_FILLER["en"] + " "


@pytest.fixture
def agent(conn: sqlite3.Connection) -> Receptionist:
    return Receptionist(ToolService(conn, lambda: NOW))


def _after_tool_round(agent: Receptionist, caller: str, said: str) -> llm.ChatContext:
    """The history LiveKit builds for the reply after a tool: line, speech, call, output."""
    ctx = agent.chat_ctx.copy()
    ctx.add_message(role="user", content=caller)
    ctx.add_message(role="assistant", content=said)
    ctx.items.append(llm.FunctionCall(call_id="c1", name="find_slots", arguments="{}"))
    ctx.items.append(
        llm.FunctionCallOutput(
            call_id="c1", name="find_slots", output="Gesucht: Dienstag.", is_error=False
        )
    )
    return ctx


def test_reply_that_starts_with_a_tool_call_gets_the_filler(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, ScriptedLLM([("tool", "find_slots")]))
    text, calls = _reply(agent, "Haben Sie morgen einen Termin frei?")
    assert text == DE
    assert calls == ["find_slots"]


def test_english_caller_gets_the_english_filler(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, ScriptedLLM([("tool", "find_slots")]))
    text, _ = _reply(agent, "Hello, do you have anything next Thursday?")
    assert text == EN


def test_no_filler_when_the_reply_speaks_first(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression (first filler draft): the filter still held this sentence when the
    # tool call chunk passed, so the caller got "Einen Moment, bitte. Einen Moment, ...".
    reply = "Einen Moment, ich schaue nach. "
    _install(monkeypatch, ScriptedLLM([*_words(reply), ("tool", "find_slots")]))
    text, _ = _reply(agent, "Haben Sie morgen einen Termin frei?")
    assert text == reply


def test_no_filler_without_a_tool_call(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = "Gern, wann passt es Ihnen? "
    _install(monkeypatch, ScriptedLLM(_words(reply)))
    text, _ = _reply(agent, "Ich hätte gern einen Termin.")
    assert text == reply


def test_at_most_once_per_caller_turn(agent: Receptionist, monkeypatch: pytest.MonkeyPatch) -> None:
    # Second tool round in the same turn: the filler is already in the history.
    _install(monkeypatch, ScriptedLLM([("tool", "book")]))
    ctx = _after_tool_round(agent, "Haben Sie Dienstag frei?", TOOL_FILLER["de"])
    text, calls = _reply_in(agent, ctx)
    assert text == ""
    assert calls == ["book"]


def test_no_filler_after_the_reply_already_spoke_in_this_turn(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, ScriptedLLM([("tool", "book")]))
    ctx = _after_tool_round(agent, "Haben Sie Dienstag frei?", "Gern, ich schaue nach.")
    text, _ = _reply_in(agent, ctx)
    assert text == ""


def test_filler_when_blocked_narration_was_all_the_reply_had(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    narration = "This is a new appointment request, so I'll search for slots. "
    _install(monkeypatch, ScriptedLLM([*_words(narration), ("tool", "find_slots")]))
    text, calls = _reply(agent, "Haben Sie nächsten Dienstag etwas frei?")
    assert text == DE  # the narration is dropped, the caller still hears something
    assert calls == ["find_slots"]


def test_fixed_safety_messages_get_no_filler(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM([("tool", "find_slots")])
    _install(monkeypatch, fake)
    text, _ = _reply(agent, "Ich habe starke Brustschmerzen.")
    assert text == FIXED_MESSAGES[("emergency", "de")]
    assert fake.contexts == []  # the LLM was not called


# --- the filler in the history changes neither the safety state nor intent_asked -------


def _conversation(agent: Receptionist, *, fillers: bool) -> llm.ChatContext:
    """A normal line, then two low-confidence mumbles (two strikes: handover)."""
    ctx = agent.chat_ctx.copy()
    low = LOW_CONFIDENCE / 2
    for text, confidence, reply in [
        ("Haben Sie morgen einen Termin frei?", None, "Morgen um neun Uhr wäre frei."),
        ("der äh wann denn", low, FIXED_MESSAGES[("repeat", "de")]),
        ("äh also der", low, FIXED_MESSAGES[("handover_strikes", "de")]),
    ]:
        msg = ctx.add_message(role="user", content=text)
        msg.transcript_confidence = confidence
        if fillers:
            ctx.add_message(role="assistant", content=TOOL_FILLER["de"])
        ctx.add_message(role="assistant", content=reply)
    return ctx


def test_filler_does_not_change_the_safety_replay(agent: Receptionist) -> None:
    with_fillers = replay(agent._events(_conversation(agent, fillers=True)))
    without = replay(agent._events(_conversation(agent, fillers=False)))
    assert with_fillers == without
    state, action = with_fillers
    assert state.mode is Mode.HANDOVER  # the strikes still count; a filler resets nothing
    assert action.kind == "handover_strikes"


def test_filler_does_not_count_as_the_new_or_existing_question(agent: Receptionist) -> None:
    ctx = _conversation(agent, fillers=True)
    assert FilterContext.from_chat(ctx, "").intent_asked is False
    ctx.add_message(role="assistant", content="Geht es um einen neuen oder bestehenden Termin?")
    assert FilterContext.from_chat(ctx, "").intent_asked is True


# --- helpers --------------------------------------------------------------------------


def test_said_since_caller_and_is_tool_filler(agent: Receptionist) -> None:
    ctx = agent.chat_ctx.copy()
    ctx.add_message(role="assistant", content="Guten Tag.")  # the greeting, before the line
    ctx.add_message(role="user", content="Hallo.")
    assert said_since_caller(ctx) is False
    ctx.add_message(role="assistant", content=TOOL_FILLER["de"])
    assert said_since_caller(ctx) is True
    assert is_tool_filler("Einen Moment, bitte.") and is_tool_filler(" One moment, please. ")
    assert not is_tool_filler("Einen Moment, ich schaue nach.")


def _reply_in(agent: Receptionist, ctx: llm.ChatContext) -> tuple[str, list[str]]:
    """Like _reply, for a history that already ends with the tool output."""
    tools = cast(list[llm.Tool], list(agent.tools))

    async def run() -> tuple[str, list[str]]:
        text: list[str] = []
        calls: list[str] = []
        async for out in agent.llm_node(ctx, tools, ModelSettings()):
            if isinstance(out, str):
                text.append(out)
            elif isinstance(out, llm.ChatChunk) and out.delta is not None:
                text.append(out.delta.content or "")
                calls += [c.name for c in out.delta.tool_calls]
        return "".join(text), calls

    return asyncio.run(run())
