"""Claim blocking in Receptionist.llm_node: no "saved/booked/moved/cancelled" before the
matching tool succeeded in this call (offline, with a scripted fake LLM)."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import pytest
from livekit.agents import Agent, ModelSettings, llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, FunctionToolCall

from praxis_telefon.agent.claim_check import take_sentences
from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_service import ToolService
from tests.conftest import NOW

Token = str | tuple[str, str]  # text, or ("tool", name)


def _chunks(tokens: Sequence[Token]) -> list[ChatChunk]:
    out: list[ChatChunk] = []
    for i, tok in enumerate(tokens):
        if isinstance(tok, tuple):
            call = FunctionToolCall(name=tok[1], arguments="{}", call_id=f"call_{i}")
            out.append(ChatChunk(id="r", delta=ChoiceDelta(role="assistant", tool_calls=[call])))
        else:
            out.append(ChatChunk(id="r", delta=ChoiceDelta(role="assistant", content=tok)))
    return out


def _words(text: str) -> list[str]:
    """Split like an LLM stream: word by word, spaces kept."""
    parts = text.split(" ")
    return [p + (" " if i < len(parts) - 1 else "") for i, p in enumerate(parts)]


class ScriptedLLM:
    """Replaces Agent.default.llm_node; returns one scripted response per call."""

    def __init__(self, *responses: Sequence[Token]) -> None:
        self.responses = list(responses)
        self.contexts: list[llm.ChatContext] = []

    async def node(
        self, agent: Agent, chat_ctx: llm.ChatContext, tools: list[llm.Tool], _: Any
    ) -> AsyncIterator[ChatChunk]:
        self.contexts.append(chat_ctx)
        for chunk in _chunks(self.responses.pop(0) if self.responses else []):
            yield chunk


@pytest.fixture
def agent(conn: sqlite3.Connection) -> Receptionist:
    return Receptionist(ToolService(conn, lambda: NOW))


def _install(monkeypatch: pytest.MonkeyPatch, fake: ScriptedLLM) -> None:
    monkeypatch.setattr(Agent.default, "llm_node", fake.node)


def _reply(
    agent: Receptionist, user: str = "Ja, bitte.", ctx: llm.ChatContext | None = None
) -> tuple[str, list[str]]:
    """Run llm_node once; return the spoken text and the tool calls passed through."""
    ctx = ctx or agent.chat_ctx.copy()
    ctx.add_message(role="user", content=user)
    tools = cast(list[llm.Tool], list(agent.tools))

    async def run() -> tuple[str, list[str]]:
        text: list[str] = []
        calls: list[str] = []
        async for out in agent.llm_node(ctx, tools, ModelSettings()):
            if isinstance(out, str):
                text.append(out)
            elif isinstance(out, ChatChunk) and out.delta is not None:
                text.append(out.delta.content or "")
                calls += [c.name for c in out.delta.tool_calls]
        return "".join(text), calls

    return asyncio.run(run())


# --- regression: live test human_request[2] ---------------------------------------------


def test_message_claim_without_take_message_is_never_spoken(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM(
        _words(
            "Vielen Dank. Ich habe Ihre Nachricht für das Praxisteam aufgenommen. Auf Wiederhören."
        ),
        _words("Wie ist bitte Ihre Rückrufnummer?"),
    )
    _install(monkeypatch, fake)
    text, _ = _reply(agent)
    assert "aufgenommen" not in text
    assert text.startswith("Vielen Dank.")


def _ctx_with_success(agent: Receptionist, tool: str, output: str = "ok") -> llm.ChatContext:
    ctx = agent.chat_ctx.copy()
    ctx.items.append(llm.FunctionCall(call_id="c1", name=tool, arguments="{}"))
    ctx.items.append(llm.FunctionCallOutput(call_id="c1", name=tool, output=output, is_error=False))
    return ctx


SLOTS = (
    "Gesucht: Dienstag, 10. November.\nFreie Termine (nur diese anbieten, wörtlich "
    "vorlesen):\nslot_id=1: Dienstag, 10. November, um 8 Uhr, bei Dr. Weber"
)


def _ctx_with_slots(agent: Receptionist) -> llm.ChatContext:
    """Realistic context: the day the agent talks about comes from a find_slots result."""
    return _ctx_with_success(agent, "find_slots", SLOTS)


def test_retry_gets_a_note_and_never_the_blocked_sentence(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM(
        _words("Danke. Ihre Nachricht ist gespeichert."),
        _words("Darf ich noch Ihre Rückrufnummer haben?"),
    )
    _install(monkeypatch, fake)
    text, _ = _reply(agent)
    assert text == "Danke. Darf ich noch Ihre Rückrufnummer haben?"
    retry_ctx = fake.contexts[1]
    messages = [m for m in retry_ctx.items if isinstance(m, llm.ChatMessage)]
    assert messages[-1].role == "system" and "take_message" in (messages[-1].text_content or "")
    assert messages[-2].role == "assistant" and messages[-2].text_content == "Danke."
    assert not any("gespeichert." in (m.text_content or "") for m in messages if m.role != "system")


def test_second_false_claim_gets_fixed_sentence(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM(
        _words("Ich habe den Termin für Sie gebucht."),
        _words("Alles klar, der Termin ist eingetragen."),
    )
    _install(monkeypatch, fake)
    text, _ = _reply(agent)
    assert text == "Entschuldigung, der Termin ist noch nicht gebucht. Soll ich ihn jetzt buchen?"
    assert len(fake.contexts) == 2  # one retry only


@pytest.mark.parametrize(
    ("tool", "sentence"),
    [
        ("take_message", "Ihre Nachricht ist gespeichert."),
        ("book", "Ihr Termin ist gebucht."),
        ("move_booking", "Ich habe den Termin verschoben."),
        ("cancel_booking", "Der Termin ist abgesagt."),
    ],
)
def test_claim_after_tool_success_is_spoken(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch, tool: str, sentence: str
) -> None:
    fake = ScriptedLLM(_words(f"Erledigt. {sentence} Kann ich sonst helfen?"))
    _install(monkeypatch, fake)
    text, _ = _reply(agent, ctx=_ctx_with_success(agent, tool))
    assert text == f"Erledigt. {sentence} Kann ich sonst helfen?"
    assert len(fake.contexts) == 1


def test_failed_tool_does_not_back_a_claim(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = agent.chat_ctx.copy()
    ctx.items.append(llm.FunctionCall(call_id="c1", name="cancel_booking", arguments="{}"))
    ctx.items.append(
        llm.FunctionCallOutput(call_id="c1", name="cancel_booking", output="x", is_error=True)
    )
    _install(monkeypatch, ScriptedLLM(_words("Der Termin ist abgesagt."), _words("Hm.")))
    text, _ = _reply(agent, ctx=ctx)
    assert "abgesagt" not in text


def test_claim_in_the_same_response_as_the_tool_call_is_dropped_without_retry(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ScriptedLLM([*_words("Gut. Ihre Nachricht ist gespeichert. "), ("tool", "take_message")])
    _install(monkeypatch, fake)
    text, calls = _reply(agent)
    assert text == "Gut. "
    assert calls == ["take_message"]
    assert len(fake.contexts) == 1


@pytest.mark.parametrize(
    "reply",
    [
        "Dienstag, 10. November, um 8 Uhr bei Dr. Weber, gebucht auf Max Mustermann. "
        "Stimmt das so?",
        "Soll ich die Nachricht für das Praxisteam speichern?",
        "Wenn Sie möchten, ist der Termin danach gebucht.",
        "Die Nachricht ist noch nicht gespeichert. Wie ist Ihre Nummer?",
        "Dieser Termin ist leider schon gebucht. Möchten Sie einen anderen?",
        "Dienstag ist leider ausgebucht.",
        "Ich kann gern eine Nachricht für das Praxisteam aufnehmen.",
        "Is that right? I have Tuesday at 8, booked for Max Mustermann. Shall I book it?",
        "Tuesday is fully booked, unfortunately.",
    ],
)
def test_questions_read_backs_and_negations_pass(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    # Realistic context: the day comes from a find_slots result, and English replies come
    # after an English caller turn (otherwise the weekday and language filters apply).
    fake = ScriptedLLM(_words(reply))
    _install(monkeypatch, fake)
    user = "Yes, please." if reply.startswith(("Is ", "Tuesday")) else "Ja, bitte."
    text, _ = _reply(agent, user, ctx=_ctx_with_slots(agent))
    assert text == reply
    assert len(fake.contexts) == 1


@pytest.mark.parametrize(
    "claim",
    [
        "Ich habe Ihre Nachricht für das Praxisteam aufgenommen.",
        "Ihre Rückrufbitte ist notiert.",
        "Ich habe die Nachricht an das Team weitergeleitet.",
        "Your message has been saved.",
        "I have booked you in for Tuesday.",
        "Your appointment is cancelled.",
        "I've rescheduled your appointment.",
    ],
)
def test_claims_are_blocked(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch, claim: str
) -> None:
    _install(monkeypatch, ScriptedLLM(_words(f"Okay. {claim}"), _words("Einen Moment.")))
    text, _ = _reply(agent)
    assert text == "Okay. Einen Moment."


def test_sentence_split_across_chunks_and_unterminated_end(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    pieces = ["Guten Tag", ". Ihre Nach", "richt ist gesp", "eichert. Bis", " bald"]
    _install(monkeypatch, ScriptedLLM(pieces, _words("Wie ist Ihr Name?")))
    text, _ = _reply(agent)
    assert text == "Guten Tag. Bis bald Wie ist Ihr Name?"
    assert "gespeichert" not in text


def test_order_and_spacing_of_normal_sentences_kept(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = "Gern. Ich schaue nach. Am Dienstag um 8 Uhr bei Dr. Weber ginge es. Passt das?"
    _install(monkeypatch, ScriptedLLM(_words(reply)))
    assert _reply(agent, ctx=_ctx_with_slots(agent))[0] == reply


# --- the blocked sentence is in neither chat history nor transcript ---------------------


def test_blocked_sentence_never_reaches_history_or_transcript(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.agents import AgentSession, ConversationItemAddedEvent
    from livekit.agents.voice.run_result import RunResult
    from livekit.plugins import openai

    fake = ScriptedLLM(
        _words("Danke. Ich habe Ihre Nachricht für das Praxisteam aufgenommen."),
        _words("Wie ist Ihre Rückrufnummer?"),
    )
    _install(monkeypatch, fake)

    async def scenario() -> tuple[list[str], list[str], list[str]]:
        # A real AgentSession in text mode; the LLM object is never called (patched node).
        dummy = openai.LLM(model="offline-dummy", api_key="sk-test-fake")
        added: list[str] = []
        async with AgentSession(llm=dummy) as session:

            @session.on("conversation_item_added")
            def _added(ev: ConversationItemAddedEvent) -> None:
                if isinstance(ev.item, llm.ChatMessage) and ev.item.role == "assistant":
                    added.append(ev.item.text_content or "")

            agent = Receptionist(ToolService(conn, lambda: NOW))
            await session.start(agent)
            result: RunResult[None] = await session.run(user_input="Ja, bitte speichern Sie das.")
            run_texts = [
                e.item.text_content or ""
                for e in result.events
                if isinstance(e.item, llm.ChatMessage) and e.item.role == "assistant"
            ]
            history = [
                m.text_content or ""
                for m in agent.chat_ctx.items
                if isinstance(m, llm.ChatMessage) and m.role == "assistant"
            ]
        return run_texts, history, added

    run_texts, history, added = asyncio.run(scenario())
    for texts in (run_texts, history, added):
        assert texts == ["Danke. Wie ist Ihre Rückrufnummer?"], texts
        assert not any("aufgenommen" in t for t in texts)


# --- latency: the delay the filter adds before the first spoken sentence -----------------


def _release_index_gate(tokens: list[str]) -> int:
    """Index of the token after which the filter releases the first sentence."""
    from praxis_telefon.agent.claim_gate import FilterContext, SentenceFilter, filter_claims

    released: list[int] = []

    async def run() -> None:
        # The day in the examples is known from a tool result, as in a real call.
        sf = SentenceFilter(FilterContext(tool_texts=(SLOTS,)))
        index = {"i": -1}

        async def stream() -> AsyncIterator[ChatChunk]:
            for i, chunk in enumerate(_chunks(tokens)):
                index["i"] = i
                yield chunk

        async for out in filter_claims(stream(), sf):
            if isinstance(out, str) and not released:
                released.append(index["i"])

    asyncio.run(run())
    return released[0]


def _release_index_livekit(tokens: list[str]) -> int:
    """Same for LiveKit's sentence tokenizer, which feeds ElevenLabs (auto_mode)."""
    from livekit.agents.tokenize import blingfire

    async def run() -> int:
        stream = blingfire.SentenceTokenizer().stream()
        for i, tok in enumerate(tokens):
            stream.push_text(tok)
            try:
                await asyncio.wait_for(stream.__anext__(), 0.01)
                return i
            except TimeoutError:
                continue
        return len(tokens)

    return asyncio.run(run())


def test_filter_adds_no_delay_before_the_first_sentence() -> None:
    """Measured in streamed tokens after the end of the first sentence; reported in ms
    at MS_PER_TOKEN. Compared with LiveKit's tokenizer that already gates ElevenLabs."""
    ms_per_token = 10.0  # ~100 tokens/s, a typical streaming rate for a small LLM
    replies = [
        "Gerne, ich schaue nach. Einen Moment bitte, ich suche einen Termin.",
        "Am Dienstag um 8 Uhr bei Dr. Weber ist frei. Passt Ihnen das?",
        "Das tut mir leid. Wenn es dringend ist, rufen Sie die eins eins sechs an.",
    ]
    for reply in replies:
        tokens = _words(reply)
        first_len = len(take_sentences(reply)[0][0].rstrip())
        sentence_end = next(
            i for i in range(len(tokens)) if len("".join(tokens[: i + 1])) >= first_len
        )
        gate = _release_index_gate(tokens) - sentence_end
        livekit = _release_index_livekit(tokens) - sentence_end
        # The TTS waits for LiveKit's tokenizer anyway, so only a later release costs time.
        extra_ms = max(0, gate - livekit) * ms_per_token
        print(
            f"\n{reply[:30]!r}: filter +{gate} token(s), LiveKit tokenizer +{livekit}; "
            f"extra before TTS: {extra_ms:.0f} ms"
        )
        assert gate <= livekit
        assert gate == 1  # released as soon as the next sentence starts


def test_claim_check_cost_per_sentence() -> None:
    import time

    from praxis_telefon.agent.claim_check import ClaimTracker

    tracker = ClaimTracker()
    sentence = "Am Dienstag, den 10. November, um 8 Uhr bei Dr. Weber ist noch etwas frei."
    n = 2000
    started = time.perf_counter()
    for _ in range(n):
        tracker.unbacked_sentence(sentence)
    per_sentence_us = (time.perf_counter() - started) / n * 1e6
    print(f"\nclaim check: {per_sentence_us:.1f} µs per sentence")
    assert per_sentence_us < 1000


def test_read_back_with_claim_word_waits_for_the_question() -> None:
    """The one case that costs time: a sentence with a claim word is held until the next
    sentence shows it is a read-back. Reported in tokens and ms at 10 ms per token."""
    reply = "Dienstag um 8 Uhr bei Dr. Weber, gebucht auf Max Mustermann. Stimmt das so?"
    tokens = _words(reply)
    first_len = len(take_sentences(reply)[0][0].rstrip())
    sentence_end = next(i for i in range(len(tokens)) if len("".join(tokens[: i + 1])) >= first_len)
    gate = _release_index_gate(tokens) - sentence_end
    livekit = _release_index_livekit(tokens) - sentence_end
    extra = max(0, gate - livekit)
    print(f"\nread-back held: +{extra} token(s) = {extra * 10:.0f} ms before TTS")
    assert gate == len(tokens) - 1 - sentence_end  # released together with the question


# --- regression: find_booking must not back a "book" claim -----------------------------


def test_new_booking_claim_after_only_find_booking_is_blocked(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The caller looked up an existing appointment, then asked for a new one; book was
    # never called. "gebucht" must not pass just because find_booking succeeded.
    fake = ScriptedLLM(
        _words("Alles klar. Ich habe Ihren neuen Termin gebucht."),
        _words("Welcher Tag passt Ihnen für den neuen Termin?"),
    )
    _install(monkeypatch, fake)
    text, _ = _reply(agent, ctx=_ctx_with_success(agent, "find_booking"))
    assert "gebucht" not in text
    assert text == "Alles klar. Welcher Tag passt Ihnen für den neuen Termin?"


def test_reading_out_an_existing_appointment_is_not_a_claim(
    agent: Receptionist, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = "Ihr Termin ist am Dienstag, 10. November, um 9 Uhr bei Dr. Weber."
    fake = ScriptedLLM(_words(reply))
    _install(monkeypatch, fake)
    found = "Gefundene Termine:\nbooking_id=1: Dienstag, 10. November, um 9 Uhr, bei Dr. Weber"
    text, _ = _reply(agent, ctx=_ctx_with_success(agent, "find_booking", found))
    assert text == reply
    assert len(fake.contexts) == 1


# --- regression: live run 2026-11-07, book_move_cancel_in_one_call[0] -------------------


def test_read_back_with_one_sentence_before_the_question_is_spoken() -> None:
    # The move read-back was blocked as claim:move_booking: the confirmation question
    # came two sentences after the claim, with the name and date of birth between.
    from praxis_telefon.agent.claim_gate import FilterContext, SentenceFilter

    sentences = [
        "Das ist Mittwoch, der 11. November, um 8 Uhr bei Dr. Brandt, und Ihr jetziger "
        "Termin am Dienstag, dem 10. November, um 8 Uhr bei Dr. Brandt wird dorthin verlegt. ",
        "Für Max Mustermann, geboren am ersten Februar neunzehnhundertneunundsiebzig. ",
        "Soll ich das so ändern?",
    ]
    callers = (
        "Ach, Dienstag passt doch nicht. Können Sie den Termin auf Mittwoch vormittags "
        "verschieben?",
        "Der erste Termin passt mir.",
    )
    found = "Gesucht: Mittwoch, 11. November.\nslot_id=12: Mittwoch, 11. November, um 8 Uhr"
    ctx = FilterContext(
        succeeded=frozenset({"book", "find_slots"}), caller_texts=callers, tool_texts=(found,)
    )
    sf = SentenceFilter(ctx)
    released = [s for sentence in sentences for s in sf.push(sentence)]
    sf.finish()
    assert released == sentences
    assert sf.blocked == []


def test_claim_without_a_question_after_one_more_sentence_is_still_blocked() -> None:
    from praxis_telefon.agent.claim_gate import FilterContext, SentenceFilter

    sentences = [
        "Ihr Termin wurde auf Mittwoch verlegt. ",
        "Für Max Mustermann. ",
        "Kann ich sonst noch etwas für Sie tun?",
    ]
    ctx = FilterContext(succeeded=frozenset({"book"}), caller_texts=("Geht auch Mittwoch?",))
    sf = SentenceFilter(ctx)
    released = [s for sentence in sentences for s in sf.push(sentence)]
    sf.finish()
    assert "Ihr Termin wurde auf Mittwoch verlegt. " not in released
    assert released == sentences[1:]  # the sentence between is spoken after all
    assert sf.blocked == ["claim:move_booking"]


def test_sentence_held_after_a_claim_at_the_end_of_a_reply_is_spoken() -> None:
    # With the read-back gap, a clean sentence may still be held when the stream ends.
    from praxis_telefon.agent.claim_gate import FilterContext, SentenceFilter, filter_claims

    reply = "Ihr Termin ist gebucht. Vielen Dank für Ihren Anruf."
    sf = SentenceFilter(FilterContext())

    async def run() -> str:
        out = []

        async def stream() -> AsyncIterator[ChatChunk]:
            for chunk in _chunks(_words(reply)):
                yield chunk

        async for chunk in filter_claims(stream(), sf):
            if isinstance(chunk, str):
                out.append(chunk)
        return "".join(out)

    assert asyncio.run(run()) == "Vielen Dank für Ihren Anruf."
    assert sf.blocked == ["claim:book"]
