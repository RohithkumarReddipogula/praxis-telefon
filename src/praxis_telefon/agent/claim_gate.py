"""Check every agent sentence before it is spoken.

The LLM's text is cut into sentences as it streams. Each sentence is checked against
what is known about the call (``FilterContext``) and dropped if it:
- claims a saved message, booking, move or cancellation whose tool has not succeeded in
  this call (``claim:<tool>``; bug live test human_request[2]),
- is the agent talking about itself: its tools, "the caller", a "Hinweis:" note,
  English while the caller speaks German (``meta:<why>``; bug naechsten_dienstag), or
  German narration about the request (``meta:narration``; live run arm A explicit_date-0),
- asks "neuer oder bestehender Termin?" although the caller named a day or it was
  already asked (``intent``; bug friday_tomorrow_closed),
- names a weekday or month that neither the caller, a tool result nor today's date
  named, i.e. a date the LLM worked out itself (``weekday``).

A sentence with only a claim is held until the next one: a following confirmation
question ("Stimmt das so?") makes it a read-back and both are released. A dropped
sentence is never yielded, so it reaches neither the TTS, nor the transcript, nor the
chat history: the LLM never sees it. Receptionist decides whether one retry is needed.

Latency: ElevenLabs (auto_mode) already waits for whole sentences, released by LiveKit's
sentence tokenizer one word after the sentence ends; this filter releases at the same
point, so it adds no delay there (tests/test_claim_gate.py measures it).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from livekit.agents import FlushSentinel, llm

from praxis_telefon.agent.claim_check import (
    BACKED_BY,
    is_confirmation_question,
    sentence_claims,
    take_sentences,
)
from praxis_telefon.agent.speech_filters import (
    call_language,
    is_new_or_existing_question,
    mentions_wanted_day,
    meta_reasons,
    unsourced_days,
)
from praxis_telefon.safety.phrases import Language

logger = logging.getLogger("praxis_telefon.safety")

Output = llm.ChatChunk | str | FlushSentinel


@dataclass(frozen=True)
class FilterContext:
    """What the filters know about the call so far."""

    succeeded: frozenset[str] = frozenset()  # tools that succeeded in this call
    caller_texts: tuple[str, ...] = ()
    tool_texts: tuple[str, ...] = ()
    today: str = ""  # "Heute ist Freitag, der 13. November 2026, ..."
    intent_asked: bool = False  # "neuer oder bestehender Termin?" already asked

    @property
    def language(self) -> Language:
        return call_language(self.caller_texts)

    @classmethod
    def from_chat(cls, chat_ctx: llm.ChatContext, today: str) -> FilterContext:
        succeeded: set[str] = set()
        callers: list[str] = []
        tools: list[str] = []
        asked = False
        for item in chat_ctx.items:
            if isinstance(item, llm.ChatMessage) and item.role == "user":
                callers.append(item.text_content or "")
            elif isinstance(item, llm.ChatMessage) and item.role == "assistant":
                asked = asked or is_new_or_existing_question(item.text_content or "")
            elif isinstance(item, llm.FunctionCallOutput):
                tools.append(item.output)
                if not item.is_error:
                    succeeded.add(item.name)
        return cls(frozenset(succeeded), tuple(callers), tuple(tools), today, asked)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["succeeded"] = sorted(self.succeeded)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FilterContext:
        return cls(
            frozenset(data["succeeded"]),
            tuple(data["caller_texts"]),
            tuple(data["tool_texts"]),
            data["today"],
            data["intent_asked"],
        )


def check_sentence(sentence: str, ctx: FilterContext) -> list[str]:
    """Reasons to drop the sentence; empty if it may be spoken."""
    reasons = [
        f"claim:{action}"
        for action in sorted(sentence_claims(sentence))
        if not BACKED_BY[action] & ctx.succeeded
    ]
    reasons += [f"meta:{why}" for why in meta_reasons(sentence, ctx.language)]
    if is_new_or_existing_question(sentence) and (
        ctx.intent_asked or any(mentions_wanted_day(t) for t in ctx.caller_texts)
    ):
        reasons.append("intent")
    if unsourced_days(sentence, (*ctx.caller_texts, *ctx.tool_texts, ctx.today)):
        reasons.append("weekday")
    return reasons


# Set by the live-test harness only, to collect every sentence the LLM produced.
SentenceHook = Callable[["SentenceFilter", str], None]
SENTENCE_HOOK: SentenceHook | None = None


@dataclass
class SentenceFilter:
    ctx: FilterContext
    stage: str = "first"  # "first" or "retry", for the log
    spoken: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)  # reasons, in order, unique
    tool_calls: set[str] = field(default_factory=set)
    _held: tuple[str, list[str]] | None = None

    def push(self, sentence: str) -> list[str]:
        if SENTENCE_HOOK is not None:
            SENTENCE_HOOK(self, sentence)
        out: list[str] = []
        reasons = check_sentence(sentence, self.ctx)
        if self._held is not None:
            held, held_reasons = self._held
            self._held = None
            if not reasons and is_confirmation_question(sentence):  # a read-back
                out += [held, sentence]
                self.spoken += out
                return out
            self._block(held_reasons)
        if reasons and all(r.startswith("claim:") for r in reasons):
            self._held = (sentence, reasons)
        elif reasons:
            self._block(reasons)
        else:
            self.spoken.append(sentence)
            out.append(sentence)
        return out

    def finish(self) -> None:
        if self._held is not None:
            self._block(self._held[1])
            self._held = None

    def _block(self, reasons: list[str]) -> None:
        for reason in reasons:
            # The reason only, never the sentence.
            record = {"event": "speech_blocked", "reason": reason, "stage": self.stage}
            logger.info(json.dumps(record, separators=(",", ":")))
            if reason not in self.blocked:
                self.blocked.append(reason)

    @property
    def claims(self) -> list[str]:
        return [r.split(":", 1)[1] for r in self.blocked if r.startswith("claim:")]

    def needs_retry(self) -> bool:
        """Whether the reply as filtered would leave the caller without a real answer."""
        if any(not (BACKED_BY[a] & self.tool_calls) for a in self.claims):
            return True  # an unbacked claim and the tool is not even being called
        if "intent" in self.blocked and not self.tool_calls:
            return True  # the dropped question was the reply
        return bool(self.blocked) and not self.spoken and not self.tool_calls


def _without_content(chunk: llm.ChatChunk) -> llm.ChatChunk | None:
    """The chunk minus its text, or None if nothing else is in it."""
    delta = chunk.delta
    assert delta is not None
    if not (delta.tool_calls or delta.extra or chunk.usage):
        return None
    return chunk.model_copy(update={"delta": delta.model_copy(update={"content": None})})


async def filter_claims(stream: AsyncIterable[Output], sf: SentenceFilter) -> AsyncIterator[Output]:
    """Pass the LLM stream through; text goes out sentence by sentence, checked."""
    buffer = ""
    async for chunk in stream:
        if isinstance(chunk, str):
            text = chunk
        elif isinstance(chunk, llm.ChatChunk) and chunk.delta is not None:
            sf.tool_calls.update(c.name for c in chunk.delta.tool_calls)
            text = chunk.delta.content or ""
            rest = _without_content(chunk) if text else chunk
            if rest is not None:
                yield rest
        else:
            yield chunk  # FlushSentinel, usage-only chunks
            continue
        if not text:
            continue
        buffer += text
        sentences, buffer = take_sentences(buffer)
        for sentence in sentences:
            for out in sf.push(sentence):
                yield out
    if buffer.strip():
        for out in sf.push(buffer):
            yield out
    sf.finish()
