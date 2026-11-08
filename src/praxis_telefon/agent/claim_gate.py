"""Check every agent sentence before it is spoken.

The LLM's text is cut into sentences as it streams. Each sentence is checked against
what is known about the call (``FilterContext``) and dropped if it:
- claims a saved message, booking, move or cancellation whose tool has not succeeded in
  this call (``claim:<tool>``; bug live test human_request[2]),
- is the agent talking about itself: its tools, "the caller", a "Hinweis:" note,
  a sentence not in the call's language (``meta:language``; bug naechsten_dienstag), or
  narration about the request (``meta:narration``; live run arm A explicit_date-0),
- asks "neuer oder bestehender Termin?" although the caller named a day or it was
  already asked (``intent``; bug friday_tomorrow_closed),
- names a weekday or month that neither the caller, a tool result nor today's date
  named, i.e. a date the LLM worked out itself (``weekday``),
- states a practice fact (opening hours, address, parking, ...) that practice_info has
  not backed in this call (``fact:<kind>``; a closed day is also backed by find_slots),
- asks for another person's date of birth outside a new booking (``privacy``),
- tells the caller which word to answer with ("Bitte sagen Sie Ja oder Nein", "Please
  answer yes or no", "I need a clear yes"; ``demand``).

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
import re
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from livekit.agents import FlushSentinel, llm

from praxis_telefon.agent import call_language, caller_identity
from praxis_telefon.agent.claim_check import (
    BACKED_BY,
    is_confirmation_question,
    sentence_claims,
    take_sentences,
)
from praxis_telefon.agent.speech_filters import (
    demands_keyword_answer,
    is_new_or_existing_question,
    mentions_wanted_day,
    meta_reasons,
    practice_facts,
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
    # Decided by Receptionist from the STT finals (agent/call_language.py); None: from
    # caller_texts by the same rule (typed input, replayed live corpus).
    decided_language: Language | None = None

    @property
    def language(self) -> Language:
        if self.decided_language is not None:
            return self.decided_language
        return call_language.from_texts(self.caller_texts).language

    @classmethod
    def from_chat(
        cls, chat_ctx: llm.ChatContext, today: str, language: Language | None = None
    ) -> FilterContext:
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
        return cls(frozenset(succeeded), tuple(callers), tuple(tools), today, asked, language)

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
            data.get("decided_language"),
        )


# Privacy (M4 baseline 2026-11-07, injection_other_patient): asking for another person's
# date of birth to look them up ("Wie lautet das Geburtsdatum von Erika Mustermann?").
# Allowed for the caller's own name, and while booking a new appointment (find_slots has
# run), where someone else's date of birth is needed.
_WHO = r"(?P<who>[A-ZÄÖÜ][\wäöüß-]+(?:\s+[A-ZÄÖÜ][\wäöüß-]+)*|(?i:the patient))"
_THIRD_PARTY_DOB = (
    re.compile(
        r"(?i:\b(?:geburtsdatum|geburtstag|date of birth|birthday)\b.{0,20}?\b(?:von|der|des|of))"
        r"\s+(?:(?i:frau|herrn?|mr|mrs|ms)\.?\s+)?" + _WHO
    ),
    re.compile(_WHO + r"'s\s+(?i:date of birth|birthday)"),
)


def asks_third_party_dob(sentence: str, ctx: FilterContext) -> bool:
    if "find_slots" in ctx.succeeded:
        return False
    own = caller_identity.self_names(ctx.caller_texts)
    for pattern in _THIRD_PARTY_DOB:
        for m in pattern.finditer(sentence):
            if caller_identity.name_key(m["who"]) not in own:
                return True
    return False


# Practice facts (FAQ): only practice_info backs them; a closed day also find_slots, whose
# result says "Samstag, 14. November: Praxis geschlossen" (user, 2026-11-07).
FACT_BACKED_BY: dict[str, frozenset[str]] = {"closed": frozenset({"practice_info", "find_slots"})}
_FACT_DEFAULT = frozenset({"practice_info"})


def check_sentence(sentence: str, ctx: FilterContext) -> list[str]:
    """Reasons to drop the sentence; empty if it may be spoken."""
    reasons = [
        f"claim:{action}"
        for action in sorted(sentence_claims(sentence))
        if not BACKED_BY[action] & ctx.succeeded
    ]
    reasons += [
        f"fact:{family}"
        for family in sorted(practice_facts(sentence))
        if not FACT_BACKED_BY.get(family, _FACT_DEFAULT) & ctx.succeeded
    ]
    reasons += [f"meta:{why}" for why in meta_reasons(sentence, ctx.language)]
    if asks_third_party_dob(sentence, ctx):
        reasons.append("privacy")
    if demands_keyword_answer(sentence):
        reasons.append("demand")
    if is_new_or_existing_question(sentence) and (
        ctx.intent_asked or any(mentions_wanted_day(t) for t in ctx.caller_texts)
    ):
        reasons.append("intent")
    if unsourced_days(sentence, (*ctx.caller_texts, *ctx.tool_texts), ctx.today):
        reasons.append("weekday")
    return reasons


# Set by the live-test harness only, to collect every sentence the LLM produced.
SentenceHook = Callable[["SentenceFilter", str], None]
SENTENCE_HOOK: SentenceHook | None = None


READ_BACK_GAP = 1  # clean sentences allowed between a claim and its confirmation question


@dataclass
class SentenceFilter:
    ctx: FilterContext
    stage: str = "first"  # "first" or "retry", for the log
    spoken: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)  # reasons, in order, unique
    tool_calls: set[str] = field(default_factory=set)
    # A claim sentence waiting for its confirmation question, the clean sentences said
    # after it (at most READ_BACK_GAP), and the claim's reasons.
    _held: tuple[str, list[str], list[str]] | None = None

    def push(self, sentence: str) -> list[str]:
        if SENTENCE_HOOK is not None:
            SENTENCE_HOOK(self, sentence)
        out: list[str] = []
        reasons = check_sentence(sentence, self.ctx)
        if self._held is not None:
            held, between, held_reasons = self._held
            self._held = None
            if not reasons and is_confirmation_question(sentence):  # a read-back
                out += [held, *between, sentence]
                self.spoken += out
                return out
            if not reasons and not sentence.strip().endswith("?") and len(between) < READ_BACK_GAP:
                # Live run 2026-11-07: "... wird dorthin verlegt. Für Max Mustermann, geboren
                # am .... Soll ich das so ändern?" The question may come one sentence later.
                self._held = (held, [*between, sentence], held_reasons)
                return out
            self._block(held_reasons)
            self.spoken += between
            out += between
        if reasons and all(r.startswith("claim:") for r in reasons):
            self._held = (sentence, [], reasons)
        elif reasons:
            self._block(reasons)
        else:
            self.spoken.append(sentence)
            out.append(sentence)
        return out

    def finish(self) -> list[str]:
        """Blocks a claim still waiting; returns the clean sentences held after it."""
        if self._held is None:
            return []
        _, between, held_reasons = self._held
        self._held = None
        self._block(held_reasons)
        self.spoken += between
        return between

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
        if any(r.startswith("fact:") for r in self.blocked) and (
            "practice_info" not in self.tool_calls
        ):
            return True  # a practice fact from memory; practice_info is not being called
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
    for out in sf.finish():
        yield out
