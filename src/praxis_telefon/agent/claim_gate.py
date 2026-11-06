"""Stop the agent from saying that something was saved before the tool succeeded.

Bug (live test human_request[2]): "Ich habe Ihre Nachricht für das Praxisteam
aufgenommen", but take_message was never called. The claim check only logged it.

The LLM's text is cut into sentences as it streams. A sentence is released only if it
does not claim a saved message, booking, move or cancellation (``claim_check``) whose
tool has not succeeded in this call. A claiming sentence is held until the next one: a
following confirmation question ("Stimmt das so?") makes it a read-back and both are
released; otherwise it is dropped. A dropped sentence is never yielded, so it reaches
neither the TTS, nor the transcript, nor the chat history: the LLM never sees it.

Latency: ElevenLabs (auto_mode) already waits for whole sentences, released by LiveKit's
sentence tokenizer one word after the sentence ends; this filter releases at the same
point, so it adds no delay there (tests/test_claim_gate.py measures it).
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field

from livekit.agents import FlushSentinel, llm

from praxis_telefon.agent.claim_check import (
    ClaimTracker,
    is_confirmation_question,
    take_sentences,
)

Output = llm.ChatChunk | str | FlushSentinel


@dataclass
class SentenceFilter:
    tracker: ClaimTracker
    spoken: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)  # actions, in order
    tool_calls: set[str] = field(default_factory=set)
    _held: tuple[str, list[str]] | None = None

    def push(self, sentence: str) -> list[str]:
        out: list[str] = []
        if self._held is not None:
            held, actions = self._held
            self._held = None
            if is_confirmation_question(sentence):  # read-back, not a claim
                out += [held, sentence]
                self.spoken += out
                return out
            self.blocked += [a for a in actions if a not in self.blocked]
        actions = self.tracker.unbacked_sentence(sentence)
        if actions:
            self._held = (sentence, actions)
            return out
        self.spoken.append(sentence)
        out.append(sentence)
        return out

    def finish(self) -> None:
        if self._held is not None:
            self.blocked += [a for a in self._held[1] if a not in self.blocked]
            self._held = None


def _without_content(chunk: llm.ChatChunk) -> llm.ChatChunk | None:
    """The chunk minus its text, or None if nothing else is in it."""
    delta = chunk.delta
    assert delta is not None
    if not (delta.tool_calls or delta.extra or chunk.usage):
        return None
    return chunk.model_copy(update={"delta": delta.model_copy(update={"content": None})})


async def filter_claims(stream: AsyncIterable[Output], sf: SentenceFilter) -> AsyncIterator[Output]:
    """Pass the LLM stream through; text goes out sentence by sentence, claims held back."""
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
