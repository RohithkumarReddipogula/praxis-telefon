"""A fixed "Einen Moment, bitte." when a reply calls a tool without saying anything.

With reasoning "low" (docs/experiments/reasoning-leak.md) the LLM no longer says a short
preamble before a tool, so the caller heard nothing until the tool round trip was done:
caller line to first speech rose from 1.0 s to 2.1 s at p50. Code now says a fixed
phrase when a reply called a tool and said nothing, if nothing was said since the
caller's last line either. The phrase is not LLM text, so it cannot leak reasoning and
needs no filter.

"Nothing was said" is read from the conversation, like the safety state: LiveKit adds
the filler as an assistant message before the tool call items, so a second tool round in
the same turn sees it and stays silent (at most once per caller turn), and a discarded
preemptive run changes nothing.
"""

from __future__ import annotations

from livekit.agents import llm

from praxis_telefon.agent.claim_gate import SentenceFilter
from praxis_telefon.agent.prompts import TOOL_FILLER


def said_since_caller(chat_ctx: llm.ChatContext) -> bool:
    """The agent said something after the caller's last line (a filler included)."""
    for item in reversed(chat_ctx.items):
        if not isinstance(item, llm.ChatMessage):
            continue
        if item.role == "user":
            return False
        if item.role == "assistant" and (item.text_content or "").strip():
            return True
    return False


def is_tool_filler(text: str) -> bool:
    """The assistant message is (or starts with) the code-spoken filler."""
    return text.strip().startswith(tuple(TOOL_FILLER.values()))


class ToolFiller:
    """Says the phrase after a reply that called a tool and spoke nothing.

    Decided when the filtered reply ends, not when the tool call chunk arrives: the
    sentence filter holds a sentence until it is complete, while tool call chunks pass at
    once, so at that moment "Einen Moment, ich schaue nach." may still be held. A tool
    call is the last thing in the LLM stream, so the end comes right after it.

    One instance per reply, shared by the first try and the retry, so it speaks at most
    once; ``allowed`` is False if something was already said since the caller's line.
    """

    def __init__(self, phrase: str, *, allowed: bool) -> None:
        self.phrase = phrase
        self.pending = allowed
        self.spoken = False

    def after(self, sf: SentenceFilter) -> str | None:
        """The phrase to say once ``sf``'s reply is done, if any."""
        if sf.spoken:
            self.pending = False  # the reply spoke
            return None
        if not (self.pending and sf.tool_calls):
            return None
        self.pending = False
        self.spoken = True
        return self.phrase + " "
