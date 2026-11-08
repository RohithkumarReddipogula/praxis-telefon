"""Consent before book, move_booking and cancel_booking, checked in code.

Bug, cloud browser test 2026-11-08: the agent called book right after the caller gave
name and date of birth, without reading back and without a yes, and said "Ihr Termin ist
am ...". The prompt rule ("Bestätige genau einmal, direkt vor book ...") was not enough.

Consent: the agent's last message before the caller's latest line ends with a confirmation
question ("Stimmt das so?", "Soll ich ...?", "Is that right?"), and the caller's latest
line is a clear yes. Polite forms count ("Ja, gerne", "Ja, bitte", "Jawohl", "Genau so",
"Ja, passt", "Yes, please", "Sure"); any "nein", "aber", "lieber", "doch nicht", "no",
"but", "rather", "maybe" makes it no consent (user decision 2026-11-08).
"""

from __future__ import annotations

import re

from livekit.agents import llm

from praxis_telefon.agent.claim_check import is_confirmation_question, split_sentences
from praxis_telefon.text import normalize

# User rule 2026-11-08 (call 1: "Ja, tus." was refused): a line starting with "Ja" or
# "Yes" is a yes whatever follows; also the listed polite and short forms at the start.
_YES = re.compile(
    r"^(?:ja|jawohl|yes|yeah|yep|okay|ok|in ordnung|machen sie das|tun sie das|go ahead|"
    r"genau|richtig|stimmt|passt|korrekt|einverstanden|gerne?|klar|natuerlich|sicher|sure|"
    r"correct|right|absolutely|of course|that s right)\b"
)
_NOT_YES = re.compile(r"\b(?:nein|aber|lieber|doch nicht|nicht|no|but|rather)\b")


def is_clear_yes(text: str) -> bool:
    n = normalize(text).strip()
    return bool(_YES.match(n)) and not _NOT_YES.search(n)


def confirmed(chat_ctx: llm.ChatContext) -> bool:
    """The caller's latest line is a clear yes to the agent's read-back question."""
    messages = [i for i in chat_ctx.items if isinstance(i, llm.ChatMessage)]
    last_user = next(
        (k for k in range(len(messages) - 1, -1, -1) if messages[k].role == "user"), None
    )
    if last_user is None:
        return False
    before = next((m for m in reversed(messages[:last_user]) if m.role == "assistant"), None)
    if before is None:
        return False
    # The last question decides, even with a sentence after it ("Should I book ...?
    # Please answer yes or no.", harness run 2026-11-08).
    questions = [q for q in split_sentences(before.text_content or "") if q.rstrip().endswith("?")]
    asked = bool(questions) and is_confirmation_question(questions[-1])
    return asked and is_clear_yes(messages[last_user].text_content or "")
