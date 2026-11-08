"""What the call is waiting for: book, move, cancel or the callback number.

After a blocked "answer yes or no" the agent asks a natural question that fits the
situation (user, 2026-11-08: "Soll ich den Termin so eintragen?", "... so verschieben?",
"... absagen?", "Ist die Nummer so richtig?", otherwise "Ist das so richtig?"). The LLM's
text does not say what is pending, so it is worked out from the conversation, in this
order:
1. a refused action tool (book, move_booking, cancel_booking, take_message) after the last
   successful one;
2. the callback number: the agent's last message talks about the number, or the caller's
   last line contains one;
3. the caller's words since the last success: cancel, or move;
4. find_slots has run since the last success: book;
5. nothing.
"""

from __future__ import annotations

import re
from typing import Literal

from livekit.agents import llm

from praxis_telefon.tools.spoken_numbers import phone_numbers_in

Pending = Literal["book", "move", "cancel", "callback"]

_ACTIONS: dict[str, Pending] = {
    "book": "book",
    "move_booking": "move",
    "cancel_booking": "cancel",
    "take_message": "callback",
}
_MOVE = re.compile(
    r"verschieb|umbuch|umlegen|verleg|reschedul|\bmove\b|change (?:my|the) appointment", re.I
)
_CANCEL = re.compile(r"absag|storni|cancel", re.I)
_NUMBER_TALK = re.compile(r"nummer|rückruf|rueckruf|number|call(?:ing)? you back|callback", re.I)


def pending_action(chat_ctx: llm.ChatContext) -> Pending | None:
    items = list(chat_ctx.items)
    outputs = [(i, it) for i, it in enumerate(items) if isinstance(it, llm.FunctionCallOutput)]
    last_ok = max((i for i, it in outputs if it.name in _ACTIONS and not it.is_error), default=-1)

    for i, it in reversed(outputs):  # 1: a refused action tool since the last success
        if i > last_ok and it.name in _ACTIONS and it.is_error:
            return _ACTIONS[it.name]

    recent = items[last_ok + 1 :]
    messages = [m for m in recent if isinstance(m, llm.ChatMessage)]
    callers = [m.text_content or "" for m in messages if m.role == "user"]
    agents = [m.text_content or "" for m in messages if m.role == "assistant"]

    if (callers and phone_numbers_in(callers[-1])) or (agents and _NUMBER_TALK.search(agents[-1])):
        return "callback"  # 2

    for line in reversed(callers):  # 3: newest line with a move or cancel word
        move, cancel = bool(_MOVE.search(line)), bool(_CANCEL.search(line))
        if cancel and not move:
            return "cancel"
        if move:
            return "move"

    if any(isinstance(m, llm.FunctionCall) and m.name == "find_slots" for m in recent):
        return "book"  # 4
    return None
