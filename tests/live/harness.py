"""Shared harness for live tests: the real Receptionist and LLM in LiveKit text mode.

Runs against an in-memory synthetic database with a fixed clock (Monday 9 November 2026,
07:00 Berlin). The scripted caller answers what the agent asks: a slot choice when
slots are offered, "Ja" to a confirmation, its name and date of birth when asked
(counted as ``identity_asked``).
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from livekit.agents import AgentSession
from livekit.agents.voice.run_result import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
    RunResult,
)
from livekit.plugins import openai

from praxis_telefon.agent import main
from praxis_telefon.agent.claim_check import ClaimTracker
from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.db.schema import connect
from praxis_telefon.db.seed import seed
from praxis_telefon.tools import scheduling
from tests.conftest import DOB, MONDAY, NAME, NOW, at, slot_at

Setup = Callable[[AgentSession[Any], Receptionist], Awaitable[None]]

RUNS = 3
MAX_REPLIES = 6
TURN_TIMEOUT_S = 60

YES = "Ja, das stimmt so."
FIRST_SLOT = "Der erste Termin passt mir."

# Spoken like a caller would (STT output), not ISO.
ERIKA = "Erika Mustermann, geboren am fünfzehnten Juni neunzehnhundertfünfundachtzig"
MAX = "Max Mustermann, geboren am ersten Februar neunzehnhundertneunundsiebzig"
PHONE_SPOKEN = "null drei null, zwei drei eins zwei fünf, vier fünf sechs"

# How the scripted caller recognises what the agent asks.
_CONFIRM = re.compile(
    r"stimmt das|ist das (so )?(richtig|korrekt)|passt das|soll ich|einverstanden|"
    r"richtig so|darf ich",
    re.I,
)
_CHOICE = re.compile(r"\boder\b|\bwelche[rnms]?\b|passt (Ihnen|einer)", re.I)
_ASKS_IDENTITY = re.compile(r"\bnamen?\b|geburtsdatum|geboren", re.I)


@dataclass
class Call:
    events: list[RunEvent] = field(default_factory=list)
    identity_asked: int = 0
    slot_chosen_after: int = -1  # index of the find_slots output last answered with a choice

    def calls(self, tool: str) -> int:
        return sum(
            1 for e in self.events if isinstance(e, FunctionCallEvent) and e.item.name == tool
        )

    def succeeded(self, tool: str) -> int:
        return sum(
            1
            for e in self.events
            if isinstance(e, FunctionCallOutputEvent)
            and e.item.name == tool
            and not e.item.is_error
        )

    def unbacked_claims(self, tracker: ClaimTracker | None = None) -> list[str]:
        tracker = tracker or ClaimTracker()
        found: list[str] = []
        for e in self.events:
            if isinstance(e, FunctionCallOutputEvent):
                tracker.tool_result(e.item.name, ok=not e.item.is_error)
            elif isinstance(e, ChatMessageEvent) and e.item.role == "assistant":
                text = e.item.text_content or ""
                found += [f"{action}: {text}" for action in tracker.unbacked(text)]
        return found

    def last_agent_text(self) -> str:
        for e in reversed(self.events):
            if isinstance(e, ChatMessageEvent) and e.item.role == "assistant":
                return e.item.text_content or ""
        return ""

    def last_slots_index(self) -> int:
        for i in range(len(self.events) - 1, -1, -1):
            e = self.events[i]
            if isinstance(e, FunctionCallOutputEvent) and e.item.name == "find_slots":
                return i if not e.item.is_error and "slot_id=" in e.item.output else -1
        return -1

    def transcript(self) -> str:
        lines = []
        for e in self.events:
            if isinstance(e, ChatMessageEvent):
                lines.append(f"{e.item.role}: {e.item.text_content}")
            elif isinstance(e, FunctionCallEvent):
                lines.append(f"  -> {e.item.name}({e.item.arguments})")
            elif isinstance(e, FunctionCallOutputEvent):
                flag = "ERROR " if e.item.is_error else ""
                lines.append(f"  <- {e.item.name}: {flag}{e.item.output}")
        return "\n".join(lines)


def caller_reply(call: Call, identity: str) -> str:
    """What the scripted caller says next, based on the agent's last reply."""
    text = call.last_agent_text()
    slots = call.last_slots_index()
    if slots > call.slot_chosen_after and _CHOICE.search(text):
        call.slot_chosen_after = slots
        return FIRST_SLOT
    if _CONFIRM.search(text):
        return YES
    if text.rstrip().endswith("?") and _ASKS_IDENTITY.search(text):
        call.identity_asked += 1
        return f"Mein Name ist {identity}."
    return YES


@dataclass(frozen=True)
class Step:
    """Caller lines, then answers (at most MAX_REPLIES) until ``until`` holds."""

    lines: tuple[str, ...]
    until: Callable[[Call], bool] | None = None


def new_db(*, erika_booked: bool) -> tuple[sqlite3.Connection, int | None]:
    conn = connect(":memory:")
    seed(conn, start=MONDAY, booked_share=0.0)
    if not erika_booked:
        return conn, None
    tuesday_9 = slot_at(conn, at(MONDAY.replace(day=10), 9), "Weber")
    result = scheduling.book(conn, now=NOW, slot_id=tuesday_9, patient_name=NAME, date_of_birth=DOB)
    return conn, result.booking_id


def converse(
    conn: sqlite3.Connection,
    identity: str,
    steps: list[Step],
    *,
    setup: Setup | None = None,
    clock: Callable[[], datetime] | None = None,
) -> list[Call]:
    """Run one call; returns the events per step.

    ``setup`` runs after the session starts; ``clock`` replaces the fixed NOW.
    """
    from dotenv import load_dotenv

    from praxis_telefon.config import Settings

    load_dotenv()
    settings = Settings.from_env()
    calls = [Call() for _ in steps]

    async def run() -> None:
        llm = openai.LLM.with_openrouter(**main.llm_options(settings), api_key=settings.llm_api_key)
        async with llm, AgentSession(llm=llm) as session:
            agent = Receptionist(ToolService(conn, clock or (lambda: NOW)))
            await session.start(agent)
            if setup is not None:
                await setup(session, agent)

            async def say(call: Call, text: str) -> None:
                pending: RunResult[None] = session.run(user_input=text)
                result = await asyncio.wait_for(_wait(pending), TURN_TIMEOUT_S)
                call.events.extend(result.events)

            for step, call in zip(steps, calls, strict=True):
                for line in step.lines:
                    await say(call, line)
                if step.until is not None:
                    for _ in range(MAX_REPLIES):
                        if step.until(call):
                            break
                        await say(call, caller_reply(call, identity))

    asyncio.run(run())
    return calls


async def _wait(pending: RunResult[None]) -> RunResult[None]:
    return await pending


def transcript_of(calls: list[Call]) -> str:
    return "\n--- next step ---\n".join(c.transcript() for c in calls)


def booking_row(conn: sqlite3.Connection, booking_id: int) -> sqlite3.Row:
    row: sqlite3.Row | None = conn.execute(
        "SELECT slot_id, status FROM bookings WHERE id = ?", (booking_id,)
    ).fetchone()
    assert row is not None
    return row


def count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def ok(tool: str) -> Callable[[Call], bool]:
    return lambda c: c.succeeded(tool) > 0


def agent_texts(call: Call) -> list[str]:
    return [
        e.item.text_content or ""
        for e in call.events
        if isinstance(e, ChatMessageEvent) and e.item.role == "assistant"
    ]


def tool_outputs(call: Call, tool: str) -> list[str]:
    """Outputs of successful calls to ``tool``."""
    return [
        e.item.output
        for e in call.events
        if isinstance(e, FunctionCallOutputEvent) and e.item.name == tool and not e.item.is_error
    ]


def tool_arguments(call: Call, tool: str) -> list[str]:
    return [
        e.item.arguments
        for e in call.events
        if isinstance(e, FunctionCallEvent) and e.item.name == tool
    ]
