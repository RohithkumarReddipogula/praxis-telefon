"""Shared harness for live tests: the real Receptionist and LLM in LiveKit text mode.

Runs against an in-memory synthetic database with a fixed clock (Monday 9 November 2026,
07:00 Berlin). The scripted caller answers what the agent asks: a slot choice when
slots are offered, "Ja" to a confirmation, its name and date of birth when asked
(counted as ``identity_asked``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
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

from praxis_telefon.agent import claim_gate, main
from praxis_telefon.agent.claim_check import ClaimTracker
from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.speech_filters import (
    day_names,
    detect_language,
    is_new_or_existing_question,
)
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.db.schema import connect
from praxis_telefon.db.seed import seed
from praxis_telefon.tools import scheduling
from tests.conftest import DOB, MONDAY, NAME, NOW, at, slot_at

Setup = Callable[[AgentSession[Any], Receptionist], Awaitable[None]]


class _SafetyLog(logging.Handler):
    """Keeps the JSON records of praxis_telefon.safety (reasons only, no text)."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(ValueError):
            self.records.append(json.loads(record.getMessage()))


_SAFETY_LOG = _SafetyLog()
_safety_logger = logging.getLogger("praxis_telefon.safety")
_safety_logger.addHandler(_SAFETY_LOG)
_safety_logger.setLevel(logging.INFO)

# Every sentence the LLM produced, before filtering, grouped per filter run. Collected
# only when LIVE_CORPUS is set; tests/live/conftest.py writes it with the test outcome.
CORPUS: dict[int, dict[str, Any]] = {}


def _collect(sf: claim_gate.SentenceFilter, sentence: str) -> None:
    group = CORPUS.setdefault(id(sf), {"stage": sf.stage, "ctx": sf.ctx.to_json(), "sentences": []})
    group["sentences"].append(sentence)


if os.environ.get("LIVE_CORPUS"):
    claim_gate.SENTENCE_HOOK = _collect


def safety_records() -> list[dict[str, Any]]:
    return _SAFETY_LOG.records


RUNS = 3
MAX_REPLIES = 6
TURN_TIMEOUT_S = 60

YES = "Ja, das stimmt so."
FIRST_SLOT = "Der erste Termin passt mir."
NEW_APPOINTMENT = "Einen neuen Termin, bitte."

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
_ASKS_IDENTITY = re.compile(
    r"\bnamen?\b|geburtsdatum|geboren|date of birth|birthday|your name", re.I
)


@dataclass
class Call:
    events: list[RunEvent] = field(default_factory=list)
    identity_asked: int = 0
    intent_asked: int = 0  # "neuer oder bestehender Termin?"
    # (position in events, "speech_blocked:<reason>"), from the safety log
    filter_notes: list[tuple[int, str]] = field(default_factory=list)
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
        notes = sorted(self.filter_notes)
        for i, e in enumerate(self.events):
            lines += [f"  [{note}]" for pos, note in notes if pos == i]
            if isinstance(e, ChatMessageEvent):
                lines.append(f"{e.item.role}: {e.item.text_content}")
            elif isinstance(e, FunctionCallEvent):
                lines.append(f"  -> {e.item.name}({e.item.arguments})")
            elif isinstance(e, FunctionCallOutputEvent):
                flag = "ERROR " if e.item.is_error else ""
                lines.append(f"  <- {e.item.name}: {flag}{e.item.output}")
        lines += [f"  [{note}]" for pos, note in notes if pos >= len(self.events)]
        return "\n".join(lines)


# Replies in the language the agent is speaking.
REPLIES: dict[str, dict[str, str]] = {
    "yes": {"de": YES, "en": "Yes, that's right."},
    "new": {"de": NEW_APPOINTMENT, "en": "A new appointment, please."},
    "first_slot": {"de": FIRST_SLOT, "en": "The first one works for me."},
    "any_time": {"de": "Egal, jede Uhrzeit passt.", "en": "Any time is fine."},
    "any_doctor": {"de": "Egal, bei wem.", "en": "Any doctor is fine."},
    "any_both": {
        "de": "Egal, jede Uhrzeit und jeder Arzt passt.",
        "en": "Any time and any doctor is fine.",
    },
    "identity": {"de": "Mein Name ist {identity}.", "en": "My name is {identity}."},
}
_ASKS_TIME = re.compile(
    r"vormittag|nachmittag|tageszeit|uhrzeit|morning|afternoon|time of day|what time", re.I
)
_ASKS_DOCTOR = re.compile(
    r"welche[nrm]? (ärztin|arzt|doktor)|bestimmte[nrm]? (ärztin|arzt)|bei wem|"
    r"which doctor|doctor would you prefer|preferred doctor|particular doctor",
    re.I,
)


def caller_reply(call: Call, identity: str) -> str:
    """What the scripted caller says next, based on the agent's last reply."""
    text = call.last_agent_text()
    lang = detect_language(text) or "de"

    def say(key: str) -> str:
        return REPLIES[key][lang].format(identity=identity)

    if is_new_or_existing_question(text):
        call.intent_asked += 1
        return say("new")
    slots = call.last_slots_index()
    if slots > call.slot_chosen_after and _CHOICE.search(text):
        call.slot_chosen_after = slots
        return say("first_slot")
    time_q, doctor_q = bool(_ASKS_TIME.search(text)), bool(_ASKS_DOCTOR.search(text))
    if text.rstrip().endswith("?") and (time_q or doctor_q):
        return say("any_both" if time_q and doctor_q else "any_time" if time_q else "any_doctor")
    if _CONFIRM.search(text):
        return say("yes")
    if text.rstrip().endswith("?") and _ASKS_IDENTITY.search(text):
        call.identity_asked += 1
        return say("identity")
    return say("yes")


def self_computed_days(call: Call, caller_line: str) -> set[str]:
    """Weekdays/months the agent named before find_slots answered, beyond the caller's.

    Compared in English, so "Donnerstag" counts as named after "next Thursday".
    Regression (friday_tomorrow_closed): "morgen, also am Samstag" before find_slots.
    """
    early = " ".join(agent_texts_before(call, "find_slots"))
    return day_names(early) - day_names(caller_line)


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
                mark = len(_SAFETY_LOG.records)
                pending: RunResult[None] = session.run(user_input=text)
                result = await asyncio.wait_for(_wait(pending), TURN_TIMEOUT_S)
                call.events.extend(result.events)
                for record in _SAFETY_LOG.records[mark:]:
                    if record.get("event") in ("speech_blocked", "speech_fallback"):
                        note = f"{record['event']}:{record.get('reason')}"
                        call.filter_notes.append((len(call.events), note))

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


def agent_texts_before(call: Call, tool: str) -> list[str]:
    """Agent messages spoken before the first result of ``tool``."""
    texts: list[str] = []
    for e in call.events:
        if isinstance(e, FunctionCallOutputEvent) and e.item.name == tool:
            break
        if isinstance(e, ChatMessageEvent) and e.item.role == "assistant":
            texts.append(e.item.text_content or "")
    return texts
