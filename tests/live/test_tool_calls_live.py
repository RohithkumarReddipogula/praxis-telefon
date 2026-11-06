"""Live tool-calling regression tests. Skipped by default; run with:

    uv run pytest -m live tests/live/test_tool_calls_live.py -v

They drive the real Receptionist through LiveKit's text-only test mode (no audio) with the
real LLM from .env, against an in-memory synthetic database with a fixed clock
(Monday 9 November 2026, 07:00 Berlin). Each scenario runs RUNS times, because one pass
proves little for an LLM. On failure the transcript is printed (synthetic data only).

The scripted caller answers what the agent asks: a slot choice when slots are offered,
"Ja" to a confirmation, its name and date of birth when asked (counted, see below).

Bugs from the voice tests on 2026-11-06:
- Call 1: the caller confirmed a move and a cancel, but move_booking / cancel_booking
  were never called; the database did not change.
- Call 2: a booking request ran find_slots twice, find_booking, then take_message twice
  (two identical staff messages) instead of book.
From the first live run (14 passed, 4 failed):
- move[2]: the agent asked for the full name five times although it was given.
- move[1]: an extra name confirmation; the scripted slot choice came at the wrong moment.
- book_move_cancel[0], says_existing_patient[2]: read-back questions flagged as claims.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
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

pytestmark = pytest.mark.live

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


def _caller_reply(call: Call, identity: str) -> str:
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


def _new_db(*, erika_booked: bool) -> tuple[sqlite3.Connection, int | None]:
    conn = connect(":memory:")
    seed(conn, start=MONDAY, booked_share=0.0)
    if not erika_booked:
        return conn, None
    tuesday_9 = slot_at(conn, at(MONDAY.replace(day=10), 9), "Weber")
    result = scheduling.book(conn, now=NOW, slot_id=tuesday_9, patient_name=NAME, date_of_birth=DOB)
    return conn, result.booking_id


def _converse(conn: sqlite3.Connection, identity: str, steps: list[Step]) -> list[Call]:
    """Run one call; returns the events per step."""
    from dotenv import load_dotenv

    from praxis_telefon.config import Settings

    load_dotenv()
    settings = Settings.from_env()
    calls = [Call() for _ in steps]

    async def run() -> None:
        llm = openai.LLM.with_openrouter(**main.llm_options(settings), api_key=settings.llm_api_key)
        async with llm, AgentSession(llm=llm) as session:
            await session.start(Receptionist(ToolService(conn, lambda: NOW)))

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
                        await say(call, _caller_reply(call, identity))

    asyncio.run(run())
    return calls


async def _wait(pending: RunResult[None]) -> RunResult[None]:
    return await pending


def _transcript(calls: list[Call]) -> str:
    return "\n--- next step ---\n".join(c.transcript() for c in calls)


def _booking_row(conn: sqlite3.Connection, booking_id: int) -> sqlite3.Row:
    row: sqlite3.Row | None = conn.execute(
        "SELECT slot_id, status FROM bookings WHERE id = ?", (booking_id,)
    ).fetchone()
    assert row is not None
    return row


def _count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def _ok(tool: str) -> Callable[[Call], bool]:
    return lambda c: c.succeeded(tool) > 0


BOOKING_OPENINGS = {
    "new_patient": f"Guten Tag, hier ist {MAX}. Ich hätte gern einen Termin am Dienstag "
    "vormittags.",
    # In the voice test find_booking ran first (no match), then take_message.
    "says_existing_patient": f"Guten Tag, hier ist {MAX}. Ich bin schon Patient bei Ihnen "
    "und bräuchte einen Termin am Dienstag vormittags.",
}


# --- Call 1: move / cancel confirmed but never executed --------------------------------


@pytest.mark.parametrize("run", range(RUNS))
def test_move_after_yes_calls_move_booking(run: int) -> None:
    conn, booking_id = _new_db(erika_booked=True)
    assert booking_id is not None
    old_slot = int(_booking_row(conn, booking_id)["slot_id"])

    calls = _converse(
        conn,
        ERIKA,
        [
            Step(
                (
                    f"Guten Tag, hier ist {ERIKA}. Ich möchte meinen Termin am Dienstag "
                    "verschieben, auf Mittwoch, den elften November, vormittags.",
                ),
                until=_ok("move_booking"),
            )
        ],
    )

    info = _transcript(calls)
    assert calls[0].succeeded("move_booking") == 1, info
    assert int(_booking_row(conn, booking_id)["slot_id"]) != old_slot, info
    assert calls[0].unbacked_claims() == [], info
    assert calls[0].identity_asked == 0, "asked again for name/date of birth\n" + info


@pytest.mark.parametrize("run", range(RUNS))
def test_cancel_after_yes_calls_cancel_booking(run: int) -> None:
    conn, booking_id = _new_db(erika_booked=True)
    assert booking_id is not None

    calls = _converse(
        conn,
        ERIKA,
        [
            Step(
                (f"Guten Tag, hier ist {ERIKA}. Ich möchte meinen Termin am Dienstag absagen.",),
                until=_ok("cancel_booking"),
            )
        ],
    )

    info = _transcript(calls)
    assert calls[0].succeeded("cancel_booking") == 1, info
    assert _booking_row(conn, booking_id)["status"] == "cancelled", info
    assert calls[0].unbacked_claims() == [], info
    assert calls[0].identity_asked == 0, "asked again for name/date of birth\n" + info


@pytest.mark.parametrize("run", range(RUNS))
def test_book_move_cancel_in_one_call_no_false_success_claim(run: int) -> None:
    """The voice-test flow: book, then move, then cancel, all in one call."""
    conn, _ = _new_db(erika_booked=False)

    calls = _converse(
        conn,
        MAX,
        [
            Step(
                (BOOKING_OPENINGS["new_patient"],),
                until=_ok("book"),
            ),
            Step(
                (
                    "Ach, Dienstag passt doch nicht. Können Sie den Termin auf Mittwoch "
                    "vormittags verschieben?",
                ),
                until=_ok("move_booking"),
            ),
            Step(
                ("Wissen Sie was, sagen Sie den Termin bitte ganz ab.",),
                until=_ok("cancel_booking"),
            ),
        ],
    )

    info = _transcript(calls)
    book, move, cancel = calls
    assert book.succeeded("book") == 1, info
    assert move.succeeded("move_booking") == 1, info
    assert cancel.succeeded("cancel_booking") == 1, info
    for step in calls:
        # Fresh tracker per step: an earlier book must not excuse a later "verschoben".
        assert step.unbacked_claims() == [], info
        assert step.identity_asked == 0, "asked again for name/date of birth\n" + info
    assert _count(conn, "SELECT count(*) FROM bookings WHERE status = 'active'") == 0, info


# --- Call 2: booking request ended in take_message, twice -------------------------------


@pytest.mark.parametrize("run", range(RUNS))
@pytest.mark.parametrize("opening", BOOKING_OPENINGS)
def test_booking_request_books_instead_of_take_message(opening: str, run: int) -> None:
    conn, _ = _new_db(erika_booked=False)

    calls = _converse(conn, MAX, [Step((BOOKING_OPENINGS[opening],), until=_ok("book"))])

    info = _transcript(calls)
    assert calls[0].calls("take_message") == 0, info
    assert _count(conn, "SELECT count(*) FROM staff_messages") == 0, info
    assert calls[0].succeeded("book") == 1, info
    assert calls[0].unbacked_claims() == [], info
    assert calls[0].identity_asked == 0, "asked again for name/date of birth\n" + info


@pytest.mark.parametrize("run", range(RUNS))
def test_take_message_saves_one_message(run: int) -> None:
    """A real callback request stores exactly one message, even after the goodbye."""
    conn, _ = _new_db(erika_booked=False)

    calls = _converse(
        conn,
        MAX,
        [
            Step(
                (
                    f"Guten Tag, hier ist {MAX}. Ich brauche ein neues Rezept, "
                    f"bitte rufen Sie mich zurück unter {PHONE_SPOKEN}.",
                ),
                until=_ok("take_message"),
            ),
            Step(("Danke, das war alles. Auf Wiederhören.",)),
        ],
    )

    info = _transcript(calls)
    assert sum(c.succeeded("take_message") for c in calls) == 1, info
    assert _count(conn, "SELECT count(*) FROM staff_messages") == 1, info
