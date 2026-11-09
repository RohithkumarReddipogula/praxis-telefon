"""Public demo protections (M5): keyed caller hash, call and daily limits, demo messages.

User decisions 2026-11-08 (ElevenLabs credits: about 350 call minutes now, 10,000
credits a month from 7 December):
- at most 5 minutes per call, a warning one minute before the end;
- 3 calls per caller and day, 15 calls and 45 minutes per day overall (Europe/Berlin);
- withheld numbers are allowed and count against the overall limits only;
- caller numbers only as a keyed hash (HMAC-SHA256 with DEMO_HASH_KEY, a deployment
  secret): a plain hash of a phone number is easy to reverse by trying all numbers.

The counts live in a small SQLite file on the container's disk (DEMO_USAGE_DB), shared
by the job processes (each call runs in its own process). A container restart resets
them, so the hard spending caps stay at the providers (prepaid balances, key limits).
A running call counts with its maximum until it ends, so concurrent calls cannot
overrun the daily minutes.

Browser calls (M5/M6 item 2, user decisions 2026-11-09) form their own channel: 20 minutes
a day (so at most four calls of 5 minutes) and 60 minutes a month, counted only in minutes;
the web page limits visitors itself. Phone and web do not use each other's limits. When a
limit is reached, or the voice (TTS) fails, the agent first sets the participant attribute
``demo_status=busy`` so the page can show "the demo is busy today", then tries to speak,
then hangs up: it never fails silently.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import re
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from praxis_telefon.db.schema import PRACTICE_TZ

logger = logging.getLogger("praxis_telefon.turns")
Channel = Literal["phone", "web"]
MIN_CALL_S = 60  # less than this left for the day: the call is refused
WITHHELD = frozenset({"", "anonymous", "unknown", "restricted", "private", "withheld"})


@dataclass(frozen=True)
class DemoLimits:
    max_call_s: int = 300
    warn_before_s: int = 60
    calls_per_caller: int = 3
    calls_per_day: int = 15
    minutes_per_day: int = 45
    web_minutes_per_day: int = 20  # browser calls; 5 minutes each: at most four a day
    web_minutes_per_month: int = 60


# --- keyed caller hash ------------------------------------------------------------------


def _canonical(number: str) -> str:
    """+49... for German numbers in any spelling (030 ..., 0049 ..., +49 ...)."""
    digits = re.sub(r"[^\d+]", "", number)
    if digits.startswith("00"):
        digits = "+" + digits[2:]
    elif digits.startswith("0"):
        digits = "+49" + digits[1:]
    return digits


def caller_hash(number: str | None, key: str) -> str | None:
    """16 hex characters of HMAC-SHA256(key, number); None for a withheld number."""
    if number is None or number.strip().lower() in WITHHELD:
        return None
    canonical = _canonical(number)
    if not re.fullmatch(r"\+\d{6,15}", canonical):
        return None
    return hmac.new(key.encode(), canonical.encode(), hashlib.sha256).hexdigest()[:16]


def channel_of(attributes: Mapping[str, str]) -> Channel:
    """ "web" only for the exact attribute the token endpoint puts into the signed token
    (a phone caller cannot set participant attributes)."""
    return "web" if attributes.get("channel") == "web" else "phone"


# --- limits -------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallDecision:
    allowed: bool
    reason: str = ""  # caller_daily_limit | daily_calls | daily_minutes
    max_s: int = 0
    call_id: int | None = None


class CallLimiter:
    def __init__(self, path: Path, limits: DemoLimits) -> None:
        self._path = path
        self.limits = limits
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, day TEXT NOT NULL, "
                "caller TEXT, max_s INTEGER NOT NULL, duration_s INTEGER, "
                "channel TEXT NOT NULL DEFAULT 'phone')"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(calls)")}
            if "channel" not in columns:  # a file from before the web channel: rows are phone
                conn.execute("ALTER TABLE calls ADD COLUMN channel TEXT NOT NULL DEFAULT 'phone'")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=10, isolation_level=None)

    def start(self, caller: str | None, now: datetime, channel: Channel = "phone") -> CallDecision:
        """``caller``: the keyed hash, or None for a withheld number or a browser."""
        day = now.astimezone(PRACTICE_TZ).date().isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")  # one decision at a time across processes
            if channel == "web":
                decision = self._start_web(conn, day)
            else:
                decision = self._start_phone(conn, caller, day)
            conn.execute("COMMIT")
            return decision
        finally:
            conn.close()

    def _insert(
        self, conn: sqlite3.Connection, day: str, caller: str | None, max_s: int, channel: str
    ) -> CallDecision:
        cur = conn.execute(
            "INSERT INTO calls (day, caller, max_s, channel) VALUES (?, ?, ?, ?)",
            (day, caller, max_s, channel),
        )
        return CallDecision(True, "", max_s, cur.lastrowid)

    def _start_phone(self, conn: sqlite3.Connection, caller: str | None, day: str) -> CallDecision:
        lim = self.limits
        calls, used = conn.execute(
            "SELECT count(*), coalesce(sum(coalesce(duration_s, max_s)), 0) FROM calls "
            "WHERE day = ? AND channel = 'phone'",
            (day,),
        ).fetchone()
        mine = (
            conn.execute(
                "SELECT count(*) FROM calls WHERE day = ? AND caller = ? AND channel = 'phone'",
                (day, caller),
            ).fetchone()[0]
            if caller is not None
            else 0
        )
        left = lim.minutes_per_day * 60 - used
        if caller is not None and mine >= lim.calls_per_caller:
            return CallDecision(False, "caller_daily_limit")
        if calls >= lim.calls_per_day:
            return CallDecision(False, "daily_calls")
        if left < MIN_CALL_S:
            return CallDecision(False, "daily_minutes")
        return self._insert(conn, day, caller, min(lim.max_call_s, left), "phone")

    def _start_web(self, conn: sqlite3.Connection, day: str) -> CallDecision:
        lim = self.limits
        used_day = conn.execute(
            "SELECT coalesce(sum(coalesce(duration_s, max_s)), 0) FROM calls "
            "WHERE day = ? AND channel = 'web'",
            (day,),
        ).fetchone()[0]
        used_month = conn.execute(
            "SELECT coalesce(sum(coalesce(duration_s, max_s)), 0) FROM calls "
            "WHERE substr(day, 1, 7) = ? AND channel = 'web'",
            (day[:7],),
        ).fetchone()[0]
        left_month = lim.web_minutes_per_month * 60 - used_month
        left_day = lim.web_minutes_per_day * 60 - used_day
        if left_month < MIN_CALL_S:
            return CallDecision(False, "web_monthly_minutes")
        if left_day < MIN_CALL_S:
            return CallDecision(False, "web_daily_minutes")
        return self._insert(conn, day, None, min(lim.max_call_s, left_day, left_month), "web")

    def end(self, decision: CallDecision, duration_s: float) -> None:
        if decision.call_id is None:
            return
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE calls SET duration_s = ? WHERE id = ?",
                (min(round(duration_s), decision.max_s), decision.call_id),
            )
        finally:
            conn.close()


def call_schedule(max_s: int, warn_before_s: int) -> tuple[int | None, int]:
    """(seconds until the warning or None, seconds until the end)."""
    warn = max_s - warn_before_s
    return (warn if warn >= warn_before_s else None), max_s


# --- what the caller hears ----------------------------------------------------------------

DEMO_GREETING = (
    "Hausarztpraxis Dr. Weber, guten Tag. Dies ist eine Demo mit einer erfundenen Praxis; "
    "bitte nennen Sie keine echten Gesundheits- oder persönlichen Daten. "
    "This is a demo of a fictional practice; you can also speak English. "
    "Wie kann ich Ihnen helfen?"
)

MESSAGES = {
    "caller_daily_limit": (
        "Sie haben die Demo heute schon dreimal angerufen. Bitte versuchen Sie es morgen "
        "wieder. Sorry, you have reached today's limit for this demo. Auf Wiederhören."
    ),
    "daily_calls": (
        "Die Demo ist für heute ausgelastet. Bitte versuchen Sie es morgen wieder. "
        "Sorry, the demo is fully used for today. Auf Wiederhören."
    ),
    "daily_minutes": (
        "Die Demo ist für heute ausgelastet. Bitte versuchen Sie es morgen wieder. "
        "Sorry, the demo is fully used for today. Auf Wiederhören."
    ),
    "web_daily_minutes": (
        "Die Demo ist heute ausgelastet. Bitte versuchen Sie es morgen wieder. "
        "Sorry, the demo is busy today; please try again tomorrow. Auf Wiederhören."
    ),
    "web_monthly_minutes": (
        "Die Demo ist in diesem Monat ausgelastet. Bitte versuchen Sie es im nächsten Monat "
        "wieder. Sorry, the demo is fully used this month; please try again next month. "
        "Auf Wiederhören."
    ),
    "warn": "Hinweis: Die Demo endet in einer Minute. The demo ends in one minute.",
    "end": (
        "Die Demo-Zeit ist abgelaufen. Vielen Dank für Ihren Anruf, auf Wiederhören. "
        "Thank you for calling, goodbye."
    ),
}


# --- the busy signal ------------------------------------------------------------------------


async def tell_busy(ctx: Any, session: Any, reason: str, *, speak: bool = True) -> None:
    """Page signal first, then the spoken message, then hang up.

    The participant attribute ``demo_status=busy`` is what the web page shows as "the demo
    is busy today". It is set before anything is spoken because speaking (the TTS) may be
    exactly what failed; a failing voice must not stop the signal or the hang-up.
    """
    with contextlib.suppress(Exception):
        await ctx.room.local_participant.set_attributes(
            {"demo_status": "busy", "demo_reason": reason}
        )
    if speak and reason in MESSAGES:
        with contextlib.suppress(Exception):
            await session.say(MESSAGES[reason], allow_interruptions=False)
    with contextlib.suppress(Exception):
        await ctx.delete_room()


def guard_tts(session: Any, on_failure: Callable[[], object]) -> None:
    """Call ``on_failure`` once when the voice fails for good (a final TTS error; LiveKit
    retries recoverable ones itself)."""
    fired: list[bool] = []

    def _on_error(ev: Any) -> None:
        error = ev.error
        final_tts = getattr(error, "type", "") == "tts_error" and not getattr(
            error, "recoverable", True
        )
        if final_tts and not fired:
            fired.append(True)
            logger.info(json.dumps({"event": "demo_tts_failed"}, separators=(",", ":")))
            on_failure()

    session.on("error", _on_error)


def busy_task(ctx: Any, session: Any, reason: str, *, speak: bool) -> Callable[[], object]:
    """A callback that starts ``tell_busy`` as a task (for event handlers)."""

    def start() -> object:
        return asyncio.get_running_loop().create_task(tell_busy(ctx, session, reason, speak=speak))

    return start
