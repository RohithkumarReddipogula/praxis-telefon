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
"""

from __future__ import annotations

import hashlib
import hmac
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from praxis_telefon.db.schema import PRACTICE_TZ

MIN_CALL_S = 60  # less than this left for the day: the call is refused
WITHHELD = frozenset({"", "anonymous", "unknown", "restricted", "private", "withheld"})


@dataclass(frozen=True)
class DemoLimits:
    max_call_s: int = 300
    warn_before_s: int = 60
    calls_per_caller: int = 3
    calls_per_day: int = 15
    minutes_per_day: int = 45


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
                "caller TEXT, max_s INTEGER NOT NULL, duration_s INTEGER)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=10, isolation_level=None)

    def start(self, caller: str | None, now: datetime) -> CallDecision:
        """``caller``: the keyed hash, or None for a withheld number."""
        day = now.astimezone(PRACTICE_TZ).date().isoformat()
        lim = self.limits
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")  # one decision at a time across processes
            calls, used = conn.execute(
                "SELECT count(*), coalesce(sum(coalesce(duration_s, max_s)), 0) FROM calls "
                "WHERE day = ?",
                (day,),
            ).fetchone()
            mine = (
                conn.execute(
                    "SELECT count(*) FROM calls WHERE day = ? AND caller = ?", (day, caller)
                ).fetchone()[0]
                if caller is not None
                else 0
            )
            left = lim.minutes_per_day * 60 - used
            if caller is not None and mine >= lim.calls_per_caller:
                decision = CallDecision(False, "caller_daily_limit")
            elif calls >= lim.calls_per_day:
                decision = CallDecision(False, "daily_calls")
            elif left < MIN_CALL_S:
                decision = CallDecision(False, "daily_minutes")
            else:
                max_s = min(lim.max_call_s, left)
                cur = conn.execute(
                    "INSERT INTO calls (day, caller, max_s) VALUES (?, ?, ?)",
                    (day, caller, max_s),
                )
                decision = CallDecision(True, "", max_s, cur.lastrowid)
            conn.execute("COMMIT")
            return decision
        finally:
            conn.close()

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
    "warn": "Hinweis: Die Demo endet in einer Minute. The demo ends in one minute.",
    "end": (
        "Die Demo-Zeit ist abgelaufen. Vielen Dank für Ihren Anruf, auf Wiederhören. "
        "Thank you for calling, goodbye."
    ),
}
