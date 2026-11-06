"""Per-call adapter between the LLM-facing tools and the plain tool functions.

No LiveKit imports, so it is testable without network. It:
- parses LLM string arguments (ISO dates),
- enforces per-call guards: ``book``/``move`` only accept slot ids that ``find_slots``
  returned in this call; ``move``/``cancel`` only accept booking ids this call verified
  via ``find_booking`` or created via ``book``,
- returns short German text for the LLM,
- logs tool name, outcome code and duration only, never argument values.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime

from praxis_telefon.db.schema import name_key
from praxis_telefon.tools import messages, scheduling
from praxis_telefon.tools.errors import (
    BookingNotVerified,
    InvalidDateOfBirth,
    InvalidDateRange,
    SlotNotOffered,
    ToolFailure,
)
from praxis_telefon.tools.validation import clean_name

logger = logging.getLogger("praxis_telefon.tools")

Clock = Callable[[], datetime]


@dataclass
class CallGuards:
    offered_slot_ids: set[int] = field(default_factory=set)
    verified_booking_ids: set[int] = field(default_factory=set)
    saved_messages: dict[tuple[str, str, str], messages.MessageResult] = field(default_factory=dict)


def _parse_date(value: str, error: type[ToolFailure]) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise error() from None


class ToolService:
    def __init__(self, conn: sqlite3.Connection, clock: Clock) -> None:
        self._conn = conn
        self._clock = clock
        self.guards = CallGuards()

    def now(self) -> datetime:
        return self._clock()

    async def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _logged(self, tool: str) -> Iterator[dict[str, str]]:
        """Log tool, outcome code and duration. The body may set ``log["outcome"]``."""
        started = time.perf_counter()
        log = {"outcome": "ok"}
        try:
            yield log
        except ToolFailure as e:
            log["outcome"] = e.code
            raise
        except Exception:
            log["outcome"] = "internal_error"
            raise
        finally:
            ms = round((time.perf_counter() - started) * 1000, 1)
            record = {"event": "tool", "tool": tool, "outcome": log["outcome"], "ms": ms}
            logger.info(json.dumps(record, separators=(",", ":")))

    def find_slots(self, date_from: str, date_to: str, part_of_day: str, doctor: str | None) -> str:
        with self._logged("find_slots"):
            options = scheduling.find_slots(
                self._conn,
                now=self._clock(),
                date_from=_parse_date(date_from, InvalidDateRange),
                date_to=_parse_date(date_to, InvalidDateRange),
                part_of_day=part_of_day,
                doctor=doctor or None,
            )
            if not options:
                return "Keine freien Termine in diesem Zeitraum. Einen anderen Zeitraum suchen."
            self.guards.offered_slot_ids.update(o.slot_id for o in options)
            lines = [f"slot_id={o.slot_id}: {o.spoken}" for o in options]
            return "Freie Termine (nur diese anbieten, wörtlich vorlesen):\n" + "\n".join(lines)

    def book(self, slot_id: int, patient_name: str, date_of_birth: str) -> str:
        with self._logged("book"):
            if slot_id not in self.guards.offered_slot_ids:
                raise SlotNotOffered()
            result = scheduling.book(
                self._conn,
                now=self._clock(),
                slot_id=slot_id,
                patient_name=patient_name,
                date_of_birth=_parse_date(date_of_birth, InvalidDateOfBirth),
            )
            self.guards.verified_booking_ids.add(result.booking_id)
            prefix = "Bereits gebucht" if result.already_booked else "Gebucht"
            return f"{prefix}: booking_id={result.booking_id}, {result.slot.spoken}."

    def find_booking(self, patient_name: str, date_of_birth: str) -> str:
        with self._logged("find_booking"):
            bookings = scheduling.find_booking(
                self._conn,
                now=self._clock(),
                patient_name=patient_name,
                date_of_birth=_parse_date(date_of_birth, InvalidDateOfBirth),
            )
            if not bookings:
                return (
                    "Kein Termin gefunden. Name und Geburtsdatum noch einmal prüfen, "
                    "aber nicht verraten, ob die Person Patient der Praxis ist."
                )
            self.guards.verified_booking_ids.update(b.booking_id for b in bookings)
            lines = [f"booking_id={b.booking_id}: {b.slot.spoken}" for b in bookings]
            return "Gefundene Termine:\n" + "\n".join(lines)

    def move(self, booking_id: int, new_slot_id: int) -> str:
        with self._logged("move"):
            if booking_id not in self.guards.verified_booking_ids:
                raise BookingNotVerified()
            if new_slot_id not in self.guards.offered_slot_ids:
                raise SlotNotOffered()
            result = scheduling.move(
                self._conn, now=self._clock(), booking_id=booking_id, new_slot_id=new_slot_id
            )
            if result.already_booked:
                return f"Der Termin liegt bereits auf {result.slot.spoken}."
            return f"Verschoben: booking_id={result.booking_id}, neu {result.slot.spoken}."

    def cancel(self, booking_id: int) -> str:
        with self._logged("cancel"):
            if booking_id not in self.guards.verified_booking_ids:
                raise BookingNotVerified()
            result = scheduling.cancel(self._conn, now=self._clock(), booking_id=booking_id)
            if result.already_cancelled:
                return f"Dieser Termin war bereits abgesagt ({result.slot.spoken})."
            return f"Abgesagt: {result.slot.spoken}."

    def take_message(self, caller_name: str, callback_number: str, reason_category: str) -> str:
        with self._logged("take_message") as log:
            # Bug (voice test 2026-11-06): the LLM called take_message twice and the
            # team got two identical notes. The same note again in this call is a no-op.
            key = (
                name_key(clean_name(caller_name)),
                messages.normalize_callback_number(callback_number),
                reason_category,
            )
            if key in self.guards.saved_messages:
                log["outcome"] = "duplicate"
                label = self.guards.saved_messages[key].category_label
                return (
                    f"Diese Nachricht ist bereits gespeichert (Anliegen: {label}). "
                    "Nicht noch einmal speichern."
                )
            result = messages.take_message(
                self._conn,
                now=self._clock(),
                caller_name=caller_name,
                callback_number=callback_number,
                reason_category=reason_category,
            )
            self.guards.saved_messages[key] = result
            return f"Nachricht für das Praxisteam gespeichert (Anliegen: {result.category_label})."
