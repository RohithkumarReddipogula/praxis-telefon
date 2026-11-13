"""Per-call adapter between the LLM-facing tools and the plain tool functions.

No LiveKit imports, so it is testable without network. It:
- turns the caller's words for a day (``when``) into dates via relative_dates,
- parses LLM string arguments (ISO dates of birth),
- enforces per-call guards: ``book``/``move`` only accept slot ids that ``find_slots``
  returned in this call; ``move``/``cancel`` only accept booking ids this call verified
  via ``find_booking`` or created via ``book``,
- returns short German text for the LLM,
- refuses all scheduling tools once ``booking_allowed()`` is False (emergency or
  message-only mode), so this does not depend on the LLM respecting the tool list,
- logs tool name, outcome code and duration only, never argument values.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime

from praxis_telefon import faq
from praxis_telefon.agent import caller_identity
from praxis_telefon.db.schema import name_key
from praxis_telefon.tools import messages, relative_dates, scheduling
from praxis_telefon.tools.errors import (
    BookingDisabled,
    BookingNotForCaller,
    BookingNotVerified,
    CallbackNumberMismatch,
    InvalidDateOfBirth,
    InvalidTopic,
    SlotNotOffered,
    ToolFailure,
)
from praxis_telefon.tools.spoken_numbers import digit_runs
from praxis_telefon.tools.validation import clean_name

logger = logging.getLogger("praxis_telefon.tools")

Clock = Callable[[], datetime]


@dataclass
class CallGuards:
    offered_slot_ids: set[int] = field(default_factory=set)
    verified_booking_ids: set[int] = field(default_factory=set)
    # (name key, number) -> (message id, reasons in the order said)
    saved_messages: dict[tuple[str, str], tuple[int, list[str]]] = field(default_factory=dict)


_MAX_PARTS = 4  # a number may be said in parts: "null drei null" ... "zwei drei eins ..."


def _callers_numbers(caller_texts: Iterable[str]) -> set[str]:
    """Valid callback numbers the caller said, normalised like take_message's: any run of
    number words, or up to _MAX_PARTS consecutive runs joined (across the caller's lines)."""
    runs = [r for text in caller_texts for r in digit_runs(text)]
    out: set[str] = set()
    for i in range(len(runs)):
        for j in range(i + 1, min(i + _MAX_PARTS, len(runs)) + 1):
            try:
                out.add(messages.normalize_callback_number("".join(runs[i:j])))
            except ToolFailure:
                continue
    return out


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
        # Set by the agent: False once the call is in emergency or message-only mode.
        self.booking_allowed: Callable[[], bool] = lambda: True

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

    def _require_booking(self) -> None:
        if not self.booking_allowed():
            raise BookingDisabled()

    def find_slots(self, when: str, part_of_day: str, doctor: str | None) -> str:
        with self._logged("find_slots"):
            self._require_booking()
            # The caller's words become dates here, in code, never in the LLM.
            found = relative_dates.resolve_for_search(when, self._clock())
            if found.closed:
                header = (
                    f"{found.requested.spoken()}: Praxis geschlossen. "
                    f"Stattdessen gesucht: {found.search.spoken()}."
                )
            else:
                header = f"Gesucht: {found.search.spoken()}."
            options = scheduling.find_slots(
                self._conn,
                now=self._clock(),
                date_from=found.search.start,
                date_to=found.search.end,
                part_of_day=part_of_day,
                doctor=doctor or None,
            )
            if not options:
                return (
                    f"{header}\nKeine freien Termine in diesem Zeitraum. "
                    "Einen anderen Zeitraum suchen."
                )
            self.guards.offered_slot_ids.update(o.slot_id for o in options)
            lines = [f"slot_id={o.slot_id}: {o.spoken}" for o in options]
            return (
                f"{header}\nFreie Termine (nur diese anbieten, wörtlich vorlesen):\n"
                + "\n".join(lines)
            )

    def book(self, slot_id: int, patient_name: str, date_of_birth: str) -> str:
        with self._logged("book"):
            self._require_booking()
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

    def find_booking(
        self, patient_name: str, date_of_birth: str, *, caller_texts: Iterable[str] = ()
    ) -> str:
        """``caller_texts``: the caller's lines in this call; only the caller's own name
        is looked up (privacy, agent/caller_identity.py)."""
        with self._logged("find_booking"):
            self._require_booking()
            if not caller_identity.is_caller(patient_name, caller_texts):
                raise BookingNotForCaller()
            bookings = scheduling.find_booking(
                self._conn,
                now=self._clock(),
                patient_name=patient_name,
                date_of_birth=_parse_date(date_of_birth, InvalidDateOfBirth),
            )
            if not bookings:
                return (
                    "Kein Termin gefunden. Lass dir den Nachnamen einmal buchstabieren (falls "
                    "noch nicht geschehen) und prüfe das Geburtsdatum, dann such noch einmal. "
                    "Verrate nicht, ob die Person Patient der Praxis ist."
                )
            self.guards.verified_booking_ids.update(b.booking_id for b in bookings)
            lines = [f"booking_id={b.booking_id}: {b.slot.spoken}" for b in bookings]
            return "Gefundene Termine:\n" + "\n".join(lines)

    def move(self, booking_id: int, new_slot_id: int) -> str:
        with self._logged("move"):
            self._require_booking()
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
            self._require_booking()
            if booking_id not in self.guards.verified_booking_ids:
                raise BookingNotVerified()
            result = scheduling.cancel(self._conn, now=self._clock(), booking_id=booking_id)
            if result.already_cancelled:
                return f"Dieser Termin war bereits abgesagt ({result.slot.spoken})."
            return f"Abgesagt: {result.slot.spoken}."

    def practice_info(self, topic: str, language: str) -> str:
        """The FAQ entry for ``topic`` in the call's language; "other": no information."""
        with self._logged("practice_info") as log:
            if topic != "other" and topic not in faq.TOPICS:
                raise InvalidTopic()
            if topic == "other":
                log["outcome"] = "no_entry"
                return (
                    "Dazu gibt es keine Auskunft der Praxis. Sag, dass du das nicht weißt, "
                    "und biete an, eine Nachricht für das Praxisteam aufzunehmen. Erfinde "
                    "nichts."
                )
            return (
                "Auskunft der Praxis (nur diese Angaben verwenden, nichts ergänzen): "
                + faq.answer_text(topic, language)
            )

    def take_message(
        self,
        caller_name: str,
        callback_number: str,
        reason_category: str,
        *,
        caller_texts: Iterable[str] = (),
    ) -> str:
        """``caller_texts``: the caller's lines; a number the caller said there must match
        (M4 harness rerun 2026-11-07, see CallbackNumberMismatch)."""
        with self._logged("take_message") as log:
            number = messages.normalize_callback_number(callback_number)
            said = _callers_numbers(caller_texts)
            if said and number not in said:
                raise CallbackNumberMismatch()
            digits = " ".join(number)
            read_back = (
                f" Gespeicherte Rückrufnummer zum Vorlesen, Ziffer für Ziffer: {digits}. "
                "Lies zur Bestätigung genau diese Nummer vor."
            )
            # Bug (voice test 2026-11-06): the LLM called take_message twice and the
            # team got two identical notes. The same note again in this call is a no-op.
            # Bug (voice test 2026-11-07): the same caller and number with another reason
            # gave a second note; now the first note gets the reason too.
            key = (
                name_key(clean_name(caller_name)),
                messages.normalize_callback_number(callback_number),
            )
            if key in self.guards.saved_messages:
                message_id, reasons = self.guards.saved_messages[key]
                if reason_category in reasons:
                    log["outcome"] = "duplicate"
                    labels = ", ".join(messages.REASON_LABELS_DE[r] for r in reasons)
                    return (
                        f"Diese Nachricht ist bereits gespeichert (Anliegen: {labels}). "
                        "Nicht noch einmal speichern." + read_back
                    )
                labels = messages.add_reason(
                    self._conn, message_id=message_id, reasons=[*reasons, reason_category]
                )
                reasons.append(reason_category)
                log["outcome"] = "updated"
                return (
                    f"Bestehende Nachricht ergänzt (Anliegen: {labels}). "
                    "Nicht noch einmal speichern." + read_back
                )
            result = messages.take_message(
                self._conn,
                now=self._clock(),
                caller_name=caller_name,
                callback_number=callback_number,
                reason_category=reason_category,
            )
            self.guards.saved_messages[key] = (result.message_id, [reason_category])
            return (
                f"Nachricht für das Praxisteam gespeichert (Anliegen: {result.category_label})."
                + read_back
            )
