"""The tool descriptions in English, as they were before 2026-11-06, for the leak experiment.

M3 Part B step 1 compared English and German tool descriptions (arms A/B used this copy,
LIVE_TOOL_DOCS=en; docs/experiments/reasoning-leak.md). German was not adopted, so
production (praxis_telefon/agent/receptionist.py) has the same English text again. This
agent is the timed live-test agent with the six tools overridden: same names,
parameters, bodies and descriptions; kept so the experiment's arms can be rerun.
"""

from __future__ import annotations

from typing import Any, Literal

from livekit.agents import RunContext, function_tool, llm

from praxis_telefon.agent.receptionist import _run
from tests.live.harness import TimedReceptionist


class EnglishToolDocsReceptionist(TimedReceptionist):
    # Each override is flagged by mypy only because FunctionTool is generic in the class.

    @function_tool
    async def find_slots(  # type: ignore[override]
        self,
        context: RunContext[Any],
        when: str,
        part_of_day: Literal["morning", "afternoon", "any"] = "any",
        doctor: str | None = None,
    ) -> str:
        """Search free appointment slots. Returns at most 3. Only offer slots returned here.

        The result starts with "Gesucht: <date>": the day the caller's words mean. Use
        that date; never work out dates yourself.

        Args:
            when: The caller's own words for the day or period, unchanged, e.g. "morgen",
                "nächsten Dienstag", "Ende der Woche", "in zwei Wochen", "12. November",
                "so bald wie möglich", "tomorrow", "next Thursday".
            part_of_day: "morning" (before 12:00), "afternoon" or "any".
            doctor: Optional doctor surname, e.g. "Weber" or "Brandt".
        """
        return _run(lambda: self._service.find_slots(when, part_of_day, doctor))

    @function_tool
    async def book(  # type: ignore[override]
        self, context: RunContext[Any], slot_id: int, patient_name: str, date_of_birth: str
    ) -> str:
        """Book a slot. Call right after the caller's clear yes to your one read-back.

        Say the appointment is booked only after this returns success.

        Args:
            slot_id: slot_id from find_slots in this call.
            patient_name: Patient's first and last name.
            date_of_birth: Patient's date of birth, ISO format YYYY-MM-DD.
        """
        return _run(lambda: self._service.book(slot_id, patient_name, date_of_birth))

    @function_tool
    async def find_booking(  # type: ignore[override]
        self, context: RunContext[Any], patient_name: str, date_of_birth: str
    ) -> str:
        """Look up the caller's own upcoming appointments by name and date of birth.

        Call as soon as both are known, without asking for confirmation first. Only for
        the caller themself: never for another person's appointment.

        Args:
            patient_name: Patient's first and last name.
            date_of_birth: Patient's date of birth, ISO format YYYY-MM-DD.
        """
        callers = [
            item.text_content or ""
            for item in self.chat_ctx.items
            if isinstance(item, llm.ChatMessage) and item.role == "user"
        ]
        return _run(
            lambda: self._service.find_booking(patient_name, date_of_birth, caller_texts=callers)
        )

    @function_tool
    async def move_booking(  # type: ignore[override]
        self, context: RunContext[Any], booking_id: int, new_slot_id: int
    ) -> str:
        """Move a booking to a new slot. Call right after the caller's clear yes.

        Say the appointment is moved only after this returns success.

        Args:
            booking_id: booking_id from find_booking or book in this call.
            new_slot_id: slot_id from find_slots in this call.
        """
        return _run(lambda: self._service.move(booking_id, new_slot_id))

    @function_tool
    async def cancel_booking(self, context: RunContext[Any], booking_id: int) -> str:  # type: ignore[override]
        """Cancel a booking. Call right after the caller's clear yes.

        Say the appointment is cancelled only after this returns success.

        Args:
            booking_id: booking_id from find_booking or book in this call.
        """
        return _run(lambda: self._service.cancel(booking_id))

    @function_tool
    async def take_message(  # type: ignore[override]
        self,
        context: RunContext[Any],
        caller_name: str,
        callback_number: str,
        reason_category: Literal[
            "prescription", "referral", "test_results", "appointment", "callback", "other"
        ],
    ) -> str:
        """Leave a callback note for the practice team. No medical details.

        Only for callback requests, prescriptions, referrals, results, or when no suitable
        slot exists. Never instead of book. Call once per request.

        Args:
            caller_name: Caller's first and last name.
            callback_number: The callback number exactly as the caller said it, e.g.
                "null drei null, zwei drei eins zwei fünf"; code converts it. Must be a
                number the caller said in this call.
            reason_category: Category of the request; never record symptoms.
        """
        callers = [
            item.text_content or ""
            for item in self.chat_ctx.items
            if isinstance(item, llm.ChatMessage) and item.role == "user"
        ]
        return _run(
            lambda: self._service.take_message(
                caller_name, callback_number, reason_category, caller_texts=callers
            )
        )
