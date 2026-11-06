"""The LiveKit Agent: function tools that delegate to ToolService."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from livekit.agents import Agent, RunContext, ToolError, function_tool

from praxis_telefon.agent.prompts import build_instructions
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.tools.errors import ToolFailure


def _run(call: Callable[[], str]) -> str:
    try:
        return call()
    except ToolFailure as e:
        # Fixed message without argument values; the LLM explains it to the caller.
        raise ToolError(f"{e.code}: {e.message}") from None


class Receptionist(Agent):
    def __init__(self, service: ToolService) -> None:
        super().__init__(instructions=build_instructions(service.now()))
        self._service = service

    @function_tool
    async def find_slots(
        self,
        context: RunContext[Any],
        date_from: str,
        date_to: str,
        part_of_day: Literal["morning", "afternoon", "any"] = "any",
        doctor: str | None = None,
    ) -> str:
        """Search free appointment slots. Returns at most 3. Only offer slots returned here.

        Args:
            date_from: First day to search, ISO format YYYY-MM-DD.
            date_to: Last day to search, ISO format YYYY-MM-DD, at most 14 days after date_from.
            part_of_day: "morning" (before 12:00), "afternoon" or "any".
            doctor: Optional doctor surname, e.g. "Weber" or "Brandt".
        """
        return _run(lambda: self._service.find_slots(date_from, date_to, part_of_day, doctor))

    @function_tool
    async def book(
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
    async def find_booking(
        self, context: RunContext[Any], patient_name: str, date_of_birth: str
    ) -> str:
        """Look up the patient's upcoming appointments by name and date of birth.

        Call as soon as both are known, without asking for confirmation first.

        Args:
            patient_name: Patient's first and last name.
            date_of_birth: Patient's date of birth, ISO format YYYY-MM-DD.
        """
        return _run(lambda: self._service.find_booking(patient_name, date_of_birth))

    @function_tool
    async def move_booking(
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
    async def cancel_booking(self, context: RunContext[Any], booking_id: int) -> str:
        """Cancel a booking. Call right after the caller's clear yes.

        Say the appointment is cancelled only after this returns success.

        Args:
            booking_id: booking_id from find_booking or book in this call.
        """
        return _run(lambda: self._service.cancel(booking_id))

    @function_tool
    async def take_message(
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
            callback_number: German phone number for the callback.
            reason_category: Category of the request; never record symptoms.
        """
        return _run(
            lambda: self._service.take_message(caller_name, callback_number, reason_category)
        )
