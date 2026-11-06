"""The LiveKit Agent: safety screening before the LLM, and function tools.

Every reply goes through ``llm_node`` (voice turns, typed input, ``generate_reply``).
There the call's safety state is replayed from the conversation (see
``safety/call_state.py``) and, if needed, a fixed message is returned instead of
calling the LLM. In emergency or message-only mode the LLM only gets ``take_message``,
and ToolService refuses scheduling tools in code as well.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterable, Callable
from typing import Any, Literal

from livekit.agents import (
    Agent,
    FlushSentinel,
    ModelSettings,
    RunContext,
    ToolError,
    function_tool,
    llm,
)

from praxis_telefon.agent.claim_check import BACKED_BY, ClaimTracker
from praxis_telefon.agent.claim_gate import SentenceFilter, filter_claims
from praxis_telefon.agent.prompts import build_instructions
from praxis_telefon.agent.speech_lock import lock_current_speech
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.safety.call_state import (
    Action,
    CallerTurn,
    Event,
    Mode,
    State,
    UnrecognizedSpeech,
    replay,
)
from praxis_telefon.safety.messages import (
    FIXED_MESSAGES,
    MODE_NOTES,
    NOT_DONE_MESSAGES,
    SUICIDE_NOTE,
    UNINTERRUPTIBLE,
    retry_note,
)
from praxis_telefon.tools.errors import ToolFailure

logger = logging.getLogger("praxis_telefon.safety")

MESSAGE_ONLY_TOOLS = frozenset({"take_message"})


def _run(call: Callable[[], str]) -> str:
    try:
        return call()
    except ToolFailure as e:
        # Fixed message without argument values; the LLM explains it to the caller.
        raise ToolError(f"{e.code}: {e.message}") from None


def _tool_name(tool: llm.Tool) -> str:
    info = getattr(tool, "info", None)
    return str(getattr(info, "name", ""))


class Receptionist(Agent):
    def __init__(self, service: ToolService) -> None:
        super().__init__(instructions=build_instructions(service.now()))
        self._service = service
        self._unrecognized_at: list[float] = []
        self._pending: Action | None = None
        self._logged_ids: set[str] = set()
        service.booking_allowed = lambda: self.safety_state().mode is Mode.NORMAL

    # --- safety state ------------------------------------------------------------------

    def _events(self, chat_ctx: llm.ChatContext) -> list[Event]:
        timed: list[tuple[float, Event]] = [
            (t, UnrecognizedSpeech()) for t in self._unrecognized_at
        ]
        for item in chat_ctx.items:
            if isinstance(item, llm.ChatMessage) and item.role == "user":
                turn = CallerTurn(item.text_content or "", item.transcript_confidence)
                timed.append((item.created_at, turn))
        timed.sort(key=lambda pair: pair[0])
        return [event for _, event in timed]

    def safety_state(self) -> State:
        """State from the committed conversation (used by the ToolService guard)."""
        return replay(self._events(self.chat_ctx))[0]

    def on_unrecognized_speech(self) -> Any:
        """Called when the caller spoke but STT produced nothing. Returns the SpeechHandle."""
        self._unrecognized_at.append(time.time())
        _, action = replay(self._events(self.chat_ctx))
        self._log(action, key=f"unrecognized-{len(self._unrecognized_at)}")
        if action.kind == "none":
            return None
        self._pending = action
        return self.session.generate_reply(allow_interruptions=True)

    def _log(self, action: Action, *, key: str) -> None:
        if action.kind == "none" or key in self._logged_ids:
            return
        self._logged_ids.add(key)
        record = {"event": "safety", "action": action.kind}  # never the caller's words
        logger.info(json.dumps(record, separators=(",", ":")))

    # --- LLM node ----------------------------------------------------------------------

    def _action_for(self, chat_ctx: llm.ChatContext) -> tuple[State, Action | None]:
        state, action = replay(self._events(chat_ctx))
        last = chat_ctx.items[-1] if chat_ctx.items else None
        if isinstance(last, llm.ChatMessage) and last.role == "user":
            self._log(action, key=last.id)
            return state, action
        pending, self._pending = self._pending, None
        return state, pending

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk | str | FlushSentinel]:
        state, action = self._action_for(chat_ctx)
        if action is not None and action.kind != "none":
            if action.kind in UNINTERRUPTIBLE:
                lock_current_speech()
            yield FIXED_MESSAGES[(action.kind, action.language)]
            return

        if state.mode is not Mode.NORMAL:
            tools = [t for t in tools if _tool_name(t) in MESSAGE_ONLY_TOOLS]
            chat_ctx = chat_ctx.copy()
            note = MODE_NOTES[state.mode]
            if state.suicide_mentioned:
                note = f"{note} {SUICIDE_NOTE}"
            chat_ctx.add_message(role="system", content=note)

        async for out in self._llm_without_false_claims(chat_ctx, tools, model_settings, state):
            yield out

    async def _llm_without_false_claims(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
        state: State,
    ) -> AsyncIterable[llm.ChatChunk | str | FlushSentinel]:
        """The LLM's reply, minus any sentence claiming an action no tool has done yet.

        Blocked once: retry with a note (the retry may call the tool). Blocked again: a
        fixed "not saved yet" sentence. A blocked sentence is never yielded, so it is in
        neither the speech, the transcript nor the chat history. See claim_gate.py.
        """
        done = {
            item.name
            for item in chat_ctx.items
            if isinstance(item, llm.FunctionCallOutput) and not item.is_error
        }
        tracker = ClaimTracker(done)

        first = SentenceFilter(tracker)
        stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        async for out in filter_claims(stream, first):
            yield out
        if not first.blocked or self._tool_follows(first):
            return
        self._log_claim("claim_blocked", first.blocked)

        retry_ctx = chat_ctx.copy()
        if first.spoken:
            retry_ctx.add_message(role="assistant", content="".join(first.spoken).strip())
        retry_ctx.add_message(role="system", content=retry_note(first.blocked))
        if first.spoken and not first.spoken[-1][-1:].isspace():
            yield " "  # keep the retry's first word apart from what was already said
        second = SentenceFilter(tracker)
        stream = Agent.default.llm_node(self, retry_ctx, tools, model_settings)
        async for out in filter_claims(stream, second):
            yield out
        if second.blocked and not self._tool_follows(second):
            self._log_claim("claim_fallback", second.blocked)
            said = "".join(second.spoken)
            prefix = " " if said and not said[-1:].isspace() else ""
            yield prefix + NOT_DONE_MESSAGES[(second.blocked[0], state.language)]

    @staticmethod
    def _tool_follows(sf: SentenceFilter) -> bool:
        """The same response calls the claimed tool; the reply after its result confirms."""
        return any(BACKED_BY[a] & sf.tool_calls for a in sf.blocked)

    def _log_claim(self, event: str, actions: list[str]) -> None:
        for action in actions:
            record = {"event": event, "action": action}  # never the sentence itself
            logger.info(json.dumps(record, separators=(",", ":")))

    # --- tools ------------------------------------------------------------------------

    @function_tool
    async def find_slots(
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
            callback_number: German phone number for the callback, as digits, e.g.
                "030 23125456". Spoken digits are converted, but digits are preferred.
            reason_category: Category of the request; never record symptoms.
        """
        return _run(
            lambda: self._service.take_message(caller_name, callback_number, reason_category)
        )
