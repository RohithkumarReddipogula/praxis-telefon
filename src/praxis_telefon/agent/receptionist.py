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

from livekit import rtc
from livekit.agents import (
    Agent,
    FlushSentinel,
    ModelSettings,
    RunContext,
    ToolError,
    function_tool,
    llm,
    stt,
)

from praxis_telefon.agent import call_language
from praxis_telefon.agent.claim_gate import FilterContext, SentenceFilter, filter_claims
from praxis_telefon.agent.prompts import LANGUAGE_NOTES, TOOL_FILLER, build_instructions
from praxis_telefon.agent.speech_lock import lock_current_speech
from praxis_telefon.agent.tool_filler import ToolFiller, said_since_caller
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
    SUICIDE_NOTE,
    UNINTERRUPTIBLE,
    fallback_for,
    retry_note_for,
)
from praxis_telefon.safety.phrases import Language
from praxis_telefon.tools.errors import ToolFailure
from praxis_telefon.tools.formatting import today_sentence

logger = logging.getLogger("praxis_telefon.safety")
language_logger = logging.getLogger("praxis_telefon.language")

MESSAGE_ONLY_TOOLS = frozenset({"take_message"})
_CALL_LANGUAGE_ACTIONS = frozenset({"repeat", "handover_strikes"})


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
    def __init__(
        self,
        service: ToolService,
        on_language_change: Callable[[Language], None] | None = None,
    ) -> None:
        super().__init__(instructions=build_instructions(service.now()))
        self._service = service
        self._unrecognized_at: list[float] = []
        self._pending: Action | None = None
        self._logged_ids: set[str] = set()
        self._finals: list[call_language.CallerFinal] = []
        # Switches the Deepgram stream when the call language changes (option B,
        # main.stt_switcher); None in text mode or with a fixed STT_LANGUAGE.
        self._on_language_change = on_language_change
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
        if self._switch_back_on_empty():
            self._pending = Action("switch_back", "de")
            return self.session.generate_reply(allow_interruptions=True)
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

    # --- call language -----------------------------------------------------------------

    def _switch_back_on_empty(self) -> bool:
        """An empty final on the en stream in a call that started German: the caller most
        likely spoke German (en heard "Sprechen Sie Deutsch?" as nothing). Back to de, no
        strike (user decision 2026-11-07)."""
        state = call_language.decide(self._finals)
        if (
            self._on_language_change is None
            or state.language != "en"
            or state.first_language != "de"
        ):
            return False
        self._finals.append(call_language.SWITCH_BACK)
        self._on_language_change("de")
        record = {"event": "stt_language_switch", "to": "de", "reason": "empty_en_final"}
        language_logger.info(json.dumps(record, separators=(",", ":")))
        return True

    def record_final(self, text: str, stt_language: str | None) -> None:
        """One final transcript, kept as language labels only (agent/call_language.py)."""
        if not text.strip():
            return
        final = call_language.CallerFinal.from_text(text, stt_language)
        before = call_language.decide(self._finals).language
        self._finals.append(final)
        after = call_language.decide(self._finals).language
        record = {
            "event": "caller_final",
            "stt": final.stt,
            "detected": final.detected,
            "request": final.request,
            "call_language": after,
        }  # labels only, never the caller's words
        language_logger.info(json.dumps(record, separators=(",", ":")))
        if after != before and self._on_language_change is not None:
            # Right after the final that switched it: the caller has just stopped talking.
            self._on_language_change(after)
            language_logger.info(
                json.dumps({"event": "stt_language_switch", "to": after}, separators=(",", ":"))
            )

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterable[stt.SpeechEvent | str]:
        """The default STT, passed through unchanged; finals are recorded on the way."""
        events = Agent.default.stt_node(self, audio, model_settings)
        async for ev in events:
            if (
                isinstance(ev, stt.SpeechEvent)
                and ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT
                and ev.alternatives
            ):
                alt = ev.alternatives[0]
                self.record_final(alt.text, str(alt.language) if alt.language else None)
            yield ev

    def language_state(self, chat_ctx: llm.ChatContext) -> call_language.LanguageState:
        """From the STT finals; in text mode (no finals) from the typed caller lines."""
        if self._finals:
            return call_language.decide(self._finals)
        return call_language.from_texts(
            item.text_content or ""
            for item in chat_ctx.items
            if isinstance(item, llm.ChatMessage) and item.role == "user"
        )

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
        lang = self.language_state(chat_ctx)
        if action is not None and action.kind != "none":
            if action.kind in UNINTERRUPTIBLE:
                lock_current_speech()
            # Crisis and human-request messages follow the phrase's language; "say that
            # again" and the strike handover follow the call's.
            said_in = lang.language if action.kind in _CALL_LANGUAGE_ACTIONS else action.language
            yield FIXED_MESSAGES[(action.kind, said_in)]
            return

        if lang.english_heard:
            chat_ctx = chat_ctx.copy()
            chat_ctx.add_message(role="system", content=LANGUAGE_NOTES[lang.language])

        if state.mode is not Mode.NORMAL:
            tools = [t for t in tools if _tool_name(t) in MESSAGE_ONLY_TOOLS]
            chat_ctx = chat_ctx.copy()
            note = MODE_NOTES[state.mode]
            if state.suicide_mentioned:
                note = f"{note} {SUICIDE_NOTE}"
            chat_ctx.add_message(role="system", content=note)

        async for out in self._llm_filtered(chat_ctx, tools, model_settings, lang.language):
            yield out

    async def _llm_filtered(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
        language: Language,
    ) -> AsyncIterable[llm.ChatChunk | str | FlushSentinel]:
        """The LLM's reply, with every sentence checked before it is spoken.

        See claim_gate.py for what is dropped. At most one retry per reply, across all
        filters together; if the retry needs one too, a fixed sentence is said instead.
        A dropped sentence is never yielded, so it is in neither the speech, the
        transcript nor the chat history.
        """
        ctx = FilterContext.from_chat(chat_ctx, today_sentence(self._service.now()), language)
        filler = ToolFiller(TOOL_FILLER[ctx.language], allowed=not said_since_caller(chat_ctx))

        first = SentenceFilter(ctx, "first")
        stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        async for out in filter_claims(stream, first):
            yield out
        if (phrase := filler.after(first)) is not None:
            yield phrase
        if not first.needs_retry():
            return

        retry_ctx = chat_ctx.copy()
        if first.spoken:
            retry_ctx.add_message(role="assistant", content="".join(first.spoken).strip())
        retry_ctx.add_message(role="system", content=retry_note_for(first.blocked, ctx.language))
        if first.spoken and not first.spoken[-1][-1:].isspace():
            yield " "  # keep the retry's first word apart from what was already said
        second = SentenceFilter(ctx, "retry")
        stream = Agent.default.llm_node(self, retry_ctx, tools, model_settings)
        async for out in filter_claims(stream, second):
            yield out
        if (phrase := filler.after(second)) is not None:
            yield phrase
        if second.needs_retry():
            reasons = second.blocked
            record = {"event": "speech_fallback", "reason": reasons[0]}  # reason only
            logger.info(json.dumps(record, separators=(",", ":")))
            said = "".join(second.spoken)
            prefix = " " if said and not said[-1:].isspace() else ""
            yield prefix + fallback_for(reasons, ctx.language)

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
    async def practice_info(
        self,
        context: RunContext[Any],
        topic: Literal[
            "opening_hours",
            "address",
            "parking",
            "repeat_prescription",
            "referral",
            "sick_note",
            "holidays",
            "insurance_card",
            "new_patients",
            "other",
        ],
    ) -> str:
        """Practice information for a question about the practice itself. Answer only
        with the returned text; never state practice facts without it.

        Args:
            topic: Exactly one of these names. opening_hours: when the practice is open,
                today, on a given weekday, at the weekend. address: where the practice
                is, how to get there, access and lift. parking: parking spaces and car
                parks. repeat_prescription: reordering regular medication.
                referral: referral to a specialist. sick_note: Krankschreibung or AU.
                holidays: public holidays, practice holidays, Christmas, covering
                practice. insurance_card: bringing or forgetting the health insurance
                card. new_patients: joining the practice as a new patient. other:
                anything else, for example home visits, vaccinations, prices,
                pharmacies or hospitals.
        """
        language = self.language_state(self.chat_ctx).language
        return _run(lambda: self._service.practice_info(topic, language))

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
