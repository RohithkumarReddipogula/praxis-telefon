"""LiveKit entrypoint for the Praxis-Telefon voice agent.

Run with ``uv run python -m praxis_telefon.agent.main console`` (see README).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
)
from livekit.agents.llm import ChatMessage
from livekit.agents.voice.turn import EndpointingOptions, InterruptionOptions
from livekit.plugins import deepgram, elevenlabs, openai, silero
from livekit.plugins.openai.models import OpenRouterProviderPreferences
from openai import AsyncClient

from praxis_telefon.agent.claim_check import ClaimTracker
from praxis_telefon.agent.console_audio import release_console_audio
from praxis_telefon.agent.prompts import GREETING_DE
from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_filler import is_tool_filler
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.agent.turn_log import build_turn_record, format_turn_record
from praxis_telefon.agent.unrecognized_speech import attach_unrecognized_speech
from praxis_telefon.config import Settings
from praxis_telefon.db.schema import PRACTICE_TZ, connect
from praxis_telefon.logging_privacy import install_pii_filter

logger = logging.getLogger("praxis_telefon.turns")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
APP_NAME = "Praxis-Telefon"
ELEVENLABS_MODEL = "eleven_flash_v2_5"
DEEPGRAM_MODEL = "nova-3"

# Phone tuning: answer within ~1.5 s even if the turn detector is unsure.
# Trade-off: long mid-sentence pauses (dates of birth, insurance numbers) may get cut off.
ENDPOINTING: EndpointingOptions = {"min_delay": 0.4, "max_delay": 1.5}

# "vad": barge-in is decided locally by Silero. The default "adaptive" detector in
# console/dev mode streams caller audio to agent-gateway.livekit.cloud.
INTERRUPTION: InterruptionOptions = {"mode": "vad"}

server = AgentServer()


def practice_now() -> datetime:
    return datetime.now(PRACTICE_TZ)


class DatabaseMissing(RuntimeError):
    pass


def open_database(settings: Settings) -> ToolService:
    path = settings.database_path
    if not path.exists():
        raise DatabaseMissing(
            f"Database {path} not found. Create it with: uv run python -m praxis_telefon.db.seed"
        )
    return ToolService(connect(path), practice_now)


def stt_options(settings: Settings) -> dict[str, Any]:
    """Deepgram options. The call's language unless STT_LANGUAGE says otherwise.

    `multi` beat `de` on synthetic German (docs/experiments/stt-language.md) but in M1
    heard real non-native German as English, so it stays opt-in until a voice test.
    """
    return {"model": DEEPGRAM_MODEL, "language": settings.stt_language or settings.default_language}


def openrouter_provider() -> OpenRouterProviderPreferences:
    # Fastest provider first; skip providers that store or train on prompts.
    return {"sort": "latency", "data_collection": "deny"}


def openrouter_body(session_id: str) -> dict[str, Any]:
    """OpenRouter fields beyond the OpenAI API, sent with every request of the call."""
    return {
        "provider": openrouter_provider(),
        # Automatic prompt caching (Anthropic): the cache breakpoint sits at the end of
        # each request, so a turn reads all the previous turn sent (tools, system
        # prompt, history) at 0.1x the input price; 5 minutes TTL, writes 1.25x.
        "cache_control": {"type": "ephemeral"},
        # Sticky routing from the first request on: later turns of the call go to the
        # provider that holds the cache. Random per call, no caller data.
        "session_id": session_id,
    }


def build_llm(
    settings: Settings, *, session_id: str | None = None, client: AsyncClient | None = None
) -> openai.LLM:
    """The LLM for one call, via OpenRouter. ``client`` is for tests only."""
    return openai.LLM(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=OPENROUTER_URL,
        client=client,
        reasoning_effort=settings.llm_reasoning_effort,
        extra_body=openrouter_body(session_id or uuid.uuid4().hex),
        extra_headers={"X-Title": APP_NAME},
    )


def turn_handling_options() -> TurnHandlingOptions:
    return TurnHandlingOptions(
        # v1-mini runs locally (supports German); "v1" would use LiveKit Cloud Inference.
        turn_detection=inference.TurnDetector(version="v1-mini"),
        endpointing=ENDPOINTING,
        interruption=INTERRUPTION,
    )


def build_session(settings: Settings) -> AgentSession[Any]:
    return AgentSession(
        stt=deepgram.STT(**stt_options(settings), api_key=settings.deepgram_api_key),
        llm=build_llm(settings),
        tts=elevenlabs.TTS(
            voice_id=settings.elevenlabs_voice_id,
            model=ELEVENLABS_MODEL,
            api_key=settings.elevenlabs_api_key,
        ),
        vad=silero.VAD.load(),
        turn_handling=turn_handling_options(),
    )


def attach_turn_logger(session: AgentSession[Any]) -> None:
    # last_user: the caller's STT/EOU metrics, for the first record after the line only.
    # caller_stopped_at: kept for every record until the next line (filler, then answer).
    state: dict[str, Any] = {"turn": 0, "last_user": None, "caller_stopped_at": None}

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if not isinstance(item, ChatMessage):
            return
        if item.role == "user":
            state["last_user"] = dict(item.metrics)
            stopped = item.metrics.get("stopped_speaking_at")
            state["caller_stopped_at"] = stopped if isinstance(stopped, int | float) else None
        elif item.role == "assistant":
            state["turn"] += 1
            record = build_turn_record(
                state["turn"],
                state["last_user"],
                dict(item.metrics),
                interrupted=item.interrupted,
                is_filler=is_tool_filler(item.text_content or ""),
                caller_stopped_at=state["caller_stopped_at"],
            )
            state["last_user"] = None
            logger.info(format_turn_record(record))


def attach_claim_check(session: AgentSession[Any]) -> None:
    """Log when the agent says a booking changed before the matching tool succeeded."""
    tracker = ClaimTracker()

    @session.on("function_tools_executed")
    def _on_tools(ev: FunctionToolsExecutedEvent) -> None:
        for output in ev.function_call_outputs:
            tracker.tool_result(output.name, ok=not output.is_error)

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if not isinstance(item, ChatMessage) or item.role != "assistant":
            return
        for action in tracker.unbacked(item.text_content or ""):
            # Action name only, never the spoken text.
            record = {"event": "unbacked_claim", "action": action}
            logger.warning(json.dumps(record, separators=(",", ":")))


@server.rtc_session(agent_name="praxis-telefon")
async def entrypoint(ctx: JobContext) -> None:
    # Runs in the job process after LiveKit configured logging there.
    install_pii_filter()
    settings = Settings.from_env()
    logger.setLevel(settings.log_level)

    service = open_database(settings)
    session = build_session(settings)
    attach_turn_logger(session)
    attach_claim_check(session)
    ctx.add_shutdown_callback(service.close)
    # Console mode: stop mic/speaker before the job loop closes (Ctrl+C traceback).
    ctx.add_shutdown_callback(release_console_audio)

    # record=False: do not upload call audio/transcripts to LiveKit Cloud observability.
    agent = Receptionist(service)
    # Caller spoke but STT produced nothing: counts as a strike (two in a row: handover).
    attach_unrecognized_speech(session, agent.on_unrecognized_speech)

    await session.start(agent=agent, room=ctx.room, record=False)
    session.say(GREETING_DE, allow_interruptions=False)


if __name__ == "__main__":
    # Only the running agent loads .env (job processes inherit the environment);
    # importing this module, e.g. in tests, never does.
    load_dotenv()
    cli.run_app(server)
