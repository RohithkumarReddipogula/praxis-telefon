"""LiveKit entrypoint for the Praxis-Telefon voice agent.

Run with ``uv run python -m praxis_telefon.agent.main console`` (see README).
"""

from __future__ import annotations

import json
import logging
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

from praxis_telefon.agent.claim_check import ClaimTracker
from praxis_telefon.agent.console_audio import release_console_audio
from praxis_telefon.agent.prompts import GREETING_DE
from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.agent.turn_log import build_turn_record, format_turn_record
from praxis_telefon.agent.unrecognized_speech import attach_unrecognized_speech
from praxis_telefon.config import Settings
from praxis_telefon.db.schema import PRACTICE_TZ, connect
from praxis_telefon.logging_privacy import install_pii_filter

logger = logging.getLogger("praxis_telefon.turns")

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
    """Deepgram options. A fixed language beats `multi` for German accuracy."""
    return {"model": DEEPGRAM_MODEL, "language": settings.default_language}


def openrouter_provider() -> OpenRouterProviderPreferences:
    # Fastest provider first; skip providers that store or train on prompts.
    return {"sort": "latency", "data_collection": "deny"}


def llm_options(settings: Settings) -> dict[str, Any]:
    return {
        "model": settings.llm_model,
        "reasoning_effort": settings.llm_reasoning_effort,
        "provider": openrouter_provider(),
        "app_name": "Praxis-Telefon",
    }


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
        llm=openai.LLM.with_openrouter(**llm_options(settings), api_key=settings.llm_api_key),
        tts=elevenlabs.TTS(
            voice_id=settings.elevenlabs_voice_id,
            model=ELEVENLABS_MODEL,
            api_key=settings.elevenlabs_api_key,
        ),
        vad=silero.VAD.load(),
        turn_handling=turn_handling_options(),
    )


def attach_turn_logger(session: AgentSession[Any]) -> None:
    state: dict[str, Any] = {"turn": 0, "last_user": None}

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if not isinstance(item, ChatMessage):
            return
        if item.role == "user":
            state["last_user"] = dict(item.metrics)
        elif item.role == "assistant":
            state["turn"] += 1
            record = build_turn_record(
                state["turn"],
                state["last_user"],
                dict(item.metrics),
                interrupted=item.interrupted,
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
