"""Session wiring checks that need no network, keys or model downloads."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2 as httpx  # what openai and the LiveKit plugin use
from livekit.agents import llm
from openai import AsyncClient

from praxis_telefon.agent import main
from praxis_telefon.config import Settings

ENV = {
    "LLM_PROVIDER": "openrouter",
    "LLM_API_KEY": "sk-test-fake",
    "LLM_MODEL": "google/gemini-3.8-flash",
    "DEEPGRAM_API_KEY": "dg-test-fake",
    "ELEVENLABS_API_KEY": "el-test-fake",
    "ELEVENLABS_VOICE_ID": "voice-test-fake",
}


def settings(**overrides: str) -> Settings:
    return Settings.from_env({**ENV, **overrides})


def test_stt_uses_german_by_default() -> None:
    # Regression: nova-3 "multi" transcribed German callers as English.
    opts = main.stt_options(settings())
    assert opts["language"] == "de"
    assert "language_hint" not in opts


def test_stt_follows_default_language() -> None:
    assert main.stt_options(settings(DEFAULT_LANGUAGE="en"))["language"] == "en"


def test_stt_multi_is_opt_in() -> None:
    # STT A/B 2026-11-07 (docs/experiments/stt-language.md): multi +8.8 points on
    # synthetic German, but the M1 failure (real non-native German heard as English)
    # needs a voice test first, so multi is chosen explicitly, not by default.
    opts = main.stt_options(settings(STT_LANGUAGE="multi"))
    assert opts["language"] == "multi"
    assert "language_hint" not in opts  # no effect in the A/B (638 of 640 identical)
    assert main.stt_options(settings(STT_LANGUAGE="de", DEFAULT_LANGUAGE="en"))["language"] == "de"


def test_stt_language_hint_is_sent_with_multi() -> None:
    # Voice test 2026-11-07, call 1: multi tagged most of the user's accented German finals
    # "te" (Telugu) and one "no"; the callback number was lost. The hint biases multi to
    # de/en; opt-in until a voice test shows it helps.
    opts = main.stt_options(settings(STT_LANGUAGE="multi", STT_LANGUAGE_HINT="de, EN"))
    assert opts["language"] == "multi"
    assert opts["language_hint"] == ["de", "en"]


def test_stt_language_hint_needs_multi() -> None:
    # Deepgram applies a hint only in multi mode; anything else would silently ignore it.
    import pytest

    from praxis_telefon.config import ConfigError

    with pytest.raises(ConfigError, match="STT_LANGUAGE_HINT"):
        settings(STT_LANGUAGE_HINT="de,en")
    with pytest.raises(ConfigError, match="STT_LANGUAGE_HINT"):
        settings(STT_LANGUAGE="de", STT_LANGUAGE_HINT="de,en")


def test_stt_language_hint_rejects_unknown_languages() -> None:
    import pytest

    from praxis_telefon.config import ConfigError

    with pytest.raises(ConfigError, match="STT_LANGUAGE_HINT"):
        settings(STT_LANGUAGE="multi", STT_LANGUAGE_HINT="de,te")


def test_stt_language_rejects_unknown_values() -> None:
    import pytest

    from praxis_telefon.config import ConfigError

    with pytest.raises(ConfigError, match="STT_LANGUAGE"):
        settings(STT_LANGUAGE="fr")


def _request_sent(s: Settings, *, session_id: str | None = None) -> httpx.Request:
    """The HTTP request build_llm's LLM sends for one reply (mock transport, no network)."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        chunk = {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": s.llm_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hallo"},
                    "finish_reason": "stop",
                }
            ],
        }
        body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    client = AsyncClient(
        api_key="sk-test-fake",
        base_url=main.OPENROUTER_URL,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    model = main.build_llm(s, session_id=session_id, client=client)

    async def reply() -> None:
        ctx = llm.ChatContext()
        ctx.add_message(role="user", content="Haben Sie morgen einen Termin frei?")
        async with model.chat(chat_ctx=ctx) as stream:
            async for _ in stream:
                pass

    asyncio.run(reply())
    assert len(sent) == 1
    return sent[0]


def _body(request: httpx.Request) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(request.content)
    return body


def test_llm_reasoning_effort_is_passed() -> None:
    # Regression: no effort was sent, so Gemini reasoned at "medium" (llm_ttft 6.8 s).
    assert _body(_request_sent(settings()))["reasoning_effort"] == "low"
    none = _body(_request_sent(settings(LLM_REASONING_EFFORT="none")))
    assert none["reasoning_effort"] == "none"


def test_openrouter_prefers_latency_and_denies_data_collection() -> None:
    body = _body(_request_sent(settings()))
    assert body["model"] == "google/gemini-3.8-flash"
    assert body["provider"] == {"sort": "latency", "data_collection": "deny"}


def test_prompt_caching_is_requested() -> None:
    # Arm A live run 2026-11-06: 0 of 1,268,848 prompt tokens came from the cache.
    body = _body(_request_sent(settings()))
    assert body["cache_control"] == {"type": "ephemeral"}


def test_each_call_gets_its_own_random_session_id() -> None:
    # Sticky routing keeps a call on the provider that holds its cache.
    assert _body(_request_sent(settings(), session_id="call-1"))["session_id"] == "call-1"
    first = _body(_request_sent(settings()))["session_id"]
    second = _body(_request_sent(settings()))["session_id"]
    assert first != second
    assert len(first) == 32 and int(first, 16) >= 0  # uuid4 hex, no caller data


def test_request_names_the_app_and_keeps_the_key_out_of_the_body() -> None:
    request = _request_sent(settings())
    assert request.headers["X-Title"] == "Praxis-Telefon"
    assert request.url.host == "openrouter.ai"
    assert b"sk-test-fake" not in request.content


def test_endpointing_tuned_for_phone() -> None:
    # Regression: default max_delay 3.0 s gave 2.5 s end-of-turn waits.
    assert main.ENDPOINTING["max_delay"] <= 1.5
    assert main.ENDPOINTING["min_delay"] <= main.ENDPOINTING["max_delay"]


def test_interruption_detection_stays_local() -> None:
    # Regression: "adaptive" mode streams caller audio to agent-gateway.livekit.cloud.
    assert main.INTERRUPTION["mode"] == "vad"


def test_false_interruption_resumes_after_one_second() -> None:
    # Voice test 2026-11-07: "mhm" and a cough paused the agent for about 2 s (the 2.0 s
    # default) before it resumed. 1.0 s; min_duration stays at LiveKit's 0.5 s.
    assert main.INTERRUPTION["false_interruption_timeout"] == 1.0
    assert "min_duration" not in main.INTERRUPTION


def test_missing_database_gives_clear_error(tmp_path: Path) -> None:
    import pytest

    s = settings(DATABASE_PATH=str(tmp_path / "missing.db"))
    with pytest.raises(main.DatabaseMissing, match=r"praxis_telefon\.db\.seed"):
        main.open_database(s)


def test_no_trace_exporter_configured() -> None:
    # Tool arguments are span attributes; spans must not leave the process.
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    assert isinstance(provider, trace.ProxyTracerProvider | trace.NoOpTracerProvider)


def test_receptionist_registers_all_tools(tmp_path: Path) -> None:
    from praxis_telefon.agent.receptionist import Receptionist
    from praxis_telefon.agent.tool_service import ToolService
    from praxis_telefon.db.schema import connect, init_db

    conn = connect(tmp_path / "t.db")
    init_db(conn)
    agent = Receptionist(ToolService(conn, main.practice_now))
    names = sorted(getattr(t, "id", "") for t in agent.tools)
    assert names == [
        "book",
        "cancel_booking",
        "find_booking",
        "find_slots",
        "move_booking",
        "practice_info",
        "take_message",
    ]
    conn.close()


def test_find_slots_takes_the_callers_words_not_dates(tmp_path: Path) -> None:
    # M3 Part B: the LLM passes "when" (caller's words); dates are computed in code.
    from livekit.agents.llm.utils import build_legacy_openai_schema

    from praxis_telefon.agent.receptionist import Receptionist
    from praxis_telefon.agent.tool_service import ToolService
    from praxis_telefon.db.schema import connect, init_db

    conn = connect(tmp_path / "t.db")
    init_db(conn)
    agent = Receptionist(ToolService(conn, main.practice_now))
    tool = next(t for t in agent.tools if getattr(t, "id", "") == "find_slots")
    params = build_legacy_openai_schema(tool)["function"]["parameters"]  # type: ignore[arg-type]
    assert "when" in params["required"]
    assert "date_from" not in params["properties"] and "date_to" not in params["properties"]
    assert "nächsten Dienstag" in params["properties"]["when"]["description"]
    conn.close()
