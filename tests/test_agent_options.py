"""Session wiring checks that need no network, keys or model downloads."""

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


def test_llm_reasoning_effort_is_passed() -> None:
    # Regression: no effort was sent, so Gemini reasoned at "medium" (llm_ttft 6.8 s).
    assert main.llm_options(settings())["reasoning_effort"] == "low"
    assert main.llm_options(settings(LLM_REASONING_EFFORT="none"))["reasoning_effort"] == "none"


def test_openrouter_prefers_latency_and_denies_data_collection() -> None:
    provider = main.llm_options(settings())["provider"]
    assert provider["sort"] == "latency"
    assert provider["data_collection"] == "deny"


def test_llm_options_do_not_contain_api_key() -> None:
    assert "sk-test-fake" not in repr(main.llm_options(settings()))


def test_endpointing_tuned_for_phone() -> None:
    # Regression: default max_delay 3.0 s gave 2.5 s end-of-turn waits.
    assert main.ENDPOINTING["max_delay"] <= 1.5
    assert main.ENDPOINTING["min_delay"] <= main.ENDPOINTING["max_delay"]


def test_interruption_detection_stays_local() -> None:
    # Regression: "adaptive" mode streams caller audio to agent-gateway.livekit.cloud.
    assert main.INTERRUPTION["mode"] == "vad"
