from pathlib import Path

import pytest

from praxis_telefon.config import ConfigError, Settings

FAKE_SECRET = "sk-test-fake-secret-123"

VALID_ENV = {
    "LLM_PROVIDER": "openrouter",
    "LLM_API_KEY": FAKE_SECRET,
    "LLM_MODEL": "google/gemini-3.8-flash",
    "DEEPGRAM_API_KEY": "dg-test-fake",
    "ELEVENLABS_API_KEY": "el-test-fake",
    "ELEVENLABS_VOICE_ID": "voice-test-fake",
}

REQUIRED = [
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "DEEPGRAM_API_KEY",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
]


def test_valid_env_parses() -> None:
    s = Settings.from_env(VALID_ENV)
    assert s.llm_provider == "openrouter"
    assert s.llm_model == "google/gemini-3.8-flash"
    assert s.llm_api_key == FAKE_SECRET
    assert s.elevenlabs_voice_id == "voice-test-fake"


def test_defaults_apply() -> None:
    s = Settings.from_env(VALID_ENV)
    assert s.tts_provider == "elevenlabs"
    assert s.default_language == "de"
    assert s.log_level == "INFO"
    assert s.database_path == Path("./data/praxis.db")


def test_optional_values_override_defaults() -> None:
    env = {**VALID_ENV, "DEFAULT_LANGUAGE": "en", "LOG_LEVEL": "DEBUG", "DATABASE_PATH": "x.db"}
    s = Settings.from_env(env)
    assert s.default_language == "en"
    assert s.log_level == "DEBUG"
    assert s.database_path == Path("x.db")


@pytest.mark.parametrize("name", REQUIRED)
def test_missing_required_variable_is_named(name: str) -> None:
    env = {k: v for k, v in VALID_ENV.items() if k != name}
    with pytest.raises(ConfigError, match=name):
        Settings.from_env(env)


@pytest.mark.parametrize("name", REQUIRED)
def test_blank_required_variable_is_rejected(name: str) -> None:
    with pytest.raises(ConfigError, match=name):
        Settings.from_env({**VALID_ENV, name: "   "})


def test_all_problems_reported_at_once() -> None:
    with pytest.raises(ConfigError) as exc:
        Settings.from_env({})
    for name in REQUIRED:
        assert name in str(exc.value)


def test_choices_are_case_insensitive() -> None:
    env = {
        **VALID_ENV,
        "LLM_PROVIDER": "OpenRouter",
        "DEFAULT_LANGUAGE": "DE",
        "LOG_LEVEL": "debug",
    }
    s = Settings.from_env(env)
    assert (s.llm_provider, s.default_language, s.log_level) == ("openrouter", "de", "DEBUG")


def test_unsupported_llm_provider_rejected() -> None:
    with pytest.raises(ConfigError, match="LLM_PROVIDER"):
        Settings.from_env({**VALID_ENV, "LLM_PROVIDER": "openai"})


def test_cartesia_not_supported_yet() -> None:
    with pytest.raises(ConfigError, match="TTS_PROVIDER"):
        Settings.from_env({**VALID_ENV, "TTS_PROVIDER": "cartesia"})


def test_invalid_language_rejected() -> None:
    with pytest.raises(ConfigError, match="DEFAULT_LANGUAGE"):
        Settings.from_env({**VALID_ENV, "DEFAULT_LANGUAGE": "fr"})


def test_repr_hides_secrets() -> None:
    text = repr(Settings.from_env(VALID_ENV))
    for secret in ("LLM_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY"):
        assert VALID_ENV[secret] not in text
    assert "google/gemini-3.8-flash" in text


def test_error_message_does_not_echo_values() -> None:
    with pytest.raises(ConfigError) as exc:
        Settings.from_env({**VALID_ENV, "LLM_PROVIDER": FAKE_SECRET})
    assert FAKE_SECRET not in str(exc.value)


def test_reads_os_environ_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in VALID_ENV.items():
        monkeypatch.setenv(k, v)
    assert Settings.from_env().llm_model == "google/gemini-3.8-flash"


def test_reasoning_effort_defaults_to_low() -> None:
    assert Settings.from_env(VALID_ENV).llm_reasoning_effort == "low"


def test_invalid_reasoning_effort_rejected() -> None:
    with pytest.raises(ConfigError, match="LLM_REASONING_EFFORT"):
        Settings.from_env({**VALID_ENV, "LLM_REASONING_EFFORT": "turbo"})
