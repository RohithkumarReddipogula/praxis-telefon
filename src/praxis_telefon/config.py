"""Typed settings loaded from environment variables.

This is the only module that reads configuration from the environment. Secrets are
excluded from ``repr`` so they never end up in logs or tracebacks, and error messages
name missing or invalid variables without echoing their values.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from praxis_telefon.agent.demo import DemoLimits

SUPPORTED_LLM_PROVIDERS = ("openrouter",)
SUPPORTED_TTS_PROVIDERS = ("elevenlabs",)
SUPPORTED_LANGUAGES = ("de", "en")
# Deepgram language; "" follows DEFAULT_LANGUAGE. "multi": docs/experiments/stt-language.md
STT_LANGUAGES = ("de", "en", "multi")
# Deepgram language_hint, multi only (voice test 2026-11-07: multi tagged accented German
# as Telugu); opt-in until a voice test shows it helps.
STT_HINT_LANGUAGES = ("de", "en")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
# OpenRouter `reasoning_effort`. Gemini 3.x Flash reasons mandatorily (lowest: "low");
# Claude Haiku 5.5 accepts "none".
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")

Language = Literal["de", "en"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]


class ConfigError(ValueError):
    """Raised when required settings are missing or invalid."""


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    llm_api_key: str = field(repr=False)
    llm_reasoning_effort: ReasoningEffort
    deepgram_api_key: str = field(repr=False)
    tts_provider: str
    elevenlabs_api_key: str = field(repr=False)
    elevenlabs_voice_id: str
    default_language: Language
    stt_language: str  # "" = default_language
    stt_language_hint: tuple[str, ...]  # empty: no hint
    log_level: str
    database_path: Path
    # M5 public demo (agent/demo.py): limits, keyed caller hash, synthetic DB at start.
    demo_mode: bool = False
    demo_hash_key: str = field(default="", repr=False)
    demo_limits: DemoLimits = field(default_factory=DemoLimits)
    demo_usage_db: Path = Path("/tmp/praxis-demo-usage.db")
    seed_on_start: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``environ`` (defaults to ``os.environ``).

        Collects every problem before raising, so one run reports all of them.
        """
        env = os.environ if environ is None else environ
        problems: list[str] = []

        def get(name: str, default: str = "") -> str:
            return env.get(name, default).strip() or default

        def require(name: str) -> str:
            value = get(name)
            if not value:
                problems.append(f"{name} is missing or empty")
            return value

        def choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
            value = get(name, default)
            value = value.upper() if allowed is LOG_LEVELS else value.lower()
            if value not in allowed:
                problems.append(f"{name} must be one of {', '.join(allowed)}")
            return value

        llm_provider = choice("LLM_PROVIDER", "", SUPPORTED_LLM_PROVIDERS)
        llm_api_key = require("LLM_API_KEY")
        llm_model = require("LLM_MODEL")
        reasoning_effort = choice("LLM_REASONING_EFFORT", "low", REASONING_EFFORTS)
        deepgram_api_key = require("DEEPGRAM_API_KEY")
        tts_provider = choice("TTS_PROVIDER", "elevenlabs", SUPPORTED_TTS_PROVIDERS)
        elevenlabs_api_key = require("ELEVENLABS_API_KEY")
        elevenlabs_voice_id = require("ELEVENLABS_VOICE_ID")
        default_language = choice("DEFAULT_LANGUAGE", "de", SUPPORTED_LANGUAGES)
        stt_language = get("STT_LANGUAGE").lower()
        if stt_language and stt_language not in STT_LANGUAGES:
            problems.append(f"STT_LANGUAGE must be empty or one of {', '.join(STT_LANGUAGES)}")
        hint = tuple(h.strip().lower() for h in get("STT_LANGUAGE_HINT").split(",") if h.strip())
        if hint and stt_language != "multi":
            problems.append("STT_LANGUAGE_HINT needs STT_LANGUAGE=multi")
        elif any(h not in STT_HINT_LANGUAGES for h in hint):
            problems.append(f"STT_LANGUAGE_HINT may only list {', '.join(STT_HINT_LANGUAGES)}")
        log_level = choice("LOG_LEVEL", "INFO", LOG_LEVELS)
        database_path = Path(get("DATABASE_PATH", "./data/praxis.db"))

        def flag(name: str) -> bool:
            return get(name).lower() in ("1", "true", "yes", "on")

        def positive(name: str, default: int) -> int:
            raw = get(name, str(default))
            if not raw.isdigit() or int(raw) <= 0:
                problems.append(f"{name} must be a positive whole number")
                return default
            return int(raw)

        demo_mode = flag("DEMO_MODE")
        demo_hash_key = get("DEMO_HASH_KEY")
        if demo_mode and len(demo_hash_key) < 16:
            problems.append("DEMO_HASH_KEY must be set (at least 16 characters) with DEMO_MODE")
        demo_limits = DemoLimits(
            max_call_s=positive("DEMO_MAX_CALL_S", 300),
            warn_before_s=positive("DEMO_WARN_BEFORE_S", 60),
            calls_per_caller=positive("DEMO_CALLS_PER_CALLER", 3),
            calls_per_day=positive("DEMO_CALLS_PER_DAY", 15),
            minutes_per_day=positive("DEMO_MINUTES_PER_DAY", 45),
            web_minutes_per_day=positive("DEMO_WEB_MINUTES_PER_DAY", 20),
            web_minutes_per_month=positive("DEMO_WEB_MINUTES_PER_MONTH", 60),
        )

        if problems:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))

        return cls(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_reasoning_effort=_as_effort(reasoning_effort),
            deepgram_api_key=deepgram_api_key,
            tts_provider=tts_provider,
            elevenlabs_api_key=elevenlabs_api_key,
            elevenlabs_voice_id=elevenlabs_voice_id,
            default_language="en" if default_language == "en" else "de",
            stt_language=stt_language,
            stt_language_hint=hint,
            log_level=log_level,
            database_path=database_path,
            demo_mode=demo_mode,
            demo_hash_key=demo_hash_key,
            demo_limits=demo_limits,
            demo_usage_db=Path(get("DEMO_USAGE_DB", "/tmp/praxis-demo-usage.db")),
            seed_on_start=flag("SEED_ON_START"),
        )


def _as_effort(value: str) -> ReasoningEffort:
    for effort in ("none", "minimal", "low", "medium", "high"):
        if value == effort:
            return effort
    raise ConfigError("LLM_REASONING_EFFORT is invalid")  # unreachable after validation
