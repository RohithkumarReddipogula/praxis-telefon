# Praxis-Telefon

German-first, English-capable AI phone receptionist for a fictional doctor's office (portfolio project).

Status: M1 – first voice loop (German/English, per-turn timing logs).

```
uv sync
uv run pytest
uv run pytest -m live   # optional: real LLM regression tests, needs .env
```

## Latency so far

Single manual test calls (MacBook mic, German, console mode). These are individual
measurements, not statistics; proper p50/p95 numbers will come from the eval harness.

| LLM (via OpenRouter) | Reasoning | Max endpointing | LLM TTFT | EOU delay | TTS TTFB | End-to-end |
|---|---|---|---|---|---|---|
| Gemini 3.8 Flash | medium (default) | 3.0 s (default) | 6.8 s | 2.5 s | – | 9.5 s |
| Gemini 3.8 Flash | low | 1.5 s | 2.96 s | – | – | 3.7 s |
| Claude Haiku 5.5 | none | 1.5 s | 0.79 s | 0.58 s | 0.16 s | 1.6 s |

STT (Deepgram nova-3, `de`) was about 0.3 s. Dashes are values not recorded for that call.
Default config is the last row.

## Run the agent

Copy `.env.example` to `.env` and fill it in, then:

```
uv run -m livekit.agents download-files                     # once: VAD + turn-detector weights
uv run python -m praxis_telefon.agent.main console          # talk via mic in the terminal
uv run python -m praxis_telefon.agent.main dev              # LiveKit Cloud (needs LIVEKIT_*)
```
