# CLAUDE.md

Praxis-Telefon: German-first, English-capable AI phone receptionist for a fictional doctor's office.
Stack: Python 3.12, LiveKit Agents, Deepgram STT, ElevenLabs/Cartesia TTS, one tool-calling LLM, SQLite, pytest.

## Rules

1. **Never read or print `.env`.** Don't cat, grep, open, or echo it or its values. Use `.env.example` for variable names.
2. **Ask before** `git commit`, `git push`, `rm`, `sudo`, or any install (`uv add`, `uv sync` with new deps, `pip`, `brew`, etc.).
3. **Synthetic data only.** Made-up German names, phone numbers in reserved/fictional ranges, no real patient data, no real call recordings.
4. **Every bug fix gets a regression test.** Name it after the bug; it must fail before the fix and pass after.

## Commands

```
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
uv run pytest -m live   # real LLM calls via .env keys; the user runs these, not CI
```

Run the agent (the user runs this; it needs real keys and a mic):

```
uv run -m livekit.agents download-files                     # once: VAD + turn-detector weights
uv run python -m praxis_telefon.db.seed                     # once: synthetic data (--reset to recreate)
uv run python -m praxis_telefon.agent.main console          # talk via mic in the terminal
uv run python -m praxis_telefon.agent.main dev              # LiveKit Cloud (needs LIVEKIT_*)
```


## Layout

- `src/praxis_telefon/agent/` – LiveKit voice agent wiring and prompts
- `src/praxis_telefon/tools/` – LLM-callable functions (typed, pure where possible)
- `src/praxis_telefon/db/` – SQLite schema, access layer, synthetic seed data
- `eval/` – scripted conversation scenarios and scoring
- `tests/` – unit and regression tests; no network or API keys required
