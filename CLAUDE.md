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
uv run pytest -m live <files> -v -ra   # real LLM calls via .env keys; not in CI
```

Live tests (`tests/live/`): Claude may run them itself with the command above (they read
`.env` in-process; never open or print it). If any fail, show the failure and a proposed
fix before changing code; when all pass, show the result and ask before committing.
`LIVE_CORPUS=<path>` additionally writes every sentence the LLM produced (before filtering)
per test, with outcome and blocked reasons; used to extend `tests/data/live_sentences.jsonl`.

Working style: plan first and wait for approval; regression test first (show it failing);
after each part run ruff, mypy and pytest, then ask before commit and push.

Run the agent (the user runs this; it needs real keys and a mic):

```
uv run -m livekit.agents download-files                     # once: VAD + turn-detector weights
uv run python -m praxis_telefon.db.seed                     # once: synthetic data (--reset to recreate)
uv run python -m praxis_telefon.agent.main console          # talk via mic in the terminal
uv run python -m praxis_telefon.agent.main dev              # LiveKit Cloud (needs LIVEKIT_*)
```


## Layout

- `src/praxis_telefon/agent/` – LiveKit voice agent wiring and prompts
  - `receptionist.py` – `llm_node`: safety screening, then the LLM through the sentence filter
  - `claim_gate.py` / `claim_check.py` / `speech_filters.py` – checks every sentence before
    it is spoken (claims, meta talk, new-or-existing question, self-computed weekdays)
- `src/praxis_telefon/safety/` – deterministic emergency/suicide/human-request checks,
  two-strike handover, fixed messages
- `src/praxis_telefon/tools/` – LLM-callable functions (typed, pure where possible);
  `relative_dates.py` turns the caller's words into dates
- `src/praxis_telefon/db/` – SQLite schema, access layer, synthetic seed data (28 days)
- `eval/` – scripted conversation scenarios and scoring
- `tests/` – unit and regression tests; no network or API keys required
  - `tests/live/` – live LLM tests (`harness.py`: scripted caller, text-mode session)
  - `tests/data/` – live sentence corpus and reviewed known-bad sentences

## Status (M3, as of 2026-11-06)

Done and pushed:
- **Part A, safety:** emergency and suicide phrases (DE/EN) get fixed, uninterruptible
  messages without the LLM; booking off for the rest of the call. Request for a human or
  two strikes (empty transcript / low STT confidence; short answers never count) hand
  over to message-only mode. Decisions are replayed from the conversation, so discarded
  preemptive runs cannot change state.
- **Part B, relative dates:** `find_slots(when=...)` takes the caller's words; code
  resolves them in Europe/Berlin (weekdays, ranges, closed days, DST, 28-day horizon).
- **Fixes since:** spoken callback numbers converted to digits; sentence filter blocks
  unbacked claims (only the matching tool backs a claim), spoken reasoning, repeated or
  unneeded new-or-existing questions, and weekdays the LLM worked out; at most one retry
  per reply, then a fixed sentence. False-positive check: corpus of 149 passing live tests.
- Last live run: 75/75 passed (safety, tool calls, dates).

Open:
- **Reasoning leak frequency:** Haiku (reasoning "none") still narrates before tool calls,
  often in English (13 sentences blocked in 75 live tests). The filter catches it; worth
  trying reasoning "low" or German tool descriptions and measuring e2e.
- **Part B, English switching (next):** A/B measurement of Deepgram `de` vs `multi` on
  synthetic ElevenLabs audio (a few cents; ask before running), then decide.
- **Part C not started:** FAQ with local retrieval, barge-in tuning.
- Afterwards: README and CLAUDE.md update for all of M3; the user's local
  `data/praxis.db` must be reseeded for 28 days (`... db.seed --reset`, the user runs it).
