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
  - `tool_filler.py` – code-spoken "Einen Moment, bitte." when a reply calls a tool silently
  - `main.py` – `build_llm`: OpenRouter with prompt caching and a per-call `session_id`
- `src/praxis_telefon/safety/` – deterministic emergency/suicide/human-request checks,
  two-strike handover, fixed messages
- `src/praxis_telefon/tools/` – LLM-callable functions (typed, pure where possible);
  `relative_dates.py` turns the caller's words into dates
- `src/praxis_telefon/db/` – SQLite schema, access layer, synthetic seed data (28 days)
- `eval/` – scripted conversation scenarios and scoring; `leak_report.py` compares live runs
- `docs/experiments/` – experiment write-ups (`reasoning-leak.md`)
- `tests/` – unit and regression tests; no network or API keys required
  - `tests/live/` – live LLM tests (`harness.py`: scripted caller, text-mode session)
  - `tests/data/` – live sentence corpus and reviewed known-bad sentences

## Status (M3, as of 2026-11-07)

Done and pushed:
- **Part A, safety:** emergency and suicide phrases (DE/EN) get fixed, uninterruptible
  messages without the LLM; booking off for the rest of the call. Request for a human or
  two strikes (empty transcript / low STT confidence; short answers never count) hand
  over to message-only mode. Decisions are replayed from the conversation, so discarded
  preemptive runs cannot change state.
- **Part B, relative dates:** `find_slots(when=...)` takes the caller's words; code
  resolves them in Europe/Berlin (weekdays, ranges, closed days, DST, 28-day horizon).
- **Part B, reasoning leak** (`docs/experiments/reasoning-leak.md`): 2×2 experiment, 5
  cached runs per arm. **Reasoning effort `low` adopted** (0 vs 27 leaked sentences, p=0.008;
  +58 ms `llm_ttft`); German tool descriptions rejected (no effect, 7 test failures). The
  user's `.env` is on `low`; `.env.example` and the code default are too.
- **Prompt caching on:** `cache_control` (Anthropic automatic caching) and a random
  `session_id` per call via OpenRouter; about 80 % of prompt tokens read from the cache,
  live run $0.14 → $0.055; no measurable latency change.
- **Code-spoken filler:** with `low` the agent no longer says a preamble before tools, so
  `tool_filler.py` says one when a reply calls a tool and said nothing since the caller's
  line (at most once per turn). Caller line to first speech p50 2128 → 1212 ms; the
  answer is not delayed. Turn log marks it (`is_filler`) and has `since_caller_s`.
- **Fixes since:** spoken callback numbers converted to digits; sentence filter blocks
  unbacked claims (only the matching tool backs a claim), spoken reasoning (incl. German
  narration such as "Das ist ein Terminwunsch …"), repeated or unneeded new-or-existing
  questions, and weekdays or months the LLM worked out (today's date backs only its
  weekday); at most one retry per reply, then a fixed sentence. False-positive check:
  corpus of 149 passing live tests.
- Live harness records timings, token counts and config per test (`LIVE_CORPUS`);
  `uv run python -m eval.leak_report eval/results/*.jsonl` compares runs.
- Last live run: 3 × 78/78 passed (reasoning `low`, caching and filler on).

Open:
- **First speech p90 for direct answers:** turns without a tool get no filler, and with
  `low` the model sometimes thinks long first (first speech p90 3.4 s). Options: a
  timer-based filler or a shorter thinking time; not built.
- **Real end-to-end check:** a console session by the user; read `e2e_s`, `is_filler` and
  `since_caller_s` from the turn log.
- **Part B, English switching (next):** A/B measurement of Deepgram `de` vs `multi` on
  synthetic ElevenLabs audio (ask before running); reject `multi` if German accuracy on
  names, dates of birth and phone digits drops by more than 2 percentage points vs `de`.
  English calls also need meta detection without the language check.
- **Part C not started:** FAQ with local retrieval, barge-in tuning.
- Afterwards: README and CLAUDE.md update for all of M3; the user's local
  `data/praxis.db` must be reseeded for 28 days (`... db.seed --reset`, the user runs it).
