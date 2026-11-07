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
  - `call_language.py` – the call's language from the STT finals (2-final rule, explicit
    requests); the LLM, filters and fixed messages follow it
  - `barge_in.py` / `late_transcript.py` – Part C measurements in the turn log (time to
    quiet on barge-in; finals after the turn was committed, summary per call)
  - `main.py` – `build_llm`: OpenRouter with prompt caching and a per-call `session_id`
- `src/praxis_telefon/safety/` – deterministic emergency/suicide/human-request checks,
  two-strike handover, fixed messages
- `src/praxis_telefon/tools/` – LLM-callable functions (typed, pure where possible);
  `relative_dates.py` turns the caller's words into dates
- `src/praxis_telefon/faq/` – synthetic practice FAQ (`faq.json`, DE/EN; opening hours
  from the seed); the LLM picks the topic via `practice_info(topic)`
- `src/praxis_telefon/db/` – SQLite schema, access layer, synthetic seed data (28 days)
- `eval/` – scripted conversation scenarios and scoring; `leak_report.py` compares live runs
- `docs/experiments/` – experiment write-ups (`reasoning-leak.md`, `stt-language.md`,
  `faq-retrieval.md`)
- `eval/faq/` – labelled FAQ questions (Claude's and a second author's) and the LLM
  topic measurement (`report.py`, paid; `--rescore` after a review)
- `eval/stt_ab/` – Deepgram language A/B on synthetic ElevenLabs audio (ask before running)
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
  weekday); at most one retry per reply, then a fixed sentence. A read-back may have one
  sentence between the claim and its question. False-positive check: the corpus of
  passing live tests (`tests/data/live_sentences.jsonl`).
- Live harness records timings, token counts and config per test (`LIVE_CORPUS`);
  `uv run python -m eval.leak_report eval/results/*.jsonl` compares runs.
- **Part B, call language** (step 2b): code decides it from the STT finals
  (`call_language.py`): an English final needs Deepgram's tag and `detect_language`
  agreeing; switch after 2 consecutive finals either way (1 English final before any
  German); neutral finals neither count nor break a run; "Can we speak English?" /
  "Sprechen Sie Deutsch?" switch at once. The LLM gets a note with the call's language;
  the language check blocks the wrong language both ways; English reasoning patterns
  (request classification, internal ids). Each final is logged with labels only.
- **Part B, English switching** (`docs/experiments/stt-language.md`): STT A/B on synthetic
  audio (`eval/stt_ab/`) passed `multi` (+8.8 points), but with the user's accented German
  `multi` tagged Telugu/Afrikaans/Norwegian and lost digits, also with `language_hint`:
  **`multi` rejected** (kept as opt-in `STT_LANGUAGE` / `STT_LANGUAGE_HINT` experiments).
  **Option B adopted:** Deepgram `de` by default (`test_stt_uses_german_by_default`); the
  stream switches to `en` on "English" / "Englisch" alone (the bilingual "say that again"
  asks for it) or "Können wir Englisch sprechen?", back to `de` on "Deutsch" / "Deutsche"
  on the `en` stream, "German" alone or in a request, or an empty `en` final in a call
  that started German (no strike, bilingual "please repeat"). Built from the user's
  own-voice recordings (`eval/my_voice.py`, audio gitignored); no audio lost switching
  after a final (`eval/stt_ab/switch_check.py`). `take_message` merges a second reason
  for the same caller into one note. **Voice test 2026-11-07 passed** (3 calls: switch
  to `en` and back, English caller via "English", German message).
- Last live run (2026-11-07): 99/99 (78 suite + 21 language tests), $0.06; no German
  sentence in English calls. False-positive corpus: 347 passing live tests.

Open:
- **First speech p90 for direct answers:** turns without a tool get no filler, and with
  `low` the model sometimes thinks long first (first speech p90 3.4 s). Options: a
  timer-based filler or a shorter thinking time; not built.
- **Real end-to-end check:** a console session by the user; read `e2e_s`, `is_filler` and
  `since_caller_s` from the turn log.
- **English line in the greeting (open option):** with option B an English caller's first
  line comes back empty from `de`; the bilingual "say that again" message tells them to
  say "English", so English takes two turns (accepted for now, user 2026-11-07). An English
  sentence in the greeting would save that turn at about 1.5 s more per call.
- **Names misheard:** about 1 in 3 German names wrong in the A/B with every setting
  (`de` and `multi` alike, e.g. "Östürk" for Öztürk); German names in English calls
  4–15 % right. Booking and lookup by name need a fallback (spelling, confirm by date
  of birth); not built.
- **Part C, in progress (not committed):** FAQ via `practice_info(topic)` (keyword
  retrieval missed the bar, `docs/experiments/faq-retrieval.md`); practice facts are
  blocked unless `practice_info` backed them (a closed day also by `find_slots`);
  barge-in and late-transcript records in the turn log. Next: FAQ live tests, the user's
  voice test, then barge-in tuning and the `min_delay` decision from the records.
- **M4: controlled barge-in harness** (moved from Part C, user 2026-11-07): real session
  with custom audio in/out, synthetic caller clips at fixed points (real interruption,
  backchannels, cough, "Stopp"), comparing interruption settings on time to quiet,
  false and missed interruptions (~$0.17 + ~2,500 ElevenLabs credits).
  - **Late transcript (for barge-in / endpointing tuning):** voice test 2026-11-07, call A
    at 16:40:14, LiveKit warned "transcript arrives after turn has been committed,
    consider raising min_delay" (`main.ENDPOINTING`); once in three calls, no visible
    effect. Check how often it happens before raising `min_delay` (costs latency).
- Afterwards: README and CLAUDE.md update for all of M3; the user's local
  `data/praxis.db` must be reseeded for 28 days (`... db.seed --reset`, the user runs it).
