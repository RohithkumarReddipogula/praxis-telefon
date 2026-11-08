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
  - `caller_identity.py` – names the caller introduced as their own; `find_booking` only
    looks those up (privacy)
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
- `eval/scenarios/` + `eval/harness/` – M4 evaluation harness: 46 YAML scenarios,
  rule-based callers, deterministic checks, Wilson/bootstrap intervals, cost cap
  (`runner`, paid; `report`, `compare`, free)
- `docs/experiments/` – experiment write-ups (`reasoning-leak.md`, `stt-language.md`,
  `faq-retrieval.md`)
- `eval/faq/` – labelled FAQ questions (Claude's and a second author's) and the LLM
  topic measurement (`report.py`, paid; `--rescore` after a review)
- `eval/stt_ab/` – Deepgram language A/B on synthetic ElevenLabs audio (ask before running)
- `tests/` – unit and regression tests; no network or API keys required
  - `tests/live/` – live LLM tests (`harness.py`: scripted caller, text-mode session)
  - `tests/data/` – live sentence corpus and reviewed known-bad sentences

## Status (M3 done, 2026-11-07)

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
- **Part C, FAQ** (`docs/experiments/faq-retrieval.md`, `faq-topic-choice.md`): keyword
  retrieval (BM25) missed the bar (82–92 % right, up to 28 % false) and was removed; the
  LLM picks the topic via `practice_info(topic)` (9 topics or `other`), code returns the
  entry from `faq/faq.json` in the call's language (opening hours from the seed). Second
  author's 10 questions 100 % / 0 % false, Claude's 122 100 % / 3.2 %. Practice facts are
  blocked unless `practice_info` backed them (a closed day also by `find_slots`); a
  crisis-sentence false positive ("nicht auf einen Termin warten") was fixed. Voice test
  passed (FAQ answers correct, `no_entry` for home visits and flu vaccination).
- **Part C, barge-in:** `false_interruption_timeout` 1.0 s (was LiveKit's 2.0), `min_duration`
  0.5 s. Voice test: real interruptions quiet in 0.40 / 0.50 s, "mhm" and a cough pause
  about 1 s, no `resumed_then_stopped`. `barge_in.py` logs every episode (time to the
  pause, outcome); `late_transcript.py` logs finals after the turn was committed (0 of 6
  turns; `min_delay` stays 0.4 s by the rule: change only above 5 % or on a wrong or
  doubled answer, and only after at least 40 turns of data, user 2026-11-08).
- Last live run (2026-11-07, M3 final): 135/135 (all of `tests/live/`), $0.18; `llm_ttft`
  p50 816 / p90 2043 ms, first speech 899 / 2328 ms, answer 1611 / 3247 ms. One false
  positive found and fixed: a move read-back naming "bestehenden" and "neuen" Termin was
  taken for the new-or-existing question (now needs "oder"). False-positive corpus: 617
  passing live tests.
- README updated for all of M3.

## Status (M4 in progress, 2026-11-07)

Done and pushed:
- **Evaluation harness** (`eval/harness/`, `eval/scenarios/`): 46 scenarios in text mode,
  N=5 each, 4 worker processes, hard cost cap (default $1). Checks from the database and
  the transcript (task success, tools, unbacked claims, `never_say` leaks, language per
  turn and actually spoken, read-back = saved number); Wilson per scenario, bootstrap over
  scenarios overall, bootstrap p50/p95 latency; `compare` pairs by scenario, uses common
  scenarios only, calls "helps/hurts" only when the interval excludes 0. Run it:
  `uv run python -m eval.harness.runner [--n 5 --max-usd 1.0 --only ids]`, then
  `uv run python -m eval.harness.report <run dir>` / `... compare <A> <B>`.
- **Baseline** (40 scenarios × 5): 200/200, $0.135, first speech p50 1.08 / p95 2.72 s.
  Its review found the privacy and tool-word issues below.
- **Privacy:** `find_booking` only for a name the caller introduced as their own
  (`caller_identity.py`); move and cancel follow (they need a found booking). The filter
  blocks asking for another person's date of birth (`privacy`), except in a new booking;
  other people's appointments go to a message for the team. Booking for someone else
  stays allowed. Limit: name + date of birth is the only check, a caller claiming to be
  the patient gets through.
- **Callback number:** `take_message` refuses a number the caller did not say (also said
  in parts); its result gives the saved number for the read-back (bug: saved 03212345456,
  read back the caller's correct words).
- **Tool word:** "book"/"Booking" in a German sentence is `meta:tool`.
- After the fixes, 46 × 5: 229/230 ($0.165; the miss was a too-strict check, fixed);
  live suite 135/135 ($0.077).

## Status (M5/M6, 2026-11-08)

M5 and M6 basic come before the rest of M4 (user 2026-11-08). Done and pushed:
- **Cloud agent** on LiveKit Cloud (project `praxis-telefon-demo`, EU data region, agent
  `CA_oRRNE5dv7jQB` in `eu-central`, free Build plan; `livekit.toml`). Deploy:
  `lk agent deploy --yes --project praxis-telefon-demo --secrets-file .env.demo .` (asks
  first). `.env.demo` holds the demo keys (OpenRouter key with a credit limit, Deepgram
  and ElevenLabs keys `praxis-demo`, `DEMO_HASH_KEY`); never open, read or print it; the
  upload excludes `.env*` and ignored files.
- **Dockerfile:** base image `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` (LiveKit's
  template; a version-pinned tag did not exist); dependencies and model weights installed
  as the runtime user. Synthetic DB seeded at start; demo mode on.
- **Demo protections** (`agent/demo.py`): 5 min per call (warning 1 min before), 3 calls
  per caller and day, 15 calls and 45 minutes per day; withheld numbers count against
  the overall limits; caller numbers only as HMAC with `DEMO_HASH_KEY`; bilingual demo
  greeting. Counts in a file on the container (reset on restart; hard caps at providers).
- **Consent before book/move/cancel** (`agent/consent.py`, bug from the browser test: book
  without a read-back): the agent's last question before the caller's line is a
  confirmation question, and the line is a yes (starts with Ja/Yes or a listed form;
  nein/aber/lieber/doch nicht/nicht/no/but/rather refuse; "Alles okay, danke. Tschüs."
  is no yes). The refusal note never asks the caller for a particular word.
- **False strike fixed:** interim transcripts count as recognised speech;
  `utterance_timing` logs times and counts per utterance.
- Browser tests on the cloud agent: greeting, FAQ, filler, no false strike, "Ja, aber
  lieber um neun Uhr" not booked, booking after "Ja".

Next: the phone number (free LiveKit US number now; Telnyx Berlin/national number when
approved), inbound only, dispatch rule to `praxis-telefon`; phone tests. Open: the agent
sometimes still says "Please answer yes or no" after a consent refusal.

Next in M4: name fix (find_booking by sound plus exact date of birth; never pick one of
several matching patients, ask for more details instead), audio layer (synthetic callers
through real STT/TTS, the user's 12 recordings as extra caller audio, gitignored) and the
controlled barge-in harness; then CLAUDE.md/README for M4.

Open for M4:
- **Controlled barge-in harness:** real session with custom audio in/out, synthetic
  caller clips at fixed points (real interruption, backchannels, cough, "Stopp"),
  comparing interruption settings on time to quiet, false and missed interruptions
  (~$0.17 + ~2,500 ElevenLabs credits). Decided 2026-11-07 to move it here from Part C.
- **Names misheard:** about 1 in 3 German names wrong in the STT A/B with every setting
  (`de` and `multi` alike, e.g. "Östürk" for Öztürk); German names in English calls
  4–15 % right. Booking and lookup by name need a fallback (spelling, confirm by date
  of birth).
- **First speech p90 for direct answers:** turns without a tool get no filler, and with
  `low` the model sometimes thinks long first (first speech p90 2.5–3.4 s). Options: a
  timer-based filler or a shorter thinking time.
- **English line in the greeting (open option):** English callers take two turns (their
  first line is empty on `de`, the bilingual "say that again" asks for "English"); an
  English sentence in the greeting would save that turn at about 1.5 s more per call.
- **FAQ gaps:** practice policies outside the FAQ ("keine Befunde telefonisch") are not
  caught by the fact check; a new entry or filter family if they recur.
- **Late transcripts:** keep reading `late_transcript_summary`; the 5 % rule applies only
  after at least 40 turns of data. So far 1 of 20 turns (three cloud browser calls 2026-11-08,
  0.047 s after the commit, no visible effect); `min_delay` stays 0.4 s.
- **Real end-to-end latency:** a longer console session; read `e2e_s`, `is_filler` and
  `since_caller_s` from the turn log.
- The user's local `data/praxis.db` must be reseeded for 28 days (`... db.seed --reset`).
