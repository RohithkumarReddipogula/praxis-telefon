# Praxis-Telefon

German-first, English-capable AI phone receptionist for a fictional doctor's office (portfolio project).

Status: **M3 done** – safety, relative dates, honest speech (sentence filters), English
calls, practice FAQ and barge-in tuning on top of the M2 appointment tools. Synthetic data
only: a fictional practice, made-up patients, phone numbers in fictional ranges.

```
uv sync
uv run pytest                    # offline, no keys needed
uv run pytest -m live tests/live # optional: real LLM tests, needs .env
```

Stack: Python 3.12, LiveKit Agents, Deepgram Nova-3 (STT), ElevenLabs Flash v2.5 (TTS),
Claude Haiku 5.5 via OpenRouter (reasoning `low`, prompt caching), SQLite, pytest.

## What the agent can do

- **Appointments:** find free slots (at most 3) and book, move or cancel, after reading
  the details back and getting a clear yes. Slots are searched with the caller's own
  words ("nächsten Dienstag", "Ende der Woche", "tomorrow afternoon"); code turns them
  into dates in Europe/Berlin (weekdays, ranges, closed days, DST, 28-day horizon), the
  LLM never computes a date.
- **Messages for the team:** name, callback number (spoken digits converted) and fixed
  categories, no medical details; a second reason for the same caller updates the note.
- **Practice questions** (opening hours, address, parking, repeat prescriptions,
  referrals, sick notes, holidays, insurance card, new patients) answered only from a
  synthetic FAQ; anything else: "I don't know, shall I take a message?".
- **German and English:** German by default; "English" or "Können wir Englisch sprechen?"
  switches the call (and the speech recognition) to English, "Deutsch, bitte" back.
- **Safety:** emergency and suicide phrases (DE/EN) get a fixed, uninterruptible message
  without the LLM, and booking is off for the rest of the call; a request for a human or
  two unrecognised turns hand over to message-only mode.
- Fictional practice (Dr. Weber, Dr. Brandt), opening hours Mon, Tue, Thu 8–12 and
  15–18, Wed and Fri 8–12.

## Guaranteed in code, not only in the prompt

- **Slot and booking guards:** `book` / `move_booking` only take slots `find_slots`
  returned in this call; `move_booking` / `cancel_booking` only bookings this call looked
  up by name and date of birth, or created. A unique index makes double bookings
  impossible.
- **Safety decisions** are replayed from the conversation on every reply, so a discarded
  preemptive LLM run cannot change the call's state.
- **Sentence filters** check every sentence before it is spoken (`agent/claim_gate.py`):
  - claims ("gebucht", "Nachricht aufgenommen") only after the matching tool succeeded,
    except in a read-back that ends with a confirmation question;
  - no spoken reasoning ("This is a new appointment request, so I'll …", internal ids),
    no sentence in the wrong language for the call;
  - no weekday or month the LLM worked out itself (only the caller, a tool result or
    today's date can name one);
  - no practice fact without `practice_info` (a closed day also from `find_slots`);
  - no repeated "new or existing appointment?" question.
  At most one retry per reply, then a fixed sentence. False-positive check: 617 passing
  live tests replayed through the filters (`tests/data/live_sentences.jsonl`).
- **The call's language** is decided by code from the speech-recognition finals, never
  by the LLM (one misheard line cannot switch a German call).
- **Privacy:** logs hold labels, times and counts, never transcripts; tracing stays
  local; interruption detection runs locally (Silero), not in the cloud.

## Experiments (M3)

Each with a decision rule set before measuring; write-ups in `docs/experiments/`.

| Write-up | Question | Result |
|---|---|---|
| [`reasoning-leak.md`](docs/experiments/reasoning-leak.md) | Why does the agent speak its reasoning before tools? 2×2, 5 runs per arm | **Reasoning `low`**: 0 vs 27 leaked sentences (p = 0.008), +58 ms LLM TTFT; German tool descriptions: no effect. A code-spoken "Einen Moment, bitte." closes the silence before tools (first speech p50 2128 → 1212 ms) |
| [`stt-language.md`](docs/experiments/stt-language.md) | Deepgram `de` vs `multi` for English callers | On 640 synthetic clips `multi` was +8.8 points on German, but with the user's real accented German it tagged Telugu/Afrikaans and lost digits (also with `language_hint`): **rejected**. Instead `de` by default and the stream switched to `en` on request; built from the user's own recordings, no audio lost when switching after a final |
| [`faq-retrieval.md`](docs/experiments/faq-retrieval.md) | Local keyword (BM25) retrieval for FAQ answers | **Negative result:** 82–92 % right with up to 28 % false answers on held-out questions (bar: ≥ 95 % / ≤ 5 %); fixes traded misses for false answers |
| [`faq-topic-choice.md`](docs/experiments/faq-topic-choice.md) | The LLM picks the FAQ topic instead | Second author's 10 questions: **100 % right, 0 % false**; Claude's 122: 100 % / 3.2 % after a description fix (changes after run 1 are marked as such) |

## Tests

- **Offline** (`uv run pytest`, about 2,300 tests, no keys): tools, date resolution,
  safety phrases and state replay, every filter with regression tests named after the
  bug, the language rule, FAQ content (opening hours generated from the seed), the
  measurement counters, and the replayed live corpus.
- **Live** (`uv run pytest -m live tests/live`, about $0.18): the real agent and LLM in
  LiveKit's text mode with a scripted caller and an in-memory synthetic database; 135
  tests (appointments, dates, safety, triage, English calls and switching, FAQ). Each
  records timings, token counts and every sentence the LLM produced (`LIVE_CORPUS`);
  `uv run python -m eval.leak_report eval/results/<run>.jsonl` summarises a run.
- **Voice tests** by the user in the console for what text mode cannot show: speech
  recognition with accented German, switching the stream, barge-in.

## Latency

Live text-mode suite (the LLM and tools, no STT/TTS), M3 final run, 135 tests, 225 turns:

| | p50 | p90 |
|---|---|---|
| LLM time to first output | 816 ms | 2043 ms |
| Caller line to first speech (incl. the "Einen Moment" filler) | 899 ms | 2328 ms |
| Caller line to the answer | 1611 ms | 3247 ms |

Around that in a real call: Deepgram's final transcript about 0.13 s (p90 0.26 s) after
the caller stops (synthetic A/B), end of turn after at least 0.4 s and at most 1.5 s,
ElevenLabs first audio about 0.16 s (single console measurement). FAQ answers take one
extra LLM round trip; the filler covers it.

**How we got here.** Single console calls in M1 (MacBook mic, German), end-to-end from
the caller stopping to the agent's first audio:

| LLM (via OpenRouter) | Reasoning | Max endpointing | LLM TTFT | End-to-end |
|---|---|---|---|---|
| Gemini 3.8 Flash | medium (default) | 3.0 s (default) | 6.8 s | 9.5 s |
| Gemini 3.8 Flash | low | 1.5 s | 2.96 s | 3.7 s |
| Claude Haiku 5.5 | none | 1.5 s | 0.79 s | 1.6 s |

Haiku with reasoning `none` was fast but sometimes spoke its reasoning before tools, so
M3 moved to reasoning `low` (0 leaks, +58 ms LLM TTFT) and added the code-spoken "Einen
Moment, bitte." filler, which brought first speech before tools back from 2.1 s to 1.2 s
(p50, text mode; `docs/experiments/reasoning-leak.md`).

**Barge-in** (voice test, `false_interruption_timeout` 1.0 s instead of LiveKit's 2.0 s):
the agent went quiet 0.40 s and 0.50 s after a real interruption, including a hesitant
"Nein … äh … lieber am Montag"; "mhm" and a cough pause it for about 1 s instead of 2 s
before it resumes; no "resume, then cut". The turn log records every barge-in
(`time_to_quiet_s`, outcome) and every final transcript that arrives after the turn was
committed (0 of 6 turns in the voice test; `min_delay` stays 0.4 s).

## Run the agent

Copy `.env.example` to `.env` and fill it in, then:

```
uv run -m livekit.agents download-files                     # once: VAD + turn-detector weights
uv run python -m praxis_telefon.db.seed                     # once: synthetic data (--reset to recreate)
uv run python -m praxis_telefon.agent.main console          # talk via mic in the terminal
uv run python -m praxis_telefon.agent.main dev              # LiveKit Cloud (needs LIVEKIT_*)
```

An older local database must be recreated for the 28-day horizon:
`uv run python -m praxis_telefon.db.seed --reset`.

## Known limits

- About 1 in 3 German names is misheard by speech recognition (with `de` and `multi`
  alike); booking by name needs a fallback (spelling, confirming by date of birth).
- English callers take two turns to switch (their first line is not understood by the
  German recognition; the agent then asks them to say "English").
- Turns without a tool get no filler; with reasoning `low` the model sometimes thinks
  long first (first speech p90 2.5–3.4 s).
- The FAQ check knows only the FAQ's topics; a practice policy on another subject
  ("we can't give results by phone") is not caught.
- Small samples throughout: 3 runs per live scenario, one non-native speaker for the
  voice tests, a scripted caller in text mode.
