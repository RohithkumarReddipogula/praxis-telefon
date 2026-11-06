# Praxis-Telefon

German-first, English-capable AI phone receptionist for a fictional doctor's office (portfolio project).

Status: M2 – appointment tools on SQLite (find, book, move, cancel, staff message), synthetic data only.

```
uv sync
uv run pytest
uv run pytest -m live   # optional: real LLM regression tests, needs .env
```

## What the agent can do

- Find free slots (at most 3) and book, move or cancel appointments, after reading the
  details back and getting a clear yes from the caller.
- Take a callback message for staff: name, number and a fixed category, no medical details.
- Guards in code, not only in the prompt: it can only book slots it found during the
  current call, and can only change bookings it looked up by name and date of birth
  during that call.
- Fictional practice (Dr. Weber, Dr. Brandt) with synthetic patients; opening hours
  Mon–Fri 8–12, plus Mon, Tue and Thu 15–18 (Europe/Berlin).

## Agent reliability

Bugs found in manual voice tests:

- **Missing tool call after "Ja":** the caller confirmed a move and a cancel, but
  `move_booking` / `cancel_booking` never ran and the database did not change.
- **Booking request ended as a staff message:** instead of `book`, the agent called
  `take_message`, twice, and saved two identical messages.

Live tests (`tests/live/test_tool_calls_live.py`) drive the real agent and LLM in
LiveKit's text-only test mode against an in-memory synthetic database. Six scenarios,
each run 3 times (18 tests): move after "Ja", cancel after "Ja", book + move + cancel in
one call, two kinds of booking request that must not become a staff message, and a
callback request that must save exactly one message. Every scenario also checks that
the agent never claims "gebucht/verschoben/abgesagt" before the tool succeeded and never
asks again for a name or date of birth already given.

| | Passed |
|---|---|
| Before the fix | 14 / 18 |
| After the fix | 18 / 18 (130 s) |

To be honest about the numbers: the improvement came from both prompt fixes (call the
tool right after "Ja", one read-back only, don't re-ask for given details, booking
requests never become messages) and a fairer scripted caller (it now answers what the
agent actually asks, and the claim checker no longer flags read-back questions). Two of
the four failures were test false positives. 3 runs per scenario is a small sample;
the LLM is not deterministic, so 18/18 does not prove the agent is always right.

What is guaranteed in code, not only in the prompt:

- **Slot guard:** `book` and `move_booking` only accept slots that `find_slots` returned
  in the current call.
- **Booking guard:** `move_booking` and `cancel_booking` only accept bookings this call
  looked up by name and date of birth, or created itself.
- **Duplicate-message guard:** the same message (name, number, category) twice in one
  call is saved once.
- **Unique index:** the database allows at most one active booking per slot, so a
  double booking is impossible even if two calls race.
- **Claim log:** a reply that claims a change before the tool succeeded is logged as
  `unbacked_claim` (action name only, no transcript). It cannot stop the sentence.

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
uv run python -m praxis_telefon.db.seed                     # once: synthetic data (--reset to recreate)
uv run python -m praxis_telefon.agent.main console          # talk via mic in the terminal
uv run python -m praxis_telefon.agent.main dev              # LiveKit Cloud (needs LIVEKIT_*)
```
