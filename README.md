# Praxis-Telefon

A German-first, English-capable AI phone receptionist for a doctor's practice: it books, moves and cancels appointments, answers practice questions and takes messages, on a real phone line and in the browser.

## Try it

- **Call this number to talk to the agent:** +1 484 998 6615
- **Live web demo (in the browser, no phone needed):** https://praxis-telefon-production.up.railway.app

This is a demo of a fictional practice (Hausarztpraxis Dr. Weber) with synthetic data. Please do not give real personal or health data; make up a name and a date of birth. Calls are limited to 5 minutes, and the number of demo calls per day and month is limited.

## What it does

- **Appointments:** finds up to three free slots from the caller's own words ("next Tuesday", "Ende der Woche", "tomorrow afternoon") and books, moves or cancels after reading the details back and getting a clear yes. Code turns the words into dates in Europe/Berlin; the LLM never computes a date.
- **Practice questions:** opening hours, address, parking, repeat prescriptions, referrals, sick notes, holidays, insurance card, new patients, answered only from a synthetic FAQ. Anything else: "I don't know, shall I take a message?".
- **Messages for the team:** name, callback number and a fixed category, no medical details.
- **German and English:** German by default. "English" or "Können wir Englisch sprechen?" switches the call, "Deutsch, bitte" switches back.
- **Safety:** emergency and suicide phrases get a fixed, uninterruptible message without the LLM; a request for a human, or two unrecognised turns in a row, hands over to message-only mode.

## Key numbers

| Measure | Result | Source |
|---|---|---|
| Offline tests (Python) | 2,554, no keys or network needed | `uv run pytest` |
| Web tests (Node 24) | 69 | `cd web && npm test` |
| Live LLM tests | 135 of 135 passed, about $0.08 per run | `tests/live/`, latest full run after the M4 fixes |
| Scenario evaluation | 229 of 230 passed (99.6 %, 95 % interval 97.6 to 100 %), 46 scenarios x 5, $0.165 | `eval/harness`; the one miss was a check that was too strict, fixed since |
| Baseline (before the privacy fixes) | 200 of 200, 40 scenarios x 5, $0.135 | same harness |
| Unbacked claims, invented facts, wrong language | 0 of 230 each | same run |
| Caller line to first speech | p50 0.99 s, p95 2.56 s | scenario run, text mode (LLM and tools, no STT or TTS) |
| Caller line to the answer | p50 1.78 s, p95 3.63 s | same |
| Barge-in: time until the agent is quiet | 0.40 s and 0.50 s | voice test, real interruptions |
| Prompt cache | about 80 % of prompt tokens read from the cache | live run $0.14 to $0.055 |

Latency in text mode excludes speech recognition and synthesis. In real calls Deepgram's final transcript arrives about 0.13 s after the caller stops (synthetic A/B) and ElevenLabs delivers first audio in about 0.16 s (a single console measurement). Samples are small: five runs per scenario, one non-native speaker for the voice tests.

### How we got here

Single console calls in the first milestone (MacBook microphone, German), measured from the caller stopping to the agent's first audio. One call per row, so read it as the story of the decisions, not as a benchmark:

| LLM (via OpenRouter) | Reasoning | Max endpointing | LLM time to first token | End to end |
|---|---|---|---|---|
| Gemini 3.8 Flash | medium (default) | 3.0 s (default) | 6.8 s | 9.5 s |
| Gemini 3.8 Flash | low | 1.5 s | 2.96 s | 3.7 s |
| Claude Haiku 5.5 | none | 1.5 s | 0.79 s | 1.6 s |

Haiku with reasoning `none` was fast but sometimes spoke its reasoning before tools. The next milestone moved to reasoning `low` (0 leaks, +58 ms LLM time to first token) and added the code-spoken "Einen Moment, bitte." filler, which brought first speech before tools back from 2.1 s to 1.2 s (p50, text mode; see the reasoning-leak experiment below).

## How it works

```
caller (phone or browser)
   |  WebRTC / SIP
LiveKit Cloud  ---  agent worker (Python, LiveKit Agents, eu-central)
                       Deepgram Nova-3 (speech to text)
                       Claude Haiku 5.5 via OpenRouter (one tool-calling LLM, reasoning "low", prompt caching)
                       ElevenLabs Flash v2.5 (text to speech)
                       SQLite (appointments, synthetic seed data)
browser: small page + token endpoint (Node, Railway, Amsterdam), Cloudflare Turnstile
```

- The agent runs on LiveKit Cloud (project in the EU data region, agent in Frankfurt). The web page and its token endpoint run on Railway in Amsterdam.
- The LLM only chooses what to say and which tool to call. Code owns state, dates, safety decisions, consent and every write to the database.
- Interruption detection (Silero) runs locally in the agent, not in the cloud.

## Safety and privacy

- **Consent before changes:** `book`, `move_booking` and `cancel_booking` run only after a read-back and a clear yes from the caller, checked in code (`agent/consent.py`). A refusal blocks the action; the agent never tells the caller which word to say.
- **No third-party data:** a booking lookup works only for a name the caller introduced as their own, and the sentence filter blocks asking for another person's date of birth. Other people's appointments go to a message for the team. Limit: name plus date of birth is the only check, so a caller who claims to be the patient gets through.
- **Honest speech:** every sentence passes filters before it is spoken. Claims such as "booked" need the matching tool result, spoken reasoning and invented weekdays or practice facts are blocked, and the language filter follows the call's language. At most one retry per reply, then a fixed sentence.
- **Logs:** labels, times and counts only, never transcripts or audio. Phone numbers are stored only as a keyed hash (HMAC-SHA256); on the web page the visitor's IP address is stored only as a keyed hash, erased after the day.
- **Demo limits:** phone calls last at most 5 minutes (warning at 4), 3 calls per caller and day, 15 calls and 45 minutes per day. Web calls: 5 minutes each, 20 minutes a day (at most 4 calls), 60 minutes a month, 3 per visitor and day, Cloudflare Turnstile before a token is issued. When a limit is reached, or the voice fails, the agent signals "busy" to the page and hangs up, never fails silently.
- **Honest wording on the page:** the privacy notice names every service that receives the caller's voice (Deepgram, OpenRouter with Claude, ElevenLabs, LiveKit, Cloudflare Turnstile) and says that Railway's technical logs may be stored outside the EU.

## Engineering highlights

- **Evaluation harness** (`eval/harness/`, `eval/scenarios/`): 46 scripted YAML scenarios with rule-based callers, run in text mode, N = 5, four worker processes and a hard cost cap. Checks come from the database and the transcript: task success, right tools, unbacked claims, forbidden phrases, language per turn, read-back equals saved value. Wilson intervals per scenario, bootstrap intervals overall, and a paired `compare` that calls a change "helps" or "hurts" only when the interval excludes zero. Its review found the privacy and tool-word issues fixed in the same milestone.
- **Experiment: reasoning leak** ([`reasoning-leak.md`](docs/experiments/reasoning-leak.md)): why does the agent speak its reasoning before tools? A 2x2 with five runs per arm. Reasoning `low` removed the leak (0 vs 27 leaked sentences, p = 0.008) for +58 ms LLM time to first token; a code-spoken "Einen Moment, bitte." brought first speech before tools from 2.1 s to 1.2 s (p50).
- **Experiment: STT language** ([`stt-language.md`](docs/experiments/stt-language.md)): Deepgram `multi` was +8.8 points on 640 synthetic clips, but mislabeled and dropped digits on real accented German, so it was rejected; the stream stays German and switches to English on request.
- **FAQ retrieval** ([`faq-retrieval.md`](docs/experiments/faq-retrieval.md), [`faq-topic-choice.md`](docs/experiments/faq-topic-choice.md)): keyword retrieval missed its bar (82 to 92 % right, up to 28 % false) and was removed; letting the LLM pick the topic and code return the entry reached 100 % right on the second author's questions.
- Decision rules were set before each measurement, and negative results are written up.

## Run it yourself

Copy `.env.example` to `.env` and fill it in, then:

```
uv sync
uv run pytest                                               # offline, no keys
uv run -m livekit.agents download-files                     # once: VAD + turn-detector weights
uv run python -m praxis_telefon.db.seed                     # once: synthetic data (--reset to recreate)
uv run python -m praxis_telefon.agent.main console          # talk via mic in the terminal
uv run python -m eval.harness.runner --n 5 --max-usd 1.0    # scenario evaluation (paid)
cd web && npm ci && npm test                                # web page and token endpoint
```

## Known limits

- About 1 in 3 German names is misheard by speech recognition; booking by name needs a fallback (spelling, confirming by date of birth). This is the next planned fix.
- English callers take two turns to switch, because their first line is not understood by the German recognition.
- Turns without a tool get no filler; with reasoning `low` the model sometimes thinks long first (first speech p90 2.5 to 3.4 s).
- The FAQ fact check knows only the FAQ's topics; a practice policy on another subject is not caught.
- Short utterances sometimes come back empty from speech recognition, which can trigger the two-strike handover.

## Links

- GitHub: https://github.com/RohithkumarReddipogula/praxis-telefon
- Portfolio: https://rohithkumarreddipogula.github.io
