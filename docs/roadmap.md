# Roadmap (as of 2026-11-08)

Decided with the user; all items stay in. ElevenLabs stays the TTS (10,000 credits a month
from 7 December, about 125 call minutes: the real limit, not money).

- [x] **1. "Please answer yes or no" fix (item 3).** Done 2026-11-09 (`e88d2fb`, deployed
      as `95qr7Q57giyX`): the filter blocks the phrase (reason `demand`), and the retry
      note and fixed fallback ask a question that fits the pending action. Live suite
      135/135, booking harness 100/100. In five console calls (logs 16:31-16:40 UTC) the
      LLM wrote no demand sentence and no booking tool was called, so the consent path and
      the new questions were not exercised live, only offline and in the harness.
- [ ] **2. Public web demo link (item 1).** LiveKit Sandbox no longer exists: a minimal
      page plus a token endpoint in `web/`, hosted on Railway (the user moved from Vercel
      on 2026-11-09). Cloudflare Turnstile captcha. Web calls have their own limits:
      20 minutes a day (5 minutes per call, at most 4 calls) and 60 minutes a month; when
      reached the page shows "The demo is busy today, please try again tomorrow" (also
      when the agent signals a failing voice). German page with an English toggle, short
      privacy notice before the call, `noindex`, no home address (contact by email only).
      Step 1 (agent side: channel-aware limiter, busy signal) is done offline.
- [ ] **3. Free LiveKit US number (item 2a).** Inbound-only trunk and a dispatch rule to
      `praxis-telefon`; phone test.
- [ ] **4. Name fix (item 4).** `find_booking` matches the birth date exactly and the name
      by Kölner Phonetik; several matches: ask for more details, never pick one; unclear
      name: ask for the spelling.
      Seen in the 2026-11-09 console calls (logs, no words): no call reached a booking.
      (a) Names and birth dates were misrecognised repeatedly. (b) Short utterances of
      0.8-1.8 s (slot choice, "Ja") came back from the speech recognition as nothing at
      all (0 interim transcripts), 3 times; two in a row triggered the two-strike
      handover in call 6 ("ich verstehe Sie leider nicht gut ... Nachricht aufnehmen"),
      which is the safety design, not an LLM decision: the agent then took a message.
      Also one low-confidence transcript counted as a strike (call 7). Not fixed; to
      measure and decide in this item (why short utterances come back empty, whether the
      handover is too eager for a demo).
- [ ] **5. M4 final comparison (item 6).** 46 scenarios x 5 against the baseline;
      `docs/experiments/m4-eval.md`.
- [ ] **6. Telnyx German number (item 2b),** once approved (SIP trunk, dispatch rule,
      phone tests).
- [ ] **7. Audio layer and barge-in harness (item 5),** before 7 December while the 30,000
      ElevenLabs credits last (about 5,000 credits per full run).
- [ ] **8. README update and demo video script (item 7).**

Each paid run, deploy, commit and push is asked first.

## Small findings, not fixed (2026-11-09)

- ~~`run_demo_call`: a participant who leaves before joining raised "room disconnected
  while waiting for participant" as an unhandled exception.~~ Fixed 2026-11-09: logged as
  `demo_no_participant`, no call or minute counted (test).
- A false-interruption pause lasted 3.75 s although `false_interruption_timeout` is 1.0 s
  (call 6): LiveKit waits for an open end-of-turn decision first. Look at it in the
  barge-in harness (item 7).
- Console calls that connect and leave at once still count as a demo call (the limiter
  records their real duration, so minutes are not wasted).
- Late transcripts: 2 of 43 turns = 4.7 % (under the 5 % rule); `min_delay` stays 0.4 s.
