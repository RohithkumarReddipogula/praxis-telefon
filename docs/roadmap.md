# Roadmap (as of 2026-11-08)

Decided with the user; all items stay in. ElevenLabs stays the TTS (10,000 credits a month
from 7 December, about 125 call minutes: the real limit, not money).

- [ ] **1. "Please answer yes or no" fix (item 3).** The sentence filter blocks that phrase
      in German and English after a consent refusal; the retry note gives a natural
      alternative ("Soll ich den Termin so eintragen?"). Regression test.
- [ ] **2. Public web demo link (item 1).** LiveKit Sandbox no longer exists: LiveKit's
      React starter app plus a token endpoint, hosted on Vercel. Cloudflare Turnstile
      captcha. Separate daily limit for web calls: 20 minutes in total; when reached the
      page shows "The demo is busy today, please try again tomorrow". Short privacy notice
      before the call.
- [ ] **3. Free LiveKit US number (item 2a).** Inbound-only trunk and a dispatch rule to
      `praxis-telefon`; phone test.
- [ ] **4. Name fix (item 4).** `find_booking` matches the birth date exactly and the name
      by Kölner Phonetik; several matches: ask for more details, never pick one; unclear
      name: ask for the spelling.
- [ ] **5. M4 final comparison (item 6).** 46 scenarios x 5 against the baseline;
      `docs/experiments/m4-eval.md`.
- [ ] **6. Telnyx German number (item 2b),** once approved (SIP trunk, dispatch rule,
      phone tests).
- [ ] **7. Audio layer and barge-in harness (item 5),** before 7 December while the 30,000
      ElevenLabs credits last (about 5,000 credits per full run).
- [ ] **8. README update and demo video script (item 7).**

Each paid run, deploy, commit and push is asked first.
