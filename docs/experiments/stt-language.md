# Experiment: Deepgram `de` vs `multi` (M3 Part B, step 2a)

Date: 2026-11-07. Model: Deepgram Nova-3, streaming, production options
(`main.stt_options`), only the language changed.

## Question

English callers need an STT that hears English. `language="de"` cannot; `multi` can, but
in M1 `multi` transcribed the user's own (non-native) German as English
(`tests/test_agent_options.py::test_stt_uses_german_by_default`).

Decision rule, set before the run (user): **reject `multi` if German accuracy on names,
dates of birth and phone digits combined drops by more than 2 percentage points against
`de`**; a single item type dropping by more than 2 points is flagged.

## Method

- **Utterances** (`eval/stt_ab/utterances.py`, committed as `utterances.json`, seed
  20261107): 100 German (33 name, 33 date of birth, 34 phone), 40 English (13/13/14) and
  20 switch calls (a German opener, then the item in English). Made-up names, phone numbers
  in fictional ranges; numbers and dates in several spoken forms (digit by digit, pairs,
  "doppel", ordinals, "oh" and "zero").
- **Audio** (`make_audio.py`): ElevenLabs `eleven_flash_v2_5`, two standard voices (Sarah,
  George); each clip clean (16 kHz) and as a phone line (8 kHz, G.711 mu-law, back to
  16 kHz). 640 clips, 47 minutes of speech.
- **Settings** (`run_stt.py`): `de` (production), `multi`, `multi` with
  `language_hint=["de","en"]`, and `en` (English and switch clips only). Each clip streamed
  in 20 ms frames at real-time pace plus 1.5 s silence. 2,160 transcripts.
- **Scoring** (`score.py`): phone digits via `spoken_to_digits`, dates compared as dates
  (German and English forms), names as consecutive words with umlauts/ß folded (strict in
  brackets). Paired comparison on the same clips, exact McNemar.
- **Cost:** ElevenLabs 4,202 credits; Deepgram about $1.85 (estimate from clip lengths).

## Results

German clips (400 per setting, paired):

| | `de` | `multi` | change |
|---|---|---|---|
| name (folded) | 64.4 % [58.3 %] | 63.6 % [54.5 %] | −0.8 [−3.8] |
| date of birth | 91.7 % | 100 % | +8.3 |
| phone digits | 76.5 % | 94.9 % | +18.4 |
| **combined** | **77.5 %** | **86.3 %** | **+8.8, p = 0.00004** |
| clean / phone line | 78.0 / 77.0 % | 85.5 / 87.0 % | |

**`multi` is not rejected and no type is flagged.** `de` lost most on numbers: it drops
repeated digits ("neun neun neun" as one "neun"), merges words ("siebzehntelfte") and
writes dates as "zweiundzwanzig Punkt fünf.1968" (the scorer reads that form; before that
fix `de` scored 74.0 % combined, same decision).

- **`language_hint`:** same text as plain `multi` on 638 of 640 clips. No effect.
- **English clips:** `de` returned an empty transcript for 110 of 160 (a real English
  caller would get the two-strike handover). `multi` 68.8 %, `en` 67.5 % right; English
  digits and dates ≥ 96 % with both.
- **Switch calls:** `de` 0 %, `multi` 58.8 %, `en` 72.5 %.
- **Deepgram's language label with `multi`:** 100 % of finals on German clips labelled
  German, 0 % on English clips. A better signal for a switch to English than our
  transcript check (`detect_language` saw English in 11 of 80 switch transcripts).
- **Final transcript after the audio ends, p50 / p90:** `de` 0.13 / 0.26 s, `multi`
  0.15 / 0.31 s (+20 / +50 ms).
- **Price:** `multi` $0.0092/min vs $0.0077/min.

## Caveats

- **Synthetic native voices.** ElevenLabs speaks clean, native German. The M1 failure was
  the user's real non-native German heard as English; this test cannot show that. `multi`
  needs a voice test with real non-native speech before it becomes the default.
- **Names are weak in every setting:** about 1 in 3 German names misheard (e.g. "Östürk"
  for Öztürk, "Anne Gret" for Annegret), equally often with `de` and `multi` (15 vs 14
  clips right only in one). German names spoken in English calls: 4–15 % right. Strict
  spelling (ß, umlauts) is 3.8 points worse with `multi`.
- One run per setting; Deepgram may vary between runs. The paired difference (53 clips right
  only with `multi`, 18 only with `de`) is far from noise.

## Decision

By the rule, `multi` passed on synthetic audio. **With the user's real accented German it
failed** (voice tests below): it tagged German as Telugu, Afrikaans, Norwegian or English
and lost callback digits; `language_hint=["de","en"]` did not fix it. **`multi` is
rejected; `de` stays the default** (M1 regression test unchanged). English calls go
through option B: `de` by default, the Deepgram stream switched to `en` on an explicit
request (planned).

The synthetic A/B could not show this: ElevenLabs speaks native German, and the user's
voice cannot be recorded for a repeatable test (synthetic data only), so accented speech
was tested live.

## Voice test (2026-11-07, `STT_LANGUAGE=multi`)

Two calls by the user, real voice, console.

- **English call:** all finals tagged `en`; `find_slots` and `book` worked.
- **German call** (the user's non-native German): message saved, callback number right,
  but 2 German finals tagged `en` (10:47:37, 10:48:09). After the first, the LLM answered
  in English and the sentence filter blocked 3 sentences (`meta:language`).

So `multi` hears both languages, but a single wrong tag must not switch the call. `multi`
stayed opt-in; the code now decides the call language with hysteresis
(`agent/call_language.py`: 2 consecutive confirmed finals, explicit requests).

## Voice tests with accented German (2026-11-07, afternoon)

The user's native language is Telugu. Calls in the console, real voice.

**`multi`, no hint** (one German call, a message with a callback number): most German
finals tagged `te` (Telugu), one `no` (Norwegian); `detect_language` found no language in
them, so the call language stayed German, but the transcripts themselves were wrong: the
callback number never came through, the agent asked for it 3 times, no `take_message`.
The language rule worked; the transcription did not.

**`multi` with `language_hint=["de","en"]`** (`STT_LANGUAGE_HINT=de,en`), three calls:

| call | result |
|---|---|
| 1, German | first German sentence transcribed as English (tag `en`, detected `en`); as the first confirmed final it switched the call to English |
| 2, German | tags `te`, `af`, `no` still present; callback number wrong by 2 digits; `take_message` called twice (same name and number, reasons "callback" then "prescription", 45 s apart: two notes for one request) |
| 3, English | tags `te`, `af`, `no` still present; phone number wrong by 2 digits |

The hint biases `multi` but does not restrict it to German and English.

**`de`** (one German call with language requests): German transcribed fine. Of the
requests, only "Können wir Englisch sprechen?" was recognised by the code and switched
the call; "Can we speak English?", "Do you speak English?" and "English, please" did
not match as `de` transcribed them. The LLM still understood them and answered in English,
which the filter blocked (`meta:language`), so `de` transcribes English requests well
enough for the LLM, not yet for our patterns.

Other notes from the logs:
- LiveKit's transcript line shows a sticky language (it updates only on finals longer
  than 5 characters), so it can say `te` for a final whose own tag is `de`; our
  `caller_final` line logs each final's own tag. The LiveKit turn detector was not
  affected (no "does not support language" line).

## Own-voice recordings (2026-11-07, for option B)

The console logs are redacted, so the call-4 transcripts could not be read back. Instead
the user recorded the sentences option B depends on.

**Method.** `eval/my_voice.py`: 12 sentences, one non-native speaker (native language
Telugu), MacBook Air microphone, 16 kHz mono, one take each (01 re-recorded: the first
take was almost silent). Each clip streamed through Deepgram Nova-3 twice, `de` and `en`,
as in a call (20 ms frames in real time, production options). The recordings and
transcripts stay in `eval/results/my_voice/` (gitignored, a test checks it); no audio is
in git. Cost about $0.03.

**Results.** Transcripts as Deepgram wrote them; "→" is what the agent's code makes of
them as it is now (after the changes below).

| # | said | `de` | `en` |
|---|---|---|---|
| 01 | Can we speak English? | (empty) | "Can we speak English?" → en |
| 02 | Do you speak English? | (empty) | "Do you speak English?" → en |
| 03 | English, please. | (empty) | "English please" → en |
| 04 | Could we speak in English, please? | (empty) | "Could we speak in English please" → en |
| 05 | English. | "Englisch." → en | "English" → en |
| 06 | Englisch. | "Englisch." → en | "English" → en |
| 07 | Können wir Englisch sprechen? | "Können wir Englisch sprechen?" → en | "English Pression." |
| 08 | Können wir wieder Deutsch sprechen? | "Können wir wieder Deutsch sprechen?" → de | "Deutsche expression." → de |
| 09 | Sprechen Sie Deutsch? | "Sprechen Sie Deutsch." → de | (empty) |
| 10 | Meine Nummer ist null drei null, zwei drei eins zwei fünf, vier fünf sechs. | all digits right | "My numeral ist null dry null zwai dry eins zwei funf fear, funf zex." |
| 11 | Deutsch. | "Deutsch." → de | "Doish" → de |
| 12 | Deutsch, bitte. | "Neulich bitte." | "Deutsche" → de |

- `de` does not hear English requests (01–04 empty), as in the A/B (110 of 160 English
  clips empty). Into English through `de` works with "English." / "Englisch." alone (05,
  06) and "Können wir Englisch sprechen?" (07).
- `en` does not hear German: requests come out as near-English ("Deutsche expression.")
  or nothing (09), German digits are lost (10), and `detect_language` finds no German in
  them, so the "2 German finals" rule cannot bring the call back.

**What changed in the code because of it** (`agent/call_language.py`,
`agent/receptionist.py`, `safety/messages.py`):
- "English" / "Englisch" / "Inglisch", also with "please" / "bitte", switch to English as
  the whole utterance (03: "English please" matched no pattern).
- The German "say that again" message ends "If you'd like to speak English, just say
  'English'." (English callers take two turns: an empty first line, then the word).
- On the `en` stream, the exact words "Deutsch", "Deutsche" and "Doish" (11) anywhere in a
  final mean back to German; "German" only alone or in a request ("I have German health
  insurance" stays English); "Deutschmann" does not switch.
- The English "say that again" message asks for "Deutsch, bitte" (12 came through as
  "Deutsche", 11 as "Doish").
- An empty final on the `en` stream in a call that started German switches back to `de`
  and the agent asks bilingually to repeat; it is no strike (09 came back empty).
- "Können wir wieder Deutsch sprechen?" is recognised ("wieder" was missing).

**Limits.** One speaker, one take per sentence, one microphone, a quiet room, no phone
line. Spellings like "Doish" or "Deutsche expression" may differ for other speakers or
another take; the empty-final fallback covers the way back when they do. The final check
is the user's voice test with the default `de`.

**Switching itself** (`eval/stt_ab/switch_check.py`, synthetic clips, $0.02): switching
the stream on a final lost no audio (the start of the following phone number intact in
10/10; one digit error also seen without a switch, one inserted digit). Switching
mid-sentence lost the words sent before it (2 of 10), so the agent switches only right
after a final.

## Voice test with option B (2026-11-07, default `de`)

Three calls by the user, real voice, console; `STT_LANGUAGE` and `STT_LANGUAGE_HINT` empty.
**All three passed.**

| call | what happened |
|---|---|
| A, German, then English | switched to `en` at 16:39:11, booked in English, switched back to `de` at 16:40:26 |
| B, English from the first line | the first line got the bilingual "say that again"; "English" switched at 16:41:27; booked |
| C, German | message saved |

One warning in call A at 16:40:14 (after the switch to `en`, before the switch back):
LiveKit's "transcript arrives after turn has been committed, consider raising
min_delay". No visible effect in the call; noted for the endpointing / barge-in work in
Part C rather than changed here (a higher `min_delay` costs latency on every turn).

**Decision: option B adopted.** `de` stays the default, the stream switches on language
requests; English switching (M3 Part B) is done. Open: an English line in the greeting
would save English callers one turn (about 1.5 s per call); names are still misheard
about 1 in 3 in every setting.
