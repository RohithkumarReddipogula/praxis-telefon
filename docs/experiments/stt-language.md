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

By the rule, `multi` passes. Before it becomes the default, a voice test with the user's
real non-native German (the M1 failure); until then `de` stays the default and the M1
regression test stays.
