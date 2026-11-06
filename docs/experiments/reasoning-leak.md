# Experiment: reasoning leak before tool calls (M3 Part B, step 1)

Date: 2026-11-06. Model: `anthropic/claude-haiku-5.5` via OpenRouter, prompt caching on.

## Question

With reasoning effort `none`, the agent sometimes spoke its own reasoning before a tool
call, mostly in English, for example:

> This is a new appointment request for November 12th, so I'll search for free slots.

The sentence filter (`agent/speech_filters.py`) blocks most of these, mainly through the
"English in a German call" check. That check is useless once English calls exist, so the
leak has to be reduced at the source. Two levers were tested:

1. **Tool descriptions in German** instead of English (the leaked sentences borrowed
   words from them, e.g. "search for slots").
2. **Reasoning effort `low`** instead of `none` (with thinking off, Claude models are
   known to write reasoning into the visible answer; with thinking on it goes into the
   hidden reasoning instead).

Decision rule, set before the runs: latency limit +150 ms on `llm_ttft` p50 against the
baseline; if two arms are within noise on leaks, prefer the faster and simpler one.

## Method

- **Suite:** the 78 live tests in `tests/live/` (safety, tool calls, relative dates,
  triage): the real `Receptionist` and LLM in LiveKit text mode, a scripted caller, an
  in-memory synthetic database, fixed clock (Monday 9 November 2026, 07:00 Berlin).
- **Arms:** 2 × 2, selected with `LIVE_TOOL_DOCS` and `LLM_REASONING_EFFORT`.

  | Arm | Tool descriptions | Reasoning |
  |---|---|---|
  | A (baseline) | English (production) | none |
  | B | English | low |
  | C | German | none |
  | D | German | low |

- **Runs:** 5 per arm, 20 in total, run interleaved (A, B, C, D, A, …) so drift in
  provider latency spreads evenly. A's first run is the caching check run (same
  settings). An earlier uncached A run is not counted.
- **Leak:** a sentence the filters blocked with a `meta:` reason, recounted by replaying
  every run's sentences through the current filters, so all runs are judged by the same
  rules. Blocked sentences are never spoken; a leak count measures how often the LLM
  produced one.
- **Latency** (recorded by `tests/live/harness.py`):
  - `llm_ttft`: time from the start of `llm_node` to its first text or tool call, as
    LiveKit's `llm_node_ttft` in the turn log. Only replies that spoke count.
  - Time to tool call: the same for replies that started with a tool call.
  - Caller line to first speech: from the caller's line to the first text the agent
    yields, across the tool round trip. This is what the caller waits for (minus STT,
    end-of-turn and TTS, which the arms do not change).
- **Report:** `uv run python -m eval.leak_report eval/results/*.jsonl`.

## Results

| Arm | Passed | Leaks per run | Total (per 75 tests) | `llm_ttft` p50 | Tool call p50 / p90 | First speech p50 / p90 | Reasoning tokens | Cost (5 runs) |
|---|---|---|---|---|---|---|---|---|
| A en / none | 390/390 | 7, 5, 6, 6, 3 | 27 (5.2) | 960 ms | 1016 / 1219 ms | 1022 / 1988 ms | 0 | $0.265 |
| B en / low | 390/390 | 0, 0, 0, 0, 0 | **0** | 1018 ms | 1126 / 2056 ms | **2128 / 3361 ms** | 41,495 | $0.228 |
| C de / none | **383/390** | 5, 9, 7, 6, 5 | 32 (6.2) | 952 ms | 1022 / 1304 ms | 1036 / 2018 ms | 0 | $0.264 |
| D de / low | 390/390 | 0, 0, 0, 0, 0 | **0** | 1028 ms | 1565 / 2138 ms | 2250 / 3669 ms | 51,440 | $0.236 |

Per-run `llm_ttft` p50 (ms): A 986, 981, 957, 957, 928; B 1064, 1013, 1016, 984, 1021;
C 1014, 926, 948, 942, 971; D 1075, 1027, 1017, 1018, 1058. 81–88 % of prompt tokens
were read from the cache in every arm.

C's 7 failures: 6 × `uebermorgen` (the agent worked out the date itself, "Übermorgen wäre
der 11. November", before `find_slots`) and 1 × `new_patient` (asked again for name and
date of birth). It had 6 weekday blocks against 0–1 in the other arms.

## Statistics

Each arm against A. Three comparisons, so Bonferroni: significant if p < 0.017.

- **Permutation test** on the per-run leak counts (exact, all 252 ways to split 10 runs
  into 5 + 5; two-sided, difference of means). It respects how much runs vary (3 to 7
  leaks in A, 5 to 9 in C). With 5 against 5 runs, the smallest possible p is 0.008.
- **Exact conditional Poisson test** on the totals (equal exposure), for reference. It
  assumes Poisson variance between runs, so it is optimistic here.

| vs A | Leaks | Permutation p | Poisson p | `llm_ttft` p50 | Run-p50 permutation p |
|---|---|---|---|---|---|
| B | 0 vs 27 | **0.008** | < 0.001 | +58 ms | 0.016 |
| C | 32 vs 27 | 0.476 | 0.603 | −8 ms | 0.952 |
| D | 0 vs 27 | **0.008** | < 0.001 | +68 ms | 0.008 |

- **Reasoning `low` removes the leak** (B and D), beyond noise. Its `llm_ttft` cost
  (+58 / +68 ms) is real but within the +150 ms limit.
- **German descriptions do not help** (C): leak counts are within noise of A, latency is
  unchanged, and they caused test failures.
- **B against D:** both have 0 leaks; B is faster (1018 vs 1028 ms `llm_ttft`, 2128 vs
  2250 ms first speech) and simpler (no change to the descriptions).

## The silence finding

With reasoning `low`, the model thinks instead of saying a short "Einen Moment, ich schaue
nach …" before a tool: 9 sentences before a tool in all of B, 0 in D, about 550 in A and
C. The caller hears nothing until the tool round trip is done:

- Caller line to first speech: p50 **+1.1 s** (1022 → 2128 ms), p90 **+1.4 s**
  (1988 → 3361 ms).
- `llm_ttft` does not show this: a reply that only calls a tool creates no message, so it
  has no `llm_node_ttft` in the turn log.
- Time to the tool call itself rises only modestly at p50 (1016 → 1126 ms), more at p90
  (1219 → 2056 ms), when the model thinks longer.

By the agreed metric, B passes. By what the caller hears, it adds about a second of
silence whenever a tool is called. A code-spoken filler at the start of a tool call is
planned to close that gap (see below).

## Decision

- **Adopt B:** English tool descriptions (unchanged), reasoning effort `low`
  (`.env.example`, and the code default in `config.py`). Each deployment's `.env` has to
  set `LLM_REASONING_EFFORT=low`.
- **German tool descriptions: not adopted.**
- **Next:** a short phrase spoken by code ("Einen Moment, bitte." / "One moment, please.")
  when a reply starts with a tool call and nothing was said yet, then measure first
  speech again.
- **Found on the way:** the weekday filter let months through when only today's date
  sentence named them, so "Übermorgen wäre der 11. November" before `find_slots` was
  spoken (C's failures). Fixed: today's date backs only today's weekday.

## Follow-up: code-spoken filler (2026-11-06)

`agent/tool_filler.py` says "Einen Moment, bitte." / "One moment, please." when a reply
called a tool and said nothing, and nothing was said since the caller's last line (so at
most once per caller turn). The decision is made when the filtered reply ends: the
sentence filter may still hold "Einen Moment, ich schaue nach." when the tool call chunk
passes, and a first draft that fired on that chunk said both. A tool call is the last
thing in the stream, so the end comes right after it. The phrase is fixed text, so it
cannot leak; fillers in the history change neither the safety replay nor `intent_asked`
(tests).

3 runs, B settings (English descriptions, reasoning `low`), filler on:

| | A (none) | B (low) | B + filler |
|---|---|---|---|
| Passed | 390/390 | 390/390 | 234/234 |
| Leaks per 75 tests | 6.0 | 0 | 0.3 |
| Caller line → first speech p50 / p90 | 1022 / 1988 ms | 2128 / 3361 ms | **1212 / 2853 ms** |
| Caller line → answer p50 / p90 | 1022 / 1988 ms | 2128 / 3361 ms | 2173 / 3432 ms |
| Time to tool call p50 / p90 | 1016 / 1219 ms | 1126 / 2056 ms | 1221 / 2177 ms |

(Leaks here count block reasons, as `eval/leak_report.py` reports them; the main results
table above counts sentences.)

- First speech p50 is 916 ms earlier than B and 190 ms later than A (the filler waits for
  the tool call). The answer is not delayed (+45 ms p50, within noise).
- Turns with the filler (262 of 406): first speech p50 1233 / p90 2185 ms, which matches
  the time to the tool call.
- The remaining p90 comes from turns **without** a tool (no filler): first speech p50
  1121 / p90 3432 ms. With `low` the model sometimes thinks long before a direct text
  answer too (`llm_ttft` p90 2.7 s against 1.2 s with `none`). Open: a timer-based filler
  or a shorter thinking time; not built.
- The one leak: the model wrote a `take_message` call as text ("[take_message aufrufen
  mit caller_name …]"); `meta:tool` blocked it and the test passed.
- Real end-to-end latency still needs a console session: the turn log now marks the
  filler (`is_filler`) and gives every message `since_caller_s`, since LiveKit reports
  `e2e_s` only for the first speech of a turn.

## Limitations

- Text mode: no STT, end-of-turn detection or TTS; all four arms share those.
- Scripted caller and 78 fixed scenarios, one day, one model and one provider route;
  counts per run are small (0–9).
- A leak is measured as "the filter blocked it". Narration the filters do not recognise
  is not counted, though the report lists unfiltered sentences before tools for review.
  B left 9 for review, all fine: an existing appointment read out before `find_slots`,
  and "Einen Moment, ich suche einen Termin am Mittwoch, den 11. November, …" with the
  day the caller had named. D left none.
- Cost: 20 runs about $0.99 at OpenRouter list prices (the uncached A run was $0.14 and
  matched the OpenRouter dashboard).
