# Experiment: the LLM picks the FAQ topic (M3 Part C)

Date: 2026-11-07. Follows `faq-retrieval.md` (keyword retrieval missed the bar).

## Design

`practice_info(topic)` takes one of 9 topic ids (`opening_hours`, `address`, `parking`,
`repeat_prescription`, `referral`, `sick_note`, `holidays`, `insurance_card`,
`new_patients`) or `other`; code returns that entry's text from `faq/faq.json` in the
call's language, or "no information, offer a message" for `other`. No new service or
package; the one tool call the design already had.

Bar (user, set before measuring): **≥ 95 % of answerable questions get the right
topic, ≤ 5 % of questions without an answer get a false answer.**

## Method

`eval/faq/report.py`: each question goes once to the production LLM (Haiku 5.5,
reasoning `low`) as the first caller line, with the Receptionist's system prompt, tools
and language note; only the reply's tool calls and text are read, no tool runs.

- **Questions:** 122 by Claude (`eval/faq/questions.json`), 10 by a second author who had
  not seen the FAQ (`eval/faq/user_questions.json`), reported separately. The second
  author's 10 are the clean check: nothing below was changed because of them.
- **Outcomes:** answerable: right only with `practice_info(<expected topic>)`. Without an
  answer: false if `practice_info` got a real topic. A text reply with no tool call at
  all is listed for review and counts as a false answer unless a reviewer marks that
  exact reply text "ok" (user: it may invent a fact, "Ja, wir machen Hausbesuche").
  Verdicts are in `eval/faq/review.json`.
- Cost about $0.03 per run.

## Run 1

| | Right | False answers |
|---|---|---|
| Claude's 122 (90 / 32) | 84/90 = 93.3 % | 1/32 = 3.1 % (after review: 4 text replies "ok") |
| Second author's 10 (7 / 3) | 7/7 = **100 %** | 0/3 = **0 %** |

Misses on Claude's questions:
- 2 × topic `weekend` (not a valid topic): the tool description read "opening_hours (when
  open, today, weekend)" and the LLM took the bracketed word for a topic. LiveKit rejects
  such a call (schema validation), so a real call would have needed a retry.
- 1 × "Gibt es einen Aufzug zur Praxis?" -> `other`: the description did not say that the
  address entry covers access and the lift.
- 3 × requests, not questions ("Ich brauche ein Folgerezept.", "Ich brauche eine
  Überweisung zum Orthopäden / ... für die Radiologie."): no tool; the agent started a
  message for the team, which the prompt asks for with prescriptions and referrals.
- 1 false answer: "Kann ich mich für meinen Mann krankschreiben lassen?" (labelled "no
  answer") -> `sick_note`.

## Changes after run 1 (made after seeing the results)

These were decided after seeing run 1, on Claude's questions only, so Claude's numbers
after them are not an independent test; the second author's 10 are.

1. **Tool description** (a bug): each topic as "name: what it covers", no brackets; the
   address topic mentions access and the lift. An unknown topic returns an error listing
   the valid ones instead of "no information". Regression tests in `tests/test_faq.py`.
2. **Scoring** (user decision): for prescription and referral requests, starting a
   message for the team counts as right (`right_message`).
3. **Label** (user decision): "Kann ich mich für meinen Mann krankschreiben lassen?" is
   `sick_note`, not "no answer"; Claude's set now has 91 answerable, 31 without answer.
4. **Review tooling:** verdicts now apply only to the exact reply text they judged (run 2
   gave new texts; verdicts keyed by question alone would have carried run 1's "ok").

Run 1 under changes 2 and 3 (same replies): Claude's 88/91 = 96.7 % right, 0/31 false;
the second author's unchanged.

## Run 2 (after all changes)

| | Right | False answers |
|---|---|---|
| Claude's 122 (91 / 31) | 91/91 = 100 % | 1/31 = 3.2 % |
| — German (55 / 19) | 100 % | 1/19 = 5.3 % |
| — English (36 / 12) | 100 % | 0 % |
| Second author's 10 (7 / 3) | 7/7 = **100 %** | 0/3 = **0 %** |

**Both meet the bar.** The one false answer is a text reply without a tool:
"Haben Sie einen Laborbefund für mich?" -> "Ich kann Ihnen leider keine Befunde
telefonisch mitteilen, aber ich kann eine Nachricht für das Praxisteam aufnehmen." —
reviewed as **invented**: a practice policy the FAQ does not contain (cautious and
plausible, but not backed). The three other text replies (weather twice, "Is the doctor
married?") decline without a practice fact and are "ok".

## Limits

- One run per configuration; the LLM's choice can vary between runs (run 1 and run 2
  gave different texts for the same questions).
- Claude wrote the FAQ, the descriptions and 122 of the questions and changed the
  description after run 1; the second author's 10 are a small clean sample (7 + 3).
- Text replies are judged by one reviewer (Claude), recorded per reply.
- First caller line only; mid-call questions are covered by the live tests
  (`tests/live/test_faq_live.py`).
- The sentence filter blocks practice facts without `practice_info` only for the FAQ's
  topics; a policy on another subject (results by phone, as above) is not caught there.

## Decision

Adopt `practice_info(topic)`. Open: the "results by phone" kind of reply (a practice
policy outside the FAQ) could become an FAQ entry or a filter family.
