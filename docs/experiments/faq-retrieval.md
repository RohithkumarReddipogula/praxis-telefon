# Experiment: local keyword retrieval for FAQ answers (M3 Part C) — negative result

Date: 2026-11-07. Outcome: **not adopted**; the LLM picks the FAQ topic instead.

## Question

The agent should answer practice questions (opening hours, address, parking, repeat
prescriptions, referrals, sick notes, holidays, insurance card, new patients) only from a
matching FAQ entry, and say it does not know otherwise. Can a local retrieval step find
the entry without a new paid service or package?

Bar, set before measuring (user): on held-out questions **at least 95 % of answerable
questions get the right entry, at most 5 % of questions without an answer get any entry.**

## Method

- **FAQ:** `src/praxis_telefon/faq/faq.json`, 9 synthetic entries in German and English
  (opening hours generated from `db.seed.OPENING_HOURS`), plus a keyword list per entry
  (German and English synonyms, e.g. "Krankschreibung, AU, gelber Schein, sick note").
- **Retrieval:** BM25 in pure Python over `text.normalize` tokens (umlauts folded),
  stopwords, light suffix stripping, prefix matches for German compounds ("park" ->
  "Parkplatz"), keywords weighted 3×. An entry counted only above a score threshold and
  with at least a share of the question's content words matched (coverage).
- **Questions:** `eval/faq/questions.json`, 122 labelled questions written by Claude
  (9 topics × 6 German + 4 English, 32 without an answer, some in STT style without
  punctuation). Even rows tuned threshold and coverage, odd rows were held out.
  `eval/faq/user_questions.json`: 10 questions by a second author who had not seen the
  FAQ, reported separately.

## Results

| Round | Tuning: right / false | Held-out: right / false | Second author: right |
|---|---|---|---|
| 1, first keyword lists | 78 % / 19 % | 82 % / 0 % | — |
| 2, keywords added from tuning misses | 86 % / 12 % | 92 % / 28 % | 6 / 10 |
| **Bar** | | **≥ 95 % / ≤ 5 %** | |

(Round 2's numbers include the second author's questions in the halves; their 10 alone:
4 of 7 answerable right, 1 of 3 without an answer wrongly answered.)

Failure types:
- **Paraphrases** with no shared word: "meine tabletten sind bald alle wie bekomme ich
  neue", "Kann ich bei Ihnen Patient werden?", "Haben Sie zwischen den Jahren geöffnet?"
  (second author: the German idiom for the days between Christmas and New Year).
- **Topic overlap:** "Haben Sie über Weihnachten geöffnet?" went to opening hours, not
  holidays: "geöffnet" outweighs "Weihnachten".
- **Near-misses sharing one word:** "Where is the nearest pharmacy?", "Wo ist das nächste
  Krankenhaus?" -> address ("where", "nächste"); "Kann ich mit Karte bezahlen?" ->
  insurance card; second author: "Is there a pharmacy close to your practice?" -> opening
  hours.
- **Fixes trade misses for false answers.** Round 2 made "wo" / "where" address words
  (more address hits, more false answers on "where is the nearest …") and made
  "Patient" a stopword because it is in many entries, which broke the second author's
  "nehmen Sie noch neue Patienten auf?". Each fix fitted the tuning questions and missed
  new ones: the same author writing the FAQ, the keywords and the questions overstates
  how well keyword retrieval generalises.

## Decision

Keyword retrieval stays far from the bar and a third round would only fit this question
set. **The LLM picks the topic instead:** `practice_info(topic)` takes one of the 9 topic
ids or `other`, and code returns that entry's text in the call's language. It reuses the
one tool call the design already had, needs no new service or package, and the LLM
understands paraphrases and whether a question is about the practice at all. The same
bar applies, measured on the same 132 questions with the real LLM. The BM25 code was
removed (it is in this branch's history only as uncommitted work); the question sets and
the report stay.

Considered and not chosen: local multilingual embeddings (a ~120 MB model download; the
"no answer" threshold stays the hard part) and more keyword rounds (overfitting).
