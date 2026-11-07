"""Offline checks of the STT A/B utterances and scoring (eval/stt_ab), no API calls."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict

import pytest
from eval.stt_ab import utterances as U
from eval.stt_ab.score import (
    dob,
    item_correct,
    mcnemar_p,
    multi_decision,
    name_found,
    phone_digits,
    wer,
)

ALL = U.build()


# --- the list ---------------------------------------------------------------------------


def test_committed_list_matches_the_generator() -> None:
    data = json.loads(U.JSON_PATH.read_text(encoding="utf-8"))
    assert data["utterances"] == [asdict(u) for u in ALL]
    assert data["voices"] == U.VOICES
    assert U.load() == ALL


def test_counts_per_group_and_kind() -> None:
    counts = Counter((u.group, u.kind) for u in ALL)
    assert counts == {
        ("de", "name"): 33, ("de", "dob"): 33, ("de", "phone"): 34,
        ("en", "name"): 13, ("en", "dob"): 13, ("en", "phone"): 14,
        ("switch", "name"): 7, ("switch", "dob"): 7, ("switch", "phone"): 6,
    }  # fmt: skip
    assert len({u.id for u in ALL}) == len(ALL) == 160


def test_switch_calls_start_german_and_carry_the_item_in_english() -> None:
    for u in (u for u in ALL if u.group == "switch"):
        assert [s.lang for s in u.segments] == ["de", "en"]
        assert u.lang == "en"


def test_phone_numbers_are_in_the_fictional_ranges() -> None:
    blocks = {p + b for p, b in U.PHONE_BLOCKS}
    for u in (u for u in ALL if u.kind == "phone"):
        assert u.expected[:-3] in blocks, u.id


@pytest.mark.parametrize("u", ALL, ids=lambda u: u.id)
def test_each_utterance_scores_as_correct_against_its_own_text(u: U.Utterance) -> None:
    # Ties the generator and the parsers together: a scoring bug cannot pass for an
    # STT error, nor the reverse.
    text = " ".join(s.text for s in u.segments)
    assert item_correct(u.kind, u.expected, text, folded=False)
    assert item_correct(u.kind, u.expected, text, folded=True)


def test_german_number_words() -> None:
    assert U.de_cardinal(85) == "fünfundachtzig"
    assert U.de_cardinal(21) == "einundzwanzig"
    assert U.de_ordinal(1, "en") == "ersten" and U.de_ordinal(23, "e") == "dreiundzwanzigste"
    assert U.de_year(1985) == "neunzehnhundertfünfundachtzig"
    assert U.de_year(2005) == "zweitausendfünf"
    # Pairs from the left, as often said in German: 456 -> "fünfundvierzig sechs".
    pairs = U.phone_de("030", "23125", "456", 2)
    assert pairs == "null drei null, dreiundzwanzig zwölf fünf, fünfundvierzig sechs"


def test_english_number_words() -> None:
    assert U.en_year(1985) == "nineteen eighty-five"
    assert U.en_year(1905) == "nineteen oh five"
    assert U.en_year(2003) == "two thousand three"
    assert U.en_ordinal(23) == "twenty-third"
    assert (
        U.phone_en("040", "66969", "011", 0)
        == "oh four oh, double six nine six nine, oh double one"
    )


# --- scoring: what Deepgram may write ---------------------------------------------------


@pytest.mark.parametrize(
    ("transcript", "digits"),
    [
        ("Meine Nummer ist 030 23125 456.", "03023125456"),
        (
            "Meine Nummer ist null drei null zwei drei eins zwei fünf vier fünf sechs.",
            "03023125456",
        ),
        ("Sie erreichen mich unter 0302 3125 456.", "03023125456"),
        ("You can call me on oh three oh, two three one two five, four five six.", "03023125456"),
        (
            "Rufen Sie mich zurück unter null drei null dreiundzwanzig zwölf fünf "
            "vier sechsundfünfzig.",
            "03023125456",
        ),
    ],
)
def test_phone_digits(transcript: str, digits: str) -> None:
    assert phone_digits(transcript) == digits


@pytest.mark.parametrize(
    "transcript",
    [
        "Ich bin am fünfzehnten Juni neunzehnhundertfünfundachtzig geboren.",
        "Ich bin am 15. Juni 1985 geboren.",
        "Geboren am 15.06.1985.",
        "Mein Geburtsdatum ist der fünfzehnte sechste neunzehnhundert fünfundachtzig.",
        "Geboren am fünfzehnten Juni fünfundachtzig.",
        "I was born on the fifteenth of June, nineteen eighty-five.",
        "My date of birth is June 15th, 1985.",
        "Date of birth, 15th June 1985.",
    ],
)
def test_dob_forms(transcript: str) -> None:
    assert dob(transcript) is not None and dob(transcript).isoformat() == "1985-06-15"  # type: ignore[union-attr]


def test_dob_2000s_and_wrong_dates() -> None:
    assert str(dob("Geboren am dritten März zweitausendfünf.")) == "2005-03-03"
    assert str(dob("I was born on March third, two thousand five.")) == "2005-03-03"
    assert dob("Geboren am Dienstag.") is None
    assert not item_correct("dob", "1985-06-15", "Geboren am sechzehnten Juni 1985.")


def test_dob_dotted_cardinal_date_as_deepgram_de_writes_it() -> None:
    # Bug: Deepgram `de` writes "22.5.1968" as "zweiundzwanzig Punkt fünf.1968"; the
    # scorer found no date, so right transcripts counted as wrong.
    assert str(dob("Mein Geburtsdatum ist der zweiundzwanzig Punkt fünf.1968.")) == "1968-05-22"
    assert str(dob("Mein Geburtsdatum ist der fünfzehn Punkt eins.2004.")) == "2004-01-15"
    # Still wrong when the month is wrong: "siebzehn Punkt eins" is 17 January, not November.
    assert not item_correct("dob", "2003-11-17", "Geburtsdatum ist der siebzehn Punkt eins.2003.")


def test_phone_compound_thousands_as_deepgram_de_writes_them() -> None:
    # Bug: Deepgram `de` merges digit groups into numbers ("neunzigtausendneun" = 90009);
    # spoken_to_digits rejects them, so right transcripts counted as wrong.
    assert phone_digits("unter null sechs neun neunzigtausendneun einundsechzig eins.") == (
        "06990009611"
    )
    assert (
        phone_digits(
            "unter null vier null sechstausendsechshundertsechsundneunzig neun dreiundvierzig zwei."
        )
        == "04066969432"
    )
    # A digit lost in the merged number stays wrong.
    assert not item_correct(
        "phone",
        "04066969969",
        "null vier null sechs millionen "
        "sechshundertneunundsechzigtausendneunhundertneunundsechzig.",
    )


def test_name_strict_and_folded() -> None:
    assert name_found("Mein Name ist Jürgen Weißmüller.", "Jürgen Weißmüller", folded=False)
    assert not name_found("Mein Name ist Jürgen Weissmüller.", "Jürgen Weißmüller", folded=False)
    assert name_found("Mein Name ist Juergen Weissmueller.", "Jürgen Weißmüller", folded=True)
    assert not name_found("Mein Name ist Jürgen Weismüller.", "Jürgen Weißmüller", folded=True)
    assert not name_found("Mein Name ist Weißmüller.", "Jürgen Weißmüller", folded=True)


def test_wrong_digit_is_wrong() -> None:
    assert not item_correct(
        "phone", "03023125456", "null drei null zwei drei eins zwei fünf vier fünf sieben"
    )


# --- statistics and the decision rule ---------------------------------------------------


def test_wer() -> None:
    assert wer("Mein Name ist Max", "Mein Name ist Max") == 0
    assert wer("Mein Name ist Max", "Mein Name Max") == 0.25


def test_mcnemar() -> None:
    assert mcnemar_p([True] * 10, [True] * 10) == 1.0
    # 8 right only in base, 0 only in other: p = 2 * 0.5**8
    assert mcnemar_p([True] * 8 + [False] * 2, [False] * 8 + [False] * 2) == pytest.approx(2 / 256)


def _results(right: int, total: int) -> list[bool]:
    return [True] * right + [False] * (total - right)


def test_multi_rejected_when_combined_drop_exceeds_2_points() -> None:
    de = {"name": _results(90, 100), "dob": _results(95, 100), "phone": _results(98, 100)}
    multi = {"name": _results(85, 100), "dob": _results(92, 100), "phone": _results(97, 100)}
    decision = multi_decision(de, multi)
    assert decision["combined_drop_points"] == 3.0 and decision["rejected"] is True
    assert decision["flagged_kinds"] == {"name": 5.0, "dob": 3.0}


def test_single_kind_over_2_points_is_flagged_not_rejected() -> None:
    de = {"name": _results(90, 100), "dob": _results(95, 100), "phone": _results(98, 100)}
    multi = {"name": _results(87, 100), "dob": _results(96, 100), "phone": _results(99, 100)}
    decision = multi_decision(de, multi)
    assert decision["combined_drop_points"] == 0.3 and decision["rejected"] is False
    assert decision["flagged_kinds"] == {"name": 3.0}


# --- scripts (offline parts) ------------------------------------------------------------


def test_telephone_keeps_length_and_drops_high_frequencies() -> None:
    import array
    import math as m

    from eval.stt_ab.make_audio import RATE, telephone

    def tone(hz: float) -> bytes:
        return array.array(
            "h", (int(8000 * m.sin(2 * m.pi * hz * i / RATE)) for i in range(RATE))
        ).tobytes()

    def level(pcm: bytes) -> float:
        samples = array.array("h", pcm)
        return sum(abs(x) for x in samples) / len(samples)

    speech, hiss = tone(400), tone(6000)
    assert abs(len(telephone(speech)) - len(speech)) <= 8
    assert level(telephone(speech)) > 0.8 * level(speech)  # 400 Hz passes
    assert level(telephone(hiss)) < 0.2 * level(hiss)  # 6 kHz is above the 4 kHz phone band


def test_clip_counts_per_setting() -> None:
    from eval.stt_ab.run_stt import SETTINGS, clips_for

    counts = {s: len(clips_for(s, ALL)) for s in SETTINGS}
    assert counts == {"de": 640, "multi": 640, "multi_hint": 640, "en": 240}


def test_report_on_made_up_transcripts() -> None:
    from eval.stt_ab.report import render, scored

    de_name = next(u for u in ALL if u.id == "de-name-00")
    records = []
    for setting, text in [("de", de_name.segments[0].text), ("multi", "Mein Name ist Jürgen.")]:
        for condition in ("clean", "phone"):
            records.append({
                "setting": setting, "clip": f"de-name-00-sarah-{condition}.wav", "id": "de-name-00",
                "voice": "sarah", "condition": condition, "text": text, "languages": ["de"],
                "final_delay_s": 0.4,
            })  # fmt: skip
    rows = scored(records, ALL)
    assert [r["right"] for r in rows] == [True, True, False, False]
    text = render(rows)
    assert "'rejected': True" in text and "combined_drop_points': 100.0" in text
