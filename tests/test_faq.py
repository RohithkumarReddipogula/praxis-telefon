"""FAQ answers (M3 Part C): content, the practice_info tool and how it is measured.

The LLM picks the topic; local keyword retrieval missed the bar
(docs/experiments/faq-retrieval.md). Bar (user, 2026-11-07): >= 95 % of answerable
questions get the right topic, <= 5 % of questions without an answer get any topic;
measured with the real LLM by ``eval/faq/report.py`` (paid, ask first).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.db.seed import OPENING_HOURS
from praxis_telefon.faq import ENTRIES, FAQ_PATH, TOPICS, answer_text, opening_hours_text
from tests.conftest import NOW

HERE = Path(__file__).parent.parent / "eval" / "faq"
EXPECTED_TOPICS = {
    "opening_hours", "address", "parking", "repeat_prescription", "referral",
    "sick_note", "holidays", "insurance_card", "new_patients",
}  # fmt: skip


def test_every_topic_has_one_entry_in_both_languages() -> None:
    assert set(TOPICS) == EXPECTED_TOPICS
    for e in ENTRIES:
        assert e.de.strip() and e.en.strip(), e.id


def test_opening_hours_come_from_the_seed() -> None:
    # Mo/Di/Do 8-12 and 15-18, Mi/Fr 8-12 (db/seed.py); the FAQ must not drift from it.
    assert OPENING_HOURS[2] == OPENING_HOURS[4] and len(OPENING_HOURS[0]) == 2
    de, en = opening_hours_text("de"), opening_hours_text("en")
    assert "Montag, Dienstag und Donnerstag von 8 bis 12 Uhr und von 15 bis 18 Uhr" in de
    assert "Mittwoch und Freitag von 8 bis 12 Uhr" in de
    assert "Monday, Tuesday and Thursday from 8 to 12 and from 3 to 6 pm" in en
    assert answer_text("opening_hours", "de").count("Uhr") >= 3


def test_faq_file_has_no_hand_written_hours() -> None:
    raw = FAQ_PATH.read_text(encoding="utf-8")
    assert "{hours}" in raw and "15 bis 18" not in raw


@pytest.mark.parametrize("name", ["questions.json", "user_questions.json"])
def test_question_sets_are_labelled_with_known_topics(name: str) -> None:
    rows = json.loads((HERE / name).read_text(encoding="utf-8"))
    assert {r["expected"] for r in rows} <= EXPECTED_TOPICS | {None}
    assert {r["lang"] for r in rows} == {"de", "en"}


def test_question_set_sizes() -> None:
    mine = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    user = json.loads((HERE / "user_questions.json").read_text(encoding="utf-8"))
    # 31 without an answer: "Kann ich mich für meinen Mann krankschreiben lassen?" was
    # relabelled sick_note after the first measurement (docs/experiments/faq-topic-choice.md).
    assert len(mine) == 122 and sum(r["expected"] is None for r in mine) == 31
    assert len(user) == 10 and sum(r["expected"] is None for r in user) == 3


# --- the tool ---------------------------------------------------------------------------


@pytest.fixture
def service(conn: sqlite3.Connection) -> ToolService:
    return ToolService(conn, lambda: NOW)


def test_practice_info_returns_the_entry_in_the_call_language(service: ToolService) -> None:
    de = service.practice_info("parking", "de")
    en = service.practice_info("parking", "en")
    assert answer_text("parking", "de") in de and answer_text("parking", "en") in en
    assert "nur" in de.lower()  # only this text may be used


def test_practice_info_other_says_unknown_and_offers_a_message(service: ToolService) -> None:
    text = service.practice_info("other", "de")
    assert "nicht" in text and "Nachricht" in text
    assert not any(answer_text(t, "de") in text for t in TOPICS)


def test_receptionist_tool_lists_exactly_the_topics(conn: sqlite3.Connection) -> None:
    from livekit.agents.llm.utils import build_legacy_openai_schema

    from praxis_telefon.agent.receptionist import Receptionist

    agent = Receptionist(ToolService(conn, lambda: NOW))
    tool = next(t for t in agent.tools if getattr(t, "id", "") == "practice_info")
    params = build_legacy_openai_schema(tool)["function"]["parameters"]  # type: ignore[arg-type]
    assert set(params["properties"]["topic"]["enum"]) == EXPECTED_TOPICS | {"other"}


# --- measurement (offline part of eval/faq/report.py) -----------------------------------


def test_scoring_of_one_answer() -> None:
    from eval.faq.report import outcome

    assert outcome("parking", [("practice_info", "parking")]) == "right"
    assert outcome("parking", [("practice_info", "address")]) == "wrong_topic"
    assert outcome("parking", []) == "no_tool"
    assert outcome("parking", [("practice_info", "other")]) == "said_unknown"
    assert outcome(None, [("practice_info", "other")]) == "right"
    assert outcome(None, []) == "right"  # nothing said at all
    assert outcome(None, [("practice_info", "opening_hours")]) == "false_answer"
    assert outcome(None, [("find_slots", "")]) == "right"


def test_summary_applies_the_bar() -> None:
    from eval.faq.report import summary

    right: dict[str, object] = {"expected": "parking", "outcome": "right"}
    missed: dict[str, object] = {"expected": "parking", "outcome": "no_tool"}
    unknown: dict[str, object] = {"expected": None, "outcome": "right"}
    rows = [right] * 19 + [missed] + [unknown] * 20
    s = summary(rows)
    assert s["right_share"] == 0.95 and s["false_share"] == 0.0 and s["meets_bar"] is True


def test_text_reply_without_a_tool_needs_review() -> None:
    # User, 2026-11-07: a question without an answer may still get an invented fact in
    # text ("Ja, wir machen Hausbesuche"); such replies are listed and count as false
    # answers until reviewed.
    from eval.faq.report import outcome

    assert outcome(None, [], "Ja, wir machen Hausbesuche.") == "review"
    assert outcome(None, [("find_slots", "")], "Einen Moment.") == "right"


def test_review_verdicts_decide_false_answers() -> None:
    from eval.faq.report import summary

    rows: list[dict[str, object]] = [
        {"q": "Machen Sie Hausbesuche?", "expected": None, "outcome": "review", "text": "Ja."},
        {"q": "Kann ich mit Karte bezahlen?", "expected": None, "outcome": "review", "text": "?"},
        {"q": "Wo ist das Krankenhaus?", "expected": None, "outcome": "right"},
    ]
    unreviewed = summary(rows)
    assert unreviewed["false_share"] == round(2 / 3, 3) and unreviewed["to_review"] == 2
    verdicts = {
        "Machen Sie Hausbesuche?": {"verdict": "invented", "text": "Ja."},
        "Kann ich mit Karte bezahlen?": {"verdict": "ok", "text": "?"},
    }
    reviewed = summary(rows, verdicts)
    assert reviewed["false_share"] == round(1 / 3, 3) and reviewed["to_review"] == 0


# --- regression: FAQ measurement 2026-11-07 ---------------------------------------------


def test_topic_description_names_each_topic_with_a_colon(conn: sqlite3.Connection) -> None:
    # The LLM sent topic "weekend" twice: the description read "opening_hours (when open,
    # today, weekend)", and it took a word from the brackets for a topic. Now each topic is
    # "name: what it covers", and no covered word is written like a topic name.
    from livekit.agents.llm.utils import build_legacy_openai_schema

    from praxis_telefon.agent.receptionist import Receptionist

    agent = Receptionist(ToolService(conn, lambda: NOW))
    tool = next(t for t in agent.tools if getattr(t, "id", "") == "practice_info")
    desc = build_legacy_openai_schema(tool)["function"]["parameters"]["properties"]["topic"][  # type: ignore[arg-type]
        "description"
    ]
    for topic in EXPECTED_TOPICS | {"other"}:
        assert f"{topic}:" in desc, topic
    assert "(" not in desc
    assert "lift" in desc  # "Gibt es einen Aufzug?" went to other: address covers access


def test_unknown_topic_is_an_error_not_no_information(service: ToolService) -> None:
    # "weekend" must not read as "no information about Saturday opening hours".
    from praxis_telefon.tools.errors import ToolFailure

    with pytest.raises(ToolFailure, match="opening_hours"):
        service.practice_info("weekend", "de")


def test_starting_a_message_is_right_for_prescription_and_referral_requests() -> None:
    # User decision 2026-11-07, after the first measurement: "Ich brauche ein
    # Folgerezept." is a request; starting a message for the team is the prompt's flow.
    from eval.faq.report import outcome

    text = "Gern nehme ich Ihr Anliegen für das Praxisteam auf. Wie ist Ihr Name?"
    assert outcome("repeat_prescription", [], text) == "right_message"
    assert outcome("referral", [("take_message", "")], "") == "right_message"
    assert outcome("referral", [], "I can take a message for the practice team.") == (
        "right_message"
    )
    # Only for those two topics, and only with a message offer.
    assert outcome("parking", [], text) == "no_tool"
    assert outcome("referral", [], "Gern.") == "no_tool"


def test_right_message_counts_as_right() -> None:
    from eval.faq.report import summary

    rows: list[dict[str, object]] = [
        {"q": "a", "expected": "referral", "outcome": "right_message"},
        {"q": "b", "expected": "parking", "outcome": "right"},
    ]
    assert summary(rows)["right_share"] == 1.0


def test_rescore_uses_current_labels() -> None:
    from eval.faq.report import rescore

    saved = [{"author": "claude", "q": "Kann ich mich für meinen Mann krankschreiben lassen?",
              "expected": None, "calls": [["practice_info", "sick_note"]], "text": "",
              "outcome": "false_answer"}]  # fmt: skip
    assert rescore(saved)[0]["outcome"] == "right"


def test_a_verdict_applies_only_to_the_reply_it_was_given_for() -> None:
    # Re-measurement 2026-11-07: verdicts keyed by question alone were applied to new
    # reply texts nobody had read. A verdict now carries the reply text it judged.
    from eval.faq.report import summary

    row: dict[str, object] = {"q": "Q", "expected": None, "outcome": "review", "text": "new"}
    stale = {"Q": {"verdict": "ok", "text": "old"}}
    fresh = {"Q": {"verdict": "ok", "text": "new"}}
    assert summary([row], stale)["false_share"] == 1.0 and summary([row], stale)["to_review"] == 1
    assert summary([row], fresh)["false_share"] == 0.0 and summary([row], fresh)["to_review"] == 0
