"""The call's language, decided by code from the caller's finals (agent/call_language.py).

Rule (user, 2026-11-07): the language changes only after 2 consecutive finals in the
other language; a final counts as English only if Deepgram's tag and detect_language
both say English. Before the caller said anything German, one English final is enough.
Finals that show no language (``Ja.``, digits, a name) neither count nor break a run.
An explicit request ("Can we speak English?") switches at once.

Regression, voice test 2026-11-07 with STT_LANGUAGE=multi: 2 of the user's German finals
were tagged ``en``; after the first the LLM answered in English.
"""

from __future__ import annotations

import pytest

from praxis_telefon.agent.call_language import (
    CallerFinal,
    decide,
    language_request,
    vote,
)

DE = "Ich hätte gern einen Termin bei Dr. Weber."
DE2 = "Mein Name ist Max Mustermann, das ist richtig."
EN = "I would like an appointment with Dr. Weber, please."
EN2 = "Yes, that is right, thank you."
NEUTRAL = "Max Mustermann."


def final(text: str, tag: str | None) -> CallerFinal:
    return CallerFinal.from_text(text, tag)


def lang(*finals: CallerFinal) -> str:
    return decide(finals).language


def test_one_wrong_en_tag_does_not_switch_a_german_call() -> None:
    # Voice test 2026-11-07: German, one final tagged en (English-looking text), German.
    finals = [final(DE, "de"), final(EN, "en"), final(DE2, "de")]
    assert [lang(*finals[: i + 1]) for i in range(3)] == ["de", "de", "de"]


def test_two_consecutive_english_finals_switch() -> None:
    assert lang(final(DE, "de"), final(EN, "en"), final(EN2, "en")) == "en"


def test_en_tag_alone_does_not_count_as_english() -> None:
    # Deepgram says en, our detector does not: not English.
    assert vote(final(NEUTRAL, "en")) is None
    assert vote(final(DE, "en")) == "de"
    assert lang(final(DE, "de"), final(DE2, "en"), final(DE, "en")) == "de"


def test_english_words_alone_do_not_count_with_a_de_tag() -> None:
    # STT_LANGUAGE=de tags everything de: the call never switches by itself.
    assert vote(final(EN, "de")) == "de"
    assert lang(final(EN, "de"), final(EN2, "de"), final(EN, "de")) == "de"


def test_neutral_finals_neither_count_nor_break_a_run() -> None:
    assert lang(final(DE, "de"), final(EN, "en"), final(NEUTRAL, "en")) == "de"
    assert lang(final(DE, "de"), final(EN, "en"), final(NEUTRAL, "en"), final(EN2, "en")) == "en"
    # A tag of de counts as German (either label saying German is enough): run broken.
    assert lang(final(DE, "de"), final(EN, "en"), final(NEUTRAL, "de"), final(EN2, "en")) == "de"


def test_first_english_final_switches_before_any_german() -> None:
    assert lang(final(EN, "en")) == "en"
    assert lang(final(NEUTRAL, "en"), final(EN, "en")) == "en"  # neutral is not German


def test_switch_back_to_german_needs_two_german_finals() -> None:
    english_call = [final(EN, "en"), final(EN2, "en")]
    assert lang(*english_call, final(DE, "de")) == "en"
    assert lang(*english_call, final(DE, "de"), final(EN, "en")) == "en"  # run broken
    assert lang(*english_call, final(DE, "de"), final(DE2, "de")) == "de"


def test_typed_input_counts_by_the_detector_alone() -> None:
    # Text mode (live tests, console text): no STT tag.
    assert lang(final(DE, None), final(EN, None)) == "de"
    assert lang(final(DE, None), final(EN, None), final(EN2, None)) == "en"


@pytest.mark.parametrize(
    ("text", "wanted"),
    [
        ("Can we speak English?", "en"),
        ("Do you speak English?", "en"),
        ("Could we speak in English, please?", "en"),
        ("Können wir Englisch sprechen?", "en"),
        ("Sprechen Sie Englisch?", "en"),
        ("Können wir Deutsch sprechen?", "de"),
        ("Sprechen Sie Deutsch?", "de"),
        ("Can we speak German?", "de"),
        ("Do you speak German?", "de"),
        ("Können wir bitte auf Deutsch weitermachen?", "de"),
    ],
)
def test_explicit_request_switches_immediately(text: str, wanted: str) -> None:
    assert language_request(text) == wanted
    start = [final(DE, "de")] if wanted == "en" else [final(EN, "en"), final(EN2, "en")]
    # One final is enough, whatever Deepgram tagged it.
    for tag in ("de", "en", None):
        assert lang(*start, final(text, tag)) == wanted


def test_after_a_request_the_two_final_rule_holds_again() -> None:
    switched = [final(DE, "de"), final("Can we speak English?", "en")]
    assert lang(*switched, final(DE, "de")) == "en"
    assert lang(*switched, final(DE, "de"), final(DE2, "de")) == "de"


@pytest.mark.parametrize(
    "text",
    [
        "Ich spreche Englisch mit meinem Sohn, aber jetzt Deutsch.",
        "I speak English.",
        "My English is not so good.",
        "Mein Sohn spricht Deutsch.",
        "Do you speak with Dr. Weber today?",
    ],
)
def test_mentioning_a_language_is_not_a_request(text: str) -> None:
    assert language_request(text) is None


def test_final_keeps_no_caller_words() -> None:
    # Only labels are kept (logging privacy): no transcript text in the record.
    f = final(EN, "en")
    assert EN not in repr(f) and "appointment" not in repr(f)


def test_each_final_is_logged_with_labels_only(
    caplog: pytest.LogCaptureFixture, conn: object
) -> None:
    # For the voice test: Deepgram's tag, our detector, and the call's language per final,
    # never the caller's words (logging privacy).
    import json
    import logging

    from praxis_telefon.agent.receptionist import Receptionist
    from praxis_telefon.agent.tool_service import ToolService
    from tests.conftest import NOW

    agent = Receptionist(ToolService(conn, lambda: NOW))  # type: ignore[arg-type]
    with caplog.at_level(logging.INFO, logger="praxis_telefon.language"):
        agent.record_final(DE, "de")
        agent.record_final(EN, "en")
        agent.record_final(EN2, "en")
    records = [json.loads(r.getMessage()) for r in caplog.records]
    assert [r["call_language"] for r in records] == ["de", "de", "en"]
    assert records[1] == {
        "event": "caller_final", "stt": "en", "detected": "en", "request": None,
        "call_language": "de",
    }  # fmt: skip
    assert all(word not in caplog.text for word in ("Termin", "appointment", "Weber"))
