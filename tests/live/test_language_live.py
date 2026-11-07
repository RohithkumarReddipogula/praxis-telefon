"""Live language tests (M3 Part B, step 2b). Skipped by default; run with:

    uv run pytest -m live tests/live/test_language_live.py -v

English calls get English answers, without spoken reasoning; a German call stays German
after one English-looking line and switches at once on an explicit request. The call's
language is decided in code (agent/call_language.py), not by the LLM. The scripted
caller keeps its own language (``caller_language``): the older English date tests passed
although the agent answered in German, because the caller followed the agent.
Text mode: no STT tags, so finals count by detect_language alone.
"""

from __future__ import annotations

import pytest

from praxis_telefon.agent.speech_filters import detect_language
from tests.live.harness import (
    MAX,
    RUNS,
    Call,
    Step,
    agent_texts,
    converse,
    count,
    new_db,
    ok,
    transcript_of,
)

pytestmark = pytest.mark.live

MAX_EN = "Max Mustermann, born on the first of February nineteen seventy-nine"
ERIKA_EN = "Erika Mustermann, born on the fifteenth of June nineteen eighty-five"
PHONE_EN = "oh three oh, two three one two five, four five six"


def german(call: Call) -> list[str]:
    return [t for t in agent_texts(call) if detect_language(t) == "de"]


def english(call: Call) -> list[str]:
    return [t for t in agent_texts(call) if detect_language(t) == "en"]


def blocked(calls: list[Call]) -> list[str]:
    return [note for c in calls for _, note in c.filter_notes]


@pytest.mark.parametrize("run", range(RUNS))
def test_english_call_books_in_english(run: int) -> None:
    conn, _ = new_db(erika_booked=False)
    line = f"Hi, this is {MAX_EN}. I'd like an appointment next Tuesday morning, please."
    calls = converse(conn, MAX_EN, [Step((line,), ok("book"))], caller_language="en")

    info = transcript_of(calls) + f"\nblocked: {blocked(calls)}"
    assert calls[0].succeeded("book") == 1, info
    assert count(conn, "SELECT count(*) FROM bookings") == 1, info
    assert german(calls[0]) == [], info


@pytest.mark.parametrize("run", range(RUNS))
def test_english_call_finds_the_booking(run: int) -> None:
    conn, _ = new_db(erika_booked=True)  # Erika: Tuesday 10 November, 9:00, Dr. Weber
    line = f"Hello, this is {ERIKA_EN}. Do I have an appointment on Tuesday?"
    calls = converse(conn, ERIKA_EN, [Step((line,), ok("find_booking"))], caller_language="en")

    info = transcript_of(calls) + f"\nblocked: {blocked(calls)}"
    assert calls[0].succeeded("find_booking") >= 1, info
    assert any("Tuesday" in t for t in agent_texts(calls[0])), info
    assert german(calls[0]) == [], info


@pytest.mark.parametrize("run", range(RUNS))
def test_english_call_cancels(run: int) -> None:
    conn, _ = new_db(erika_booked=True)
    line = f"Hello, this is {ERIKA_EN}. I need to cancel my appointment on Tuesday."
    calls = converse(conn, ERIKA_EN, [Step((line,), ok("cancel_booking"))], caller_language="en")

    info = transcript_of(calls) + f"\nblocked: {blocked(calls)}"
    assert calls[0].succeeded("cancel_booking") == 1, info
    assert german(calls[0]) == [], info


@pytest.mark.parametrize("run", range(RUNS))
def test_english_call_moves(run: int) -> None:
    conn, _ = new_db(erika_booked=True)
    line = (
        f"Hello, this is {ERIKA_EN}. Could I move my appointment on Tuesday to Wednesday morning?"
    )
    calls = converse(conn, ERIKA_EN, [Step((line,), ok("move_booking"))], caller_language="en")

    info = transcript_of(calls) + f"\nblocked: {blocked(calls)}"
    assert calls[0].succeeded("move_booking") == 1, info
    assert german(calls[0]) == [], info


@pytest.mark.parametrize("run", range(RUNS))
def test_english_call_leaves_a_message(run: int) -> None:
    conn, _ = new_db(erika_booked=False)
    line = (
        f"Hello, this is {MAX_EN}. Could Dr. Weber call me back about my lab results? "
        f"My number is {PHONE_EN}."
    )
    calls = converse(conn, MAX_EN, [Step((line,), ok("take_message"))], caller_language="en")

    info = transcript_of(calls) + f"\nblocked: {blocked(calls)}"
    assert calls[0].succeeded("take_message") == 1, info
    number = conn.execute("SELECT callback_number FROM staff_messages").fetchone()[0]
    assert number.replace(" ", "") == "03023125456", info
    assert german(calls[0]) == [], info


@pytest.mark.parametrize("run", range(RUNS))
def test_explicit_request_switches_a_german_call_to_english(run: int) -> None:
    conn, _ = new_db(erika_booked=False)
    calls = converse(
        conn,
        MAX_EN,
        [
            Step((f"Guten Tag, hier ist {MAX}. Ich hätte gern einen Termin.",)),
            Step(("Können wir Englisch sprechen?",)),
        ],
    )

    info = transcript_of(calls) + f"\nblocked: {blocked(calls)}"
    assert english(calls[0]) == [], info
    assert english(calls[1]) and german(calls[1]) == [], info


@pytest.mark.parametrize("run", range(RUNS))
def test_one_english_line_keeps_a_german_call_german(run: int) -> None:
    # Voice test 2026-11-07: after one German final heard as English the LLM switched.
    conn, _ = new_db(erika_booked=False)
    calls = converse(
        conn,
        MAX,
        [
            Step((f"Guten Tag, hier ist {MAX}. Ich hätte gern morgen Vormittag einen Termin.",)),
            Step(("Yes, that is right, thank you.",)),
        ],
    )

    info = transcript_of(calls) + f"\nblocked: {blocked(calls)}"
    assert english(calls[1]) == [], info
    assert german(calls[1]), info
