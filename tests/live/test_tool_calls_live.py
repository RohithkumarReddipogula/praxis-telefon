"""Live tool-calling regression tests. Skipped by default; run with:

    uv run pytest -m live tests/live/test_tool_calls_live.py -v

They drive the real Receptionist through LiveKit's text-only test mode (no audio) with the
real LLM from .env, against an in-memory synthetic database with a fixed clock
(Monday 9 November 2026, 07:00 Berlin). Each scenario runs RUNS times, because one pass
proves little for an LLM. On failure the transcript is printed (synthetic data only).

The scripted caller answers what the agent asks: a slot choice when slots are offered,
"Ja" to a confirmation, its name and date of birth when asked (counted, see below).

Bugs from the voice tests on 2026-11-06:
- Call 1: the caller confirmed a move and a cancel, but move_booking / cancel_booking
  were never called; the database did not change.
- Call 2: a booking request ran find_slots twice, find_booking, then take_message twice
  (two identical staff messages) instead of book.
From the first live run (14 passed, 4 failed):
- move[2]: the agent asked for the full name five times although it was given.
- move[1]: an extra name confirmation; the scripted slot choice came at the wrong moment.
- book_move_cancel[0], says_existing_patient[2]: read-back questions flagged as claims.
"""

from __future__ import annotations

import pytest

from tests.live.harness import (
    ERIKA,
    MAX,
    PHONE_SPOKEN,
    RUNS,
    Step,
    booking_row,
    converse,
    count,
    new_db,
    ok,
    transcript_of,
)

pytestmark = pytest.mark.live


BOOKING_OPENINGS = {
    "new_patient": f"Guten Tag, hier ist {MAX}. Ich hätte gern einen Termin am Dienstag "
    "vormittags.",
    # In the voice test find_booking ran first (no match), then take_message.
    "says_existing_patient": f"Guten Tag, hier ist {MAX}. Ich bin schon Patient bei Ihnen "
    "und bräuchte einen Termin am Dienstag vormittags.",
}


# --- Call 1: move / cancel confirmed but never executed --------------------------------


@pytest.mark.parametrize("run", range(RUNS))
def test_move_after_yes_calls_move_booking(run: int) -> None:
    conn, booking_id = new_db(erika_booked=True)
    assert booking_id is not None
    old_slot = int(booking_row(conn, booking_id)["slot_id"])

    calls = converse(
        conn,
        ERIKA,
        [
            Step(
                (
                    f"Guten Tag, hier ist {ERIKA}. Ich möchte meinen Termin am Dienstag "
                    "verschieben, auf Mittwoch, den elften November, vormittags.",
                ),
                until=ok("move_booking"),
            )
        ],
    )

    info = transcript_of(calls)
    assert calls[0].succeeded("move_booking") == 1, info
    assert int(booking_row(conn, booking_id)["slot_id"]) != old_slot, info
    assert calls[0].unbacked_claims() == [], info
    assert calls[0].identity_asked == 0, "asked again for name/date of birth\n" + info


@pytest.mark.parametrize("run", range(RUNS))
def test_cancel_after_yes_calls_cancel_booking(run: int) -> None:
    conn, booking_id = new_db(erika_booked=True)
    assert booking_id is not None

    calls = converse(
        conn,
        ERIKA,
        [
            Step(
                (f"Guten Tag, hier ist {ERIKA}. Ich möchte meinen Termin am Dienstag absagen.",),
                until=ok("cancel_booking"),
            )
        ],
    )

    info = transcript_of(calls)
    assert calls[0].succeeded("cancel_booking") == 1, info
    assert booking_row(conn, booking_id)["status"] == "cancelled", info
    assert calls[0].unbacked_claims() == [], info
    assert calls[0].identity_asked == 0, "asked again for name/date of birth\n" + info


@pytest.mark.parametrize("run", range(RUNS))
def test_book_move_cancel_in_one_call_no_false_success_claim(run: int) -> None:
    """The voice-test flow: book, then move, then cancel, all in one call."""
    conn, _ = new_db(erika_booked=False)

    calls = converse(
        conn,
        MAX,
        [
            Step(
                (BOOKING_OPENINGS["new_patient"],),
                until=ok("book"),
            ),
            Step(
                (
                    "Ach, Dienstag passt doch nicht. Können Sie den Termin auf Mittwoch "
                    "vormittags verschieben?",
                ),
                until=ok("move_booking"),
            ),
            Step(
                ("Wissen Sie was, sagen Sie den Termin bitte ganz ab.",),
                until=ok("cancel_booking"),
            ),
        ],
    )

    info = transcript_of(calls)
    book, move, cancel = calls
    assert book.succeeded("book") == 1, info
    assert move.succeeded("move_booking") == 1, info
    assert cancel.succeeded("cancel_booking") == 1, info
    for step in calls:
        # Fresh tracker per step: an earlier book must not excuse a later "verschoben".
        assert step.unbacked_claims() == [], info
        assert step.identity_asked == 0, "asked again for name/date of birth\n" + info
    assert count(conn, "SELECT count(*) FROM bookings WHERE status = 'active'") == 0, info


# --- Call 2: booking request ended in take_message, twice -------------------------------


@pytest.mark.parametrize("run", range(RUNS))
@pytest.mark.parametrize("opening", BOOKING_OPENINGS)
def test_booking_request_books_instead_of_take_message(opening: str, run: int) -> None:
    conn, _ = new_db(erika_booked=False)

    calls = converse(conn, MAX, [Step((BOOKING_OPENINGS[opening],), until=ok("book"))])

    info = transcript_of(calls)
    assert calls[0].calls("take_message") == 0, info
    assert count(conn, "SELECT count(*) FROM staff_messages") == 0, info
    assert calls[0].succeeded("book") == 1, info
    assert calls[0].unbacked_claims() == [], info
    assert calls[0].identity_asked == 0, "asked again for name/date of birth\n" + info


@pytest.mark.parametrize("run", range(RUNS))
def test_take_message_saves_one_message(run: int) -> None:
    """A real callback request stores exactly one message, even after the goodbye."""
    conn, _ = new_db(erika_booked=False)

    calls = converse(
        conn,
        MAX,
        [
            Step(
                (
                    f"Guten Tag, hier ist {MAX}. Ich brauche ein neues Rezept, "
                    f"bitte rufen Sie mich zurück unter {PHONE_SPOKEN}.",
                ),
                until=ok("take_message"),
            ),
            Step(("Danke, das war alles. Auf Wiederhören.",)),
        ],
    )

    info = transcript_of(calls)
    assert sum(c.succeeded("take_message") for c in calls) == 1, info
    assert count(conn, "SELECT count(*) FROM staff_messages") == 1, info
