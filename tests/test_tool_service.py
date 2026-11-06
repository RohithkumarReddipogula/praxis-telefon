"""Per-call guards, LLM-facing text, and privacy of the tool layer."""

import io
import json
import logging
import sqlite3
from collections.abc import Callable, Iterator

import pytest
from livekit.agents import ToolError
from livekit.agents.cli.log import JsonFormatter

from praxis_telefon.agent.receptionist import _run
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.logging_privacy import RedactPiiFilter
from praxis_telefon.tools import errors
from tests.conftest import NOW, PHONE

# Unique synthetic values so a leak is easy to spot.
PII_NAME = "Wilhelmina Quellenberg"
PII_DOB = "1971-04-23"
PII_PHONE = PHONE
PII_VALUES = (PII_NAME, "Quellenberg", PII_DOB, "1971", "23125", "456")


@pytest.fixture
def service(conn: sqlite3.Connection) -> ToolService:
    return ToolService(conn, lambda: NOW)


@pytest.fixture
def captured_logs() -> Iterator[io.StringIO]:
    """Everything logged at DEBUG, formatted by LiveKit's JSON formatter (prints extras)."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactPiiFilter())
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield stream
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


def _offer_first(service: ToolService) -> int:
    text = service.find_slots("2026-11-09", "morning", None)
    return int(text.split("slot_id=")[1].split(":")[0])


def _book(service: ToolService) -> int:
    slot_id = _offer_first(service)
    text = service.book(slot_id, PII_NAME, PII_DOB)
    return int(text.split("booking_id=")[1].split(",")[0])


# --- guards ----------------------------------------------------------------------------


def test_book_rejects_slot_not_offered_in_this_call(service: ToolService) -> None:
    # The LLM cannot book a slot id it invented or remembered from elsewhere.
    with pytest.raises(errors.SlotNotOffered):
        service.book(1, PII_NAME, PII_DOB)


def test_book_accepts_offered_slot(service: ToolService) -> None:
    assert service.book(_offer_first(service), PII_NAME, PII_DOB).startswith("Gebucht")


def test_cancel_rejects_unverified_booking(conn: sqlite3.Connection, service: ToolService) -> None:
    booking_id = _book(service)
    fresh_call = ToolService(conn, lambda: NOW)
    with pytest.raises(errors.BookingNotVerified):
        fresh_call.cancel(booking_id)


def test_find_booking_verifies_for_cancel(conn: sqlite3.Connection, service: ToolService) -> None:
    booking_id = _book(service)
    fresh_call = ToolService(conn, lambda: NOW)
    assert f"booking_id={booking_id}" in fresh_call.find_booking(PII_NAME, PII_DOB)
    assert fresh_call.cancel(booking_id).startswith("Abgesagt")


def test_move_needs_verified_booking_and_offered_slot(
    conn: sqlite3.Connection, service: ToolService
) -> None:
    booking_id = _book(service)
    fresh_call = ToolService(conn, lambda: NOW)
    with pytest.raises(errors.BookingNotVerified):
        fresh_call.move(booking_id, 2)
    fresh_call.find_booking(PII_NAME, PII_DOB)
    with pytest.raises(errors.SlotNotOffered):
        fresh_call.move(booking_id, 999)
    new_slot = int(
        fresh_call.find_slots("2026-11-10", "any", None).split("slot_id=")[1].split(":")[0]
    )
    assert fresh_call.move(booking_id, new_slot).startswith("Verschoben")


def test_wrong_birthday_does_not_verify(conn: sqlite3.Connection, service: ToolService) -> None:
    booking_id = _book(service)
    fresh_call = ToolService(conn, lambda: NOW)
    assert "Kein Termin gefunden" in fresh_call.find_booking(PII_NAME, "1971-04-24")
    with pytest.raises(errors.BookingNotVerified):
        fresh_call.cancel(booking_id)


@pytest.mark.parametrize("bad", ["23.04.1971", "1971-13-01", "gestern"])
def test_unparseable_birthday(service: ToolService, bad: str) -> None:
    with pytest.raises(errors.InvalidDateOfBirth):
        service.find_booking(PII_NAME, bad)


def test_unparseable_date(service: ToolService) -> None:
    with pytest.raises(errors.DateNotUnderstood):
        service.find_slots("irgendwann mal", "any", None)


def test_no_slots_message(service: ToolService) -> None:
    # Wednesday has no afternoon hours.
    text = service.find_slots("2026-11-11", "afternoon", None)
    assert "Keine freien Termine" in text
    assert text.startswith("Gesucht: Mittwoch, 11. November.")


def test_find_slots_text_lists_only_returned_slots(service: ToolService) -> None:
    text = service.find_slots("2026-11-09 bis 2026-11-13", "any", None)
    assert text.count("slot_id=") == 3
    assert len(service.guards.offered_slot_ids) == 3


def test_take_message_text(service: ToolService) -> None:
    assert "Rezept" in service.take_message(PII_NAME, PII_PHONE, "prescription")


def _message_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT count(*) FROM staff_messages").fetchone()[0])


def test_take_message_twice_in_one_call_saves_one_message(
    conn: sqlite3.Connection, service: ToolService
) -> None:
    # Regression (voice test 2026-11-06): two identical staff messages from one call.
    service.take_message(PII_NAME, PII_PHONE, "appointment")
    # Same note, written differently: case, spacing, number separators.
    text = service.take_message("wilhelmina  quellenberg", "030-23125-456", "appointment")
    assert _message_count(conn) == 1
    assert "bereits gespeichert" in text


def test_take_message_different_category_is_a_new_message(
    conn: sqlite3.Connection, service: ToolService
) -> None:
    service.take_message(PII_NAME, PII_PHONE, "prescription")
    service.take_message(PII_NAME, PII_PHONE, "referral")
    assert _message_count(conn) == 2


def test_take_message_duplicate_is_per_call(conn: sqlite3.Connection) -> None:
    ToolService(conn, lambda: NOW).take_message(PII_NAME, PII_PHONE, "callback")
    ToolService(conn, lambda: NOW).take_message(PII_NAME, PII_PHONE, "callback")
    assert _message_count(conn) == 2


def test_duplicate_message_is_logged_as_duplicate(
    service: ToolService, captured_logs: io.StringIO
) -> None:
    service.take_message(PII_NAME, PII_PHONE, "callback")
    service.take_message(PII_NAME, PII_PHONE, "callback")
    assert '\\"outcome\\":\\"duplicate\\"' in captured_logs.getvalue()
    _assert_no_pii(captured_logs.getvalue())


# --- privacy ---------------------------------------------------------------------------


def _assert_no_pii(text: str) -> None:
    for value in PII_VALUES:
        assert value not in text, f"leaked {value!r}"


def test_tool_logs_contain_no_arguments(service: ToolService, captured_logs: io.StringIO) -> None:
    booking_id = _book(service)
    service.find_booking(PII_NAME, PII_DOB)
    service.cancel(booking_id)
    service.take_message(PII_NAME, PII_PHONE, "callback")
    with pytest.raises(errors.InvalidName):
        service.take_message("Quellenberg", PII_PHONE, "callback")
    with pytest.raises(errors.InvalidCallbackNumber):
        service.take_message(PII_NAME, "030 12", "callback")

    output = captured_logs.getvalue()
    _assert_no_pii(output)
    events = [json.loads(line)["message"] for line in output.splitlines() if '"event' in line]
    tools = [json.loads(e)["tool"] for e in events]
    assert tools.count("take_message") == 3 and "book" in tools


def test_tool_log_records_outcome_code(service: ToolService, captured_logs: io.StringIO) -> None:
    with pytest.raises(errors.SlotNotOffered):
        service.book(1, PII_NAME, PII_DOB)
    assert '\\"outcome\\":\\"slot_not_offered\\"' in captured_logs.getvalue()


def test_livekit_tool_argument_extras_are_redacted(captured_logs: io.StringIO) -> None:
    # LiveKit logs tool arguments under lk.pii.arguments on tool errors.
    logging.getLogger("livekit.agents").error(
        "exception occurred while executing tool",
        extra={"function": "book", "lk.pii.arguments": json.dumps({"patient_name": PII_NAME})},
    )
    output = captured_logs.getvalue()
    _assert_no_pii(output)
    assert "exception occurred while executing tool" in output


def test_failure_messages_never_echo_arguments(service: ToolService) -> None:
    attempts: list[Callable[[], object]] = [
        lambda: service.book(1, PII_NAME, PII_DOB),
        lambda: service.find_booking("Quellenberg", PII_DOB),
        lambda: service.find_booking(PII_NAME, "1971-23-04"),
        lambda: service.take_message(PII_NAME, "0302", "callback"),
        lambda: service.take_message(PII_NAME, PII_PHONE, "Quellenberg 1971"),
        lambda: service.find_slots("2026-11-09", "any", "Dr. Quellenberg"),
    ]
    for attempt in attempts:
        with pytest.raises(errors.ToolFailure) as exc:
            attempt()
        _assert_no_pii(str(exc.value))


def test_tool_error_for_llm_has_code_and_no_arguments() -> None:
    def fail() -> str:
        raise errors.SlotTaken()

    with pytest.raises(ToolError) as exc:
        _run(fail)
    assert str(exc.value).startswith("slot_taken:")
    _assert_no_pii(str(exc.value))


# --- relative dates (M3 Part B) ---------------------------------------------------------


def test_find_slots_resolves_the_callers_words(service: ToolService) -> None:
    text = service.find_slots("nächsten Dienstag", "morning", None)
    assert text.startswith("Gesucht: Dienstag, 10. November.\n")
    assert "Dienstag, 10. November, um 8 Uhr" in text


def test_find_slots_on_friday_for_tomorrow_says_closed(conn: sqlite3.Connection) -> None:
    friday = NOW.replace(day=13, hour=10)
    service = ToolService(conn, lambda: friday)
    text = service.find_slots("morgen", "any", None)
    assert text.startswith(
        "Samstag, 14. November: Praxis geschlossen. Stattdessen gesucht: Montag, 16. November."
    )
    assert "Montag, 16. November, um 8 Uhr" in text


def test_find_slots_range_header(service: ToolService) -> None:
    text = service.find_slots("Ende der Woche", "any", None)
    assert text.startswith("Gesucht: Donnerstag, 12. November bis Freitag, 13. November.")


def test_find_slots_too_far(service: ToolService) -> None:
    with pytest.raises(errors.DateTooFar):
        service.find_slots("in sechs Wochen", "any", None)


def test_date_errors_never_echo_the_phrase(service: ToolService) -> None:
    with pytest.raises(errors.DateNotUnderstood) as exc:
        service.find_slots("wenn Quellenberg Zeit hat", "any", None)
    assert "Quellenberg" not in str(exc.value)
