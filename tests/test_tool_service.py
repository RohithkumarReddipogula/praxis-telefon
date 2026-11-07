"""Per-call guards, LLM-facing text, and privacy of the tool layer."""

import io
import json
import logging
import re
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
# Privacy (M4, 2026-11-07): find_booking only looks up the caller's own name.
SELF = (f"Hier ist {PII_NAME}.",)
PII_DOB_DE = "23.04.1971"  # the same date as spoken/written in German
PII_VALUES = (PII_NAME, "Quellenberg", PII_DOB, PII_DOB_DE, "1971", "23125", "456")


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
    assert f"booking_id={booking_id}" in fresh_call.find_booking(
        PII_NAME, PII_DOB, caller_texts=SELF
    )
    assert fresh_call.cancel(booking_id).startswith("Abgesagt")


def test_move_needs_verified_booking_and_offered_slot(
    conn: sqlite3.Connection, service: ToolService
) -> None:
    booking_id = _book(service)
    fresh_call = ToolService(conn, lambda: NOW)
    with pytest.raises(errors.BookingNotVerified):
        fresh_call.move(booking_id, 2)
    fresh_call.find_booking(PII_NAME, PII_DOB, caller_texts=SELF)
    with pytest.raises(errors.SlotNotOffered):
        fresh_call.move(booking_id, 999)
    new_slot = int(
        fresh_call.find_slots("2026-11-10", "any", None).split("slot_id=")[1].split(":")[0]
    )
    assert fresh_call.move(booking_id, new_slot).startswith("Verschoben")


def test_wrong_birthday_does_not_verify(conn: sqlite3.Connection, service: ToolService) -> None:
    booking_id = _book(service)
    fresh_call = ToolService(conn, lambda: NOW)
    assert "Kein Termin gefunden" in fresh_call.find_booking(
        PII_NAME, "1971-04-24", caller_texts=SELF
    )
    with pytest.raises(errors.BookingNotVerified):
        fresh_call.cancel(booking_id)


@pytest.mark.parametrize("bad", ["23.04.1971", "1971-13-01", "gestern"])
def test_unparseable_birthday(service: ToolService, bad: str) -> None:
    with pytest.raises(errors.InvalidDateOfBirth):
        service.find_booking(PII_NAME, bad, caller_texts=SELF)


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


def test_second_reason_for_the_same_caller_updates_the_note(
    conn: sqlite3.Connection, service: ToolService
) -> None:
    # Regression (voice test 2026-11-07, call 2): take_message "callback", then 45 s later
    # "prescription" for the same name and number: two notes for one request. Now one
    # note keeps both reasons (user decision 2026-11-07; replaces "different category is
    # a new message").
    service.take_message(PII_NAME, PII_PHONE, "callback")
    text = service.take_message(PII_NAME, "030-23125-456", "prescription")
    assert _message_count(conn) == 1
    row = conn.execute("SELECT reason_category FROM staff_messages").fetchone()
    assert row["reason_category"] == "callback,prescription"
    assert "Rückruf" in text and "Rezept" in text and "nicht noch einmal" in text.lower()


def test_reasons_are_kept_once_each(conn: sqlite3.Connection, service: ToolService) -> None:
    for reason in ("callback", "prescription", "callback", "referral", "prescription"):
        service.take_message(PII_NAME, PII_PHONE, reason)
    row = conn.execute("SELECT reason_category FROM staff_messages").fetchone()
    assert row["reason_category"] == "callback,prescription,referral"
    assert _message_count(conn) == 1


def test_another_number_is_a_new_message(conn: sqlite3.Connection, service: ToolService) -> None:
    service.take_message(PII_NAME, PII_PHONE, "callback")
    service.take_message(PII_NAME, "040 66969 432", "callback")
    assert _message_count(conn) == 2


def test_updated_message_is_logged_as_updated(
    service: ToolService, captured_logs: io.StringIO
) -> None:
    service.take_message(PII_NAME, PII_PHONE, "callback")
    service.take_message(PII_NAME, PII_PHONE, "prescription")
    assert '\\"outcome\\":\\"updated\\"' in captured_logs.getvalue()
    _assert_no_pii(captured_logs.getvalue())


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


# Fields LiveKit's JSON formatter adds to every line; they never hold tool arguments.
_LOG_METADATA = ("timestamp", "pid", "level", "name")


def _without_metadata(line: str) -> str:
    try:
        record = json.loads(line)
    except ValueError:
        return line
    if not isinstance(record, dict):
        return line
    kept = {k: v for k, v in record.items() if k not in _LOG_METADATA and not k.startswith("code.")}
    return json.dumps(kept, ensure_ascii=False)


def _whole(value: str) -> re.Pattern[str]:
    """The value on its own: not inside a word or a longer number, and not the digits of
    a decimal such as a duration "1.456" (a timestamp's "197186" holds "1971")."""
    return re.compile(rf"(?<!\w)(?<!\d[.,]){re.escape(value)}(?!\w)(?![.,]\d)")


def _assert_no_pii(text: str) -> None:
    searched = "\n".join(_without_metadata(line) for line in text.splitlines())
    for value in (*PII_VALUES, PII_PHONE):
        assert not _whole(value).search(searched), f"leaked {value!r}"


def test_privacy_check_ignores_pii_digits_inside_log_metadata() -> None:
    # Regression (flaky test_tool_logs_contain_no_arguments, 2026-11-06): the synthetic
    # birth year "1971" appeared inside the timestamp "...T21:19:50.197186+00:00".
    # pid, source line and the tool's duration are plain numbers that can hit too.
    record = {
        "message": json.dumps({"event": "tool", "tool": "book", "outcome": "ok", "ms": 1.456}),
        "level": "INFO",
        "pid": 456,
        "code.line.number": 1971,
        "timestamp": "2026-11-06T21:19:50.197186+00:00",
    }
    _assert_no_pii(json.dumps(record))


@pytest.mark.parametrize(
    "leak",
    [
        "Termin für Wilhelmina Quellenberg",
        "geboren 1971",
        "geboren am 1971-04-23",
        # German form: the year alone no longer matches after "04." (decimal rule).
        "geboren am 23.04.1971",
        "Rückruf unter 030 23125 456",
        "Nummer endet auf 456.",
    ],
)
def test_privacy_check_still_finds_real_leaks(leak: str) -> None:
    line = json.dumps({"message": leak, "pid": 1, "timestamp": "2026-11-06T21:19:50.000001"})
    with pytest.raises(AssertionError, match="leaked"):
        _assert_no_pii(line)


def _tool_records(output: str) -> list[dict[str, object]]:
    """Our tool log records only. Other loggers (asyncio, LiveKit) may write lines at any
    time, e.g. during garbage collection, so lines are filtered, not assumed."""
    records = []
    for line in output.splitlines():
        try:
            inner = json.loads(json.loads(line)["message"])
        except (ValueError, KeyError, TypeError):
            continue
        if isinstance(inner, dict) and inner.get("event") == "tool":
            records.append(inner)
    return records


def test_tool_logs_contain_no_arguments(service: ToolService, captured_logs: io.StringIO) -> None:
    booking_id = _book(service)
    service.find_booking(PII_NAME, PII_DOB, caller_texts=SELF)
    service.cancel(booking_id)
    service.take_message(PII_NAME, PII_PHONE, "callback")
    with pytest.raises(errors.InvalidName):
        service.take_message("Quellenberg", PII_PHONE, "callback")
    with pytest.raises(errors.InvalidCallbackNumber):
        service.take_message(PII_NAME, "030 12", "callback")

    output = captured_logs.getvalue()
    _assert_no_pii(output)
    tools = [record["tool"] for record in _tool_records(output)]
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
        lambda: service.find_booking(PII_NAME, "1971-23-04", caller_texts=SELF),
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


def test_spoken_number_duplicate_is_recognised(
    conn: sqlite3.Connection, service: ToolService
) -> None:
    # Regression (live test human_request[1]): spoken digits are the same number.
    service.take_message(PII_NAME, PII_PHONE, "callback")
    spoken = "null drei null, zwei drei eins zwei fünf, vier fünf sechs"
    assert "bereits gespeichert" in service.take_message(PII_NAME, spoken, "callback")
    assert _message_count(conn) == 1
