import sqlite3

import pytest

from praxis_telefon.tools import errors
from praxis_telefon.tools.messages import normalize_callback_number, take_message
from tests.conftest import NAME, NOW, PHONE


def test_message_stored(conn: sqlite3.Connection) -> None:
    result = take_message(
        conn, now=NOW, caller_name=NAME, callback_number=PHONE, reason_category="prescription"
    )
    assert result.category_label == "Rezept"
    row = conn.execute("SELECT * FROM staff_messages WHERE id = ?", (result.message_id,)).fetchone()
    assert (row["caller_name"], row["callback_number"], row["reason_category"], row["status"]) == (
        NAME,
        "03023125456",
        "prescription",
        "open",
    )


def test_message_table_has_no_free_text_column(conn: sqlite3.Connection) -> None:
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(staff_messages)")}
    assert columns == {
        "id",
        "caller_name",
        "callback_number",
        "reason_category",
        "created_at",
        "status",
    }


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("030 23125 456", "03023125456"),
        ("030/23125-456", "03023125456"),
        ("+49 30 23125456", "+493023125456"),
        ("0049 (30) 23125 456", "+493023125456"),
    ],
)
def test_number_normalization(raw: str, normalized: str) -> None:
    assert normalize_callback_number(raw) == normalized


@pytest.mark.parametrize("raw", ["", "12345", "0030 1234", "+1 555 0100", "030 abc", "0" * 20])
def test_invalid_number(conn: sqlite3.Connection, raw: str) -> None:
    with pytest.raises(errors.InvalidCallbackNumber):
        take_message(conn, now=NOW, caller_name=NAME, callback_number=raw, reason_category="other")


@pytest.mark.parametrize("category", ["symptoms", "Rezept", "", "Ich habe Schmerzen"])
def test_invalid_category(conn: sqlite3.Connection, category: str) -> None:
    with pytest.raises(errors.InvalidReasonCategory):
        take_message(
            conn, now=NOW, caller_name=NAME, callback_number=PHONE, reason_category=category
        )
    assert conn.execute("SELECT COUNT(*) FROM staff_messages").fetchone()[0] == 0


def test_invalid_name(conn: sqlite3.Connection) -> None:
    with pytest.raises(errors.InvalidName):
        take_message(
            conn, now=NOW, caller_name="Erika", callback_number=PHONE, reason_category="other"
        )


# --- regression: spoken callback number (live test human_request[1], 2026-11-06) --------
# The LLM passed "null drei null, zwei drei eins zwei fünf, vier fünf sechs", got
# invalid_callback_number and kept asking for the number without saving.


@pytest.mark.parametrize(
    ("spoken", "normalized"),
    [
        ("null drei null, zwei drei eins zwei fünf, vier fünf sechs", "03023125456"),
        ("Null Drei Null Zwei Drei Eins Zwei Fuenf Vier Fuenf Sechs", "03023125456"),
        (
            "null dreißig, dreiundzwanzig, eins fünfundzwanzig, vierhundertsechsundfünfzig",
            "03023125456",
        ),
        ("null drei null zwo drei eins zwo fünf vier fünf sechs", "03023125456"),
        ("null drei null, zwei drei eins zwei fünf, doppel fünf sechs", "030231255 56"),
        ("030 zwei drei eins zwei fünf 456", "03023125456"),
        ("plus vier neun drei null zwei drei eins zwei fünf vier fünf sechs", "+493023125456"),
        ("zero three zero, two three one two five, four five six", "03023125456"),
        ("oh three oh two three one two five four five six", "03023125456"),
        ("zero thirty, twenty three, one twenty five, four fifty six", "03023125456"),
        ("zero three zero two three one two five four double five", "0302312545 5"),
        ("plus four nine three zero two three one two five four five six", "+493023125456"),
    ],
)
def test_spoken_callback_number_is_converted_to_digits(spoken: str, normalized: str) -> None:
    assert normalize_callback_number(spoken) == normalized.replace(" ", "")


def test_take_message_accepts_spoken_number(conn: sqlite3.Connection) -> None:
    result = take_message(
        conn,
        now=NOW,
        caller_name=NAME,
        callback_number="null drei null, zwei drei eins zwei fünf, vier fünf sechs",
        reason_category="callback",
    )
    row = conn.execute(
        "SELECT callback_number FROM staff_messages WHERE id = ?", (result.message_id,)
    ).fetchone()
    assert row["callback_number"] == "03023125456"


@pytest.mark.parametrize(
    "raw", ["null drei null bitte rufen sie an", "zwei drei", "my number is unknown", "null drei"]
)
def test_spoken_number_with_other_words_or_too_short_is_invalid(raw: str) -> None:
    with pytest.raises(errors.InvalidCallbackNumber):
        normalize_callback_number(raw)


def test_invalid_number_message_asks_for_digits() -> None:
    message = errors.InvalidCallbackNumber.message
    assert "Ziffern" in message
