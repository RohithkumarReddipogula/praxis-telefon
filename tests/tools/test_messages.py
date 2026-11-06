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
