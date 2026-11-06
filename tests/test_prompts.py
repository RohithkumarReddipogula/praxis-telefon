"""Static checks on the prompt text. They do not test model behaviour (see tests/live)."""

import pytest

from praxis_telefon.agent.prompts import GREETING_DE, SYSTEM_PROMPT

# Prompt with line breaks collapsed, so checks don't depend on wrapping.
FLAT = " ".join(SYSTEM_PROMPT.split())


def test_greeting_discloses_ai() -> None:
    assert "KI-Assistent" in GREETING_DE


def test_greeting_names_practice() -> None:
    assert "Dr. Weber" in GREETING_DE


@pytest.mark.parametrize("number", ["112", "116117"])
def test_prompt_names_emergency_numbers(number: str) -> None:
    assert number in SYSTEM_PROMPT


def test_prompt_forbids_symptom_questions() -> None:
    # Regression: agent asked about swelling/redness for "starke Schmerzen in den Beinen".
    assert "NIEMALS Rückfragen zu Beschwerden" in SYSTEM_PROMPT
    for symptom in ("Schwellung", "Rötung", "seit wann", "wie stark"):
        assert symptom in SYSTEM_PROMPT


def test_prompt_offers_appointment_or_callback() -> None:
    assert "Termin" in SYSTEM_PROMPT
    assert "Rückruf" in SYSTEM_PROMPT


def test_prompt_forbids_medical_advice() -> None:
    assert "Keine Diagnosen" in SYSTEM_PROMPT


def test_prompt_has_correct_and_wrong_example() -> None:
    assert "Falsch:" in SYSTEM_PROMPT
    assert "Richtig:" in SYSTEM_PROMPT


def test_prompt_only_offers_find_slots_results() -> None:
    assert "Biete nur Termine an, die find_slots" in SYSTEM_PROMPT
    assert "Erfinde niemals Uhrzeiten" in SYSTEM_PROMPT


def test_prompt_requires_clear_yes_before_changes() -> None:
    assert "klaren Ja" in SYSTEM_PROMPT
    for tool in ("book", "move_booking", "cancel_booking"):
        assert tool in SYSTEM_PROMPT


def test_prompt_hides_patient_status_and_internal_ids() -> None:
    assert "Verrate nie" in SYSTEM_PROMPT
    assert "slot_id oder booking_id" in SYSTEM_PROMPT


def test_prompt_calls_tool_right_after_yes() -> None:
    # Regression (voice test 2026-11-06): "Ja" to a move/cancel, but no tool call.
    assert "Nach einem klaren Ja rufst du das Werkzeug sofort auf" in FLAT


def test_prompt_claims_success_only_after_tool_success() -> None:
    assert "wenn das Werkzeug Erfolg gemeldet hat" in FLAT
    assert "(ohne move_booking aufzurufen)" in FLAT


def test_prompt_does_not_ask_twice_for_given_details() -> None:
    # Regression (live test move[2]): the full name was asked for five times.
    assert "frage nie zweimal nach demselben" in FLAT
    assert "find_booking ohne Rückfrage" in FLAT


def test_prompt_confirms_once_with_date_of_birth() -> None:
    assert "Bestätige genau einmal" in FLAT
    assert "Namen und Geburtsdatum in einem Satz" in FLAT


def test_prompt_read_back_avoids_claim_words() -> None:
    # Regression (live test false positives): "gebucht auf ..." in a read-back question.
    assert 'nicht "gebucht auf"' in FLAT


def test_prompt_booking_request_is_not_a_message() -> None:
    # Regression (voice test 2026-11-06): a booking request ended in take_message, twice.
    assert "Ein Terminwunsch ist nie ein Fall für take_message" in FLAT
    assert "Rufe take_message pro Anliegen nur einmal auf" in FLAT


def test_instructions_start_with_today() -> None:
    from datetime import datetime

    from praxis_telefon.agent.prompts import build_instructions
    from praxis_telefon.db.schema import PRACTICE_TZ

    text = build_instructions(datetime(2026, 11, 6, 9, 0, tzinfo=PRACTICE_TZ))
    assert text.startswith("Heute ist Freitag, der 6. November 2026")
    assert SYSTEM_PROMPT in text


def test_prompt_passes_callers_date_words_to_find_slots() -> None:
    assert "bei when die Worte des Anrufers" in FLAT
    assert "Rechne Daten nie selbst aus" in FLAT
    assert 'unter "Gesucht"' in FLAT
