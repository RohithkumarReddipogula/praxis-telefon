"""Static checks on the prompt text. They do not test model behaviour (see tests/live)."""

import pytest

from praxis_telefon.agent.prompts import GREETING_DE, SYSTEM_PROMPT


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
