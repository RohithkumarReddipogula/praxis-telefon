"""An English tool word in a German call is spoken reasoning (meta:tool).

Regression, M4 baseline run 2026-11-07 (injection_role_claim[1] and [3]): "Für book brauche
ich noch ..." and "Für das Booking brauche ich ..." were spoken; the reasoning filter only
knew tool names like find_slots. In English calls "book" is an ordinary verb.
"""

from __future__ import annotations

import pytest

from praxis_telefon.agent.claim_gate import FilterContext, check_sentence


def ctx(callers: tuple[str, ...]) -> FilterContext:
    return FilterContext(caller_texts=callers)


@pytest.mark.parametrize(
    "sentence",
    [
        "Für book brauche ich noch den Vor- und Nachnamen sowie das Geburtsdatum des Patienten.",
        "Für das Booking brauche ich den Namen des Patienten.",
    ],
)
def test_english_tool_word_in_a_german_call_is_meta(sentence: str) -> None:
    assert "meta:tool" in check_sentence(sentence, ctx(("Guten Tag.",)))


def test_book_in_an_english_call_is_fine() -> None:
    c = ctx(("Hello, I would like an appointment, please.",))
    assert check_sentence("I can book that for you right away.", c) == []
