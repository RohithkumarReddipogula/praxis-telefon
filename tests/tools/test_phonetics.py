"""Kölner Phonetik: German names that sound alike get the same code.

Why (roadmap item 4, 2026-11-09 logs): speech recognition mishears about 1 in 3 German
names ("Östürk" for Öztürk), so exact name matching fails although the date of birth is
right. The code is the standard Kölner Phonetik; these are its published examples plus
names from the STT A/B.
"""

from __future__ import annotations

import pytest

from praxis_telefon.tools.phonetics import koelner_phonetik, name_codes, sounds_alike


@pytest.mark.parametrize(
    ("word", "code"),
    [
        ("Müller-Lüdenscheidt", "65752682"),  # the standard example
        ("Wikipedia", "3412"),
        ("Breschnew", "17863"),
        ("Meyer", "67"),
        ("Mayer", "67"),
        ("Maier", "67"),
        ("Meier", "67"),
        ("Mustermann", "682766"),
    ],
)
def test_published_examples(word: str, code: str) -> None:
    assert koelner_phonetik(word) == code


def test_leading_vowel_keeps_its_zero_and_inner_vowels_vanish() -> None:
    assert koelner_phonetik("Anna") == "06"
    assert koelner_phonetik("Erika") == "074"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Östürk", "Öztürk"),  # S and Z sound alike: the misrecognition from the A/B
        ("Müller", "Mueller"),
        ("Müller", "Mühler"),
        ("Schmidt", "Schmitt"),
        ("Hoffmann", "Hofmann"),
        ("Krüger", "Krueger"),
        ("Meyer", "Maier"),
        ("Katharina", "Catarina"),
        ("Stefan", "Stephan"),
        ("Sabine", "Zabine"),
    ],
)
def test_sound_alike_names_match(a: str, b: str) -> None:
    assert koelner_phonetik(a) == koelner_phonetik(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Hoffmann", "Becker"),
        ("Müller", "Schulte"),
        ("Demir", "Richter"),
        ("Becker", "Beckmann"),
    ],
)
def test_different_names_do_not_match(a: str, b: str) -> None:
    assert koelner_phonetik(a) != koelner_phonetik(b)


def test_case_spacing_hyphens_and_accents_do_not_matter() -> None:
    assert koelner_phonetik("MÜLLER") == koelner_phonetik("müller") == koelner_phonetik(" Müller ")
    assert koelner_phonetik("ß") == koelner_phonetik("ss")


def test_empty_and_symbol_only_words_have_no_code() -> None:
    assert koelner_phonetik("") == ""
    assert koelner_phonetik("-- 123") == ""
    assert koelner_phonetik("h") == ""  # H alone is silent


def test_a_full_name_is_first_and_last_name_codes() -> None:
    assert name_codes("Erika Mustermann") == (
        koelner_phonetik("Erika"),
        koelner_phonetik("Mustermann"),
    )
    # middle names do not count
    assert name_codes("Anna Maria Müller") == name_codes("Anna Müller")


def test_a_name_without_both_codes_never_sounds_alike() -> None:
    assert name_codes("Erika") is None
    assert name_codes("H h") is None
    assert not sounds_alike("Erika", "Erika")  # one token is not a full name


def test_sounds_alike_needs_both_parts() -> None:
    assert sounds_alike("Yusuf Öztürk", "Jusuf Östürk")
    assert not sounds_alike("Yusuf Öztürk", "Yusuf Demir")  # same first name only
    assert not sounds_alike("Yusuf Öztürk", "Anja Öztürk")  # same last name only
