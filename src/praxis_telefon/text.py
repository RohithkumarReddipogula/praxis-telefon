"""Text normalisation shared by the safety phrases and the date parser."""

from __future__ import annotations

import re

_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "\u2019": "", "'": ""})
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, fold umlauts and ß, drop apostrophes, keep only letters and digits.

    The result has one space at each end, so patterns can anchor words with spaces.
    """
    folded = text.casefold().translate(_FOLD)
    return " " + _NON_WORD.sub(" ", folded).strip() + " "
