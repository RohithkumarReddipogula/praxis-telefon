"""Kölner Phonetik: a sound code for German names (Postel, 1969).

Speech recognition mishears about 1 in 3 German names ("Östürk" for Öztürk), so patient
lookup compares names by sound next to the exact date of birth. Names that sound alike
get the same digit string; first and last name are compared separately.
"""

from __future__ import annotations

import re

_VOWELS = frozenset("aeijouyäöü")
_MAP = {
    **dict.fromkeys(_VOWELS, "0"),
    "b": "1",
    **dict.fromkeys("fvw", "3"),
    **dict.fromkeys("gkq", "4"),
    "l": "5",
    **dict.fromkeys("mn", "6"),
    "r": "7",
    **dict.fromkeys("sz", "8"),
}


def _normalise(word: str) -> str:
    return re.sub(r"[^a-zäöü]", "", word.casefold().replace("ß", "ss"))


def koelner_phonetik(word: str) -> str:
    """The Kölner Phonetik code of one word; empty if it has no sounding letter."""
    letters = _normalise(word)
    codes: list[str] = []
    for i, ch in enumerate(letters):
        prev = letters[i - 1] if i > 0 else ""
        nxt = letters[i + 1] if i + 1 < len(letters) else ""
        if ch == "h":
            continue  # silent
        if ch in _MAP:
            code = _MAP[ch]
        elif ch == "p":
            code = "3" if nxt == "h" else "1"
        elif ch in "dt":
            code = "8" if nxt in "csz" and nxt else "2"
        elif ch == "x":
            code = "8" if prev in ("c", "k", "q") and prev else "48"
        elif ch == "c":
            if i == 0:
                code = "4" if nxt in "ahkloqrux" and nxt else "8"
            elif prev in ("s", "z"):
                code = "8"
            else:
                code = "4" if nxt in "ahkoqux" and nxt else "8"
        else:  # unreachable: _normalise keeps only the letters handled above
            continue
        codes.extend(code)
    squeezed = [c for j, c in enumerate(codes) if j == 0 or c != codes[j - 1]]
    return "".join(c for j, c in enumerate(squeezed) if c != "0" or j == 0)


def name_codes(full_name: str) -> tuple[str, str] | None:
    """(code of the first name, code of the last name); None unless both have a code.
    Middle names are ignored."""
    parts = full_name.split()
    if len(parts) < 2:
        return None
    first, last = koelner_phonetik(parts[0]), koelner_phonetik(parts[-1])
    return (first, last) if first and last else None


def sounds_alike(a: str, b: str) -> bool:
    """First and last name of both sound alike (a single word is never a full name)."""
    codes = name_codes(a)
    return codes is not None and codes == name_codes(b)
