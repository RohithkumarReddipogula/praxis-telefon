"""Phrase groups for emergencies, suicide, requests for a human, and short answers.

Plain regular expressions over normalised text, German and English. No LLM decides
these. The emergency and suicide groups over-trigger on purpose: negation is not
handled ("keine Brustschmerzen" matches too), because "kann nicht atmen" contains a
negation and must still match. A false alarm is safer than a miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Language = Literal["de", "en"]
CrisisKind = Literal["suicide", "emergency"]

_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "\u2019": "", "'": ""})
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, fold umlauts and ß, drop apostrophes, keep only letters and digits."""
    folded = text.casefold().translate(_FOLD)
    return " " + _NON_WORD.sub(" ", folded).strip() + " "


def _compile(patterns: list[str]) -> re.Pattern[str]:
    # Patterns are matched on normalize() output, which has a space at both ends,
    # so " word" anchors a word start and "word " a word end.
    return re.compile("|".join(f"(?:{p})" for p in patterns))


# Up to N filler words between two parts of a phrase.
def _gap(n: int) -> str:
    return rf"(?:\w+ ){{0,{n}}}"


_EMERGENCY_DE = _compile(
    [
        # chest pain / heart
        r"brust ?schmerz",
        rf" schmerz\w* {_gap(4)}(?:in|an) (?:der |die |meiner )?{_gap(1)}brust",
        rf" (?:druck|engegefuehl|enge) {_gap(3)}brust",
        r"herzinfarkt",
        r"herzstillstand",
        # breathing
        r" (?:keine|kaum) luft ",
        r"atemnot",
        rf" (?:kann|koennen|kriege?|bekomm\w*) {_gap(3)}nicht (?:mehr )?(?:richtig )?atmen",
        r" atmet (?:nicht|kaum)",
        r"erstick",
        # stroke
        r"schlaganfall",
        r"gelaehmt",
        r"laehmung",
        rf" gesicht {_gap(3)}(?:haengt|schief)",
        rf" kann {_gap(3)}nicht mehr (?:richtig )?sprechen",
        rf" (?:spricht|sprechen|sprache) {_gap(2)}(?:verwaschen|undeutlich|lallend)",
        r"sprachstoerung",
        # unconscious
        r"bewusstlos",
        r"ohnmaechtig",
        r" nicht (?:mehr )?ansprechbar",
        rf" reagiert {_gap(2)}nicht mehr",
        r"krampfanfall",
        # heavy bleeding
        r" (?:starke|starker|starken|heftige|heftiger) blutung",
        rf" blutet {_gap(2)}(?:stark|heftig|sehr|ununterbrochen)",
        rf" hoert {_gap(2)}nicht auf zu bluten",
        r"allergisch\w* schock",
        r"anaphyla",
    ]
)

_EMERGENCY_EN = _compile(
    [
        r" chest pain",
        rf" pain {_gap(3)}(?:in|on) (?:my |the |his |her )?chest",
        rf" (?:pressure|tightness) {_gap(3)}chest",
        r" heart attack",
        rf" (?:cant|cannot|can not|unable to|couldnt) {_gap(1)}breathe",
        r" (?:trouble|difficulty|problems?) breathing",
        r" short(?:ness)? of breath",
        r" not breathing",
        r" stroke ",
        rf" face {_gap(2)}(?:drooping|droops|drooped)",
        r" slurred speech",
        r" slurring",
        r" paraly",
        r" unconscious",
        r" passed out",
        r" not responding",
        r" fainted",
        r" seizure",
        r" bleeding (?:heavily|a lot|badly|wont stop)",
        r" heavy bleeding",
        r" wont stop bleeding",
        r" anaphyla",
    ]
)

_SUICIDE_DE = _compile(
    [
        r"suizid",
        r"selbstmord",
        rf" (?:mich|mir) {_gap(2)}umbringen",
        rf" (?:mir|mich) {_gap(2)}das leben nehmen",
        rf" (?:mir|mich) {_gap(1)}(?:etwas|was) antu",
        rf" (?:will|moechte|mag) {_gap(2)}nicht mehr leben",
        rf" (?:will|moechte) {_gap(2)}sterben",
        r"lebensmuede",
        rf" (?:mein|das) leben {_gap(2)}(?:beenden|ein ende setzen)",
    ]
)

_SUICIDE_EN = _compile(
    [
        r"suicid",
        r" kill myself",
        r" end my (?:own )?life",
        r" take my (?:own )?life",
        rf" want {_gap(1)}to die",
        r" dont want to (?:live|be alive)",
        r" hurt myself",
    ]
)

_HUMAN_DE = _compile(
    [
        rf" mit {_gap(1)}(?:einem|einer) (?:echten |richtigen |realen )?(?:menschen|person)",
        r" (?:echte|echten|richtige|richtigen|reale|realen) (?:person|mensch)",
        r" mitarbeiter",
        r" sprechstundenhilfe",
        r" arzthelfer",
        r" rezeption",
        rf" mit {_gap(1)}(?:dem |der )?praxisteam {_gap(1)}(?:sprechen|reden)",
        r" (?:verbinden|durchstellen)",
        r" kein\w* (?:roboter|computer|maschine|bot|ki) ",
        rf" nicht mit {_gap(1)}(?:einem |einer )?(?:roboter|computer|maschine|bot|ki) ",
    ]
)

_HUMAN_EN = _compile(
    [
        r" (?:real|actual|live) (?:person|human)",
        r" (?:a|to a) human",
        r" (?:speak|talk) (?:to|with) (?:someone|somebody|a person|staff|reception)",
        r" receptionist",
        r" transfer me",
        r" (?:not|no) (?:a )?(?:robot|bot|machine) ",
    ]
)


@dataclass(frozen=True)
class Crisis:
    kind: CrisisKind
    language: Language


def detect_crisis(text: str) -> Crisis | None:
    """Suicide wins over a general emergency: its message names 112 as well."""
    norm = normalize(text)
    for kind, de, en in (
        ("suicide", _SUICIDE_DE, _SUICIDE_EN),
        ("emergency", _EMERGENCY_DE, _EMERGENCY_EN),
    ):
        if de.search(norm):
            return Crisis(kind, "de")  # type: ignore[arg-type]
        if en.search(norm):
            return Crisis(kind, "en")  # type: ignore[arg-type]
    return None


def asks_for_human(text: str) -> Language | None:
    norm = normalize(text)
    if _HUMAN_DE.search(norm):
        return "de"
    if _HUMAN_EN.search(norm):
        return "en"
    return None


# --- short answers: never a strike ------------------------------------------------------

_SHORT_WORDS = frozenset(
    {
        # German
        *("ja", "nein", "ne", "nee", "noe", "genau", "okay", "ok", "okey", "richtig"),
        *("stimmt", "korrekt", "jawohl", "gut", "passt", "gerne", "bitte", "danke"),
        *("doch", "klar", "sicher", "mhm", "aha", "der", "die", "das", "den", "dem"),
        # English
        *("yes", "yeah", "yep", "yup", "no", "nope", "right", "correct", "exactly"),
        *("sure", "fine", "please", "thanks", "alright", "the", "a"),
    }
)
_NUMBER_WORD = re.compile(
    r"^(?:null|eins?|eine[mnrs]?|zwei|zwo|drei|vier|fuenf|sechs|sieben|acht|neun|zehn|elf|"
    r"zwoelf|zwanzig|dreissig|vierzig|fuenfzig|sechzig|siebzig|achtzig|neunzig|hundert|"
    r"tausend|und|zehnte[nr]?|ste[nr]?|te[nr]?|erste[nr]?|dritte[nr]?|siebte[nr]?|achte[nr]?)+$"
    r"|^(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|first|second|third|fourth|fifth)$"
)
_DIGITS = re.compile(r"^[0-9]+$")
_FILLERS = frozenset(["aeh", "aehm", "oeh", "oehm", "hm", "hmm", "uh", "um", "erm"])
_NAME_TOKEN = re.compile(r"^[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?$")
MAX_SHORT_ANSWER_WORDS = 4
MAX_NUMBER_WORDS = 16


def is_short_answer(text: str) -> bool:
    """ "Ja", "Nein", "Genau", a number or a name (and English equivalents).

    Such answers are valid even with low STT confidence and never count as a strike.
    """
    words = normalize(text).split()
    if not words or any(w in _FILLERS for w in words):
        return False
    # A number, e.g. digits of a phone number or date of birth, may be long.
    if all(_DIGITS.match(w) or _NUMBER_WORD.match(w) for w in words):
        return len(words) <= MAX_NUMBER_WORDS
    if len(words) > MAX_SHORT_ANSWER_WORDS:
        return False
    if all(w in _SHORT_WORDS or _DIGITS.match(w) or _NUMBER_WORD.match(w) for w in words):
        return True
    # A name: 1-3 capitalised words, e.g. "Mustermann" or "Erika Mustermann".
    raw = re.sub(r"[.,!?;:]", " ", text).split()
    return 1 <= len(raw) <= 3 and all(_NAME_TOKEN.match(w) for w in raw)
