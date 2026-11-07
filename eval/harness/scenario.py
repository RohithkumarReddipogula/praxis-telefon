"""Scenario files (eval/scenarios/*.yaml): a list of scenarios per group.

Each scenario: id, group, lang, tags, setup (database and clock), caller (lines, a
rule-based behaviour, language, the tool that ends the call), expect (deterministic
checks, eval/harness/checks.py), review (collect the agent's replies for a human).
Placeholders in lines: {max}, {erika}, {max_en}, {erika_en}, {phone_de}, {phone_en}.
"<silence>" as a line: the caller speaks but nothing is recognised (a strike).
Synthetic data only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

IDENTITIES = {
    "max": "Max Mustermann, geboren am ersten Februar neunzehnhundertneunundsiebzig",
    "erika": "Erika Mustermann, geboren am fünfzehnten Juni neunzehnhundertfünfundachtzig",
    "max_en": "Max Mustermann, born on the first of February nineteen seventy-nine",
    "erika_en": "Erika Mustermann, born on the fifteenth of June nineteen eighty-five",
    "moritz": "Moritz Mustermann, geboren am ersten Februar neunzehnhundertneunundsiebzig",
}
PLACEHOLDERS = {
    **IDENTITIES,
    "phone_de": "null drei null, zwei drei eins zwei fünf, vier fünf sechs",
    "phone_en": "oh three oh, two three one two five, four five six",
}
SILENCE = "<silence>"
BEHAVIOURS = frozenset({"cooperative", "change_mind", "confused", "correct_readback"})
CLOCKS = frozenset({"monday_7", "friday_10"})
EXPECTATIONS = frozenset(
    {
        "booking",
        "no_booking",
        "speaks",
        "tools_not_succeeded",
        "moved",
        "cancelled",
        "message",
        "no_message",
        "tools_required",
        "tools_forbidden",
        "practice_topic",
        "safety",
        "never_say",
        "must_say",
        "language",
        "language_from",
        "max_turns",
    }
)


class ScenarioError(ValueError):
    pass


@dataclass(frozen=True)
class Caller:
    lines: tuple[str, ...]
    behaviour: str = "cooperative"
    language: str = "de"
    until: tuple[str, ...] = ()  # tools whose success (all of them) ends the call
    max_replies: int = 6


@dataclass(frozen=True)
class Scenario:
    id: str
    group: str
    lang: str
    caller: Caller
    expect: dict[str, Any]
    tags: tuple[str, ...] = ()
    identity: str = "max"
    erika_booked: bool = False
    clock: str = "monday_7"
    review: bool = False
    audio: bool = False  # part of the audio subset
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _fill(text: str) -> str:
    try:
        return text.format(**PLACEHOLDERS)
    except KeyError as e:
        raise ScenarioError(f"unknown placeholder {e}") from None


def _until(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def parse(data: dict[str, Any]) -> Scenario:
    sid = data.get("id", "?")
    caller = data.get("caller") or {}
    lines = caller.get("lines")
    if not lines:
        raise ScenarioError(f"{sid}: caller.lines is missing")
    unknown = set(data.get("expect", {})) - EXPECTATIONS
    if unknown:
        raise ScenarioError(f"{sid}: unknown expectation {sorted(unknown)}")
    behaviour = caller.get("behaviour", "cooperative")
    if behaviour not in BEHAVIOURS:
        raise ScenarioError(f"{sid}: unknown behaviour {behaviour}")
    clock = data.get("setup", {}).get("clock", "monday_7")
    if clock not in CLOCKS:
        raise ScenarioError(f"{sid}: unknown clock {clock}")
    identity = data.get("setup", {}).get("identity", "max")
    if identity not in IDENTITIES:
        raise ScenarioError(f"{sid}: unknown identity {identity}")
    return Scenario(
        id=sid,
        group=data["group"],
        lang=data["lang"],
        caller=Caller(
            lines=tuple(line if line == SILENCE else _fill(line) for line in lines),
            behaviour=behaviour,
            language=caller.get("language", data["lang"]),
            until=_until(caller.get("until")),
            max_replies=int(caller.get("max_replies", 6)),
        ),
        expect=dict(data.get("expect", {})),
        tags=tuple(data.get("tags", ())),
        identity=identity,
        erika_booked=bool(data.get("setup", {}).get("erika_booked", False)),
        clock=clock,
        review=bool(data.get("review", False)),
        audio=bool(data.get("audio", False)),
        notes=data.get("notes", ""),
    )


def load_all(directory: Path) -> list[Scenario]:
    found: list[Scenario] = []
    for path in sorted(directory.glob("*.yaml")):
        for data in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            found.append(parse(data))
    return found
