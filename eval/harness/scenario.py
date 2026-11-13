"""Scenario files (eval/scenarios/*.yaml): a list of scenarios per group.

Each scenario: id, group, lang, tags, setup (database and clock), caller (lines, a
rule-based behaviour, language, the tool that ends the call), expect (deterministic
checks, eval/harness/checks.py), review (collect the agent's replies for a human).
Placeholders in lines: {max}, {erika}, {max_en}, {erika_en}, {phone_de}, {phone_en} and the
name-fix identities. Setup: clock, identity, erika_booked, seed_patients, seed_bookings.
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
    # Roadmap item 4: names as the speech recognition heard them (the booking is under
    # another spelling), always born on the third of March 1990.
    "yusuf_oestuerk": "Yusuf Östürk, geboren am dritten März neunzehnhundertneunzig",
    "anna_mayer": "Anna Mayer, geboren am dritten März neunzehnhundertneunzig",
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
class Step:
    """Caller lines, then answers (at most max_replies) until every ``until`` tool has
    succeeded. Steps keep a multi-part call in line with the agent (M4 rerun 2026-11-08:
    fixed lines spoken whatever the agent said got out of step after a read-back)."""

    lines: tuple[str, ...]
    until: tuple[str, ...] = ()
    max_replies: int = 6


@dataclass(frozen=True)
class Caller:
    lines: tuple[str, ...]
    behaviour: str = "cooperative"
    language: str = "de"
    until: tuple[str, ...] = ()  # tools whose success (all of them) ends the call
    max_replies: int = 6
    step_list: tuple[Step, ...] = ()
    spelling: str = ""  # what the caller says when asked to spell the surname

    @property
    def steps(self) -> tuple[Step, ...]:
        """The call's steps; a scenario without ``steps`` is one step."""
        return self.step_list or (Step(self.lines, self.until, self.max_replies),)


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
    seed_patients: tuple[dict[str, Any], ...] = ()  # {name, dob}: patients without a booking
    seed_bookings: tuple[dict[str, Any], ...] = ()  # {name, dob, day, hour}: Dr. Weber
    extra: dict[str, Any] = field(default_factory=dict)


def _fill(text: str) -> str:
    try:
        return text.format(**PLACEHOLDERS)
    except KeyError as e:
        raise ScenarioError(f"unknown placeholder {e}") from None


def _line(line: str) -> str:
    return line if line == SILENCE else _fill(line)


def _until(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def parse(data: dict[str, Any]) -> Scenario:
    sid = data.get("id", "?")
    caller = data.get("caller") or {}
    raw_steps = caller.get("steps") or []
    lines = caller.get("lines") or [line for st in raw_steps for line in st.get("lines", [])]
    if not lines:
        raise ScenarioError(f"{sid}: caller.lines (or steps with lines) is missing")
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
            lines=tuple(_line(line) for line in lines),
            behaviour=behaviour,
            language=caller.get("language", data["lang"]),
            until=_until(caller.get("until")),
            max_replies=int(caller.get("max_replies", 6)),
            spelling=str(caller.get("spelling", "")),
            step_list=tuple(
                Step(
                    tuple(_line(line) for line in st.get("lines", [])),
                    _until(st.get("until")),
                    int(st.get("max_replies", 6)),
                )
                for st in raw_steps
            ),
        ),
        expect=dict(data.get("expect", {})),
        tags=tuple(data.get("tags", ())),
        identity=identity,
        erika_booked=bool(data.get("setup", {}).get("erika_booked", False)),
        clock=clock,
        review=bool(data.get("review", False)),
        audio=bool(data.get("audio", False)),
        notes=data.get("notes", ""),
        seed_patients=tuple(data.get("setup", {}).get("seed_patients", ())),
        seed_bookings=tuple(data.get("setup", {}).get("seed_bookings", ())),
    )


def load_all(directory: Path) -> list[Scenario]:
    found: list[Scenario] = []
    for path in sorted(directory.glob("*.yaml")):
        for data in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            found.append(parse(data))
    return found
