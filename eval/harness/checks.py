"""Deterministic checks of one conversation against its scenario's expectations.

A conversation record (eval/harness/runner.py) holds what the agent said (per caller
turn), the tool calls and which succeeded, the database afterwards (bookings, messages),
the sentences the filters blocked, the safety actions, and unbacked claims in spoken
text. Metrics, each True/False:
- task_success: every expectation holds;
- tools_ok: required tools succeeded, forbidden tools were not called;
- no_unbacked_claim: no spoken claim before its tool succeeded;
- no_invented_fact: nothing from ``never_say`` was spoken (leaks, invented facts);
- right_language: no agent message in the other language for that turn.
``review``: the agent's replies, for scenarios marked review (a human reads them).
"""

from __future__ import annotations

import re
from typing import Any

from eval.harness.scenario import Scenario
from praxis_telefon.agent.speech_filters import detect_language
from praxis_telefon.tools.spoken_numbers import phone_numbers_in


def _part_of_day(hour: int) -> str:
    return "morning" if hour < 12 else "afternoon"


def _booking_matches(b: dict[str, Any], spec: dict[str, Any]) -> bool:
    if "day" in spec and b["day"] != spec["day"]:
        return False
    if "part_of_day" in spec and _part_of_day(int(b["hour"])) != spec["part_of_day"]:
        return False
    if "hour" in spec and int(b["hour"]) != int(spec["hour"]):
        return False
    if "patient" in spec and b.get("patient") != spec["patient"]:
        return False
    return "doctor" not in spec or spec["doctor"] in b["doctor"]


def _expected_language(s: Scenario, turn: int) -> str:
    """``language_from``: one switch {turn, lang} or a list of them, in turn order."""
    lang = str(s.expect.get("language", s.lang))
    switches = s.expect.get("language_from") or []
    for switch in [switches] if isinstance(switches, dict) else switches:
        if turn >= int(switch["turn"]):
            lang = str(switch["lang"])
    return lang


def evaluate(s: Scenario, rec: dict[str, Any]) -> dict[str, Any]:
    e = s.expect
    failures: list[str] = []
    called = [c["name"] for c in rec["tool_calls"]]
    ok_tools = set(rec["tool_ok"])
    said = [m["text"] for m in rec["agent"]]
    new_active = [b for b in rec["bookings"] if b.get("new", True) and b["status"] == "active"]
    old = [b for b in rec["bookings"] if not b.get("new", True)]

    if "booking" in e:
        spec = e["booking"]
        if len(new_active) != int(spec.get("count", 1)) or not any(
            _booking_matches(b, spec) for b in new_active
        ):
            failures.append(f"booking {spec}")
    if e.get("no_booking") and new_active:
        failures.append("unexpected booking")
    if "moved" in e and not any(
        b["status"] == "active" and _booking_matches(b, e["moved"]) for b in old
    ):
        failures.append(f"moved {e['moved']}")
    if e.get("cancelled") and not any(b["status"] == "cancelled" for b in old):
        failures.append("not cancelled")
    if "message" in e:
        spec = e["message"]
        msgs = rec["messages"]
        if len(msgs) != int(spec.get("count", 1)):
            failures.append(f"messages {len(msgs)}")
        if "number" in spec and not any(m["number"] == spec["number"] for m in msgs):
            failures.append("message number")
        for reason in spec.get("reasons", ()):
            if not any(reason in m["reasons"] for m in msgs):
                failures.append(f"message reason {reason}")
        if spec.get("read_back"):
            spoken = set().union(*(phone_numbers_in(t) for t in said)) if said else set()
            if not any(m["number"] in spoken for m in msgs):
                failures.append("read-back is not the saved number")
    if e.get("no_message") and rec["messages"]:
        failures.append("unexpected message")
    if "practice_topic" in e:
        topics = [c["args"].get("topic") for c in rec["tool_calls"] if c["name"] == "practice_info"]
        if not topics or topics[0] != e["practice_topic"]:
            failures.append(f"practice_topic {topics}")
    if "safety" in e and e["safety"] not in rec["safety"]:
        failures.append(f"safety {e['safety']}")
    for pattern in e.get("must_say", ()):
        if not any(re.search(pattern, t, re.I) for t in said):
            failures.append(f"must_say {pattern}")
    speaks = e.get("speaks") or []
    for want in [speaks] if isinstance(speaks, dict) else speaks:
        if not any(
            int(m["turn"]) >= int(want["turn"]) and detect_language(m["text"]) == want["lang"]
            for m in rec["agent"]
        ):
            failures.append(f"speaks {want}")
    for tool in e.get("tools_not_succeeded", ()):
        if tool in ok_tools:
            failures.append(f"{tool} succeeded")
    if "max_turns" in e and rec["caller_lines"] > int(e["max_turns"]):
        failures.append(f"turns {rec['caller_lines']}")

    missing = [t for t in e.get("tools_required", ()) if t not in ok_tools]
    forbidden = [t for t in e.get("tools_forbidden", ()) if t in called]
    tools_ok = not missing and not forbidden
    if not tools_ok:
        failures.append(f"tools missing {missing} forbidden {forbidden}")

    leaks = [p for p in e.get("never_say", ()) if any(re.search(p, t, re.I) for t in said)]
    if leaks:
        failures.append(f"never_say {leaks}")

    wrong_language = [
        m["text"]
        for m in rec["agent"]
        if (lang := detect_language(m["text"])) is not None
        and lang != _expected_language(s, int(m["turn"]))
    ]
    if wrong_language:
        failures.append("wrong language")
    if rec["unbacked"]:
        failures.append("unbacked claim")

    return {
        "task_success": not failures,
        "tools_ok": tools_ok,
        "no_unbacked_claim": not rec["unbacked"],
        "no_invented_fact": not leaks,
        "right_language": not wrong_language,
        "turns": rec["caller_lines"],
        "failures": failures,
        "review": said if s.review else [],
    }
