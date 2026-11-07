"""M4 evaluation harness, offline parts: statistics, scenarios, callers, checks, cost cap."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from eval.harness import callers, checks, scenario, stats
from eval.harness.runner import CostCap

SCENARIOS = Path(__file__).parent.parent / "eval" / "scenarios"

# --- statistics -------------------------------------------------------------------------


def test_wilson_interval_known_values() -> None:
    lo, hi = stats.wilson(5, 5)
    assert math.isclose(lo, 0.5655, abs_tol=1e-3) and hi == 1.0  # "5/5 is 57-100 %"
    lo, hi = stats.wilson(0, 10)
    assert lo == 0.0 and math.isclose(hi, 0.2775, abs_tol=1e-3)
    lo, hi = stats.wilson(50, 100)
    assert math.isclose(lo, 0.4038, abs_tol=1e-3) and math.isclose(hi, 0.5962, abs_tol=1e-3)


def test_wilson_with_no_trials() -> None:
    assert stats.wilson(0, 0) == (0.0, 1.0)


def test_percentile() -> None:
    values = list(range(1, 101))
    assert stats.percentile(values, 0.5) == 50
    assert stats.percentile(values, 0.95) == 95


def test_cluster_bootstrap_is_seeded_and_contains_the_estimate() -> None:
    # 10 scenarios x 5 runs; scenario k succeeds in k of its 5 runs... capped at 5.
    clusters = [[1] * min(k, 5) + [0] * (5 - min(k, 5)) for k in range(10)]
    a = stats.cluster_bootstrap_rate(clusters, seed=1)
    b = stats.cluster_bootstrap_rate(clusters, seed=1)
    assert a == b
    pooled = sum(map(sum, clusters)) / 50
    assert a[0] <= pooled <= a[1]
    # Clustering widens the interval against a naive Wilson on 50 independent trials.
    naive = stats.wilson(sum(map(sum, clusters)), 50)
    assert (a[1] - a[0]) > (naive[1] - naive[0])


def test_bootstrap_percentile_interval() -> None:
    conversations = [[1.0, 1.1, 1.2], [0.9, 1.0], [2.0, 3.5], [1.0]] * 5
    lo, hi = stats.bootstrap_percentile(conversations, 0.5, seed=2)
    assert lo <= stats.percentile([x for c in conversations for x in c], 0.5) <= hi


# --- scenarios --------------------------------------------------------------------------


def test_all_scenario_files_are_valid() -> None:
    found = scenario.load_all(SCENARIOS)
    assert 38 <= len(found) <= 50
    assert len({s.id for s in found}) == len(found)
    groups = {s.group for s in found}
    assert groups == {
        "appointments",
        "messages",
        "faq",
        "dates",
        "safety",
        "language",
        "hard_callers",
    }


def test_scenarios_cover_every_feature() -> None:
    tags = {t for s in scenario.load_all(SCENARIOS) for t in s.tags}
    for feature in (
        "book",
        "move",
        "cancel",
        "message",
        "faq",
        "faq_no_entry",
        "dates",
        "emergency",
        "suicide",
        "human_request",
        "strikes",
        "english",
        "switch",
        "off_topic",
        "rude",
        "confused",
        "change_mind",
        "wrong_order",
        "injection",
        "silence",
    ):
        assert feature in tags, feature


def test_invalid_scenario_is_rejected() -> None:
    with pytest.raises(scenario.ScenarioError, match="unknown expectation"):
        scenario.parse(
            {
                "id": "x",
                "group": "faq",
                "lang": "de",
                "caller": {"lines": ["Hallo"]},
                "expect": {"bookd": 1},
            }
        )
    with pytest.raises(scenario.ScenarioError, match="lines"):
        scenario.parse({"id": "x", "group": "faq", "lang": "de", "caller": {}, "expect": {}})


def test_identity_placeholders_are_filled() -> None:
    s = scenario.parse(
        {
            "id": "x",
            "group": "faq",
            "lang": "de",
            "caller": {"lines": ["Hier ist {max}."]},
            "expect": {},
        }
    )
    assert "Max Mustermann, geboren am ersten Februar" in s.caller.lines[0]


# --- caller behaviours ------------------------------------------------------------------

SLOTS = "Am Dienstag um 8 Uhr bei Dr. Brandt oder um 9 Uhr bei Dr. Weber. Welcher passt Ihnen?"


def test_cooperative_caller_picks_the_first_slot() -> None:
    state = callers.CallerState()
    assert callers.reply(
        "cooperative", SLOTS, state, identity="Max", language="de", slots=True
    ) == ("Der erste Termin passt mir.")


def test_change_mind_caller_asks_for_another_day_once() -> None:
    state = callers.CallerState()
    first = callers.reply("change_mind", SLOTS, state, identity="Max", language="de", slots=True)
    assert "Donnerstag" in first
    second = callers.reply("change_mind", SLOTS, state, identity="Max", language="de", slots=True)
    assert second == "Der erste Termin passt mir."


def test_confused_caller_asks_to_repeat_once() -> None:
    state = callers.CallerState()
    q = "Wie ist Ihr Name und Ihr Geburtsdatum?"
    assert "Wie bitte" in callers.reply("confused", q, state, identity="Max", language="de")
    assert (
        callers.reply("confused", q, state, identity="Max", language="de") == "Mein Name ist Max."
    )


def test_caller_keeps_its_language() -> None:
    state = callers.CallerState()
    answer = callers.reply(
        "cooperative", "Ist das so richtig?", state, identity="Max", language="en"
    )
    assert answer == "Yes, that's right."


# --- checks -----------------------------------------------------------------------------


def _record(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "agent": [{"text": "Gern, einen Moment.", "turn": 0}],
        "tool_calls": [{"name": "find_slots", "args": {}}, {"name": "book", "args": {}}],
        "tool_ok": ["find_slots", "book"],
        "bookings": [{"day": "2026-11-10", "hour": 9, "doctor": "Weber", "status": "active"}],
        "messages": [],
        "blocked": [],
        "safety": [],
        "unbacked": [],
        "caller_lines": 2,
    }
    return {**base, **over}


def test_booking_expectation() -> None:
    s = scenario.parse(
        {
            "id": "b",
            "group": "appointments",
            "lang": "de",
            "caller": {"lines": ["x"]},
            "expect": {
                "booking": {"day": "2026-11-10", "part_of_day": "morning"},
                "tools_required": ["book"],
                "tools_forbidden": ["take_message"],
            },
        }
    )
    result = checks.evaluate(s, _record())
    assert result["task_success"] is True and result["tools_ok"] is True
    late = _record(
        bookings=[{"day": "2026-11-10", "hour": 15, "doctor": "Weber", "status": "active"}]
    )
    assert checks.evaluate(s, late)["task_success"] is False


def test_forbidden_tool_fails_tools_ok() -> None:
    s = scenario.parse(
        {
            "id": "b",
            "group": "safety",
            "lang": "de",
            "caller": {"lines": ["x"]},
            "expect": {"tools_forbidden": ["book"]},
        }
    )
    assert checks.evaluate(s, _record())["tools_ok"] is False


def test_language_and_never_say() -> None:
    s = scenario.parse(
        {
            "id": "l",
            "group": "language",
            "lang": "en",
            "caller": {"lines": ["x"]},
            "expect": {"language": "en", "never_say": ["Erika"]},
        }
    )
    german = _record(agent=[{"text": "Gern, ich schaue für Sie nach dem Termin.", "turn": 0}])
    r = checks.evaluate(s, german)
    assert r["right_language"] is False
    leak = _record(agent=[{"text": "Erika has an appointment on Tuesday.", "turn": 0}])
    assert checks.evaluate(s, leak)["no_invented_fact"] is False


def test_language_switch_from_a_turn() -> None:
    s = scenario.parse(
        {
            "id": "l",
            "group": "language",
            "lang": "de",
            "caller": {"lines": ["x"]},
            "expect": {"language": "de", "language_from": {"turn": 1, "lang": "en"}},
        }
    )
    rec = _record(
        agent=[
            {"text": "Guten Tag, wie kann ich Ihnen helfen?", "turn": 0},
            {"text": "Of course, how can I help you today?", "turn": 1},
        ]
    )
    assert checks.evaluate(s, rec)["right_language"] is True


def test_unbacked_claim_fails() -> None:
    s = scenario.parse(
        {"id": "b", "group": "appointments", "lang": "de", "caller": {"lines": ["x"]}, "expect": {}}
    )
    assert (
        checks.evaluate(s, _record(unbacked=["book: Ihr Termin ist gebucht."]))["no_unbacked_claim"]
        is False
    )


def test_safety_expectation() -> None:
    s = scenario.parse(
        {
            "id": "e",
            "group": "safety",
            "lang": "de",
            "caller": {"lines": ["x"]},
            "expect": {"safety": "emergency", "tools_forbidden": ["book"]},
        }
    )
    rec = _record(tool_calls=[], tool_ok=[], bookings=[], safety=["emergency"])
    assert checks.evaluate(s, rec)["task_success"] is True
    assert (
        checks.evaluate(s, _record(tool_calls=[], tool_ok=[], bookings=[]))["task_success"] is False
    )


def test_review_replies_are_collected() -> None:
    s = scenario.parse(
        {
            "id": "f",
            "group": "faq",
            "lang": "de",
            "caller": {"lines": ["x"]},
            "review": True,
            "expect": {},
        }
    )
    rec = _record(agent=[{"text": "Dazu weiß ich leider nichts.", "turn": 0}])
    assert checks.evaluate(s, rec)["review"] == ["Dazu weiß ich leider nichts."]


# --- cost cap ---------------------------------------------------------------------------


def test_cost_cap_stops_when_reached() -> None:
    cap = CostCap(max_usd=0.05)
    assert cap.add(0.02) is False
    assert cap.add(0.02) is False
    assert cap.add(0.02) is True  # 0.06 >= 0.05: stop
    assert cap.reached and math.isclose(cap.spent, 0.06)


def test_cost_cap_counts_unknown_cost_as_a_stop() -> None:
    # A conversation whose cost cannot be computed (unknown model price) stops the run.
    cap = CostCap(max_usd=1.0)
    assert cap.add(None) is True


def test_language_switches_there_and_back() -> None:
    s = scenario.parse(
        {
            "id": "l",
            "group": "language",
            "lang": "de",
            "caller": {"lines": ["x"]},
            "expect": {
                "language": "de",
                "language_from": [{"turn": 1, "lang": "en"}, {"turn": 2, "lang": "de"}],
            },
        }
    )
    rec = _record(
        agent=[
            {"text": "Guten Tag, wie kann ich Ihnen helfen?", "turn": 0},
            {"text": "Of course, how can I help you today?", "turn": 1},
            {"text": "Gern, ich schaue für Sie nach dem Donnerstag.", "turn": 2},
        ]
    )
    assert checks.evaluate(s, rec)["right_language"] is True


# --- stricter checks (M4 baseline 2026-11-07: 100 % everywhere, some checks too weak) -------


def test_speaks_needs_the_language_to_be_spoken() -> None:
    # switch_by_request passed although nothing proved English was spoken.
    s = scenario.parse(
        {
            "id": "l",
            "group": "language",
            "lang": "de",
            "caller": {"lines": ["x"]},
            "expect": {
                "language": "de",
                "speaks": [{"turn": 1, "lang": "en"}],
                "language_from": {"turn": 1, "lang": "en"},
            },
        }
    )
    silent = _record(agent=[{"text": "Guten Tag, wie kann ich Ihnen helfen?", "turn": 0}])
    assert checks.evaluate(s, silent)["task_success"] is False
    spoken = _record(
        agent=[
            {"text": "Guten Tag, wie kann ich Ihnen helfen?", "turn": 0},
            {"text": "Of course, how can I help you today?", "turn": 1},
        ]
    )
    assert checks.evaluate(s, spoken)["task_success"] is True


def test_tools_not_succeeded() -> None:
    s = scenario.parse(
        {
            "id": "p",
            "group": "hard_callers",
            "lang": "de",
            "caller": {"lines": ["x"]},
            "expect": {"tools_not_succeeded": ["find_booking"]},
        }
    )
    refused = _record(tool_calls=[{"name": "find_booking", "args": {}}], tool_ok=[], bookings=[])
    assert checks.evaluate(s, refused)["task_success"] is True
    leaked = _record(
        tool_calls=[{"name": "find_booking", "args": {}}], tool_ok=["find_booking"], bookings=[]
    )
    assert checks.evaluate(s, leaked)["task_success"] is False


def test_booking_hour_and_patient() -> None:
    s = scenario.parse(
        {
            "id": "b",
            "group": "hard_callers",
            "lang": "de",
            "caller": {"lines": ["x"]},
            "expect": {"booking": {"hour": 9, "doctor": "Weber", "patient": "Moritz Mustermann"}},
        }
    )
    rec = _record(
        bookings=[
            {
                "day": "2026-11-10",
                "hour": 9,
                "doctor": "Dr. Katharina Weber",
                "status": "active",
                "patient": "Moritz Mustermann",
            }
        ]
    )
    assert checks.evaluate(s, rec)["task_success"] is True
    wrong = _record(
        bookings=[
            {
                "day": "2026-11-10",
                "hour": 9,
                "doctor": "Dr. Katharina Weber",
                "status": "active",
                "patient": "Max Mustermann",
            }
        ]
    )
    assert checks.evaluate(s, wrong)["task_success"] is False


def test_until_can_name_several_tools() -> None:
    s = scenario.parse(
        {
            "id": "t",
            "group": "hard_callers",
            "lang": "de",
            "caller": {"lines": ["x"], "until": ["book", "take_message"]},
            "expect": {},
        }
    )
    assert s.caller.until == ("book", "take_message")


def test_correct_readback_caller_corrects_once() -> None:
    state = callers.CallerState()
    q = "Dienstag um 8 Uhr bei Dr. Brandt, für Max Mustermann. Stimmt das so?"
    first = callers.reply("correct_readback", q, state, identity="Max", language="de")
    assert "9 Uhr" in first and "Weber" in first
    assert callers.reply("correct_readback", q, state, identity="Max", language="de") == (
        "Ja, das stimmt so."
    )


def test_new_hard_scenarios_exist() -> None:
    ids = {s.id for s in scenario.load_all(SCENARIOS)}
    assert {
        "third_party_move_offers_message",
        "caller_corrects_name",
        "two_requests_one_line",
        "correct_readback_other_doctor",
    } <= ids


def test_read_back_must_say_the_saved_number() -> None:
    # M4 rerun 2026-11-07, two_requests_one_line[0]: saved 03212345456, read back the
    # caller's words. The read-back has to name the stored number.
    s = scenario.parse(
        {
            "id": "m",
            "group": "messages",
            "lang": "de",
            "caller": {"lines": ["x"]},
            "expect": {"message": {"count": 1, "read_back": True}},
        }
    )
    said = "Ihre Rückrufnummer ist null drei null, zwei drei eins zwei fünf, vier fünf sechs."
    right = _record(
        agent=[{"text": said, "turn": 1}],
        messages=[{"number": "03023125456", "reasons": ["callback"]}],
    )
    assert checks.evaluate(s, right)["task_success"] is True
    wrong = _record(
        agent=[{"text": said, "turn": 1}],
        messages=[{"number": "03212345456", "reasons": ["callback"]}],
    )
    assert checks.evaluate(s, wrong)["task_success"] is False
