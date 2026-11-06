"""Offline checks of eval/leak_report.py on synthetic live-test entries."""

from __future__ import annotations

from typing import Any

import pytest
from eval.leak_report import arm_of, cost_usd, is_preamble, percentile, render, summarize

from praxis_telefon.agent.claim_gate import FilterContext

CTX = FilterContext(
    caller_texts=("Haben Sie nächsten Dienstag einen Termin frei?",),
    today="Heute ist Montag, der 9. November 2026, 7 Uhr.",
).to_json()
CONFIG = {"model": "anthropic/claude-haiku-5.5", "reasoning_effort": "none", "tool_docs": "de"}


def _entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "test": "tests/live/test_dates_live.py::test_x[0]",
        "outcome": "passed",
        "config": CONFIG,
        "groups": [
            {
                "stage": "first",
                "ctx": CTX,
                "tool_calls": ["find_slots"],
                "sentences": [
                    "This is a new appointment request, so I'll search for slots. ",
                    "Einen Moment, ich schaue nach. ",
                    "Das ist eine neue Terminanfrage, also suche ich freie Termine. ",
                ],
            },
            {"stage": "first", "ctx": CTX, "tool_calls": [], "sentences": ["Gern. "]},
        ],
        "blocked": ["meta:language"],
        "fallbacks": [],
        "generations": [
            {"ttft_s": 0.9, "text_s": None, "tools": ["find_slots"], "llm_requests": 1},
            {"ttft_s": 0.4, "text_s": 0.4, "tools": [], "llm_requests": 1},
            {"ttft_s": 0.0, "text_s": 0.0, "tools": [], "llm_requests": 0},  # fixed message
        ],
        "turns": [
            {"first_speech_s": 1.6, "llm_requests": 2},
            {"first_speech_s": 0.0, "llm_requests": 0},
        ],
        "llm_usage": [
            {"prompt_tokens": 5000, "prompt_cached_tokens": 4000, "completion_tokens": 100},
        ],
    }
    return {**entry, **overrides}


def test_summary_counts_narration_before_tools_in_any_language() -> None:
    report = summarize([_entry()])[arm_of(_entry())]
    assert report.tests == 1 and report.passed == 1
    assert report.meta_blocked == 1
    assert report.pre_tool_sentences == 3
    assert report.pre_tool_blocked == 1  # the English one
    assert report.pre_tool_preambles == 1
    # German narration passes the filters today; it is listed for review.
    assert [s for _, s, _ in report.pre_tool_review] == [
        "Das ist eine neue Terminanfrage, also suche ich freie Termine."
    ]


def test_latency_leaves_out_fixed_messages() -> None:
    report = summarize([_entry()])[arm_of(_entry())]
    assert report.llm_ttft == [0.4]
    assert report.tool_ttft == [0.9]
    assert report.first_speech == [1.6]


def test_entries_are_grouped_by_arm() -> None:
    low = _entry(config={**CONFIG, "reasoning_effort": "low"}, outcome="failed")
    reports = summarize([_entry(), _entry(), low])
    assert sorted(r.tests for r in reports.values()) == [1, 2]
    assert reports[arm_of(low)].passed == 0
    assert "reasoning=low" in arm_of(low)


def test_render_mentions_the_review_sentence() -> None:
    text = render(summarize([_entry()])[arm_of(_entry())])
    assert "1/1 passed" in text
    assert "75.0 per 75 tests" in text
    assert "Terminanfrage" in text


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("Einen Moment, ich schaue nach.", True),
        ("One moment, let me check.", True),
        ("Gerne, ich schaue nach.", True),
        (
            "Einen Moment, ich schaue nach, denn das ist eine neue Terminanfrage für Dienstag.",
            False,
        ),
        ("This is a new appointment request, so I'll search for slots.", False),
    ],
)
def test_preamble(sentence: str, expected: bool) -> None:
    assert is_preamble(sentence) is expected


def test_percentile_nearest_rank() -> None:
    assert percentile([], 0.5) is None
    assert percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert percentile([float(i) for i in range(1, 11)], 0.9) == 9.0


def test_cost_with_cache_reads() -> None:
    usage = [{"prompt_tokens": 5000, "prompt_cached_tokens": 4000, "completion_tokens": 100}]
    # 1000 fresh * 0.10 + 4000 cached * 0.01 + 100 out * 0.50, per million
    assert cost_usd(usage, "anthropic/claude-haiku-5.5") == pytest.approx(0.00019)
    assert cost_usd(usage, "other/model") is None
