"""Live-test reporting: with LIVE_CORPUS=<path>, each test appends one JSON line with its
outcome, every sentence the LLM produced (before filtering, with the filter context and
the tools that reply called), the reasons the filters blocked, and the experiment
measurements (config, timings, token counts; see harness.py). Synthetic data only.

Experiment arms: LIVE_TOOL_DOCS=production|en and LLM_REASONING_EFFORT=none|low in the
environment (it overrides .env); eval/leak_report.py compares the files."""

from __future__ import annotations

import json
import os
from collections.abc import Generator, Iterator

import pytest

from tests.live import harness


@pytest.fixture(autouse=True)
def _fresh_buffers() -> Iterator[None]:
    harness.clear_measurements()
    harness.safety_records().clear()
    yield


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    _write_entry(item, report)
    return report


def _write_entry(item: pytest.Item, report: pytest.TestReport) -> None:
    path = os.environ.get("LIVE_CORPUS")
    if not path or report.when != "call":
        return
    records = harness.safety_records()
    entry = {
        "test": item.nodeid,
        "outcome": report.outcome,
        "config": dict(harness.CONFIG),
        "groups": harness.corpus_groups(),
        "blocked": [r["reason"] for r in records if r.get("event") == "speech_blocked"],
        "fallbacks": [r["reason"] for r in records if r.get("event") == "speech_fallback"],
        "generations": [harness.public(g) for g in harness.GENERATIONS],
        "turns": list(harness.TURNS),
        "llm_usage": list(harness.LLM_USAGE),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
