"""False-positive check: real agent sentences from passing live runs must pass the filters.

tests/data/live_sentences.jsonl holds every sentence the LLM produced in passing live
tests (before filtering), grouped per reply, with the call context the filters saw.
Collected with LIVE_CORPUS=<path> uv run pytest -m live ... (tests/live/conftest.py).
Synthetic data only.

Replaying them through the same SentenceFilter must block nothing except the reviewed
sentences in tests/data/known_bad_sentences.json (real problems the filters are
meant to catch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from praxis_telefon.agent.claim_gate import FilterContext, SentenceFilter

CORPUS = Path(__file__).parent / "data" / "live_sentences.jsonl"

# Reviewed: sentences the LLM really produced that must not be spoken, with the reason.
KNOWN_BAD: dict[str, str] = json.loads(
    (Path(__file__).parent / "data" / "known_bad_sentences.json").read_text(encoding="utf-8")
)


def _groups() -> list[dict[str, Any]]:
    groups = []
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["outcome"] == "passed":
            groups += [{**g, "test": entry["test"]} for g in entry["groups"]]
    return groups


def _blocked(group: dict[str, Any]) -> list[str]:
    sf = SentenceFilter(FilterContext.from_json(group["ctx"]), group["stage"])
    released: list[str] = []
    for sentence in group["sentences"]:
        released += sf.push(sentence)
    sf.finish()
    return [s.strip() for s in group["sentences"] if s not in released]


def test_corpus_is_not_empty() -> None:
    groups = _groups()
    assert len(groups) >= 50
    assert sum(len(g["sentences"]) for g in groups) >= 200


@pytest.mark.parametrize("group", _groups(), ids=lambda g: g["test"].split("::")[-1])
def test_filters_block_no_good_sentence(group: dict[str, Any]) -> None:
    unexpected = [s for s in _blocked(group) if s not in KNOWN_BAD]
    assert unexpected == []


def test_known_bad_sentences_are_still_blocked() -> None:
    found = {s for g in _groups() for s in _blocked(g)}
    assert set(KNOWN_BAD) <= found
