"""Compare live-test runs of the reasoning-leak experiment (M3 Part B, step 1).

Input: JSONL files the live tests write with LIVE_CORPUS=<path> (tests/live/conftest.py).
Entries are grouped into arms by their config: tool descriptions, reasoning effort, prompt
cache, model.

Per arm:
- pass rate;
- sentences blocked as meta talk, per 75 tests, by reason;
- narration before a tool, in any language: every sentence the LLM wrote in a reply that
  then called a tool. The "language" check only sees English in a German call, so German
  narration would pass it; the ones the filters let through, other than greetings,
  thanks and "I'm looking it up" (is_preamble), are listed for review;
- retries and fixed fallback sentences;
- llm_ttft (as in the turn log: replies that spoke, the code filler not counted), time
  to a tool call, time from the caller's line to the first spoken text (filler
  included) and to the answer (first text that is not the filler); p50 and p90;
- tokens and estimated cost (OpenRouter's dashboard is the reference).

    uv run python -m eval.leak_report eval/results/*.jsonl
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from praxis_telefon.agent.claim_gate import FilterContext, SentenceFilter

# USD per million tokens: (input, cache read, cache write, output). OpenRouter listing,
# 2026-11-06, prompts under 100k tokens.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "anthropic/claude-haiku-5.5": (0.10, 0.01, 0.125, 0.50),
}

# What is fine before a tool (system prompt, "# Stil"), clause by clause: a greeting,
# thanks, an acknowledgement, "einen Moment", or saying what it now looks up. Arm A's
# first report listed 93 sentences for review, nearly all "Guten Tag, Herr Mustermann."
# or "Einen Moment, ich suche Ihnen einen Termin für morgen Vormittag."
_NAME = r"(?: (?:herr|frau|mr|mrs|ms) \w+)?"
_OK_CLAUSE = re.compile(
    "^(?:"
    + "|".join(
        [
            rf"(?:guten (?:tag|morgen|abend)|hallo|hello|hi|good (?:morning|afternoon)){_NAME}",
            rf"(?:vielen dank|danke(?: schön)?|thank you|thanks){_NAME}",
            rf"(?:gern|gerne|sehr gern|alles klar|in ordnung|okay|sure|of course){_NAME}",
            r"(?:herr|frau|mr|mrs|ms) \w+",  # "Guten Tag, Herr Mustermann."
            r"(?:bitte|please)",  # "One moment, please."
            r"(?:einen moment|moment|one moment|just a moment)(?: bitte| please)?",
            r"(?:ich )?(?:suche|schaue|sehe|prüfe)\b.*",
            r"ich nehme (?:die|eine|den|ihre|ihren) (?:nachricht|rückruf)\b.*",
            r"(?:let me|i'?ll|i will|i can|i'?d be happy to) (?:just )?(?:check|look)\b.*",
        ]
    )
    + ")$",
    re.I,
)


def is_preamble(sentence: str) -> bool:
    """Every clause is a greeting, thanks, acknowledgement or "I'm looking it up"."""
    clauses = [c.strip(" .!?\"'„“") for c in re.split(r"[,;]", sentence)]
    clauses = [c for c in clauses if c]
    return bool(clauses) and all(_OK_CLAUSE.match(c) for c in clauses)


def arm_of(entry: dict[str, Any]) -> str:
    c = entry.get("config") or {}
    if not c:
        return "unknown"
    tools, effort, cache = c.get("tool_docs"), c.get("reasoning_effort"), c.get("prompt_cache")
    filler = c.get("tool_filler", "off")
    return f"tools={tools} reasoning={effort} cache={cache} filler={filler} {c.get('model')}"


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile; None without values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def released(group: dict[str, Any]) -> list[str]:
    """Sentences of the group the filters let through (replayed as in the live run)."""
    sf = SentenceFilter(FilterContext.from_json(group["ctx"]), group["stage"])
    out: list[str] = []
    for sentence in group["sentences"]:
        out += sf.push(sentence)
    sf.finish()
    return out


def cost_usd(usage: Iterable[dict[str, Any]], model: str) -> float | None:
    if model not in PRICES:
        return None
    p_in, p_read, p_write, p_out = PRICES[model]
    total = 0.0
    for u in usage:
        read, write = u.get("prompt_cached_tokens", 0), u.get("cache_creation_tokens", 0)
        fresh = max(0, u.get("prompt_tokens", 0) - read - write)
        total += fresh * p_in + read * p_read + write * p_write
        total += u.get("completion_tokens", 0) * p_out
    return total / 1_000_000


@dataclass
class ArmReport:
    arm: str
    tests: int = 0
    passed: int = 0
    blocked: Counter[str] = field(default_factory=Counter)
    retries: int = 0
    fallbacks: int = 0
    pre_tool_sentences: int = 0
    pre_tool_preambles: int = 0
    pre_tool_blocked: int = 0
    pre_tool_review: list[tuple[str, str, list[str]]] = field(default_factory=list)
    llm_ttft: list[float] = field(default_factory=list)
    tool_ttft: list[float] = field(default_factory=list)
    first_speech: list[float] = field(default_factory=list)
    first_answer: list[float] = field(default_factory=list)
    filler_turns: int = 0
    turns: int = 0
    usage: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""

    @property
    def meta_blocked(self) -> int:
        return sum(n for reason, n in self.blocked.items() if reason.startswith("meta:"))

    def per_75(self, n: int) -> float:
        return 75 * n / self.tests if self.tests else 0.0

    def cost(self) -> float | None:
        return cost_usd(self.usage, self.model)


def summarize(entries: Iterable[dict[str, Any]]) -> dict[str, ArmReport]:
    reports: dict[str, ArmReport] = {}
    for entry in entries:
        arm = arm_of(entry)
        r = reports.setdefault(arm, ArmReport(arm))
        r.model = (entry.get("config") or {}).get("model", "")
        r.tests += 1
        r.passed += entry["outcome"] == "passed"
        r.blocked.update(entry.get("blocked", []))
        r.fallbacks += len(entry.get("fallbacks", []))
        test = entry["test"].split("::")[-1]
        for group in entry.get("groups", []):
            r.retries += group["stage"] == "retry"
            if not group.get("tool_calls"):
                continue
            spoken = released(group)
            for sentence in group["sentences"]:
                r.pre_tool_sentences += 1
                if sentence not in spoken:
                    r.pre_tool_blocked += 1
                elif is_preamble(sentence):
                    r.pre_tool_preambles += 1
                else:
                    r.pre_tool_review.append((test, sentence.strip(), group["tool_calls"]))
        for g in entry.get("generations", []):
            if not g.get("llm_requests") or g.get("ttft_s") is None:
                continue  # a fixed message, or nothing came
            if g.get("text_s") is not None and not g.get("filler"):
                r.llm_ttft.append(g["ttft_s"])
            elif g.get("tools"):  # started with a tool call (only the code filler spoken)
                r.tool_ttft.append(g["ttft_s"])
        turns = [t for t in entry.get("turns", []) if t.get("llm_requests")]
        r.first_speech += [t["first_speech_s"] for t in turns if t["first_speech_s"] is not None]
        r.first_answer += [
            t.get("first_answer_s", t["first_speech_s"])  # no filler before: the same
            for t in turns
            if t.get("first_answer_s", t["first_speech_s"]) is not None
        ]
        r.filler_turns += sum(bool(t.get("filler")) for t in turns)
        r.turns += len(turns)
        r.usage += entry.get("llm_usage", [])
    return reports


def _ms(values: Sequence[float]) -> str:
    p50, p90 = percentile(values, 0.5), percentile(values, 0.9)
    if p50 is None or p90 is None:
        return "n/a"
    return f"p50 {p50 * 1000:.0f} ms, p90 {p90 * 1000:.0f} ms (n={len(values)})"


def render(report: ArmReport) -> str:
    r = report
    tokens = Counter[str]()
    for u in r.usage:
        tokens.update({k: u.get(k, 0) for k in ("prompt_tokens", "prompt_cached_tokens")})
        tokens.update({k: u.get(k, 0) for k in ("completion_tokens", "reasoning_tokens")})
    cost = r.cost()
    reasons = ", ".join(f"{k} {v}" for k, v in sorted(r.blocked.items())) or "none"
    lines = [
        f"== {r.arm}",
        f"tests: {r.passed}/{r.tests} passed",
        f"meta blocked: {r.meta_blocked} ({r.per_75(r.meta_blocked):.1f} per 75 tests)",
        f"all blocked: {reasons}",
        f"retries: {r.retries}, fallbacks: {r.fallbacks}",
        f"before a tool: {r.pre_tool_sentences} sentences, {r.pre_tool_blocked} blocked, "
        f"{r.pre_tool_preambles} preambles, {len(r.pre_tool_review)} passed for review "
        f"({r.per_75(r.pre_tool_sentences - r.pre_tool_preambles):.1f} non-preamble per 75)",
        f"llm_ttft: {_ms(r.llm_ttft)}",
        f"time to tool call: {_ms(r.tool_ttft)}",
        f"caller line to first speech: {_ms(r.first_speech)}",
        f"caller line to answer: {_ms(r.first_answer)}",
        f"turns with the code filler: {r.filler_turns}/{r.turns}",
        f"LLM requests: {len(r.usage)}, tokens: prompt {tokens['prompt_tokens']} "
        f"(cached {tokens['prompt_cached_tokens']}), completion {tokens['completion_tokens']} "
        f"(reasoning {tokens['reasoning_tokens']})",
        "estimated cost: " + (f"${cost:.3f}" if cost is not None else "unknown model price"),
    ]
    lines += [
        f"  review [{test}] -> {','.join(tools)}: {s}" for test, s, tools in r.pre_tool_review
    ]
    return "\n".join(lines)


def load(paths: Iterable[Path]) -> list[dict[str, Any]]:
    entries = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        entries += with_cache_flag([json.loads(line) for line in lines])
    return entries


def with_cache_flag(run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Files from before config.prompt_cache: the run was cached if any request hit."""
    hit = any(u.get("prompt_cached_tokens") for e in run for u in e.get("llm_usage", []))
    for entry in run:
        config = entry.get("config")
        if config and "prompt_cache" not in config:
            config["prompt_cache"] = "on" if hit else "off"
    return run


def main(argv: Sequence[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for report in summarize(load(Path(a) for a in argv)).values():
        print(render(report), end="\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
