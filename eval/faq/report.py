"""Measure FAQ topic choice by the real LLM (M3 Part C). Costs LLM tokens; ask first.

    uv run python -m eval.faq.report             # asks the LLM (paid)
    uv run python -m eval.faq.report --rescore   # re-score saved replies after a review

Each labelled question (eval/faq/questions.json by Claude, user_questions.json by a
second author who had not seen the FAQ) goes to the production LLM once, as the first
caller line of a call: the Receptionist's system prompt and tools, plus the language note
the agent adds once English came up. Only the reply's tool calls are read; no tool runs.

Outcomes: answerable questions are right only with practice_info(<expected topic>).
Questions without an answer are false answers if practice_info got a real topic; "other"
or another tool are right. A text reply with no tool call at all is "review" (user,
2026-11-07: it may invent a fact, "Ja, wir machen Hausbesuche"): every such reply is
listed, and it counts as a false answer until eval/faq/review.json marks it "ok"
("invented" keeps it false). Bar (user): >= 95 % right, <= 5 % false answers, reported
separately for both authors.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
# User decision 2026-11-07, after the first run: for prescription and referral requests,
# starting a message for the team is the prompt's flow, so it counts as right.
MESSAGE_TOPICS = frozenset({"repeat_prescription", "referral"})
_OFFER = re.compile(
    r"\b(anliegen|nachricht|praxisteam|rückruf)\b.{0,60}\b(auf|aufnehmen|weiter\w*)\b|"
    r"\b(take|leave|pass on) (a )?message\b",
    re.I,
)
RESULTS = Path("eval/results/faq_llm.jsonl")
# {question: {"verdict": "ok" | "invented", "text": the reply judged}}, written after
# reading the replies; a verdict counts only for that exact reply text.
REVIEW = HERE / "review.json"
Verdicts = dict[str, dict[str, str]]
CONCURRENCY = 4


def outcome(expected: str | None, calls: list[tuple[str, str]], text: str = "") -> str:
    topics = [topic for name, topic in calls if name == "practice_info"]
    if expected is None:
        if any(t != "other" for t in topics):
            return "false_answer"
        return "review" if not calls and text.strip() else "right"
    if not topics:
        names = {name for name, _ in calls}
        if expected in MESSAGE_TOPICS and ("take_message" in names or _OFFER.search(text)):
            return "right_message"
        return "no_tool"
    if topics[0] == expected:
        return "right"
    return "said_unknown" if topics[0] == "other" else "wrong_topic"


def verdict(row: dict[str, Any], verdicts: Verdicts) -> str | None:
    v = verdicts.get(str(row["q"]))
    return v["verdict"] if v is not None and v.get("text") == row.get("text") else None


def _is_false(row: dict[str, Any], verdicts: Verdicts) -> bool:
    if row["outcome"] == "false_answer":
        return True
    return row["outcome"] == "review" and verdict(row, verdicts) != "ok"


def summary(rows: list[dict[str, Any]], verdicts: Verdicts | None = None) -> dict[str, Any]:
    verdicts = verdicts or {}
    answerable = [r for r in rows if r["expected"] is not None]
    no_answer = [r for r in rows if r["expected"] is None]
    right = sum(r["outcome"] in ("right", "right_message") for r in answerable) / max(
        1, len(answerable)
    )
    false = sum(_is_false(r, verdicts) for r in no_answer) / max(1, len(no_answer))
    return {
        "answerable": len(answerable),
        "no_answer": len(no_answer),
        "right_share": round(right, 3),
        "false_share": round(false, 3),
        "to_review": sum(
            r["outcome"] == "review" and verdict(r, verdicts) is None for r in no_answer
        ),
        "meets_bar": right >= 0.95 and false <= 0.05,
    }


def questions() -> list[dict[str, Any]]:
    mine = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    user = json.loads((HERE / "user_questions.json").read_text(encoding="utf-8"))
    return [{**r, "author": "claude"} for r in mine] + [{**r, "author": "user"} for r in user]


async def _ask(model: Any, agent: Any, question: str) -> tuple[list[tuple[str, str]], str]:
    from livekit.agents import llm

    from praxis_telefon.agent import call_language
    from praxis_telefon.agent.prompts import LANGUAGE_NOTES

    ctx = llm.ChatContext()
    ctx.add_message(role="system", content=agent.instructions)
    ctx.add_message(role="user", content=question)
    state = call_language.from_texts([question])
    if state.english_heard:
        ctx.add_message(role="system", content=LANGUAGE_NOTES[state.language])
    calls: list[tuple[str, str]] = []
    text: list[str] = []
    async with model.chat(chat_ctx=ctx, tools=list(agent.tools)) as stream:
        async for chunk in stream:
            if chunk.delta is None:
                continue
            text.append(chunk.delta.content or "")
            for c in chunk.delta.tool_calls:
                args = json.loads(c.arguments or "{}")
                calls.append((c.name, str(args.get("topic", ""))))
    return calls, "".join(text).strip()


async def run() -> list[dict[str, Any]]:
    from praxis_telefon.agent import main
    from praxis_telefon.agent.receptionist import Receptionist
    from praxis_telefon.agent.tool_service import ToolService
    from praxis_telefon.config import Settings
    from praxis_telefon.db.schema import connect, init_db

    settings = Settings.from_env()
    conn = connect(":memory:")
    init_db(conn)
    agent = Receptionist(ToolService(conn, main.practice_now))
    model = main.build_llm(settings)
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        async with gate:
            calls, text = await _ask(model, agent, row["q"])
        return {**row, "calls": calls, "text": text,
                "outcome": outcome(row["expected"], calls, text)}  # fmt: skip

    async with model:
        return list(await asyncio.gather(*(one(r) for r in questions())))


def rescore(saved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Saved replies scored again with the current labels and outcome rules."""
    labels = {(r["author"], r["q"]): r["expected"] for r in questions()}
    out = []
    for r in saved:
        expected = labels[(r["author"], r["q"])]
        calls = [tuple(c) for c in r["calls"]]
        out.append({**r, "expected": expected,
                    "outcome": outcome(expected, calls, r.get("text", ""))})  # fmt: skip
    return out


def report(rows: list[dict[str, Any]], verdicts: Verdicts) -> str:
    lines = []
    for author in ("claude", "user"):
        sel = [r for r in rows if r["author"] == author]
        lines.append(f"{author}: {summary(sel, verdicts)}")
        for lang in ("de", "en"):
            part = [r for r in sel if r["lang"] == lang]
            lines.append(f"  {lang}: {summary(part, verdicts)}")
    lines.append(f"all: {summary(rows, verdicts)}")
    lines.append("\nNot right:")
    for r in rows:
        if r["outcome"] not in ("right", "right_message", "review"):
            lines.append(f"  {r['outcome']:12} [{r['author']}] {r['q']!r} "
                         f"expected {r['expected']}, calls {r['calls']}")  # fmt: skip
    lines.append("\nText replies without a tool call (review; false until marked ok):")
    for r in rows:
        if r["outcome"] == "review":
            seen = verdict(r, verdicts) or "NOT REVIEWED"
            lines.append(f"  [{seen}] [{r['author']}] {r['q']!r}\n      -> {r['text']!r}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    import os

    from dotenv import load_dotenv

    load_dotenv(".env")  # read in-process only; never printed
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    verdicts = json.loads(REVIEW.read_text(encoding="utf-8")) if REVIEW.exists() else {}
    if "--rescore" in argv:  # current labels, scoring and review.json; no LLM calls
        rows = rescore(
            [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines()]
        )
    else:
        rows = asyncio.run(run())
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(report(rows, verdicts))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
