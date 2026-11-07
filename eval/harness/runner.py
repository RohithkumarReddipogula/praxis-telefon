"""Run scenarios in text mode, N times each, with a hard cost cap. Costs LLM tokens.

    uv run python -m eval.harness.runner --estimate            # free: what would run
    uv run python -m eval.harness.runner                       # all, N=5, cap $1.00
    uv run python -m eval.harness.runner --n 2 --max-usd 0.3 --only book_next_tuesday

Each conversation runs the real Receptionist and LLM in LiveKit's text mode (as the
live tests do, tests/live/harness.py) against a fresh in-memory synthetic database, in
one of WORKERS processes (the live harness keeps its measurements per process). After
every conversation its LLM cost is added; when the cap is reached, no new conversation
starts, running ones are stopped, and the run is reported as stopped at the cap.

Output: eval/results/runs/<timestamp>/ with meta.json (git commit, model, settings,
prompt hash, cap, spent) and conversations.jsonl (one record per conversation:
transcript, tool calls, database afterwards, blocked reasons, timings, cost, metrics).
``uv run python -m eval.harness.report <run dir>`` summarises it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.harness.scenario import SILENCE, Scenario, load_all

SCENARIOS = Path(__file__).parent.parent / "scenarios"
RUNS_DIR = Path("eval/results/runs")
WORKERS = 4
DEFAULT_N = 5
DEFAULT_MAX_USD = 1.0
EST_USD_PER_CONVERSATION = 0.002  # M3: ~$0.0002 per cached LLM request, ~8 per call


@dataclass
class CostCap:
    """Stops a run once the cost reaches ``max_usd``. An unknown cost stops it too."""

    max_usd: float
    spent: float = 0.0
    reached: bool = False

    def add(self, usd: float | None) -> bool:
        if usd is None:
            self.reached = True
            return True
        self.spent += usd
        if self.spent >= self.max_usd:
            self.reached = True
        return self.reached


# --- one conversation (in a worker process) ---------------------------------------------


def _snapshot(conn: Any, initial_booking_ids: set[int]) -> dict[str, Any]:
    from praxis_telefon.db.schema import PRACTICE_TZ

    bookings = []
    for row in conn.execute(
        "SELECT b.id, b.status, s.start_utc, d.name, p.full_name FROM bookings b "
        "JOIN slots s ON s.id = b.slot_id JOIN doctors d ON d.id = s.doctor_id "
        "JOIN patients p ON p.id = b.patient_id"
    ):
        start = datetime.fromisoformat(row[2]).astimezone(PRACTICE_TZ)
        bookings.append(
            {
                "id": row[0],
                "status": row[1],
                "day": start.date().isoformat(),
                "hour": start.hour,
                "doctor": row[3],
                "patient": row[4],
                "new": row[0] not in initial_booking_ids,
            }
        )
    messages = [
        {"number": r[0], "reasons": r[1].split(",")}
        for r in conn.execute("SELECT callback_number, reason_category FROM staff_messages")
    ]
    return {"bookings": bookings, "messages": messages}


def run_conversation(scenario_id: str, run: int) -> dict[str, Any]:
    """One scripted call; returns its record (never raises: errors are recorded)."""
    import asyncio
    import os

    os.environ.setdefault("LOG_LEVEL", "WARNING")
    from dotenv import load_dotenv

    load_dotenv(".env")  # read in-process only; never printed
    from livekit.agents import AgentSession, llm
    from tests.conftest import NOW
    from tests.live import harness as h
    from tests.test_speech_filters import FRIDAY_10

    from eval.harness import callers, checks
    from eval.leak_report import cost_usd
    from praxis_telefon.agent import claim_gate, main
    from praxis_telefon.agent.claim_check import ClaimTracker
    from praxis_telefon.agent.tool_service import ToolService
    from praxis_telefon.config import Settings

    s = next(x for x in load_all(SCENARIOS) if x.id == scenario_id)
    h.clear_measurements()
    claim_gate.SENTENCE_HOOK = h._collect
    mark = len(h.safety_records())
    conn, _ = h.new_db(erika_booked=s.erika_booked)
    initial = {r[0] for r in conn.execute("SELECT id FROM bookings")}
    clock = (lambda: FRIDAY_10) if s.clock == "friday_10" else (lambda: NOW)
    settings = Settings.from_env()
    call = h.Call()
    state = callers.CallerState()
    record: dict[str, Any] = {"scenario": s.id, "run": run, "error": None}
    started = time.perf_counter()

    async def conversation() -> Any:
        model = main.build_llm(settings)
        h._measure_llm(model)
        async with model, AgentSession(llm=model) as session:
            agent = h.agent_class()(ToolService(conn, clock))
            await session.start(agent)

            async def say(text: str) -> None:
                first, t0 = len(h.GENERATIONS), time.perf_counter()
                result = await asyncio.wait_for(
                    h._wait(session.run(user_input=text)), h.TURN_TIMEOUT_S
                )
                h.TURNS.append(h.turn_record(h.GENERATIONS[first:], t0))
                call.events.extend(result.events)

            for line in s.caller.lines:
                if line == SILENCE:
                    handle = agent.on_unrecognized_speech()
                    if handle is not None:
                        await handle
                else:
                    await say(line)
            seen_slots = -1
            for _ in range(s.caller.max_replies if s.caller.until else 0):
                if all(call.succeeded(tool) for tool in s.caller.until):
                    break
                slots = call.last_slots_index()
                text = call.last_agent_text()
                new_slots = slots > seen_slots
                line = callers.reply(
                    s.caller.behaviour,
                    text,
                    state,
                    identity=_identity(s),
                    language=s.caller.language,
                    slots=new_slots,
                )
                if new_slots and line != callers.REPLIES["other_day"][_lang(s)]:
                    seen_slots = slots
                await say(line)
            return session.history

    try:
        history = asyncio.run(conversation())
    except Exception as e:  # recorded, the run goes on
        record["error"] = f"{type(e).__name__}: {e}"[:300]
        history = None

    # Everything from the session history, including replies to silences (strikes),
    # which are not in the per-line run results.
    agent_msgs: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_ok: set[str] = set()
    tracker = ClaimTracker()
    unbacked: list[str] = []
    turn = -1
    for item in history.items if history is not None else []:
        if isinstance(item, llm.ChatMessage) and item.role == "user":
            turn += 1
        elif isinstance(item, llm.ChatMessage) and item.role == "assistant":
            text = item.text_content or ""
            agent_msgs.append({"text": text, "turn": max(turn, 0)})
            unbacked += [f"{a}: {text}" for a in tracker.unbacked(text)]
        elif isinstance(item, llm.FunctionCall):
            tool_calls.append({"name": item.name, "args": json.loads(item.arguments or "{}")})
        elif isinstance(item, llm.FunctionCallOutput):
            tracker.tool_result(item.name, ok=not item.is_error)
            if not item.is_error:
                tool_ok.add(item.name)
    safety = h.safety_records()[mark:]
    record.update(
        _snapshot(conn, initial),
        agent=agent_msgs,
        tool_calls=tool_calls,
        tool_ok=sorted(tool_ok),
        blocked=[r["reason"] for r in safety if r.get("event") == "speech_blocked"],
        safety=[r["action"] for r in safety if r.get("event") == "safety"],
        unbacked=unbacked,
        caller_lines=len(h.TURNS),  # caller lines spoken (silences are not turns)
        turns=[h.public(t) for t in h.TURNS],
        # Every sentence the LLM produced, before filtering, per reply (live-corpus format):
        # which sentence a block reason belongs to (M4 rerun: a meta:tool block whose
        # sentence was not kept).
        sentence_groups=h.corpus_groups(),
        llm_usage=list(h.LLM_USAGE),
        cost_usd=cost_usd(h.LLM_USAGE, settings.llm_model),
        seconds=round(time.perf_counter() - started, 1),
    )
    record["metrics"] = checks.evaluate(s, record) if record["error"] is None else None
    return record


def _identity(s: Scenario) -> str:
    from eval.harness.scenario import IDENTITIES

    return IDENTITIES[s.identity]


def _lang(s: Scenario) -> str:
    return s.caller.language if s.caller.language in ("de", "en") else "de"


def _task(args: tuple[str, int]) -> dict[str, Any]:
    return run_conversation(*args)


# --- the run ------------------------------------------------------------------------------


def _meta(n: int, max_usd: float) -> dict[str, Any]:
    from praxis_telefon.agent import main, prompts
    from praxis_telefon.config import Settings

    settings = Settings.from_env()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        ).stdout.strip()
    )
    return {
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit + ("+dirty" if dirty else ""),
        "model": settings.llm_model,
        "reasoning_effort": settings.llm_reasoning_effort,
        "prompt_sha": hashlib.sha256(prompts.SYSTEM_PROMPT.encode()).hexdigest()[:12],
        "interruption": dict(main.INTERRUPTION),
        "endpointing": dict(main.ENDPOINTING),
        "n": n,
        "max_usd": max_usd,
    }


def plan(scenarios: list[Scenario], n: int) -> list[tuple[str, int]]:
    """Run k of every scenario before run k+1, so a stop at the cap leaves even coverage."""
    return [(s.id, run) for run in range(n) for s in scenarios]


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--only", default="", help="comma-separated scenario ids")
    p.add_argument("--tag", default="", help="only scenarios with this tag")
    p.add_argument("--estimate", action="store_true")
    a = p.parse_args(argv)

    scenarios = load_all(SCENARIOS)
    if a.only:
        scenarios = [s for s in scenarios if s.id in set(a.only.split(","))]
    if a.tag:
        scenarios = [s for s in scenarios if a.tag in s.tags]
    tasks = plan(scenarios, a.n)
    print(
        f"{len(scenarios)} scenarios x {a.n} = {len(tasks)} conversations, "
        f"estimate ${len(tasks) * EST_USD_PER_CONVERSATION:.2f}, cap ${a.max_usd:.2f}"
    )
    if a.estimate or not tasks:
        return 0

    from dotenv import load_dotenv

    load_dotenv(".env")  # read in-process only; never printed
    out = RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    meta = _meta(a.n, a.max_usd)
    cap = CostCap(a.max_usd)
    done = 0
    with (
        mp.get_context("spawn").Pool(a.workers) as pool,
        (out / "conversations.jsonl").open("w", encoding="utf-8") as f,
    ):
        for record in pool.imap_unordered(_task, tasks):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            stop = cap.add(record["cost_usd"])
            if done % 10 == 0 or stop:
                print(f"  {done}/{len(tasks)} conversations, ${cap.spent:.3f}")
            if stop:
                print(f"cost cap reached (${cap.spent:.3f} >= ${a.max_usd:.2f}): stopping")
                pool.terminate()
                break
    meta.update(
        spent_usd=round(cap.spent, 4),
        conversations=done,
        planned=len(tasks),
        stopped_at_cap=cap.reached,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
