"""Web demo, agent side (M5/M6 item 2, step 1): channel-aware limits and the busy signal.

User decisions 2026-11-09:
- browser calls have their own limits, separate from the phone limits: 20 minutes a day
  (5 minutes per call, so at most 4 calls) and a monthly cap of 60 minutes;
- when a limit is reached the page shows "the demo is busy today"; a silent mid-call
  failure must never happen, also not when the speech synthesis (TTS) fails.
The agent tells the page through a participant attribute (``demo_status=busy``) that it
sets first, before it tries to speak, because speaking may be exactly what fails.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from livekit.agents import AgentSession
from livekit.agents.tts import TTSError
from livekit.agents.voice.events import ErrorEvent

from praxis_telefon.agent import demo
from praxis_telefon.config import ConfigError, Settings
from praxis_telefon.db.schema import PRACTICE_TZ
from tests.test_agent_options import ENV

DAY = datetime(2026, 11, 9, 10, 0, tzinfo=PRACTICE_TZ)
KEY = "test-hash-key-0123456789abcdef"


@pytest.fixture
def limiter(tmp_path: Path) -> demo.CallLimiter:
    return demo.CallLimiter(tmp_path / "usage.db", demo.DemoLimits())


# --- channel ---------------------------------------------------------------------------


def test_channel_is_web_only_with_the_signed_attribute() -> None:
    assert demo.channel_of({"channel": "web"}) == "web"
    assert demo.channel_of({}) == "phone"
    assert demo.channel_of({"sip.phoneNumber": "+493023125456"}) == "phone"
    assert demo.channel_of({"channel": "phone"}) == "phone"
    assert demo.channel_of({"channel": "WEB "}) == "phone"  # exact value only


# --- web daily limit --------------------------------------------------------------------


def test_four_web_calls_of_five_minutes_fill_the_twenty_minutes(
    limiter: demo.CallLimiter,
) -> None:
    for _ in range(4):
        call = limiter.start(None, DAY, "web")
        assert call.allowed and call.max_s == 300
        limiter.end(call, 300)
    refused = limiter.start(None, DAY, "web")
    assert not refused.allowed and refused.reason == "web_daily_minutes"


def test_running_web_calls_reserve_their_maximum(limiter: demo.CallLimiter) -> None:
    for _ in range(4):
        assert limiter.start(None, DAY, "web").allowed  # four at once, none ended
    assert limiter.start(None, DAY, "web").reason == "web_daily_minutes"


def test_a_short_web_call_gives_its_minutes_back(limiter: demo.CallLimiter) -> None:
    calls = [limiter.start(None, DAY, "web") for _ in range(4)]
    limiter.end(calls[0], 60)  # hung up after a minute
    again = limiter.start(None, DAY, "web")
    assert again.allowed and again.max_s == 240  # 20 min - 3 x 5 min - 1 min = 4 min left


def test_last_web_call_of_the_day_is_shortened_and_too_little_is_refused(
    tmp_path: Path,
) -> None:
    def used(seconds_of_last_call: int) -> demo.CallLimiter:
        lim = demo.CallLimiter(tmp_path / f"u{seconds_of_last_call}.db", demo.DemoLimits())
        for _ in range(3):
            lim.end(lim.start(None, DAY, "web"), 300)
        lim.end(lim.start(None, DAY, "web"), seconds_of_last_call)
        return lim

    one_minute_left = used(240).start(None, DAY, "web")  # 19 of 20 minutes used
    assert one_minute_left.allowed and one_minute_left.max_s == 60
    assert used(250).start(None, DAY, "web").reason == "web_daily_minutes"  # 50 s left


def test_web_calls_have_no_per_caller_limit(limiter: demo.CallLimiter) -> None:
    # The page limits visitors (keyed IP hash); the agent only counts minutes.
    for _ in range(4):
        assert limiter.start(None, DAY, "web").allowed


# --- web and phone are separate ---------------------------------------------------------


def test_web_calls_do_not_use_the_phone_limits(limiter: demo.CallLimiter) -> None:
    for _ in range(4):
        limiter.end(limiter.start(None, DAY, "web"), 300)
    for i in range(15):  # the full phone allowance is still there
        call = limiter.start(f"caller-{i}", DAY, "phone")
        assert call.allowed
        limiter.end(call, 10)
    assert limiter.start("caller-new", DAY, "phone").reason == "daily_calls"


def test_phone_calls_do_not_use_the_web_minutes(limiter: demo.CallLimiter) -> None:
    for i in range(9):  # all 45 phone minutes
        limiter.end(limiter.start(f"c{i}", DAY, "phone"), 300)
    assert limiter.start("c-new", DAY, "phone").reason == "daily_minutes"
    assert limiter.start(None, DAY, "web").allowed


# --- monthly cap ------------------------------------------------------------------------


def _use_web_minutes(limiter: demo.CallLimiter, day: datetime, calls: int) -> None:
    for _ in range(calls):
        limiter.end(limiter.start(None, day, "web"), 300)


def test_monthly_cap_of_sixty_minutes_stops_the_web_demo(limiter: demo.CallLimiter) -> None:
    for d in range(3):  # 3 days x 4 calls x 5 min = 60 min
        _use_web_minutes(limiter, DAY + timedelta(days=d), 4)
    fourth_day = DAY + timedelta(days=3)
    refused = limiter.start(None, fourth_day, "web")
    assert not refused.allowed and refused.reason == "web_monthly_minutes"


def test_the_monthly_cap_wins_over_a_daily_allowance(limiter: demo.CallLimiter) -> None:
    for d in range(2):
        _use_web_minutes(limiter, DAY + timedelta(days=d), 4)  # 40 min
    _use_web_minutes(limiter, DAY + timedelta(days=2), 3)  # 55 min
    day4 = DAY + timedelta(days=3)  # a fresh day: daily allowance is 20 min
    call = limiter.start(None, day4, "web")
    assert call.allowed and call.max_s == 300  # but only 5 minutes of the month are left
    limiter.end(call, 290)
    assert limiter.start(None, day4, "web").reason == "web_monthly_minutes"


def test_last_call_of_the_month_budget_is_shortened(limiter: demo.CallLimiter) -> None:
    for d in range(2):
        _use_web_minutes(limiter, DAY + timedelta(days=d), 4)
    limiter.end(limiter.start(None, DAY + timedelta(days=2), "web"), 300)  # 45 min
    for _ in range(2):
        limiter.end(limiter.start(None, DAY + timedelta(days=2), "web"), 300)  # 55 min
    limiter.end(limiter.start(None, DAY + timedelta(days=2), "web"), 180)  # 58 min
    call = limiter.start(None, DAY + timedelta(days=3), "web")
    assert call.allowed and call.max_s == 120


def test_the_monthly_cap_resets_with_the_month(limiter: demo.CallLimiter) -> None:
    for d in range(3):
        _use_web_minutes(limiter, DAY + timedelta(days=d), 4)
    december = datetime(2026, 12, 1, 9, 0, tzinfo=PRACTICE_TZ)
    assert limiter.start(None, december, "web").allowed


def test_month_boundary_follows_berlin_time(limiter: demo.CallLimiter) -> None:
    for d in range(3):
        _use_web_minutes(limiter, datetime(2026, 11, 26 + d, 12, 0, tzinfo=PRACTICE_TZ), 4)
    late_nov = datetime(2026, 11, 30, 23, 30, tzinfo=PRACTICE_TZ)
    assert limiter.start(None, late_nov, "web").reason == "web_monthly_minutes"
    after_midnight = datetime(2026, 12, 1, 0, 30, tzinfo=PRACTICE_TZ)
    assert limiter.start(None, after_midnight, "web").allowed


# --- storage ----------------------------------------------------------------------------


def test_an_older_usage_file_without_channel_keeps_working(tmp_path: Path) -> None:
    # The container's file is recreated on restart, but a file from before this change
    # (no channel column) must not break the agent: its rows count as phone calls.
    path = tmp_path / "usage.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE calls (id INTEGER PRIMARY KEY, day TEXT NOT NULL, caller TEXT, "
        "max_s INTEGER NOT NULL, duration_s INTEGER)"
    )
    old.execute(
        "INSERT INTO calls (day, caller, max_s, duration_s) VALUES ('2026-11-09', 'a', 300, 60)"
    )
    old.commit()
    old.close()
    lim = demo.CallLimiter(path, demo.DemoLimits())
    assert lim.start("a", DAY, "phone").allowed
    assert lim.start(None, DAY, "web").allowed


def test_the_web_channel_is_stored_without_a_caller(tmp_path: Path) -> None:
    path = tmp_path / "usage.db"
    lim = demo.CallLimiter(path, demo.DemoLimits())
    lim.start(None, DAY, "web")
    rows = sqlite3.connect(path).execute("SELECT channel, caller FROM calls").fetchall()
    assert rows == [("web", None)]


# --- settings ---------------------------------------------------------------------------


def test_web_limit_defaults_and_overrides() -> None:
    lim = Settings.from_env({**ENV, "DEMO_MODE": "1", "DEMO_HASH_KEY": KEY}).demo_limits
    assert (lim.web_minutes_per_day, lim.web_minutes_per_month) == (20, 60)
    custom = Settings.from_env(
        {**ENV, "DEMO_MODE": "1", "DEMO_HASH_KEY": KEY, "DEMO_WEB_MINUTES_PER_DAY": "10",
         "DEMO_WEB_MINUTES_PER_MONTH": "30"}
    ).demo_limits  # fmt: skip
    assert (custom.web_minutes_per_day, custom.web_minutes_per_month) == (10, 30)


def test_web_limits_must_be_positive_numbers() -> None:
    with pytest.raises(ConfigError, match="DEMO_WEB_MINUTES_PER_MONTH"):
        Settings.from_env(
            {**ENV, "DEMO_MODE": "1", "DEMO_HASH_KEY": KEY, "DEMO_WEB_MINUTES_PER_MONTH": "0"}
        )


# --- what the page and the caller get ---------------------------------------------------


def test_busy_messages_exist_in_both_languages() -> None:
    for reason in ("web_daily_minutes", "web_monthly_minutes"):
        text = demo.MESSAGES[reason]
        assert "Demo" in text and "demo" in text
    assert "diesem Monat" in demo.MESSAGES["web_monthly_minutes"]
    assert "this month" in demo.MESSAGES["web_monthly_minutes"]


class FakeRoom:
    def __init__(self) -> None:
        self.attributes: dict[str, str] = {}
        self.log: list[str] = []

    async def set_attributes(self, attributes: dict[str, str]) -> None:
        self.attributes.update(attributes)
        self.log.append("attributes")


class FakeCtx:
    def __init__(self, room: FakeRoom) -> None:
        self.room = type("R", (), {"local_participant": room})()
        self.deleted = False
        self._log = room.log

    async def delete_room(self) -> None:
        self.deleted = True
        self._log.append("delete_room")


class Handle:
    """What AgentSession.say returns: awaitable, and not awaited for the greeting."""

    def __init__(self, fails: bool) -> None:
        self._fails = fails

    def __await__(self) -> Any:
        if self._fails:
            raise RuntimeError("tts down")
        return iter(())


class FakeSession:
    def __init__(self, log: list[str], *, fails: bool = False) -> None:
        self.said: list[str] = []
        self._log = log
        self._fails = fails

    def say(self, text: str, **_: Any) -> Handle:
        self._log.append("say")
        self.said.append(text)
        return Handle(self._fails)


def test_busy_sets_the_page_signal_before_speaking_and_hangs_up() -> None:
    room = FakeRoom()
    ctx, session = FakeCtx(room), FakeSession(room.log)
    asyncio.run(demo.tell_busy(ctx, session, "web_monthly_minutes"))
    assert room.log == ["attributes", "say", "delete_room"]
    assert room.attributes == {"demo_status": "busy", "demo_reason": "web_monthly_minutes"}
    assert session.said == [demo.MESSAGES["web_monthly_minutes"]]


def test_a_failing_voice_never_stops_the_signal_or_the_hang_up() -> None:
    room = FakeRoom()
    ctx, session = FakeCtx(room), FakeSession(room.log, fails=True)
    asyncio.run(demo.tell_busy(ctx, session, "web_daily_minutes"))
    assert room.attributes["demo_status"] == "busy" and ctx.deleted
    assert "say" in room.log  # it tried to speak; the failure changed nothing else


def test_busy_without_speaking_for_a_tts_failure() -> None:
    room = FakeRoom()
    ctx, session = FakeCtx(room), FakeSession(room.log)
    asyncio.run(demo.tell_busy(ctx, session, "tts_unavailable", speak=False))
    assert room.log == ["attributes", "delete_room"] and session.said == []
    assert room.attributes["demo_reason"] == "tts_unavailable"


# --- a failing voice is noticed ---------------------------------------------------------


def _error(recoverable: bool, kind: str = "tts_error") -> ErrorEvent:
    if kind == "tts_error":
        error: Any = TTSError(
            timestamp=0.0, label="elevenlabs", error=Exception("quota"), recoverable=recoverable
        )
    else:
        error = type("E", (), {"type": kind, "recoverable": recoverable})()
    return ErrorEvent(error=error, source=None)


def test_a_final_tts_error_triggers_the_busy_path_once() -> None:
    async def scenario() -> list[int]:
        session: AgentSession[None] = AgentSession(loop=asyncio.get_running_loop())
        hits: list[int] = []
        demo.guard_tts(session, lambda: hits.append(1))
        session.emit("error", _error(recoverable=True))  # LiveKit retries these
        session.emit("error", _error(recoverable=False, kind="llm_error"))
        assert hits == []
        session.emit("error", _error(recoverable=False))
        session.emit("error", _error(recoverable=False))
        return hits

    assert asyncio.run(scenario()) == [1]


# --- the whole demo call, with fakes ----------------------------------------------------


class FakeParticipant:
    def __init__(self, attributes: dict[str, str]) -> None:
        self.attributes = attributes


class FakeJob(FakeCtx):
    def __init__(self, room: FakeRoom, attributes: dict[str, str]) -> None:
        super().__init__(room)
        self._participant = FakeParticipant(attributes)
        self.shutdown: list[Any] = []

    async def wait_for_participant(self) -> FakeParticipant:
        return self._participant

    def add_shutdown_callback(self, fn: Any) -> None:
        self.shutdown.append(fn)


class FakeCallSession(FakeSession):
    def __init__(self, log: list[str]) -> None:
        super().__init__(log)
        self.handlers: dict[str, Any] = {}

    def on(self, name: str, fn: Any = None) -> Any:
        if fn is not None:  # session.on("error", handler)
            self.handlers[name] = fn
            return fn

        def register(handler: Any) -> Any:  # @session.on("error")
            self.handlers[name] = handler
            return handler

        return register


def _run_call(tmp_path: Path, attributes: dict[str, str], **env: str) -> tuple[Any, ...]:
    from praxis_telefon.agent.main import run_demo_call

    settings = Settings.from_env(
        {
            **ENV,
            "DEMO_MODE": "1",
            "DEMO_HASH_KEY": KEY,
            "DEMO_USAGE_DB": str(tmp_path / "u.db"),
            **env,
        }
    )
    room = FakeRoom()
    ctx, session = FakeJob(room, attributes), FakeCallSession(room.log)

    async def go() -> None:
        await run_demo_call(cast(Any, ctx), cast(Any, session), settings)
        for fn in ctx.shutdown:  # the call ends: timers cancelled, usage recorded
            await fn()

    asyncio.run(go())
    return room, ctx, session, settings


def test_a_browser_call_is_counted_as_web(tmp_path: Path) -> None:
    room, ctx, session, settings = _run_call(tmp_path, {"channel": "web"})
    assert session.said == [demo.DEMO_GREETING]
    rows = (
        sqlite3.connect(settings.demo_usage_db)
        .execute("SELECT channel, caller FROM calls")
        .fetchall()
    )
    assert rows == [("web", None)] and room.attributes == {} and not ctx.deleted


def test_a_phone_call_is_still_counted_as_phone(tmp_path: Path) -> None:
    _, _, _, settings = _run_call(tmp_path, {"sip.phoneNumber": "+493023125456"})
    rows = (
        sqlite3.connect(settings.demo_usage_db)
        .execute("SELECT channel, caller IS NOT NULL FROM calls")
        .fetchall()
    )
    assert rows == [("phone", 1)]


def test_a_browser_call_over_the_monthly_cap_gets_the_busy_signal(tmp_path: Path) -> None:
    from praxis_telefon.agent.main import practice_now

    env = {"DEMO_WEB_MINUTES_PER_DAY": "5", "DEMO_WEB_MINUTES_PER_MONTH": "5"}
    settings = Settings.from_env({**ENV, "DEMO_MODE": "1", "DEMO_HASH_KEY": KEY, **env})
    lim = demo.CallLimiter(tmp_path / "u.db", settings.demo_limits)
    lim.end(lim.start(None, practice_now(), "web"), 300)  # the 5 minutes are gone
    room, ctx, session, _ = _run_call(tmp_path, {"channel": "web"}, **env)
    assert room.attributes == {"demo_status": "busy", "demo_reason": "web_monthly_minutes"}
    assert session.said == [demo.MESSAGES["web_monthly_minutes"]] and ctx.deleted


def test_a_final_tts_error_during_a_call_ends_it_with_the_busy_signal(tmp_path: Path) -> None:
    from praxis_telefon.agent.main import run_demo_call

    settings = Settings.from_env(
        {**ENV, "DEMO_MODE": "1", "DEMO_HASH_KEY": KEY, "DEMO_USAGE_DB": str(tmp_path / "u.db")}
    )
    room = FakeRoom()
    ctx, session = FakeJob(room, {"channel": "web"}), FakeCallSession(room.log)

    async def go() -> None:
        await run_demo_call(cast(Any, ctx), cast(Any, session), settings)
        session.handlers["error"](_error(recoverable=False))
        await asyncio.sleep(0.05)  # the busy path runs as a task
        for fn in ctx.shutdown:
            await fn()

    asyncio.run(go())
    assert room.attributes == {"demo_status": "busy", "demo_reason": "tts_unavailable"}
    assert ctx.deleted


# --- a visitor who leaves before joining -------------------------------------------------


def test_a_participant_who_leaves_before_joining_ends_the_job_quietly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Cloud logs 2026-11-09 16:31: "RuntimeError: room disconnected while waiting for
    # participant" was logged as an unhandled exception (a console that left at once). With
    # a web page it happens whenever a visitor closes the tab: no call, no minutes.
    import json
    import logging

    from praxis_telefon.agent.main import run_demo_call

    class LeavesEarly(FakeJob):
        async def wait_for_participant(self) -> FakeParticipant:
            raise RuntimeError("room disconnected while waiting for participant")

    settings = Settings.from_env(
        {**ENV, "DEMO_MODE": "1", "DEMO_HASH_KEY": KEY, "DEMO_USAGE_DB": str(tmp_path / "u.db")}
    )
    room = FakeRoom()
    ctx, session = LeavesEarly(room, {}), FakeCallSession(room.log)
    with caplog.at_level(logging.INFO, logger="praxis_telefon.turns"):
        asyncio.run(run_demo_call(cast(Any, ctx), cast(Any, session), settings))
    records = [json.loads(r.getMessage()) for r in caplog.records]
    assert records == [{"event": "demo_no_participant"}]
    assert room.log == [] and not ctx.deleted  # nothing said, set or deleted
    assert (
        not (tmp_path / "u.db").exists()
        or not sqlite3.connect(tmp_path / "u.db")
        .execute("SELECT count(*) FROM calls")
        .fetchone()[0]
    )  # no call counted


def test_other_errors_while_waiting_are_not_swallowed(tmp_path: Path) -> None:
    from praxis_telefon.agent.main import run_demo_call

    class Broken(FakeJob):
        async def wait_for_participant(self) -> FakeParticipant:
            raise ValueError("a real bug")

    settings = Settings.from_env({**ENV, "DEMO_MODE": "1", "DEMO_HASH_KEY": KEY})
    room = FakeRoom()
    with pytest.raises(ValueError, match="a real bug"):
        asyncio.run(
            run_demo_call(
                cast(Any, Broken(room, {})), cast(Any, FakeCallSession(room.log)), settings
            )
        )
