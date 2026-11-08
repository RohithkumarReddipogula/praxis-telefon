"""M5 demo deployment, offline: keyed caller hash, call limits, greeting, seeding, image.

User decisions 2026-11-08: 5 minutes per call (a warning one minute before), 3 calls per
caller and day, 15 calls and 45 minutes per day overall, withheld numbers allowed and
counted against the overall limits; caller numbers only as a keyed hash (HMAC with a
deployment secret), since a plain hash of a phone number is easy to reverse.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from praxis_telefon.agent import demo
from praxis_telefon.config import ConfigError, Settings
from praxis_telefon.db.schema import PRACTICE_TZ
from tests.test_agent_options import ENV

KEY = "test-hash-key-0123456789abcdef"
ROOT = Path(__file__).parent.parent


# --- keyed caller hash ------------------------------------------------------------------


def test_caller_hash_is_keyed_and_stable() -> None:
    a = demo.caller_hash("+49 30 23125456", KEY)
    assert a == demo.caller_hash("+493023125456", KEY) == demo.caller_hash("030 23125456", KEY)
    assert a != demo.caller_hash("+493023125457", KEY)
    assert a != demo.caller_hash("+493023125456", "another-key-0123456789abcdef")


def test_caller_hash_is_not_a_plain_hash_and_shows_no_digits() -> None:
    import hashlib

    h = demo.caller_hash("+493023125456", KEY)
    assert h is not None
    assert h != hashlib.sha256(b"+493023125456").hexdigest()[: len(h)]
    assert "23125456" not in h and len(h) == 16


def test_withheld_number_has_no_hash() -> None:
    assert demo.caller_hash(None, KEY) is None
    assert demo.caller_hash("", KEY) is None
    assert demo.caller_hash("anonymous", KEY) is None


def test_demo_mode_needs_a_hash_key() -> None:
    with pytest.raises(ConfigError, match="DEMO_HASH_KEY"):
        Settings.from_env({**ENV, "DEMO_MODE": "1"})
    with pytest.raises(ConfigError, match="DEMO_HASH_KEY"):
        Settings.from_env({**ENV, "DEMO_MODE": "1", "DEMO_HASH_KEY": "short"})


def test_hash_key_is_not_in_the_settings_repr() -> None:
    s = Settings.from_env({**ENV, "DEMO_MODE": "1", "DEMO_HASH_KEY": KEY})
    assert KEY not in repr(s)


def test_demo_settings_defaults() -> None:
    s = Settings.from_env({**ENV, "DEMO_MODE": "1", "DEMO_HASH_KEY": KEY})
    lim = s.demo_limits
    assert (
        lim.max_call_s,
        lim.warn_before_s,
        lim.calls_per_caller,
        lim.calls_per_day,
        lim.minutes_per_day,
    ) == (300, 60, 3, 15, 45)
    assert Settings.from_env(ENV).demo_mode is False


# --- call limits ------------------------------------------------------------------------

DAY = datetime(2026, 11, 9, 10, 0, tzinfo=PRACTICE_TZ)


@pytest.fixture
def limiter(tmp_path: Path) -> demo.CallLimiter:
    return demo.CallLimiter(tmp_path / "usage.db", demo.DemoLimits())


def test_three_calls_per_caller_and_day(limiter: demo.CallLimiter) -> None:
    for _ in range(3):
        call = limiter.start("caller-a", DAY)
        assert call.allowed
        limiter.end(call, 60)
    refused = limiter.start("caller-a", DAY)
    assert not refused.allowed and refused.reason == "caller_daily_limit"
    assert limiter.start("caller-b", DAY).allowed


def test_fifteen_calls_per_day_overall(limiter: demo.CallLimiter) -> None:
    for i in range(15):
        call = limiter.start(f"caller-{i}", DAY)
        assert call.allowed
        limiter.end(call, 10)
    refused = limiter.start("caller-new", DAY)
    assert not refused.allowed and refused.reason == "daily_calls"


def test_withheld_numbers_count_against_the_overall_limit(limiter: demo.CallLimiter) -> None:
    for _ in range(15):
        call = limiter.start(None, DAY)
        assert call.allowed  # no per-caller limit without a number
        limiter.end(call, 10)
    assert limiter.start(None, DAY).reason == "daily_calls"


def test_minutes_per_day(limiter: demo.CallLimiter) -> None:
    for i in range(9):  # 9 x 5 min = 45 min
        call = limiter.start(f"c{i}", DAY)
        limiter.end(call, 300)
    refused = limiter.start("c-new", DAY)
    assert not refused.allowed and refused.reason == "daily_minutes"


def test_last_call_of_the_day_is_shortened_to_the_remaining_minutes(
    limiter: demo.CallLimiter,
) -> None:
    for i in range(8):
        limiter.end(limiter.start(f"c{i}", DAY), 300)  # 40 min used, 5 min left
    limiter.end(limiter.start("c8", DAY), 180)  # 43 min used, 2 min left
    call = limiter.start("c9", DAY)
    assert call.allowed and call.max_s == 120


def test_too_little_left_for_a_call_is_refused(limiter: demo.CallLimiter) -> None:
    for i in range(8):
        limiter.end(limiter.start(f"c{i}", DAY), 300)
    limiter.end(limiter.start("c8", DAY), 270)  # 30 s left
    assert limiter.start("c9", DAY).reason == "daily_minutes"


def test_a_running_call_reserves_its_maximum(limiter: demo.CallLimiter) -> None:
    # Concurrent calls must not overrun the daily minutes: a call counts with its
    # maximum until it ends.
    for i in range(9):
        assert limiter.start(f"c{i}", DAY).allowed  # 9 running calls x 5 min = 45 min
    assert limiter.start("c9", DAY).reason == "daily_minutes"


def test_limits_reset_at_berlin_midnight(limiter: demo.CallLimiter) -> None:
    for _ in range(3):
        limiter.end(limiter.start("caller-a", DAY), 60)
    late = DAY.replace(hour=23, minute=59)
    assert limiter.start("caller-a", late).reason == "caller_daily_limit"
    next_day = (DAY + timedelta(days=1)).replace(hour=0, minute=1)
    assert limiter.start("caller-a", next_day).allowed


def test_usage_database_holds_no_phone_numbers(tmp_path: Path) -> None:
    path = tmp_path / "usage.db"
    lim = demo.CallLimiter(path, demo.DemoLimits())
    lim.end(lim.start(demo.caller_hash("+493023125456", KEY), DAY), 60)
    dump = "\n".join(sqlite3.connect(path).iterdump())
    assert "23125456" not in dump


def test_limiter_is_shared_between_processes_through_the_file(tmp_path: Path) -> None:
    # Each LiveKit job runs in its own process: two limiters on one file see each other.
    path = tmp_path / "usage.db"
    a = demo.CallLimiter(path, demo.DemoLimits())
    b = demo.CallLimiter(path, demo.DemoLimits())
    for _ in range(3):
        a.end(a.start("caller-a", DAY), 60)
    assert b.start("caller-a", DAY).reason == "caller_daily_limit"


def test_call_schedule() -> None:
    assert demo.call_schedule(300, 60) == (240, 300)
    assert demo.call_schedule(120, 60) == (60, 120)
    assert demo.call_schedule(50, 60) == (None, 50)  # too short for a warning


# --- greeting and fixed messages --------------------------------------------------------


def test_demo_greeting_says_it_is_a_demo_in_both_languages() -> None:
    g = demo.DEMO_GREETING
    assert "Demo" in g and "erfunden" in g and "keine echten" in g
    assert "demo" in g.lower() and "fictional" in g


def test_refusal_and_time_messages_exist_in_both_languages() -> None:
    for key in ("caller_daily_limit", "daily_calls", "daily_minutes", "warn", "end"):
        text = demo.MESSAGES[key]
        assert any(w in text for w in ("Demo", "demo")), key
        assert any(w in text for w in ("This", "this", "The demo", "Thank", "Sorry")), key


# --- seeding at start -------------------------------------------------------------------


def test_database_is_seeded_at_start_when_asked(tmp_path: Path) -> None:
    from praxis_telefon.agent.main import ensure_database

    path = tmp_path / "praxis.db"
    s = Settings.from_env({**ENV, "DATABASE_PATH": str(path), "SEED_ON_START": "1"})
    ensure_database(s)
    rows = sqlite3.connect(path).execute("SELECT count(*) FROM slots").fetchone()[0]
    assert rows > 0


def test_existing_database_is_not_reseeded(tmp_path: Path) -> None:
    from praxis_telefon.agent.main import ensure_database

    path = tmp_path / "praxis.db"
    s = Settings.from_env({**ENV, "DATABASE_PATH": str(path), "SEED_ON_START": "1"})
    ensure_database(s)
    sqlite3.connect(path).execute("DELETE FROM bookings").connection.commit()
    before = path.stat().st_mtime_ns
    ensure_database(s)
    assert path.stat().st_mtime_ns == before


def test_without_seed_on_start_a_missing_database_is_an_error(tmp_path: Path) -> None:
    from praxis_telefon.agent.main import DatabaseMissing, ensure_database

    s = Settings.from_env({**ENV, "DATABASE_PATH": str(tmp_path / "missing.db")})
    with pytest.raises(DatabaseMissing):
        ensure_database(s)


# --- container image --------------------------------------------------------------------


def test_dockerignore_keeps_secrets_and_local_data_out() -> None:
    lines = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    for entry in (".env", ".env.*", "eval/results", "data", ".venv", ".git"):
        assert entry in lines, entry
    assert "!.env.example" not in lines  # not even the example goes into the image


def test_dockerfile_never_copies_env_and_runs_the_agent() -> None:
    text = (ROOT / "Dockerfile").read_text()
    assert ".env" not in text
    assert "download-files" in text  # VAD and turn-detector weights baked into the image
    assert "uv sync --frozen --no-dev" in text
    assert '"start"' in text  # production mode, not dev/console
    assert "SEED_ON_START=1" in text and "DEMO_MODE=1" in text


def test_dockerfile_base_image_is_the_livekit_template_one() -> None:
    # Bug, first deployment 2026-11-08: "ghcr.io/astral-sh/uv:0.13.5-python3.12-bookworm-slim:
    # not found". LiveKit's own template (lk agent dockerfile) uses this tag.
    text = (ROOT / "Dockerfile").read_text()
    assert "FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim" in text
    assert "0.13.5-python" not in text


def test_model_weights_are_downloaded_as_the_runtime_user() -> None:
    # Found with the first deployment: download-files ran as root, so the turn-detector
    # weights went to root's cache while the agent runs as "agent" and reads its own.
    lines = [line.strip() for line in (ROOT / "Dockerfile").read_text().splitlines()]
    user = next(i for i, line in enumerate(lines) if line.startswith("USER agent"))
    download = next(i for i, line in enumerate(lines) if "download-files" in line)
    sync = next(i for i, line in enumerate(lines) if line.startswith("RUN uv sync"))
    assert user < sync < download
