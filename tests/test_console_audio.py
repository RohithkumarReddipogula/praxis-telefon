"""Console-mode audio must stop before the job's event loop closes.

Bug: stopping ``console`` with Ctrl+C printed "RuntimeError: Event loop is closed" from
the sounddevice callbacks. LiveKit's ``_audio_mode`` stops the streams only after its
loop ends normally (not in a ``finally``), so on Ctrl+C they kept running and posted
audio frames to the job's closed loop.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from praxis_telefon.agent import main
from praxis_telefon.agent.console_audio import release_console_audio, stop_console_audio


class FakeConsole:
    def __init__(self, *, enabled: bool = True, fail_mic: bool = False) -> None:
        self.enabled = enabled
        self.mic = True
        self.speaker = True
        self.fail_mic = fail_mic

    def set_microphone_enabled(self, enable: bool) -> None:
        if self.fail_mic:
            raise OSError("PortAudio error")
        self.mic = enable

    def set_speaker_enabled(self, enable: bool) -> None:
        self.speaker = enable


def test_ctrl_c_shutdown_stops_console_audio_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.agents.cli._legacy import AgentsConsole

    console = FakeConsole()
    monkeypatch.setattr(AgentsConsole, "get_instance", classmethod(lambda cls: console))

    # LiveKit passes the shutdown reason to callbacks that take an argument.
    asyncio.run(release_console_audio("user_initiated"))

    assert console.mic is False
    assert console.speaker is False


def test_console_audio_stop_does_not_block_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: "event loop blocked for 273ms" in sounddevice close at shutdown.
    from livekit.agents.cli._legacy import AgentsConsole

    stop_threads: list[int] = []

    class RecordingConsole(FakeConsole):
        def set_microphone_enabled(self, enable: bool) -> None:
            stop_threads.append(threading.get_ident())
            super().set_microphone_enabled(enable)

    console = RecordingConsole()
    monkeypatch.setattr(AgentsConsole, "get_instance", classmethod(lambda cls: console))

    async def shutdown() -> int:
        await release_console_audio("user_initiated")
        return threading.get_ident()

    loop_thread = asyncio.run(shutdown())
    assert stop_threads and stop_threads[0] != loop_thread
    assert console.mic is False and console.speaker is False


def test_entrypoint_registers_console_audio_release() -> None:
    assert "release_console_audio" in main.entrypoint.__code__.co_names


def test_dev_mode_leaves_console_alone() -> None:
    console = FakeConsole(enabled=False)
    stop_console_audio(console)
    assert console.mic is True and console.speaker is True


def test_failing_microphone_still_stops_speaker() -> None:
    console = FakeConsole(fail_mic=True)
    stop_console_audio(console)
    assert console.speaker is False


@pytest.mark.parametrize("console", [None, object()])
def test_missing_console_api_is_ignored(console: object) -> None:
    stop_console_audio(console)


def test_livekit_console_api_still_exists() -> None:
    """Fails loudly after a LiveKit upgrade that removes what the fix relies on."""
    from livekit.agents.cli._legacy import AgentsConsole

    assert callable(AgentsConsole.get_instance)
    assert isinstance(AgentsConsole.enabled, property)
    assert callable(AgentsConsole.set_microphone_enabled)
    assert callable(AgentsConsole.set_speaker_enabled)
