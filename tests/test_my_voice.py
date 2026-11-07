"""Offline checks of eval/my_voice.py (the user's own recordings; no API calls, no mic)."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

from eval import my_voice


def test_twelve_sentences_with_unique_ids() -> None:
    assert len(my_voice.SENTENCES) == 12
    assert len({s.id for s in my_voice.SENTENCES}) == 12


def test_recordings_are_ignored_by_git() -> None:
    # The user's voice must never be committed (CLAUDE.md rule 3: no real recordings).
    probe = my_voice.OUT / "probe.wav"
    result = subprocess.run(["git", "check-ignore", "-q", str(probe)], check=False)
    assert result.returncode == 0, f"{my_voice.OUT} is not gitignored"


def test_wav_is_16k_mono_16bit(tmp_path: Path) -> None:
    path = tmp_path / "x.wav"
    my_voice.write_wav(path, b"\x01\x00" * 1600)
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16_000, 1, 2)
        assert w.getnframes() == 1600


def test_what_the_code_makes_of_a_transcript() -> None:
    # The report shows, per transcript, whether our request patterns and detector see it.
    seen = my_voice.code_view("Können wir Englisch sprechen?")
    assert seen == {"request": "en", "detected": "de"}
    assert my_voice.code_view("Ja.") == {"request": None, "detected": None}
    # On the en stream "Deutsche" means back to German (agent/call_language.py).
    assert my_voice.code_view("Deutsche expression.", "en")["request"] == "de"
