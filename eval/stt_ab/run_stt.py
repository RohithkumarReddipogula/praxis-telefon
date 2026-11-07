"""Stream every A/B clip through Deepgram, as the agent does, once per setting.

Costs Deepgram minutes; ask before running. Estimate: about 69 billed minutes per full
setting (speech plus lead and tail silence), about 3.5 hours in all, about $2.05 at list
prices ($0.0077/min de and en, $0.0092/min multi). The exact figure follows from the
clip lengths once the audio exists (``--estimate``).
Results already in the output file are skipped, so a rerun only does what is missing.

    uv run python -m eval.stt_ab.run_stt                 # all settings
    uv run python -m eval.stt_ab.run_stt de multi_hint   # some settings

Each clip goes through ``deepgram.STT(...).stream()`` with the production options from
``main.stt_options`` (model, endpointing defaults), only the language changed. Audio is
pushed in 20 ms frames at real-time pace, then 1.5 s of silence, like a caller who stops
talking. Settings run in parallel, clips within a setting one after another.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

from eval.stt_ab.make_audio import RATE, clip_path
from eval.stt_ab.utterances import VOICES, Utterance, load

RESULTS = Path("eval/results/stt_ab/transcripts.jsonl")
SETTINGS: dict[str, dict[str, Any]] = {
    "de": {"language": "de"},  # production
    "multi": {"language": "multi"},
    "multi_hint": {"language": "multi", "language_hint": ["de", "en"]},
    "en": {"language": "en"},  # reference for English, and for "de, then switch to en"
}
EN_ONLY = {"en"}  # settings that only get the English and switch clips
CONDITIONS = ("clean", "phone")
FRAME_S = 0.02
TAIL_S = 1.5


def clips_for(setting: str, utterances: list[Utterance]) -> list[tuple[Utterance, str, str]]:
    return [
        (u, voice, condition)
        for u in utterances
        if setting not in EN_ONLY or u.group != "de"
        for voice in VOICES
        for condition in CONDITIONS
    ]


def _done() -> set[tuple[str, str]]:
    if not RESULTS.exists():
        return set()
    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    return {(r["setting"], r["clip"]) for r in map(json.loads, lines)}


def _pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.readframes(w.getnframes())


async def transcribe(stt: Any, pcm: bytes) -> dict[str, Any]:
    from livekit import rtc
    from livekit.agents import stt as lk_stt

    stream = stt.stream()
    finals: list[dict[str, Any]] = []
    audio_end: list[float] = []

    async def push() -> None:
        step = int(RATE * FRAME_S) * 2
        tick = time.perf_counter()
        for i in range(0, len(pcm), step):
            chunk = pcm[i : i + step].ljust(step, b"\x00")
            stream.push_frame(rtc.AudioFrame(chunk, RATE, 1, step // 2))
            tick += FRAME_S
            await asyncio.sleep(max(0.0, tick - time.perf_counter()))
        audio_end.append(time.perf_counter())
        silent = b"\x00" * step
        for _ in range(int(TAIL_S / FRAME_S)):
            stream.push_frame(rtc.AudioFrame(silent, RATE, 1, step // 2))
            await asyncio.sleep(FRAME_S)
        stream.end_input()

    pusher = asyncio.create_task(push())
    async for ev in stream:
        if ev.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
            alt = ev.alternatives[0]
            if alt.text.strip():
                finals.append({"text": alt.text, "language": str(alt.language),
                               "at": time.perf_counter()})  # fmt: skip
    await pusher
    await stream.aclose()
    end = audio_end[0] if audio_end else time.perf_counter()
    return {
        "text": " ".join(f["text"] for f in finals),
        "languages": [f["language"] for f in finals],
        "final_delay_s": round(finals[-1]["at"] - end, 3) if finals else None,
    }


async def run_setting(setting: str, done: set[tuple[str, str]], lock: asyncio.Lock) -> int:
    import aiohttp
    from livekit.plugins import deepgram

    from praxis_telefon.agent import main as agent_main
    from praxis_telefon.config import Settings

    settings = Settings.from_env()
    todo = [c for c in clips_for(setting, load()) if (setting, clip_path(*c).name) not in done]
    async with aiohttp.ClientSession() as session:
        stt = deepgram.STT(
            **{**agent_main.stt_options(settings), **SETTINGS[setting]},
            api_key=settings.deepgram_api_key,
            http_session=session,
        )
        for n, (u, voice, condition) in enumerate(todo, 1):
            path = clip_path(u, voice, condition)
            result = await transcribe(stt, _pcm(path))
            record = {"setting": setting, "clip": path.name, "id": u.id, "voice": voice,
                      "condition": condition, **result}  # fmt: skip
            async with lock:
                with RESULTS.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if n % 40 == 0:
                print(f"  {setting}: {n}/{len(todo)}")
    return len(todo)


async def run(settings: list[str]) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done, lock = _done(), asyncio.Lock()
    counts = await asyncio.gather(*(run_setting(s, done, lock) for s in settings))
    print("done:", dict(zip(settings, counts, strict=True)))


PRICE_PER_MIN = {"de": 0.0077, "en": 0.0077, "multi": 0.0092, "multi_hint": 0.0092}


def estimate(settings: list[str]) -> float:
    """Billed minutes and dollars from the clip files, including the silence pushed."""
    total = 0.0
    for setting in settings:
        seconds = 0.0
        for u, voice, condition in clips_for(setting, load()):
            with wave.open(str(clip_path(u, voice, condition)), "rb") as w:
                seconds += w.getnframes() / w.getframerate() + TAIL_S
        cost = seconds / 60 * PRICE_PER_MIN[setting]
        total += cost
        print(f"{setting:11} {seconds / 60:6.1f} min  ${cost:.2f}")
    print(f"{'total':11} {'':6}      ${total:.2f}")
    return total


def main(argv: list[str]) -> int:
    from dotenv import load_dotenv

    if "--estimate" in argv:
        estimate([a for a in argv if a != "--estimate"] or list(SETTINGS))
        return 0
    chosen = argv or list(SETTINGS)
    if unknown := set(chosen) - set(SETTINGS):
        print(f"unknown settings: {sorted(unknown)}")
        return 2
    load_dotenv(".env")  # read in-process only; never printed
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    asyncio.run(run(chosen))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
