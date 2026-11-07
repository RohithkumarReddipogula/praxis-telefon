"""Does switching the Deepgram stream's language lose audio? (option B, real API)

Costs Deepgram minutes (about 4, ~$0.03); ask before running.

    uv run python -m eval.stt_ab.switch_check            # 5 + 5 clips, two timings
    uv run python -m eval.stt_ab.switch_check --estimate

One stream per run, as in a call: a German name clip (the "request") on `de`, then
``stt.update_options(language=...)`` (the plugin reconnects its WebSocket), then a phone
clip in the new language. Timings:
- ``after_final``: switch on the first clip's final, then 0.3 s pause, then the phone clip
  (what Receptionist does: the caller has just stopped talking);
- ``mid_speech``: switch 1 s into the phone clip (worst case, for comparison).
The phone digits after the switch are scored like the A/B (eval/stt_ab/score.py);
reconnect time is from the switch to the first transcript of the phone clip.
Synthetic clips from eval/stt_ab/make_audio.py (sarah, clean).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from eval.stt_ab.make_audio import RATE, clip_path
from eval.stt_ab.run_stt import FRAME_S, TAIL_S, _pcm
from eval.stt_ab.score import phone_digits
from eval.stt_ab.utterances import Utterance, load

RESULTS = Path("eval/results/stt_ab/switch_check.jsonl")
PAUSE_S = 0.3
MID_S = 1.0
N = 5


def cases(utterances: list[Utterance]) -> list[tuple[Utterance, Utterance, str]]:
    """(request clip, phone clip, language after the switch)."""
    first = next(u for u in utterances if u.group == "de" and u.kind == "name")
    en_phone = [u for u in utterances if u.group == "en" and u.kind == "phone"][:N]
    de_phone = [u for u in utterances if u.group == "de" and u.kind == "phone"][:N]
    return [(first, u, "en") for u in en_phone] + [(first, u, "de") for u in de_phone]


async def run_one(
    stt: Any, request: bytes, phone: bytes, start: str, to: str, timing: str
) -> dict[str, Any]:
    from livekit import rtc
    from livekit.agents import stt as lk_stt

    stt.update_options(language=start)
    stream = stt.stream()
    step = int(RATE * FRAME_S) * 2
    first_final = asyncio.Event()
    finals: list[tuple[float, str]] = []
    switched_at: list[float] = []

    def frame(chunk: bytes) -> rtc.AudioFrame:
        return rtc.AudioFrame(chunk.ljust(step, b"\x00"), RATE, 1, step // 2)

    async def push(pcm: bytes, switch_after_s: float | None = None) -> None:
        tick = time.perf_counter()
        for n, i in enumerate(range(0, len(pcm), step)):
            if switch_after_s is not None and not switched_at and n * FRAME_S >= switch_after_s:
                stt.update_options(language=to)
                switched_at.append(time.perf_counter())
            stream.push_frame(frame(pcm[i : i + step]))
            tick += FRAME_S
            await asyncio.sleep(max(0.0, tick - time.perf_counter()))

    async def pusher() -> None:
        await push(request)
        silence = b"\x00" * int(RATE * 2 * PAUSE_S)
        if timing == "after_final":
            # Silence until the request's final, as the caller pauses after it.
            while not first_final.is_set():
                await push(b"\x00" * step)
            stt.update_options(language=to)
            switched_at.append(time.perf_counter())
            await push(silence)
            await push(phone)
        else:
            await push(silence)
            await push(phone, switch_after_s=MID_S)
        await push(b"\x00" * int(RATE * 2 * TAIL_S))
        stream.end_input()

    task = asyncio.create_task(pusher())
    async for ev in stream:
        if ev.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
            text = ev.alternatives[0].text
            if text.strip():
                finals.append((time.perf_counter(), text))
                first_final.set()
    await task
    await stream.aclose()
    t = switched_at[0] if switched_at else 0.0
    after = " ".join(text for at, text in finals if at > t)
    return {
        "timing": timing,
        "to": to,
        "text_after_switch": after,
        "digits": phone_digits(after),
        "first_final_after_switch_s": next((round(at - t, 3) for at, _ in finals if at > t), None),
    }


async def run_all() -> list[dict[str, Any]]:
    import aiohttp
    from livekit.plugins import deepgram

    from praxis_telefon.agent import main as agent_main
    from praxis_telefon.config import Settings

    settings = Settings.from_env()
    rows = []
    async with aiohttp.ClientSession() as session:
        options = {**agent_main.stt_options(settings), "language": "de"}
        options.pop("language_hint", None)
        stt = deepgram.STT(**options, api_key=settings.deepgram_api_key, http_session=session)
        for timing in ("after_final", "mid_speech"):
            for request, phone, to in cases(load()):
                result = await run_one(
                    stt,
                    _pcm(clip_path(request, "sarah", "clean")),
                    _pcm(clip_path(phone, "sarah", "clean")),
                    "de",
                    to,
                    timing,
                )
                rows.append({"id": phone.id, "expected": phone.expected, **result,
                             "right": result["digits"] == phone.expected})  # fmt: skip
                print(f"  {timing:11} -> {to}  {phone.id}  right={rows[-1]['right']}")
    return rows


def main(argv: list[str]) -> int:
    utterances = load()
    if "--estimate" in argv:
        seconds = 0.0
        for request, phone, _ in cases(utterances):
            for u in (request, phone):
                with open(clip_path(u, "sarah", "clean"), "rb") as f:
                    seconds += (len(f.read()) - 44) / 2 / RATE
            seconds += PAUSE_S + TAIL_S + 1.0
        minutes = 2 * seconds / 60  # two timings
        print(f"{minutes:.1f} min, ${minutes * 0.0077:.2f}")
        return 0
    from dotenv import load_dotenv

    load_dotenv(".env")  # read in-process only; never printed
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    rows = asyncio.run(run_all())
    RESULTS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for timing in ("after_final", "mid_speech"):
        sel = [r for r in rows if r["timing"] == timing]
        delays = sorted(
            r["first_final_after_switch_s"] for r in sel if r["first_final_after_switch_s"]
        )
        print(
            f"{timing}: digits right {sum(r['right'] for r in sel)}/{len(sel)}, "
            f"first final after switch median {delays[len(delays) // 2] if delays else None} s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
