"""Record the user's own voice saying 12 sentences and transcribe them with Deepgram de/en.

For option B (docs/experiments/stt-language.md): how `de` transcribes English language
requests and how `en` transcribes German ones, in the user's accented speech. The
recordings and transcripts stay in eval/results/my_voice/, which is gitignored: never
commit them (CLAUDE.md rule 3). Delete them when done.

    uv run python -m eval.my_voice record          # all missing sentences, one by one
    uv run python -m eval.my_voice record 03       # record (or redo) one sentence
    uv run python -m eval.my_voice transcribe      # costs Deepgram minutes; ask first

Recording: the sentence is shown; press Enter, say it once at normal speed, press Enter.
Each clip is streamed like a call (eval/stt_ab/run_stt.py: 20 ms frames, real time,
1.5 s silence after), once with `de` and once with `en`, production options otherwise.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from praxis_telefon.agent.call_language import CallerFinal

OUT = Path("eval/results/my_voice")
RESULTS = OUT / "transcripts.jsonl"
RATE = 16_000
SETTINGS = ("de", "en")


@dataclass(frozen=True)
class Sentence:
    id: str
    text: str
    why: str


SENTENCES = (
    Sentence("01", "Can we speak English?", "English request, through de"),
    Sentence("02", "Do you speak English?", "English request, through de"),
    Sentence("03", "English, please.", "English request, through de"),
    Sentence("04", "Could we speak in English, please?", "English request, through de"),
    Sentence("05", "English.", "one word, the whole utterance"),
    Sentence("06", "Englisch.", "one word, German"),
    Sentence("07", "Können wir Englisch sprechen?", "German request (worked in call 4)"),
    Sentence("08", "Können wir wieder Deutsch sprechen?", "back to German, through en"),
    Sentence("09", "Sprechen Sie Deutsch?", "back to German, through en"),
    Sentence(
        "10",
        "Meine Nummer ist null drei null, zwei drei eins zwei fünf, vier fünf sechs.",
        "German digits (fictional number), through de and en",
    ),
    Sentence("11", "Deutsch.", "one word, back to German, through en"),
    Sentence("12", "Deutsch, bitte.", "back to German, through en"),
)


def clip(s: Sentence) -> Path:
    return OUT / f"{s.id}.wav"


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def code_view(transcript: str, stt_language: str | None = None) -> dict[str, Any]:
    """What the agent's code makes of the transcript, on a stream in ``stt_language``
    (the en-stream rule for "Deutsch" needs it)."""
    final = CallerFinal.from_text(transcript, stt_language)
    return {"request": final.request, "detected": final.detected}


# --- record ------------------------------------------------------------------------------


def _record_one(s: Sentence) -> None:
    import sounddevice as sd  # type: ignore[import-untyped]

    chunks: list[bytes] = []

    def callback(data: Any, frames: int, time: Any, status: Any) -> None:
        chunks.append(bytes(data))

    print(f"\n[{s.id}] Say:  {s.text}")
    input("  Enter to start ...")
    with sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", callback=callback):
        input("  Recording. Enter to stop.")
    pcm = b"".join(chunks)
    write_wav(clip(s), pcm)
    print(f"  saved {clip(s)} ({len(pcm) / 2 / RATE:.1f} s)")


def record(ids: list[str]) -> int:
    chosen = (
        [s for s in SENTENCES if s.id in ids]
        if ids
        else [s for s in SENTENCES if not clip(s).exists()]
    )
    if not chosen:
        print("All recorded. To redo one: record <id>")
    for s in chosen:
        _record_one(s)
    return 0


# --- transcribe -------------------------------------------------------------------------


async def _transcribe_all() -> list[dict[str, Any]]:
    import aiohttp
    from livekit.plugins import deepgram

    from eval.stt_ab.run_stt import _pcm, transcribe
    from praxis_telefon.agent import main as agent_main
    from praxis_telefon.config import Settings

    settings = Settings.from_env()
    rows: list[dict[str, Any]] = []
    async with aiohttp.ClientSession() as session:
        for setting in SETTINGS:
            options = {**agent_main.stt_options(settings), "language": setting}
            options.pop("language_hint", None)
            stt = deepgram.STT(**options, api_key=settings.deepgram_api_key, http_session=session)
            for s in SENTENCES:
                if not clip(s).exists():
                    continue
                result = await transcribe(stt, _pcm(clip(s)))
                rows.append({"id": s.id, "setting": setting, **result,
                             **code_view(result["text"], setting)})  # fmt: skip
    return rows


def transcribe_main() -> int:
    from dotenv import load_dotenv

    missing = [s.id for s in SENTENCES if not clip(s).exists()]
    if missing:
        print(f"not recorded yet: {', '.join(missing)} (transcribing the others)")
    load_dotenv(".env")  # read in-process only; never printed
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    rows = asyncio.run(_transcribe_all())
    RESULTS.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    by = {(r["id"], r["setting"]): r for r in rows}
    for s in SENTENCES:
        print(f"\n[{s.id}] said: {s.text}   ({s.why})")
        for setting in SETTINGS:
            r = by.get((s.id, setting))
            if r:
                print(
                    f"  {setting}: {r['text']!r}  tags={r['languages']}  "
                    f"request={r['request']}  detected={r['detected']}"
                )
    print(f"\nsaved {RESULTS} (gitignored)")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["record"]:
        return record(argv[1:])
    if argv[:1] == ["transcribe"]:
        return transcribe_main()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
