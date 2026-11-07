"""Make the A/B audio: ElevenLabs TTS for every utterance and voice, clean and phone.

Costs ElevenLabs characters (about 21,000 for both voices); ask before running.
Clips that exist are skipped, so a rerun costs only what is missing.

    uv run python -m eval.stt_ab.make_audio --dry-run   # count only
    uv run python -m eval.stt_ab.make_audio

Clean: 16 kHz, 16-bit mono, as ElevenLabs returns it. Phone: down to 8 kHz, through
G.711 mu-law and back to 16 kHz, as a narrowband phone call would arrive. Segments of a
switch call are joined with a pause. Synthetic data only; audio stays out of git
(eval/results/ is ignored).
"""

from __future__ import annotations

import os
import sys
import warnings
import wave
from pathlib import Path

with warnings.catch_warnings():  # mu-law only; deprecated, removed in 3.13; fine on 3.12
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop

from eval.stt_ab.utterances import VOICES, Utterance, load

OUT = Path("eval/results/stt_ab/audio")
RATE = 16_000
MODEL = "eleven_flash_v2_5"
PAUSE_S = 1.2  # between the German opener and the English part of a switch call
LEAD_S = 0.3  # silence before the first word


def silence(seconds: float, rate: int = RATE) -> bytes:
    return b"\x00\x00" * int(seconds * rate)


def _resample(pcm: bytes, src: int, dst: int) -> bytes:
    """Band-limited resampling (soxr in LiveKit). audioop.ratecv has no anti-alias
    filter: a 6 kHz hiss folded into the speech band instead of being removed."""
    from livekit import rtc

    resampler = rtc.AudioResampler(src, dst, quality=rtc.AudioResamplerQuality.HIGH)
    frames = [*resampler.push(bytearray(pcm)), *resampler.flush()]
    return b"".join(bytes(f.data.cast("B")) for f in frames)


def telephone(pcm16k: bytes) -> bytes:
    """16 kHz PCM -> 8 kHz -> G.711 mu-law -> back to 16 kHz PCM."""
    narrow = _resample(pcm16k, RATE, 8_000)
    narrow = audioop.ulaw2lin(audioop.lin2ulaw(narrow, 2), 2)  # no rate change here
    return _resample(narrow, 8_000, RATE)


def write_wav(path: Path, pcm: bytes, rate: int = RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def clip_path(u: Utterance, voice: str, condition: str) -> Path:
    return OUT / f"{u.id}-{voice}-{condition}.wav"


def _tts(key: str, voice_id: str, text: str, lang: str) -> tuple[bytes, int]:
    """PCM audio and what ElevenLabs billed (its ``character-cost`` header)."""
    import httpx2 as httpx

    r = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_16000",
        headers={"xi-api-key": key},
        json={"text": text, "model_id": MODEL, "language_code": lang},
        timeout=60,
    )
    r.raise_for_status()
    return r.content[: len(r.content) // 2 * 2], int(r.headers.get("character-cost", "0"))


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    todo = [
        (u, voice)
        for u in load()
        for voice in VOICES
        if not (clip_path(u, voice, "clean").exists() and clip_path(u, voice, "phone").exists())
    ]
    chars = sum(u.chars for u, _ in todo)
    print(f"{len(todo)} clips to make, {chars} characters")
    if dry or not todo:
        return 0
    from dotenv import load_dotenv

    load_dotenv(".env")  # read in-process only; never printed
    key = os.environ["ELEVENLABS_API_KEY"]
    sent = billed = 0
    for n, (u, voice) in enumerate(todo, 1):
        parts = [silence(LEAD_S)]
        for i, seg in enumerate(u.segments):
            if i:
                parts.append(silence(PAUSE_S))
            pcm_part, cost = _tts(key, VOICES[voice], seg.text, seg.lang)
            parts.append(pcm_part)
            sent += len(seg.text)
            billed += cost
        pcm = b"".join(parts)
        write_wav(clip_path(u, voice, "clean"), pcm)
        write_wav(clip_path(u, voice, "phone"), telephone(pcm))
        if n % 40 == 0:
            print(f"  {n}/{len(todo)} clips, {sent} characters sent, {billed} billed")
    print(f"done: {len(todo)} clips, {sent} characters sent, {billed} billed (character-cost)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
