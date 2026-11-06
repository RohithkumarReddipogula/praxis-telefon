"""Build one structured timing record per conversational turn.

Uses the per-message ``metrics`` LiveKit attaches to each ``ChatMessage``. Only timings
are recorded, never transcript text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

# (output key, source key) pairs.
USER_FIELDS = (
    ("stt_delay_s", "transcription_delay"),
    ("eou_delay_s", "end_of_turn_delay"),
)
ASSISTANT_FIELDS = (
    ("llm_ttft_s", "llm_node_ttft"),
    ("tts_ttfb_s", "tts_node_ttfb"),
    ("e2e_s", "e2e_latency"),
)


def _seconds(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return round(float(value), 3)
    return None


def build_turn_record(
    turn: int,
    user_metrics: Mapping[str, Any] | None,
    assistant_metrics: Mapping[str, Any],
    *,
    interrupted: bool = False,
    is_filler: bool = False,
    caller_stopped_at: float | None = None,
) -> dict[str, Any]:
    """Combine user and assistant message metrics into one flat, JSON-safe record.

    Missing values become ``None`` (for example the greeting has no user turn).
    ``is_filler``: the message is the code-spoken "Einen Moment, bitte." before a tool
    (agent/tool_filler.py); the answer follows as its own record. ``since_caller_s``:
    from the caller stopping speaking to this message's first audio. LiveKit reports
    ``e2e_s`` only for the first speech of a turn, so after a filler the answer's
    latency is only in ``since_caller_s``.
    """
    user = user_metrics or {}
    record: dict[str, Any] = {"event": "turn", "turn": turn}
    for out_key, src_key in USER_FIELDS:
        record[out_key] = _seconds(user, src_key)
    for out_key, src_key in ASSISTANT_FIELDS:
        record[out_key] = _seconds(assistant_metrics, src_key)
    started = _seconds(assistant_metrics, "started_speaking_at")
    record["since_caller_s"] = (
        round(started - caller_stopped_at, 3)
        if started is not None and caller_stopped_at is not None
        else None
    )
    record["is_filler"] = is_filler
    record["interrupted"] = interrupted
    return record


def format_turn_record(record: Mapping[str, Any]) -> str:
    return json.dumps(record, separators=(",", ":"), sort_keys=False)
