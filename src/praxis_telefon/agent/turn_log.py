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
) -> dict[str, Any]:
    """Combine user and assistant message metrics into one flat, JSON-safe record.

    Missing values become ``None`` (for example the greeting has no user turn).
    """
    user = user_metrics or {}
    record: dict[str, Any] = {"event": "turn", "turn": turn}
    for out_key, src_key in USER_FIELDS:
        record[out_key] = _seconds(user, src_key)
    for out_key, src_key in ASSISTANT_FIELDS:
        record[out_key] = _seconds(assistant_metrics, src_key)
    record["interrupted"] = interrupted
    return record


def format_turn_record(record: Mapping[str, Any]) -> str:
    return json.dumps(record, separators=(",", ":"), sort_keys=False)
