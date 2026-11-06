import json

from praxis_telefon.agent.turn_log import build_turn_record, format_turn_record


def test_full_record() -> None:
    record = build_turn_record(
        3,
        {"transcription_delay": 0.2104, "end_of_turn_delay": 0.48},
        {"llm_node_ttft": 0.39, "tts_node_ttfb": 0.18, "e2e_latency": 1.3149},
    )
    assert record == {
        "event": "turn",
        "turn": 3,
        "stt_delay_s": 0.21,
        "eou_delay_s": 0.48,
        "llm_ttft_s": 0.39,
        "tts_ttfb_s": 0.18,
        "e2e_s": 1.315,
        "since_caller_s": None,
        "is_filler": False,
        "interrupted": False,
    }


def test_missing_user_turn_gives_nulls() -> None:
    record = build_turn_record(1, None, {"tts_node_ttfb": 0.2})
    assert record["stt_delay_s"] is None
    assert record["eou_delay_s"] is None
    assert record["llm_ttft_s"] is None
    assert record["tts_ttfb_s"] == 0.2


def test_non_numeric_values_ignored() -> None:
    record = build_turn_record(1, {"transcription_delay": "fast"}, {"llm_node_ttft": True})
    assert record["stt_delay_s"] is None
    assert record["llm_ttft_s"] is None


def test_interrupted_flag() -> None:
    assert build_turn_record(1, None, {}, interrupted=True)["interrupted"] is True


def test_format_is_single_line_json() -> None:
    line = format_turn_record(build_turn_record(2, {}, {"e2e_latency": 1.0}))
    assert "\n" not in line
    assert json.loads(line)["e2e_s"] == 1.0


def test_filler_and_answer_records() -> None:
    # The filler gets the caller's STT/EOU metrics and e2e; the answer after the tool
    # has no e2e from LiveKit, so its latency is since_caller_s.
    filler = build_turn_record(
        4,
        {"transcription_delay": 0.2, "stopped_speaking_at": 100.0},
        {"e2e_latency": 1.2, "started_speaking_at": 101.2},
        is_filler=True,
        caller_stopped_at=100.0,
    )
    answer = build_turn_record(5, None, {"started_speaking_at": 102.35}, caller_stopped_at=100.0)
    assert filler["is_filler"] is True and filler["since_caller_s"] == 1.2
    assert answer["is_filler"] is False and answer["since_caller_s"] == 2.35
    assert answer["e2e_s"] is None and answer["stt_delay_s"] is None
