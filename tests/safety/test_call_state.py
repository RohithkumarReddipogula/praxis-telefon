"""The call-state reducer: crisis, human request, two-strike handover."""

from praxis_telefon.safety.call_state import (
    LOW_CONFIDENCE,
    CallerTurn,
    Event,
    Mode,
    UnrecognizedSpeech,
    replay,
)

LOW = LOW_CONFIDENCE / 2
OK = 0.95
MUMBLE = CallerTurn("der äh wann denn", LOW)
CLEAR = CallerTurn("Ich hätte gern einen Termin am Dienstag.", OK)


def _modes_and_actions(events: list[Event]) -> list[tuple[Mode, str]]:
    return [
        (state.mode, action.kind)
        for state, action in (replay(events[: i + 1]) for i in range(len(events)))
    ]


def test_ja_twice_in_a_row_with_low_confidence_is_no_handover() -> None:
    state, action = replay([CallerTurn("Ja", LOW), CallerTurn("Ja.", LOW)])
    assert state.mode is Mode.NORMAL
    assert state.strikes == 0
    assert action.kind == "none"


def test_mumble_then_ja_resets_the_counter() -> None:
    events: list[Event] = [MUMBLE, CallerTurn("Ja", LOW), MUMBLE]
    assert _modes_and_actions(events) == [
        (Mode.NORMAL, "repeat"),
        (Mode.NORMAL, "none"),
        (Mode.NORMAL, "repeat"),  # only one strike again, not a handover
    ]


def test_two_mumbles_in_a_row_hand_over() -> None:
    assert _modes_and_actions([MUMBLE, MUMBLE]) == [
        (Mode.NORMAL, "repeat"),
        (Mode.HANDOVER, "handover_strikes"),
    ]


def test_clear_turn_between_mumbles_resets() -> None:
    state, action = replay([MUMBLE, CLEAR, MUMBLE])
    assert state.mode is Mode.NORMAL and action.kind == "repeat"


def test_empty_transcript_is_a_strike() -> None:
    state, action = replay([CallerTurn("   ", None), UnrecognizedSpeech()])
    assert state.mode is Mode.HANDOVER and action.kind == "handover_strikes"


def test_unrecognized_speech_counts_with_low_confidence_turns() -> None:
    state, _ = replay([UnrecognizedSpeech(), MUMBLE])
    assert state.mode is Mode.HANDOVER


def test_short_answers_with_low_confidence_are_never_strikes() -> None:
    for text in ("Ja", "Nein", "Genau", "Okay", "Richtig", "12", "Mustermann", "Yes", "No"):
        state, action = replay([CallerTurn(text, LOW), CallerTurn(text, LOW)])
        assert (state.strikes, action.kind) == (0, "none"), text


def test_typed_input_without_confidence_is_not_a_strike() -> None:
    state, _ = replay([CallerTurn("der äh wann denn", None)] * 3)
    assert state.mode is Mode.NORMAL


def test_human_request_hands_over_at_once() -> None:
    state, action = replay([CallerTurn("Ich will mit einem Menschen sprechen.", OK)])
    assert state.mode is Mode.HANDOVER
    assert (action.kind, action.language) == ("human_request", "de")


def test_human_request_in_english() -> None:
    _, action = replay([CallerTurn("Can I talk to a real person?", OK)])
    assert (action.kind, action.language) == ("human_request", "en")


def test_emergency_turns_booking_off_for_the_rest_of_the_call() -> None:
    events: list[Event] = [CallerTurn("Ich habe starke Brustschmerzen.", OK), CLEAR, CLEAR]
    assert _modes_and_actions(events) == [
        (Mode.EMERGENCY, "emergency"),
        (Mode.EMERGENCY, "none"),
        (Mode.EMERGENCY, "none"),
    ]


def test_emergency_check_stays_active_after_handover() -> None:
    state, action = replay(
        [CallerTurn("Verbinden Sie mich bitte.", OK), CallerTurn("Ich kriege keine Luft.", OK)]
    )
    assert state.mode is Mode.EMERGENCY and action.kind == "emergency"


def test_repeated_emergency_repeats_the_message() -> None:
    crisis = CallerTurn("Ich bekomme keine Luft.", OK)
    _, action = replay([crisis, CLEAR, crisis])
    assert action.kind == "emergency"


def test_suicide_is_its_own_action_and_remembered() -> None:
    state, _ = replay([CallerTurn("Ich will nicht mehr leben.", OK), CLEAR])
    assert state.mode is Mode.EMERGENCY
    assert state.suicide_mentioned
    _, first = replay([CallerTurn("Ich will nicht mehr leben.", OK)])
    assert first.kind == "suicide"


def test_human_request_in_emergency_mode_does_not_leave_emergency_mode() -> None:
    state, action = replay(
        [CallerTurn("Er ist bewusstlos.", OK), CallerTurn("Verbinden Sie mich.", OK)]
    )
    assert state.mode is Mode.EMERGENCY and action.kind == "none"


def test_strikes_in_handover_mode_only_ask_to_repeat() -> None:
    state, action = replay([MUMBLE, MUMBLE, MUMBLE, MUMBLE])
    assert state.mode is Mode.HANDOVER and action.kind == "repeat"


def test_crisis_language_is_kept_for_later_messages() -> None:
    state, action = replay([CallerTurn("I can't breathe.", OK), MUMBLE])
    assert state.language == "en" and action.language == "en"
