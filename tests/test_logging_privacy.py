import json
import logging

import pytest
from livekit.agents.cli.log import ColoredFormatter, JsonFormatter

from praxis_telefon.agent.turn_log import build_turn_record
from praxis_telefon.logging_privacy import RedactPiiFilter, install_pii_filter

TRANSCRIPT = "Ich habe starke Schmerzen in den Beinen"


def make_record(level: int = logging.DEBUG) -> logging.LogRecord:
    logger = logging.getLogger("livekit.agents.test")
    return logger.makeRecord(
        logger.name,
        level,
        __file__,
        1,
        "received user transcript",
        (),
        None,
        extra={"lk.pii.user_transcript": TRANSCRIPT, "language": "de"},
    )


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR])
@pytest.mark.parametrize(
    "formatter",
    [JsonFormatter(), ColoredFormatter("%(levelname)s %(name)s %(message)s %(extra)s")],
)
def test_transcript_not_in_formatted_output(level: int, formatter: logging.Formatter) -> None:
    # Regression: LiveKit formatters print lk.pii.* extras (transcripts) at DEBUG and above.
    record = make_record(level)
    assert RedactPiiFilter().filter(record) is True
    output = formatter.format(record)
    assert TRANSCRIPT not in output
    assert "received user transcript" in output  # record kept, only PII removed


def test_unfiltered_record_would_leak() -> None:
    # Guards the test above: without the filter, LiveKit's formatter does print the text.
    assert TRANSCRIPT in JsonFormatter().format(make_record())


def test_non_pii_extras_kept() -> None:
    record = make_record()
    RedactPiiFilter().filter(record)
    assert record.__dict__["language"] == "de"
    assert record.__dict__["pii_redacted"] == 1


def test_install_is_idempotent_and_covers_root_handlers() -> None:
    root = logging.getLogger()
    handler = logging.StreamHandler()
    root.addHandler(handler)
    try:
        install_pii_filter()
        install_pii_filter()
        assert sum(isinstance(f, RedactPiiFilter) for f in handler.filters) == 1
    finally:
        root.removeHandler(handler)


def test_turn_record_has_no_text_fields() -> None:
    record = build_turn_record(1, {"transcription_delay": 0.2}, {"e2e_latency": 1.0})
    assert TRANSCRIPT not in json.dumps(record)
    assert all(not isinstance(v, str) or k == "event" for k, v in record.items())
