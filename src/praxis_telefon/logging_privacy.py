"""Keep caller speech out of logs.

LiveKit attaches transcripts, tool arguments and similar data to log records as
``extra`` fields whose names start with ``lk.pii.``, at DEBUG and at WARNING/ERROR,
and its formatters print every extra field. ``RedactPiiFilter`` deletes those fields
before any handler formats the record.

Limitation: text that a library writes straight into the log *message* is not caught.
"""

from __future__ import annotations

import logging

PII_PREFIX = "lk.pii."


class RedactPiiFilter(logging.Filter):
    """Remove ``lk.pii.*`` extras from a record. Never drops the record itself."""

    def filter(self, record: logging.LogRecord) -> bool:
        pii_keys = [k for k in record.__dict__ if k.startswith(PII_PREFIX)]
        for key in pii_keys:
            del record.__dict__[key]
        if pii_keys:
            record.__dict__["pii_redacted"] = len(pii_keys)
        return True


def _all_handlers() -> list[logging.Handler]:
    handlers = list(logging.getLogger().handlers)
    for obj in logging.Logger.manager.loggerDict.values():
        if isinstance(obj, logging.Logger):
            handlers.extend(obj.handlers)
    return handlers


def install_pii_filter() -> int:
    """Attach the filter to every existing handler (idempotent).

    Call it after logging is configured, in every process that handles calls.
    Returns the number of handlers that newly received the filter.
    """
    added = 0
    for handler in _all_handlers():
        if not any(isinstance(f, RedactPiiFilter) for f in handler.filters):
            handler.addFilter(RedactPiiFilter())
            added += 1
    return added
