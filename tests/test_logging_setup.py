"""Tests for CSV-style logging format."""

from __future__ import annotations

import logging
import re
from io import StringIO

from open_enricher.logging_setup import CsvLogFormatter, PACKAGE_LOGGER, configure_logging

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$")


def test_csv_formatter_has_timestamp_level_logger_message() -> None:
    """Each line is timestamp;level;logger;message with readable time."""
    record = logging.LogRecord(
        name="open_enricher.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello;world\nnext",
        args=(),
        exc_info=None,
    )
    line = CsvLogFormatter().format(record)
    parts = line.split(";")
    assert len(parts) == 4
    assert _TS_RE.match(parts[0]), parts[0]
    assert parts[1] == "INFO"
    assert parts[2] == "open_enricher.test"
    # Semicolon and newline sanitized for CSV.
    assert parts[3] == "hello,world next"
    assert "\n" not in line


def test_configure_logging_writes_csv_to_stream() -> None:
    """Package logger emits CSV rows after configure_logging."""
    stream = StringIO()
    logger = configure_logging(level="INFO", force=True)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.stream = stream
    child = logging.getLogger(f"{PACKAGE_LOGGER}.unit")
    child.info("ready")
    output = stream.getvalue().strip()
    parts = output.split(";")
    assert len(parts) == 4
    assert _TS_RE.match(parts[0]), parts[0]
    assert parts[1] == "INFO"
    assert parts[2].endswith("unit")
    assert parts[3] == "ready"
