"""Standard logging setup for Open-Enricher.

Configures the ``open_enricher`` logger hierarchy with readable timestamps
and a CSV-friendly line format using ``;`` as the field separator:

    YYYY-MM-DD HH:MM:SS.mmm;LEVEL;logger.name;message

Messages are sanitized so each log record stays on one CSV row (newlines and
semicolons inside the message are replaced).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Root name for all package loggers (``logging.getLogger(__name__)`` inherits).
PACKAGE_LOGGER = "open_enricher"

# Default level when OPEN_ENRICHER_LOG_LEVEL is unset.
_DEFAULT_LEVEL = "INFO"

_CONFIGURED = False


class CsvLogFormatter(logging.Formatter):
    """
    Format log records as semicolon-separated CSV rows.

    Columns: timestamp;level;logger;message
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Build one CSV log line for ``record``.

        Args:
            record: Standard library log record.

        Returns:
            Single-line string with ``;`` separators and no embedded newlines.
        """
        timestamp = self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S")
        timestamp = f"{timestamp}.{int(record.msecs):03d}"
        level = record.levelname
        name = record.name
        message = record.getMessage()
        # Keep one CSV row per record: strip separators and line breaks.
        message = (
            message.replace("\r\n", " ")
            .replace("\n", " ")
            .replace("\r", " ")
            .replace(";", ",")
        )
        # Exception text, if any, folded onto the same row.
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            exc_text = (
                exc_text.replace("\r\n", " | ")
                .replace("\n", " | ")
                .replace("\r", " | ")
                .replace(";", ",")
            )
            message = f"{message} | {exc_text}"
        return f"{timestamp};{level};{name};{message}"


def _resolve_level(level: str | int | None) -> int:
    """
    Resolve a log level from env, argument, or default.

    Args:
        level: Explicit level name/int, or None to read ``OPEN_ENRICHER_LOG_LEVEL``.

    Returns:
        Numeric logging level.
    """
    if level is None:
        level = os.environ.get("OPEN_ENRICHER_LOG_LEVEL", _DEFAULT_LEVEL)
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def configure_logging(
    *,
    level: str | int | None = None,
    log_file: Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """
    Configure package logging once for console and optional file output.

    Args:
        level: Log level name (DEBUG, INFO, ...) or int. Defaults to
            ``OPEN_ENRICHER_LOG_LEVEL`` or INFO.
        log_file: Optional path that also receives the same CSV lines.
        force: When True, reconfigure even if already set up.

    Returns:
        The package root logger (``open_enricher``).
    """
    global _CONFIGURED
    logger = logging.getLogger(PACKAGE_LOGGER)
    if _CONFIGURED and not force:
        return logger

    resolved = _resolve_level(level)
    logger.setLevel(resolved)
    logger.handlers.clear()
    logger.propagate = False

    formatter = CsvLogFormatter()

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(resolved)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(resolved)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Quiet noisy third-party libraries unless the user asks for DEBUG.
    if resolved > logging.DEBUG:
        for name in ("urllib3", "httpx", "httpcore", "filelock", "huggingface_hub"):
            logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a logger under the ``open_enricher`` hierarchy.

    Args:
        name: Usually ``__name__``. When None, returns the package root logger.

    Returns:
        Configured child logger (or package root).
    """
    if name is None or name == PACKAGE_LOGGER:
        return logging.getLogger(PACKAGE_LOGGER)
    if name.startswith(PACKAGE_LOGGER + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER}.{name}")
