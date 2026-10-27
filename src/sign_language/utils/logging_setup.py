"""Centralized logging configuration for the sign_language package."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from sign_language.utils.paths import ensure_dir


def configure_logging(
    level: str = "INFO",
    console: bool = True,
    to_file: bool = True,
    directory: Path | str = "logs",
    filename: str = "app.log",
    max_bytes: int = 5242880,
    backup_count: int = 3,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
) -> None:
    """Configure root logger with console and rotating file handlers.

    Args:
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        console: Whether to attach a StreamHandler to stdout/stderr.
        to_file: Whether to attach a RotatingFileHandler.
        directory: Directory for log files.
        filename: Log file name.
        max_bytes: Maximum size of a log file before rotation.
        backup_count: Number of rotated backup files to preserve.
        fmt: Log record formatting string.
    """
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(fmt)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if to_file:
        log_dir = Path(directory)
        ensure_dir(log_dir)
        log_file = log_dir / filename
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
