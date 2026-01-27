"""
Centralized logging configuration for the Helpdesk Chatbot backend.

This module provides a comprehensive logging setup with:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Console and file output
- Rotating file handlers to manage log file size
- Structured formatting with timestamps and context
- Performance tracking capabilities
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        if sys.stdout.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                )
        return super().format(record)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up and configure a logger with console and file handlers.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to backend/logs)
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep

    Returns:
        Configured logger instance

    Example:
        logger = setup_logger(__name__, level="DEBUG")
        logger.info("Application started")
        logger.debug("Detailed debug information")
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = ColoredFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if file_output:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"

        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file (all levels)
        log_file = log_dir / f"{name.replace('.', '_')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        # Error log file (errors and above only)
        error_log_file = log_dir / f"{name.replace('.', '_')}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)

    return logger


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get or create a logger with the specified name and level.

    This is a convenience function that uses sensible defaults.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Logger instance

    Example:
        from utils.logger import get_logger
        logger = get_logger(__name__)
    """
    return setup_logger(name, level=level)


class PerformanceLogger:
    """Context manager for logging performance metrics."""

    def __init__(
        self, logger: logging.Logger, operation: str, level: int = logging.DEBUG
    ):
        """
        Initialize performance logger.

        Args:
            logger: Logger instance to use
            operation: Name of the operation being measured
            level: Logging level for performance metrics

        Example:
            with PerformanceLogger(logger, "Database query"):
                results = db.query(...)
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type is not None:
            self.logger.error(
                f"Failed: {self.operation} (after {elapsed:.3f}s) - {exc_val}"
            )
        else:
            self.logger.log(
                self.level, f"Completed: {self.operation} in {elapsed:.3f}s"
            )


# Pre-configured loggers for common use cases
def get_api_logger() -> logging.Logger:
    """Get logger configured for API operations."""
    return get_logger("api", level="INFO")


def get_database_logger() -> logging.Logger:
    """Get logger configured for database operations."""
    return get_logger("database", level="INFO")


def get_processing_logger() -> logging.Logger:
    """Get logger configured for data processing operations."""
    return get_logger("processing", level="INFO")


# Suppress noisy third-party loggers
def configure_third_party_loggers():
    """Reduce verbosity of third-party library loggers."""
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


# Initialize third-party logger configuration
configure_third_party_loggers()
