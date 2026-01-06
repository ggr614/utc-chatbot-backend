"""
Tests for the logger module.
"""

import pytest
import logging
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.logger import (
    setup_logger,
    get_logger,
    PerformanceLogger,
    get_api_logger,
    get_database_logger,
    get_processing_logger,
)


class TestSetupLogger:
    """Test suite for setup_logger function."""

    def test_setup_logger_creates_logger(self):
        """Test that setup_logger creates a logger instance."""
        logger = setup_logger("test_logger", level="INFO")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO

    def test_setup_logger_respects_log_level(self):
        """Test that different log levels are respected."""
        logger_debug = setup_logger("test_debug", level="DEBUG")
        logger_warning = setup_logger("test_warning", level="WARNING")

        assert logger_debug.level == logging.DEBUG
        assert logger_warning.level == logging.WARNING

    def test_setup_logger_with_invalid_level(self):
        """Test handling of invalid log level."""
        with pytest.raises(AttributeError):
            setup_logger("test_invalid", level="INVALID_LEVEL")

    def test_setup_logger_creates_log_directory(self, tmp_path):
        """Test that log directory is created if it doesn't exist."""
        log_dir = tmp_path / "test_logs"
        assert not log_dir.exists()

        logger = setup_logger("test_dir", level="INFO", log_dir=log_dir)

        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_setup_logger_without_console_output(self, tmp_path):
        """Test logger without console output."""
        logger = setup_logger(
            "test_no_console", level="INFO", log_dir=tmp_path, console_output=False
        )

        # Check that no StreamHandler is added
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        # Note: FileHandlers are also StreamHandlers, so we need to be more specific
        console_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) == 0

    def test_setup_logger_without_file_output(self):
        """Test logger without file output."""
        logger = setup_logger("test_no_file", level="INFO", file_output=False)

        # Check that no FileHandler is added
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 0

    def test_setup_logger_avoids_duplicate_handlers(self):
        """Test that calling setup_logger twice doesn't add duplicate handlers."""
        logger1 = setup_logger("test_duplicate", level="INFO")
        initial_handler_count = len(logger1.handlers)

        logger2 = setup_logger("test_duplicate", level="INFO")

        # Should return the same logger without adding new handlers
        assert logger1 is logger2
        assert len(logger2.handlers) == initial_handler_count


class TestGetLogger:
    """Test suite for get_logger convenience function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_get_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_get_logger"

    def test_get_logger_default_level(self):
        """Test that get_logger uses INFO as default level."""
        logger = get_logger("test_default_level")

        assert logger.level == logging.INFO

    def test_get_logger_custom_level(self):
        """Test that get_logger accepts custom level."""
        logger = get_logger("test_custom_level", level="DEBUG")

        assert logger.level == logging.DEBUG


class TestPerformanceLogger:
    """Test suite for PerformanceLogger context manager."""

    def test_performance_logger_measures_time(self):
        """Test that PerformanceLogger measures execution time."""
        logger = get_logger("test_performance")

        with patch.object(logger, "log") as mock_log:
            with PerformanceLogger(logger, "Test operation"):
                time.sleep(0.01)  # Simulate work

            # Check that start and completion messages were logged
            assert mock_log.call_count >= 2
            # Last call should contain timing information
            last_call_args = mock_log.call_args_list[-1]
            assert "Completed" in str(last_call_args)
            assert "Test operation" in str(last_call_args)

    def test_performance_logger_logs_on_exception(self):
        """Test that PerformanceLogger logs when exception occurs."""
        logger = get_logger("test_performance_error")

        with patch.object(logger, "log") as mock_log:
            with patch.object(logger, "error") as mock_error:
                try:
                    with PerformanceLogger(logger, "Failing operation"):
                        raise ValueError("Test error")
                except ValueError:
                    pass

                # Error method should have been called
                mock_error.assert_called_once()
                # Should contain failure message
                error_call_args = str(mock_error.call_args)
                assert "Failed" in error_call_args

    def test_performance_logger_custom_level(self):
        """Test that PerformanceLogger respects custom log level."""
        logger = get_logger("test_performance_level")

        with patch.object(logger, "log") as mock_log:
            with PerformanceLogger(logger, "Test operation", level=logging.WARNING):
                pass

            # Should use WARNING level
            first_call = mock_log.call_args_list[0]
            assert first_call[0][0] == logging.WARNING


class TestPreConfiguredLoggers:
    """Test suite for pre-configured logger functions."""

    def test_get_api_logger(self):
        """Test that get_api_logger returns configured logger."""
        logger = get_api_logger()

        assert isinstance(logger, logging.Logger)
        assert "api" in logger.name
        assert logger.level == logging.INFO

    def test_get_database_logger(self):
        """Test that get_database_logger returns configured logger."""
        logger = get_database_logger()

        assert isinstance(logger, logging.Logger)
        assert "database" in logger.name
        assert logger.level == logging.INFO

    def test_get_processing_logger(self):
        """Test that get_processing_logger returns configured logger."""
        logger = get_processing_logger()

        assert isinstance(logger, logging.Logger)
        assert "processing" in logger.name
        assert logger.level == logging.INFO


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def test_logger_actually_logs_to_file(self, tmp_path):
        """Test that logger actually writes to file."""
        # Log file name is based on logger name with underscores
        log_file = tmp_path / "test_file_logging.log"
        logger = setup_logger(
            "test_file_logging", level="INFO", log_dir=tmp_path, console_output=False
        )

        test_message = "This is a test log message"
        logger.info(test_message)

        # Force flush
        for handler in logger.handlers:
            handler.flush()

        # Check that log file was created and contains message
        assert log_file.exists()
        content = log_file.read_text()
        assert test_message in content

    def test_logger_separates_error_logs(self, tmp_path):
        """Test that errors are logged to separate file."""
        error_log_file = tmp_path / "test_errors_errors.log"
        logger = setup_logger(
            "test_errors", level="DEBUG", log_dir=tmp_path, console_output=False
        )

        logger.info("This is info")
        logger.error("This is an error")

        # Force flush
        for handler in logger.handlers:
            handler.flush()

        # Check that error log file exists and contains only error
        assert error_log_file.exists()
        error_content = error_log_file.read_text()
        assert "This is an error" in error_content
        assert "This is info" not in error_content

    def test_logger_handles_different_log_levels(self, tmp_path):
        """Test that logger handles different log levels correctly."""
        logger = setup_logger("test_levels", level="DEBUG", log_dir=tmp_path)

        with patch.object(logger, "_log") as mock_log:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")

            assert mock_log.call_count == 5
