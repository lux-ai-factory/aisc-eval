"""Logging configuration and utilities for VERA evaluation.

This module provides custom logging configuration for the VERA evaluation system,
including colored console output and queue-based logging setup. It supports
asynchronous logging through a queue handler.
"""

import atexit
import logging
import logging.config
import logging.handlers
import pathlib
from logging import Logger

import yaml

# Main application logger instances
app_logger = logging.getLogger("aisc_eval")
root_logger = logging.getLogger()


COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[41m",  # Red background
}
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in COLORS:
            record.colored_levelname = f"{COLORS[levelname]}{levelname}{RESET}"
        else:
            record.colored_levelname = levelname
        return super().format(record)


def setup_logging() -> None:
    """Set up logging configuration from the YAML config file.

    This function reads the logging configuration from config/logging.yaml
    and sets up the logging system accordingly. It also handles the setup
    of queue-based logging if configured.
    """
    config_file = (
        pathlib.Path(__file__).parent.parent.parent / "config" / "logging.yaml"
    )
    with open(config_file) as f_in:
        logger_config = yaml.safe_load(f_in)

    logging.config.dictConfig(logger_config)

    # Start the queue handler's listener if it exists
    queue_handler = logging.getHandlerByName("queue_handler")
    if (
        (queue_handler is not None)
        and isinstance(queue_handler, logging.handlers.QueueHandler)
        and (queue_handler.listener is not None)
    ):
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)


def get_logger() -> Logger:
    """Get the application logger, setting it up if necessary.

    Returns:
        Logger: The configured application logger instance
    """
    if not root_logger.handlers:
        setup_logging()

    return app_logger
