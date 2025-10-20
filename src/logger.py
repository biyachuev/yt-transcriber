"""
Logging configuration helpers.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from .config import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes for terminal output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Orange/Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[1;31m', # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format log record with color for terminal output."""
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        # Format the message
        formatted = super().format(record)

        # Reset levelname back to original (important for handlers)
        record.levelname = levelname

        return formatted


def setup_logger(name: str = "yt") -> logging.Logger:
    """
    Configure a logger that writes both to stdout and to a log file.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Avoid adding duplicate handlers.
    if logger.handlers:
        return logger

    # Console handler with colored output
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with plain text (no colors in file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    log_file = settings.LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


# Global logger instance.
logger = setup_logger()
