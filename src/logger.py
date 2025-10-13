"""
Logging configuration helpers.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from .config import settings


def setup_logger(name: str = "youtube_transcriber") -> logging.Logger:
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
    
    # Log formatting.
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler.
    log_file = settings.LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Global logger instance.
logger = setup_logger()
