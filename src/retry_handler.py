"""
Retry mechanism with exponential backoff for network operations.
"""
import time
import functools
from typing import Callable, Any, Type, Tuple, Optional
from datetime import datetime, timedelta
import requests
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError

from .logger import logger


class APICallTracker:
    """Track API calls to prevent excessive usage with unstable internet."""

    def __init__(self):
        self.call_count = 0
        self.error_count = 0
        self.last_reset = datetime.now()
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10  # Circuit breaker threshold
        self.circuit_open = False
        self.circuit_open_until = None
        self.reset_interval = timedelta(minutes=5)

    def record_call(self, success: bool = True):
        """Record an API call."""
        self.call_count += 1

        if not success:
            self.error_count += 1
            self.consecutive_errors += 1

            # Check if we should open the circuit
            if self.consecutive_errors >= self.max_consecutive_errors:
                self._open_circuit()
        else:
            # Reset consecutive errors on success
            self.consecutive_errors = 0
            if self.circuit_open:
                self._close_circuit()

        # Auto-reset stats every 5 minutes
        if datetime.now() - self.last_reset > self.reset_interval:
            self._reset_stats()

    def _open_circuit(self):
        """Open the circuit breaker - stop making API calls."""
        self.circuit_open = True
        # Keep circuit open for 2 minutes
        self.circuit_open_until = datetime.now() + timedelta(minutes=2)
        logger.error(
            f"🚨 CIRCUIT BREAKER ACTIVATED! Too many consecutive errors ({self.consecutive_errors}). "
            f"Pausing API calls for 2 minutes to prevent excessive usage."
        )

    def _close_circuit(self):
        """Close the circuit breaker - resume API calls."""
        if self.circuit_open:
            logger.info("✅ Circuit breaker closed. Resuming API calls.")
        self.circuit_open = False
        self.circuit_open_until = None
        self.consecutive_errors = 0

    def _reset_stats(self):
        """Reset statistics."""
        logger.info(
            f"📊 API Call Statistics (last 5 min): "
            f"{self.call_count} calls, {self.error_count} errors "
            f"({self.error_count / max(self.call_count, 1) * 100:.1f}% error rate)"
        )
        self.call_count = 0
        self.error_count = 0
        self.last_reset = datetime.now()

    def check_circuit(self):
        """Check if circuit breaker allows calls."""
        if not self.circuit_open:
            return True

        # Check if cooldown period has passed
        if datetime.now() >= self.circuit_open_until:
            logger.info("⏰ Circuit breaker cooldown period ended. Attempting to resume...")
            self.circuit_open = False
            self.consecutive_errors = 0
            return True

        remaining = (self.circuit_open_until - datetime.now()).total_seconds()
        logger.warning(
            f"⚠️  Circuit breaker is OPEN. Waiting {remaining:.0f}s before retrying."
        )
        return False

    def get_stats(self):
        """Get current statistics."""
        return {
            "total_calls": self.call_count,
            "total_errors": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "circuit_open": self.circuit_open,
            "error_rate": self.error_count / max(self.call_count, 1),
        }


# Global API call tracker
_api_tracker = APICallTracker()


# Network-related exceptions to retry
NETWORK_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.RequestException,
    APIConnectionError,
    APITimeoutError,
    ConnectionError,
    TimeoutError,
)

# API rate limit exceptions
RATE_LIMIT_EXCEPTIONS = (
    RateLimitError,
)


def retry_with_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: Tuple[Type[Exception], ...] = NETWORK_EXCEPTIONS,
    rate_limit_retry_on: Tuple[Type[Exception], ...] = RATE_LIMIT_EXCEPTIONS,
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        retry_on: Tuple of exceptions that should trigger a retry
        rate_limit_retry_on: Tuple of rate limit exceptions (uses longer delays)

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retry_count = 0
            delay = initial_delay

            while retry_count <= max_retries:
                # Check circuit breaker before making call
                if not _api_tracker.check_circuit():
                    # Circuit is open - wait for cooldown
                    time.sleep(5)  # Check again in 5 seconds
                    continue

                try:
                    result = func(*args, **kwargs)
                    # Record successful call
                    _api_tracker.record_call(success=True)
                    return result

                except rate_limit_retry_on as e:
                    # Record failed call
                    _api_tracker.record_call(success=False)

                    # Rate limit errors - use longer delays
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(
                            f"Rate limit exceeded after {max_retries} retries in {func.__name__}: {e}"
                        )
                        raise

                    # For rate limits, wait longer (at least 60 seconds)
                    wait_time = max(60, delay)
                    logger.warning(
                        f"Rate limit hit in {func.__name__}. "
                        f"Retry {retry_count}/{max_retries} after {wait_time:.1f}s. "
                        f"Error: {e}"
                    )
                    time.sleep(wait_time)
                    delay = min(delay * exponential_base, max_delay)

                except retry_on as e:
                    # Record failed call
                    _api_tracker.record_call(success=False)

                    # Network errors - use exponential backoff
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(
                            f"Failed after {max_retries} retries in {func.__name__}: {e}"
                        )
                        raise

                    logger.warning(
                        f"Network error in {func.__name__}. "
                        f"Retry {retry_count}/{max_retries} after {delay:.1f}s. "
                        f"Error: {type(e).__name__}: {e}"
                    )
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)

                except Exception as e:
                    # Non-retryable exceptions - raise immediately
                    logger.error(f"Non-retryable error in {func.__name__}: {type(e).__name__}: {e}")
                    raise

            # Should not reach here, but just in case
            raise RuntimeError(f"Unexpected retry loop exit in {func.__name__}")

        return wrapper
    return decorator


def check_internet_connection(timeout: float = 5.0) -> bool:
    """
    Check if internet connection is available.

    Args:
        timeout: Timeout in seconds for the check

    Returns:
        True if internet is available, False otherwise
    """
    test_urls = [
        "https://www.google.com",
        "https://www.cloudflare.com",
        "https://api.openai.com",
    ]

    for url in test_urls:
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True
        except Exception:
            continue

    return False


def wait_for_internet(
    max_wait_time: float = 300.0,
    check_interval: float = 5.0,
    show_progress: bool = True
) -> bool:
    """
    Wait for internet connection to be restored.

    Args:
        max_wait_time: Maximum time to wait in seconds (default: 5 minutes)
        check_interval: Time between connection checks in seconds
        show_progress: Whether to show progress messages

    Returns:
        True if connection restored, False if timeout reached
    """
    if check_internet_connection():
        return True

    logger.warning("Internet connection lost. Waiting for reconnection...")

    elapsed_time = 0.0
    last_log_time = 0.0

    while elapsed_time < max_wait_time:
        time.sleep(check_interval)
        elapsed_time += check_interval

        # Log progress every 30 seconds
        if show_progress and (elapsed_time - last_log_time) >= 30:
            remaining = max_wait_time - elapsed_time
            logger.info(
                f"Still waiting for internet... ({elapsed_time:.0f}s elapsed, "
                f"{remaining:.0f}s remaining)"
            )
            last_log_time = elapsed_time

        if check_internet_connection():
            logger.info(f"Internet connection restored after {elapsed_time:.1f}s")
            return True

    logger.error(f"Internet connection not restored after {max_wait_time:.0f}s")
    return False


class RetryableOperation:
    """
    Context manager for operations that should retry on network failures.

    Example:
        with RetryableOperation("Downloading video"):
            download_video(url)
    """

    def __init__(
        self,
        operation_name: str,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """
        Initialize retryable operation.

        Args:
            operation_name: Name of the operation (for logging)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay before first retry
            max_delay: Maximum delay between retries
        """
        self.operation_name = operation_name
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.retry_count = 0
        self.delay = initial_delay

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success - reset retry count
            self.retry_count = 0
            return False

        # Check if this is a retryable exception
        if exc_type in NETWORK_EXCEPTIONS or exc_type in RATE_LIMIT_EXCEPTIONS:
            self.retry_count += 1

            if self.retry_count <= self.max_retries:
                # Check if internet is available
                if not check_internet_connection():
                    logger.warning(f"No internet connection during {self.operation_name}")
                    # Wait for internet to come back (up to 5 minutes)
                    if not wait_for_internet(max_wait_time=300):
                        logger.error("Internet connection timeout. Cannot continue.")
                        return False

                logger.warning(
                    f"Retrying {self.operation_name} "
                    f"(attempt {self.retry_count}/{self.max_retries}) "
                    f"after {self.delay:.1f}s"
                )
                time.sleep(self.delay)
                self.delay = min(self.delay * 2, self.max_delay)
                return True  # Suppress the exception, will retry

            else:
                logger.error(
                    f"{self.operation_name} failed after {self.max_retries} retries"
                )
                return False  # Re-raise the exception

        # Non-retryable exception
        return False


# Convenience decorators for common retry scenarios

def retry_network_operation(max_retries: int = 5):
    """Decorator for network operations with standard retry logic."""
    return retry_with_backoff(
        max_retries=max_retries,
        initial_delay=1.0,
        max_delay=60.0,
        retry_on=NETWORK_EXCEPTIONS,
        rate_limit_retry_on=RATE_LIMIT_EXCEPTIONS,
    )


def retry_api_call(max_retries: int = 3):
    """Decorator for API calls with shorter retry logic."""
    return retry_with_backoff(
        max_retries=max_retries,
        initial_delay=2.0,
        max_delay=30.0,
        retry_on=NETWORK_EXCEPTIONS,
        rate_limit_retry_on=RATE_LIMIT_EXCEPTIONS,
    )


def get_api_tracker() -> APICallTracker:
    """Get the global API call tracker."""
    return _api_tracker


def print_api_stats():
    """Print current API usage statistics."""
    stats = _api_tracker.get_stats()
    logger.info("=" * 60)
    logger.info("📊 API USAGE STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total API calls: {stats['total_calls']}")
    logger.info(f"Failed calls: {stats['total_errors']}")
    logger.info(f"Success rate: {(1 - stats['error_rate']) * 100:.1f}%")
    logger.info(f"Consecutive errors: {stats['consecutive_errors']}")
    logger.info(f"Circuit breaker: {'🔴 OPEN' if stats['circuit_open'] else '🟢 CLOSED'}")
    logger.info("=" * 60)
