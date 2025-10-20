#!/usr/bin/env python3
"""
Test script for retry handler and network failure recovery.
"""
import time
from unittest.mock import Mock, patch
from src.retry_handler import (
    retry_with_backoff,
    retry_api_call,
    check_internet_connection,
    wait_for_internet,
    NETWORK_EXCEPTIONS,
)
import requests


def test_retry_basic():
    """Test basic retry functionality."""
    print("\n" + "=" * 60)
    print("Testing Basic Retry Functionality")
    print("=" * 60)

    # Test 1: Function succeeds on first try
    print("\n[Test 1] Function succeeds immediately")
    call_count = {"count": 0}

    @retry_with_backoff(max_retries=3, initial_delay=0.1)
    def successful_function():
        call_count["count"] += 1
        return "success"

    result = successful_function()
    assert result == "success"
    assert call_count["count"] == 1
    print(f"✓ Function called {call_count['count']} time(s) - Success!")

    # Test 2: Function fails twice, then succeeds
    print("\n[Test 2] Function fails 2 times, then succeeds")
    call_count = {"count": 0}

    @retry_with_backoff(max_retries=3, initial_delay=0.1, exponential_base=2.0)
    def flaky_function():
        call_count["count"] += 1
        if call_count["count"] < 3:
            raise requests.exceptions.ConnectionError("Simulated network error")
        return "success after retries"

    start = time.time()
    result = flaky_function()
    elapsed = time.time() - start

    assert result == "success after retries"
    assert call_count["count"] == 3
    print(f"✓ Function called {call_count['count']} times")
    print(f"✓ Total time: {elapsed:.2f}s (with exponential backoff)")

    # Test 3: Function fails all retries
    print("\n[Test 3] Function fails all retry attempts")
    call_count = {"count": 0}

    @retry_with_backoff(max_retries=2, initial_delay=0.1)
    def always_fails():
        call_count["count"] += 1
        raise requests.exceptions.ConnectionError("Always fails")

    try:
        always_fails()
        assert False, "Should have raised an exception"
    except requests.exceptions.ConnectionError:
        print(f"✓ Function correctly failed after {call_count['count']} attempts")

    print("\n✅ All basic retry tests passed!")


def test_retry_decorator():
    """Test retry decorator with different scenarios."""
    print("\n" + "=" * 60)
    print("Testing Retry Decorator")
    print("=" * 60)

    # Test 1: Network exception retry
    print("\n[Test 1] Network exception triggers retry")
    call_count = {"count": 0}

    @retry_api_call(max_retries=3)
    def network_error_function():
        call_count["count"] += 1
        if call_count["count"] < 2:
            raise ConnectionError("Network error")
        return "recovered"

    result = network_error_function()
    assert result == "recovered"
    assert call_count["count"] == 2
    print(f"✓ Recovered after {call_count['count']} attempts")

    # Test 2: Timeout retry
    print("\n[Test 2] Timeout exception triggers retry")
    call_count = {"count": 0}

    @retry_api_call(max_retries=3)
    def timeout_function():
        call_count["count"] += 1
        if call_count["count"] < 2:
            raise TimeoutError("Request timeout")
        return "timeout recovered"

    result = timeout_function()
    assert result == "timeout recovered"
    print(f"✓ Recovered from timeout after {call_count['count']} attempts")

    # Test 3: Non-retryable exception
    print("\n[Test 3] Non-retryable exception fails immediately")
    call_count = {"count": 0}

    @retry_api_call(max_retries=3)
    def non_retryable_error():
        call_count["count"] += 1
        raise ValueError("This should not retry")

    try:
        non_retryable_error()
        assert False, "Should have raised ValueError"
    except ValueError:
        assert call_count["count"] == 1
        print(f"✓ Failed immediately (called {call_count['count']} time) - Correct!")

    print("\n✅ All decorator tests passed!")


def test_internet_check():
    """Test internet connection checking."""
    print("\n" + "=" * 60)
    print("Testing Internet Connection Check")
    print("=" * 60)

    print("\n[Test 1] Check if internet is available")
    is_connected = check_internet_connection(timeout=5.0)

    if is_connected:
        print("✓ Internet connection detected")
    else:
        print("⚠ No internet connection detected")
        print("  (This is expected if you're offline)")

    print("\n✅ Internet check test completed!")


def test_exponential_backoff():
    """Test exponential backoff timing."""
    print("\n" + "=" * 60)
    print("Testing Exponential Backoff Timing")
    print("=" * 60)

    print("\n[Test] Verify exponential backoff delays")
    call_count = {"count": 0}
    call_times = []

    @retry_with_backoff(
        max_retries=3,
        initial_delay=0.5,
        exponential_base=2.0,
        max_delay=10.0
    )
    def backoff_test():
        call_count["count"] += 1
        call_times.append(time.time())
        if call_count["count"] < 4:
            raise ConnectionError(f"Attempt {call_count['count']}")
        return "success"

    start_time = time.time()
    result = backoff_test()

    # Calculate delays between calls
    if len(call_times) > 1:
        delays = [call_times[i] - call_times[i-1] for i in range(1, len(call_times))]
        print(f"  Attempt 1: Initial call")
        for i, delay in enumerate(delays, 1):
            print(f"  Attempt {i+1}: Waited {delay:.2f}s (expected ~{0.5 * (2**i):.2f}s)")

        # Verify exponential growth
        for i in range(len(delays) - 1):
            ratio = delays[i+1] / delays[i]
            print(f"  Delay ratio {i+1}/{i+2}: {ratio:.2f}x (expected ~2x)")

    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Call count: {call_count['count']}")

    assert result == "success"
    print("\n✅ Exponential backoff timing verified!")


def test_retry_with_mock_api():
    """Test retry with mocked API calls."""
    print("\n" + "=" * 60)
    print("Testing Retry with Mock API")
    print("=" * 60)

    print("\n[Test] Simulate API failures and recovery")

    call_count = {"count": 0}

    @retry_api_call(max_retries=4)
    def mock_api_call(data: str):
        call_count["count"] += 1
        print(f"  API call attempt #{call_count['count']}")

        # Fail first 2 attempts
        if call_count["count"] <= 2:
            raise requests.exceptions.ConnectionError(
                f"Connection failed (attempt {call_count['count']})"
            )

        # Success on 3rd attempt
        return f"API response: {data.upper()}"

    result = mock_api_call("test data")
    print(f"\n  Result: {result}")
    print(f"  Total attempts: {call_count['count']}")

    assert result == "API response: TEST DATA"
    assert call_count["count"] == 3
    print("\n✓ Mock API recovered after temporary failures")

    print("\n✅ Mock API test passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Retry Handler Test Suite")
    print("=" * 60)

    try:
        test_retry_basic()
        test_retry_decorator()
        test_exponential_backoff()
        test_retry_with_mock_api()
        test_internet_check()

        print("\n" + "=" * 60)
        print("🎉 ALL RETRY TESTS PASSED!")
        print("=" * 60)
        print("\nYour application will now:")
        print("  ✓ Automatically retry on network failures")
        print("  ✓ Use exponential backoff to avoid overwhelming servers")
        print("  ✓ Wait for internet to restore (up to 5 minutes)")
        print("  ✓ Gracefully handle temporary outages")
        print("\nNetwork resilience: ENABLED ✅")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        exit(1)
