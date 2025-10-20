#!/usr/bin/env python3
"""
Test script for circuit breaker and API call tracking.
"""
import time
from src.retry_handler import get_api_tracker, retry_api_call, print_api_stats
import requests


def test_circuit_breaker():
    """Test circuit breaker protection against excessive API calls."""
    print("\n" + "=" * 60)
    print("Testing Circuit Breaker Protection")
    print("=" * 60)

    tracker = get_api_tracker()

    # Reset tracker for clean test
    tracker.call_count = 0
    tracker.error_count = 0
    tracker.consecutive_errors = 0
    tracker.circuit_open = False

    print("\n[Test] Simulating unstable internet with 15 consecutive errors")
    print("Expected: Circuit breaker should open after 10 errors\n")

    call_count = {"count": 0}

    @retry_api_call(max_retries=2)
    def unstable_api_call():
        call_count["count"] += 1
        print(f"  Attempt #{call_count['count']}")

        # Simulate constant failures (like unstable internet)
        raise requests.exceptions.ConnectionError("Simulated network error")

    # Try to make 15 calls (should stop at 10 due to circuit breaker)
    errors_caught = 0
    for i in range(15):
        try:
            unstable_api_call()
        except requests.exceptions.ConnectionError:
            errors_caught += 1

        # Check if circuit breaker is open
        if tracker.circuit_open:
            print(f"\n🚨 Circuit breaker opened after {tracker.consecutive_errors} consecutive errors!")
            print(f"   Total attempts made: {call_count['count']}")
            break

        time.sleep(0.1)  # Small delay between attempts

    # Verify circuit breaker activated
    assert tracker.circuit_open, "Circuit breaker should be open"
    assert tracker.consecutive_errors >= 10, "Should have at least 10 consecutive errors"

    print(f"\n✓ Circuit breaker correctly activated")
    print(f"✓ Protected against excessive API calls")

    # Show stats
    stats = tracker.get_stats()
    print(f"\nStatistics:")
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Total errors: {stats['total_errors']}")
    print(f"  Consecutive errors: {stats['consecutive_errors']}")
    print(f"  Circuit status: {'OPEN 🔴' if stats['circuit_open'] else 'CLOSED 🟢'}")

    print("\n✅ Circuit breaker test passed!")


def test_circuit_breaker_recovery():
    """Test that circuit breaker allows recovery after cooldown."""
    print("\n" + "=" * 60)
    print("Testing Circuit Breaker Recovery")
    print("=" * 60)

    tracker = get_api_tracker()

    # Reset first
    tracker.call_count = 0
    tracker.error_count = 0
    tracker.consecutive_errors = 0
    tracker.circuit_open = False

    # Open the circuit manually
    tracker.consecutive_errors = 10
    tracker._open_circuit()

    print("\n[Test 1] Circuit is open - calls should be blocked")
    assert tracker.circuit_open, "Circuit should be open"
    assert not tracker.check_circuit(), "Circuit check should return False"
    print("✓ Circuit correctly blocks calls when open")

    # Fast-forward time by setting circuit_open_until to past
    print("\n[Test 2] Simulating cooldown period completion")
    from datetime import datetime, timedelta
    tracker.circuit_open_until = datetime.now() - timedelta(seconds=1)

    assert tracker.check_circuit(), "Circuit should close after cooldown"
    assert not tracker.circuit_open, "Circuit should be closed"
    print("✓ Circuit correctly closes after cooldown")

    # Test that successful call resets consecutive errors
    print("\n[Test 3] Successful call should reset error counter")
    tracker.consecutive_errors = 5
    tracker.record_call(success=True)
    assert tracker.consecutive_errors == 0, "Consecutive errors should reset"
    print("✓ Successful call resets error counter")

    print("\n✅ Circuit breaker recovery test passed!")


def test_api_call_tracking():
    """Test API call tracking and statistics."""
    print("\n" + "=" * 60)
    print("Testing API Call Tracking")
    print("=" * 60)

    tracker = get_api_tracker()

    # Reset tracker
    tracker.call_count = 0
    tracker.error_count = 0
    tracker.consecutive_errors = 0
    tracker.circuit_open = False

    print("\n[Test] Recording 10 calls: 7 successful, 3 failed")

    # Record calls
    for i in range(7):
        tracker.record_call(success=True)

    for i in range(3):
        tracker.record_call(success=False)

    stats = tracker.get_stats()

    assert stats['total_calls'] == 10, "Should have 10 total calls"
    assert stats['total_errors'] == 3, "Should have 3 errors"
    assert abs(stats['error_rate'] - 0.3) < 0.01, "Error rate should be 30%"

    print(f"\n✓ Call tracking working correctly:")
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Errors: {stats['total_errors']}")
    print(f"  Error rate: {stats['error_rate'] * 100:.1f}%")
    print(f"  Success rate: {(1 - stats['error_rate']) * 100:.1f}%")

    print("\n✅ API call tracking test passed!")


def test_circuit_breaker_prevents_excessive_calls():
    """Test that circuit breaker truly prevents excessive API calls during unstable internet."""
    print("\n" + "=" * 60)
    print("Testing Protection Against Excessive Calls")
    print("=" * 60)

    tracker = get_api_tracker()
    tracker.call_count = 0
    tracker.error_count = 0
    tracker.consecutive_errors = 0
    tracker.circuit_open = False
    tracker.max_consecutive_errors = 5  # Lower threshold for faster test

    print("\n[Scenario] Simulating very unstable internet")
    print("Without protection: Would make 100+ failed API calls")
    print("With circuit breaker: Should stop around 5-10 calls\n")

    actual_api_calls = {"count": 0}

    @retry_api_call(max_retries=2)
    def failing_api():
        actual_api_calls["count"] += 1
        raise ConnectionError("Network unstable")

    # Try to make many calls
    attempts = 0
    max_attempts = 50

    while attempts < max_attempts:
        attempts += 1

        # Check circuit before attempting
        if not tracker.check_circuit():
            print(f"  Attempt {attempts}: Circuit OPEN - Call blocked ✋")
            time.sleep(0.1)
            continue

        try:
            failing_api()
        except Exception:
            pass

        # If circuit opened, we're done
        if tracker.circuit_open:
            print(f"\n🛡️  Circuit breaker activated after {actual_api_calls['count']} actual API calls")
            print(f"   (Would have made {max_attempts * 3} calls without protection!)")
            break

    # Verify protection worked
    assert actual_api_calls["count"] < 20, f"Should make fewer than 20 API calls, made {actual_api_calls['count']}"
    assert tracker.circuit_open, "Circuit should be open"

    savings = ((max_attempts * 3) - actual_api_calls["count"]) / (max_attempts * 3) * 100
    print(f"\n✅ Protection successful!")
    print(f"   Prevented ~{savings:.0f}% of unnecessary API calls")
    print(f"   Saved approximately ${(max_attempts * 3 - actual_api_calls['count']) * 0.002:.2f}")

    # Reset for other tests
    tracker.max_consecutive_errors = 10

    print("\n✅ Excessive call prevention test passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Circuit Breaker & API Protection Test Suite")
    print("=" * 60)

    try:
        test_api_call_tracking()
        test_circuit_breaker()
        test_circuit_breaker_recovery()
        test_circuit_breaker_prevents_excessive_calls()

        print("\n" + "=" * 60)
        print("🎉 ALL CIRCUIT BREAKER TESTS PASSED!")
        print("=" * 60)
        print("\nYour application is now protected against:")
        print("  ✅ Excessive API calls with unstable internet")
        print("  ✅ Runaway costs from repeated failures")
        print("  ✅ Circuit breaker stops calls after 10 consecutive errors")
        print("  ✅ Automatic recovery after 2-minute cooldown")
        print("  ✅ Real-time statistics tracking")
        print("\nCircuit Breaker Protection: ACTIVE 🛡️")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        exit(1)
