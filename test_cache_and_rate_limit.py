#!/usr/bin/env python3
"""
Simple test script to verify caching and rate limiting functionality.
"""
import time
from src.api_cache import APICache, RateLimiter


def test_cache():
    """Test caching functionality."""
    print("\n" + "=" * 60)
    print("Testing Cache Functionality")
    print("=" * 60)

    cache = APICache(ttl_days=7)

    # Test 1: Set and get
    print("\n[Test 1] Set and get cache entry")
    cache.set("test", {"key": "value1"}, "result1")
    result = cache.get("test", {"key": "value1"})
    assert result == "result1", "Cache get failed"
    print("✓ Cache set and get working")

    # Test 2: Cache miss
    print("\n[Test 2] Cache miss")
    result = cache.get("test", {"key": "nonexistent"})
    assert result is None, "Cache should return None for missing key"
    print("✓ Cache miss returns None")

    # Test 3: Cache hit
    print("\n[Test 3] Cache hit")
    cache.set("translation", {"text": "Hello", "lang": "ru"}, "Привет")
    result = cache.get("translation", {"text": "Hello", "lang": "ru"})
    assert result == "Привет", "Cache hit failed"
    print("✓ Cache hit working correctly")

    # Test 4: Different namespaces
    print("\n[Test 4] Different namespaces")
    cache.set("namespace1", {"data": "test"}, "result_ns1")
    cache.set("namespace2", {"data": "test"}, "result_ns2")
    result1 = cache.get("namespace1", {"data": "test"})
    result2 = cache.get("namespace2", {"data": "test"})
    assert result1 == "result_ns1", "Namespace 1 failed"
    assert result2 == "result_ns2", "Namespace 2 failed"
    print("✓ Different namespaces work correctly")

    # Test 5: Clear cache
    print("\n[Test 5] Clear namespace")
    cache.clear("namespace1")
    result = cache.get("namespace1", {"data": "test"})
    assert result is None, "Clear cache failed"
    result2 = cache.get("namespace2", {"data": "test"})
    assert result2 == "result_ns2", "Other namespace should not be affected"
    print("✓ Clear cache working")

    print("\n✅ All cache tests passed!")


def test_rate_limiter():
    """Test rate limiting functionality."""
    print("\n" + "=" * 60)
    print("Testing Rate Limiter Functionality")
    print("=" * 60)

    # Test 1: Basic rate limiting
    print("\n[Test 1] Basic rate limiting (5 calls per 2 seconds)")
    limiter = RateLimiter(max_calls=5, time_window=2)

    start_time = time.time()

    # Make 5 calls - should be instant
    for i in range(5):
        limiter.wait_if_needed()
        print(f"  Call {i + 1} - OK")

    elapsed = time.time() - start_time
    assert elapsed < 0.5, "First 5 calls should be instant"
    print(f"✓ First 5 calls completed in {elapsed:.2f}s (expected < 0.5s)")

    # Make 6th call - should wait
    print("\n  Call 6 - Should wait...")
    start_wait = time.time()
    limiter.wait_if_needed()
    wait_time = time.time() - start_wait
    print(f"  Call 6 - Waited {wait_time:.2f}s")
    assert wait_time >= 1.5, "6th call should have waited at least 1.5 seconds"
    print(f"✓ Rate limiting enforced (waited {wait_time:.2f}s)")

    # Test 2: Remaining calls
    print("\n[Test 2] Check remaining calls")
    limiter2 = RateLimiter(max_calls=10, time_window=60)
    remaining = limiter2.get_remaining_calls()
    assert remaining == 10, f"Should have 10 remaining calls, got {remaining}"
    print(f"✓ Initially {remaining} calls remaining")

    limiter2.wait_if_needed()
    remaining = limiter2.get_remaining_calls()
    assert remaining == 9, f"Should have 9 remaining calls, got {remaining}"
    print(f"✓ After 1 call: {remaining} calls remaining")

    print("\n✅ All rate limiter tests passed!")


def test_integration():
    """Test caching with rate limiting together."""
    print("\n" + "=" * 60)
    print("Testing Cache + Rate Limiter Integration")
    print("=" * 60)

    cache = APICache()
    limiter = RateLimiter(max_calls=3, time_window=2)

    print("\n[Test] Simulate API calls with caching")

    def simulated_api_call(text: str, use_cache: bool = True):
        """Simulate an API call with caching."""
        # Check cache
        if use_cache:
            cached = cache.get("api", {"text": text})
            if cached is not None:
                print(f"  ✓ Cache HIT for '{text}' - No API call needed!")
                return cached

        # Apply rate limiting
        print(f"  → Making API call for '{text}'...")
        limiter.wait_if_needed()

        # Simulate API response
        time.sleep(0.1)  # Simulate network delay
        result = f"Translated: {text}"

        # Cache result
        if use_cache:
            cache.set("api", {"text": text}, result)

        return result

    # Make calls
    start = time.time()

    print("\n  First batch (will hit API):")
    simulated_api_call("Hello")
    simulated_api_call("World")
    simulated_api_call("Test")

    print("\n  Second batch (same texts - should use cache):")
    simulated_api_call("Hello")  # Cache hit
    simulated_api_call("World")  # Cache hit
    simulated_api_call("Test")   # Cache hit

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.2f}s")
    print(f"  Expected: ~0.3s (3 API calls) without waits")
    print(f"  If all 6 were API calls: ~0.6s + rate limit waits")

    # The cached calls should be nearly instant
    assert elapsed < 1.0, "With caching, should complete quickly"

    print("\n✅ Integration test passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("API Cache and Rate Limiter Test Suite")
    print("=" * 60)

    try:
        test_cache()
        test_rate_limiter()
        test_integration()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nCaching and rate limiting are working correctly.")
        print("Your API costs should be significantly reduced!")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        exit(1)
