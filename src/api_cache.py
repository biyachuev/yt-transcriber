"""
API response caching and rate limiting module.
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
from collections import deque, OrderedDict

from .logger import logger
from .config import settings


class APICache:
    """Cache for API responses with TTL support and LRU eviction."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl_days: int = 7, max_entries: int = 1000):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files (default: BASE_DIR/cache)
            ttl_days: Time to live for cache entries in days
            max_entries: Maximum number of cache entries (LRU eviction when exceeded)
        """
        self.cache_dir = cache_dir or settings.BASE_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.ttl_days = ttl_days
        self.max_entries = max_entries
        # Track access order for LRU
        self._access_order: OrderedDict[str, float] = OrderedDict()
        self._load_access_order()
        self._clean_expired_cache()
        self._enforce_size_limit()

    def _get_cache_key(self, namespace: str, data: Any) -> str:
        """
        Generate a cache key from namespace and data.

        Args:
            namespace: Cache namespace (e.g., 'translation', 'refinement')
            data: Data to hash (will be converted to JSON)

        Returns:
            Cache key (SHA256 hash)
        """
        # Convert data to canonical JSON string
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        # Create hash
        hash_obj = hashlib.sha256(f"{namespace}:{json_str}".encode('utf-8'))
        return hash_obj.hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache key.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.json"

    def _load_access_order(self) -> None:
        """Load access order from cache files on initialization."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_entry = json.load(f)

                    cache_key = cache_file.stem
                    timestamp = datetime.fromisoformat(cache_entry['timestamp']).timestamp()
                    self._access_order[cache_key] = timestamp
                except Exception:
                    continue

            # Sort by timestamp (oldest first for LRU)
            self._access_order = OrderedDict(
                sorted(self._access_order.items(), key=lambda x: x[1])
            )

        except Exception as e:
            logger.warning(f"Failed to load cache access order: {e}")

    def _update_access_time(self, cache_key: str) -> None:
        """Update access time for LRU tracking."""
        # Move to end (most recently used)
        if cache_key in self._access_order:
            self._access_order.move_to_end(cache_key)
        self._access_order[cache_key] = time.time()

    def _enforce_size_limit(self) -> None:
        """Evict least recently used entries if cache exceeds max size."""
        if len(self._access_order) <= self.max_entries:
            return

        evict_count = len(self._access_order) - self.max_entries

        # Remove oldest entries
        keys_to_evict = list(self._access_order.keys())[:evict_count]

        for cache_key in keys_to_evict:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
            del self._access_order[cache_key]

        if evict_count > 0:
            logger.info(f"Evicted {evict_count} cache entries (LRU policy, max={self.max_entries})")

    def get(self, namespace: str, data: Any) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            namespace: Cache namespace
            data: Data to use as cache key

        Returns:
            Cached value or None if not found/expired
        """
        cache_key = self._get_cache_key(namespace, data)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)

            # Check if expired
            cached_time = datetime.fromisoformat(cache_entry['timestamp'])
            if datetime.now() - cached_time > timedelta(days=self.ttl_days):
                logger.debug(f"Cache entry expired: {cache_key[:8]}...")
                cache_path.unlink()
                if cache_key in self._access_order:
                    del self._access_order[cache_key]
                return None

            # Update access time for LRU
            self._update_access_time(cache_key)

            logger.debug(f"Cache hit: {namespace} - {cache_key[:8]}...")
            return cache_entry['value']

        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None

    def set(self, namespace: str, data: Any, value: Any) -> None:
        """
        Store a value in cache.

        Args:
            namespace: Cache namespace
            data: Data to use as cache key
            value: Value to cache
        """
        cache_key = self._get_cache_key(namespace, data)
        cache_path = self._get_cache_path(cache_key)

        try:
            cache_entry = {
                'timestamp': datetime.now().isoformat(),
                'namespace': namespace,
                'value': value
            }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, ensure_ascii=False, indent=2)

            # Update access time and enforce size limit
            self._update_access_time(cache_key)
            self._enforce_size_limit()

            logger.debug(f"Cache stored: {namespace} - {cache_key[:8]}...")

        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")

    def _clean_expired_cache(self) -> None:
        """Remove expired cache entries."""
        try:
            count = 0
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_entry = json.load(f)

                    cached_time = datetime.fromisoformat(cache_entry['timestamp'])
                    if datetime.now() - cached_time > timedelta(days=self.ttl_days):
                        cache_file.unlink()
                        count += 1
                except Exception:
                    # If we can't read the file, delete it
                    cache_file.unlink()
                    count += 1

            if count > 0:
                logger.info(f"Cleaned {count} expired cache entries")

        except Exception as e:
            logger.warning(f"Failed to clean cache: {e}")

    def clear(self, namespace: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            namespace: If provided, only clear entries for this namespace
        """
        try:
            count = 0
            for cache_file in self.cache_dir.glob("*.json"):
                if namespace:
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_entry = json.load(f)

                        if cache_entry.get('namespace') == namespace:
                            cache_file.unlink()
                            cache_key = cache_file.stem
                            if cache_key in self._access_order:
                                del self._access_order[cache_key]
                            count += 1
                    except Exception:
                        continue
                else:
                    cache_file.unlink()
                    count += 1

            if not namespace:
                self._access_order.clear()

            logger.info(f"Cleared {count} cache entries" + (f" for namespace '{namespace}'" if namespace else ""))

        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (total entries, size limit, usage percentage)
        """
        total_entries = len(self._access_order)
        usage_pct = (total_entries / self.max_entries * 100) if self.max_entries > 0 else 0

        # Calculate total cache size
        total_size_bytes = 0
        for cache_file in self.cache_dir.glob("*.json"):
            total_size_bytes += cache_file.stat().st_size

        total_size_mb = total_size_bytes / (1024 * 1024)

        return {
            "total_entries": total_entries,
            "max_entries": self.max_entries,
            "usage_percentage": round(usage_pct, 1),
            "total_size_mb": round(total_size_mb, 2),
            "ttl_days": self.ttl_days
        }


class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, max_calls: int = 60, time_window: int = 60):
        """
        Initialize the rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self._last_log_time = 0

    def wait_if_needed(self) -> None:
        """
        Wait if rate limit would be exceeded.
        Blocks until a call can be made within the rate limit.
        """
        now = time.time()

        # Remove calls outside the time window
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()

        # Check if we've hit the limit
        if len(self.calls) >= self.max_calls:
            # Calculate how long to wait
            oldest_call = self.calls[0]
            wait_time = self.time_window - (now - oldest_call)

            if wait_time > 0:
                # Log only once per minute to avoid spam
                if now - self._last_log_time > 60:
                    logger.warning(
                        f"Rate limit reached ({self.max_calls} calls/{self.time_window}s). "
                        f"Waiting {wait_time:.1f} seconds..."
                    )
                    self._last_log_time = now

                time.sleep(wait_time)

                # Clean up old calls after waiting
                now = time.time()
                while self.calls and self.calls[0] < now - self.time_window:
                    self.calls.popleft()

        # Record this call
        self.calls.append(time.time())

    def get_remaining_calls(self) -> int:
        """
        Get the number of remaining calls in the current window.

        Returns:
            Number of calls that can still be made
        """
        now = time.time()

        # Remove calls outside the time window
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()

        return max(0, self.max_calls - len(self.calls))


# Global cache instance
_global_cache = APICache()

# Global rate limiters for different APIs
_openai_rate_limiter = RateLimiter(max_calls=50, time_window=60)  # 50 calls per minute


def get_cache() -> APICache:
    """Get the global cache instance."""
    return _global_cache


def get_openai_rate_limiter() -> RateLimiter:
    """Get the global OpenAI rate limiter."""
    return _openai_rate_limiter
