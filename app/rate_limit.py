"""A small in-memory rate limiter.

One counter per client key (an IP address here), counting requests inside
a sliding window. It lives in memory, so each worker process keeps its own
counts — good enough for a single-instance bot; a shared store is a later
upgrade if the bot ever runs behind more than one worker.
"""

import time
from collections import deque


class RateLimiter:
    """Allow at most ``per_minute`` requests per key per minute."""

    def __init__(self, per_minute, window_seconds=60, clock=None):
        self.per_minute = per_minute
        self.window_seconds = window_seconds
        self._clock = clock or time.monotonic
        self._hits = {}

    def allow(self, key):
        """True when this request is inside the limit for ``key``."""
        if self.per_minute <= 0:
            return True
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window_seconds:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        return True

    def reset(self, key=None):
        """Forget recorded hits (all keys, or one)."""
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)
