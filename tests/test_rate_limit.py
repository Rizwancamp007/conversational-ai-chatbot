from app.rate_limit import RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_allows_up_to_limit_then_blocks():
    clock = FakeClock()
    limiter = RateLimiter(3, clock=clock)
    assert limiter.allow("1.2.3.4")
    assert limiter.allow("1.2.3.4")
    assert limiter.allow("1.2.3.4")
    assert not limiter.allow("1.2.3.4")


def test_limits_are_per_client():
    clock = FakeClock()
    limiter = RateLimiter(1, clock=clock)
    assert limiter.allow("1.1.1.1")
    assert limiter.allow("2.2.2.2")
    assert not limiter.allow("1.1.1.1")


def test_window_slides_after_a_minute():
    clock = FakeClock()
    limiter = RateLimiter(1, clock=clock)
    assert limiter.allow("1.2.3.4")
    assert not limiter.allow("1.2.3.4")
    clock.now = 61
    assert limiter.allow("1.2.3.4")


def test_zero_limit_disables_limiting():
    limiter = RateLimiter(0)
    assert limiter.allow("1.2.3.4")
    assert limiter.allow("1.2.3.4")


def test_reset_forgets_hits():
    clock = FakeClock()
    limiter = RateLimiter(1, clock=clock)
    assert limiter.allow("1.2.3.4")
    assert not limiter.allow("1.2.3.4")
    limiter.reset("1.2.3.4")
    assert limiter.allow("1.2.3.4")
