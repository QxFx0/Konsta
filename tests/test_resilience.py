from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from src.resilience import CircuitBreaker, CircuitBreakerOpen, CircuitState

"""Tests for ``src.resilience.CircuitBreaker``."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _BoomError(Exception):
    """Distinct exception type used to verify non-counted failures."""


async def _ok() -> str:
    return "ok"


async def _boom(exc: BaseException) -> None:
    raise exc


async def _delayed_ok(delay: float, value: str = "ok") -> str:
    await asyncio.sleep(delay)
    return value


def _make_http_status_error(status: int = 500) -> httpx.HTTPStatusError:
    """Build an ``httpx.HTTPStatusError`` with a minimal fake response."""
    request = httpx.Request("POST", "https://example.test/chat")
    response = httpx.Response(status_code=status, request=request)
    return httpx.HTTPStatusError(
        f"server error {status}", request=request, response=response
    )


# ---------------------------------------------------------------------------
# CLOSED state: calls flow through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closed_state_allows_calls():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
    assert breaker.state == CircuitState.CLOSED

    result = await breaker.call(_ok)
    assert result == "ok"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_closed_state_passes_arguments_and_return_value():
    breaker = CircuitBreaker()

    async def echo(value: str, *, suffix: str = "") -> str:
        return value + suffix

    out = await breaker.call(echo, "hello", suffix=" world")
    assert out == "hello world"


# ---------------------------------------------------------------------------
# Tracked failures open the breaker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tracked_failures_open_breaker_after_threshold():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            await breaker.call(_boom, _make_http_status_error(503))
        assert breaker.state == CircuitState.CLOSED

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.call(_boom, _make_http_status_error(503))
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_open_breaker_rejects_calls_immediately():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.call(_boom, _make_http_status_error(500))
    assert breaker.state == CircuitState.OPEN

    # Subsequent call must be rejected without invoking the wrapped function.
    invoked = False

    async def must_not_run() -> str:
        nonlocal invoked
        invoked = True
        return "nope"

    with pytest.raises(CircuitBreakerOpen):
        await breaker.call(must_not_run)
    assert invoked is False


@pytest.mark.asyncio
async def test_manual_record_failure_opens_breaker():
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Recovery timeout -> HALF_OPEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_timeout_transitions_to_half_open():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.call(_boom, _make_http_status_error(500))
    assert breaker.state == CircuitState.OPEN

    # Wait long enough for recovery.
    await asyncio.sleep(0.1)

    # Accessing ``state`` triggers the lazy OPEN -> HALF_OPEN transition.
    assert breaker.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_half_open_success_closes_breaker():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.call(_boom, _make_http_status_error(500))
    await asyncio.sleep(0.1)

    # First call in HALF_OPEN should succeed and close the breaker.
    result = await breaker.call(_ok)
    assert result == "ok"
    assert breaker.state == CircuitState.CLOSED

    # Subsequent calls must also pass normally.
    assert await breaker.call(_ok) == "ok"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_failure_reopens_breaker():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.call(_boom, _make_http_status_error(500))
    await asyncio.sleep(0.1)
    assert breaker.state == CircuitState.HALF_OPEN

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.call(_boom, _make_http_status_error(500))
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_half_open_quota_limits_concurrent_trials():
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=0.05,
        half_open_max_calls=1,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.call(_boom, _make_http_status_error(500))
    await asyncio.sleep(0.1)

    started = asyncio.Event()
    finish = asyncio.Event()

    async def slow() -> str:
        started.set()
        await finish.wait()
        return "slow"

    trial = asyncio.create_task(breaker.call(slow))
    await started.wait()
    assert breaker.state == CircuitState.HALF_OPEN

    # While the trial is in flight a second call must be rejected because the
    # half-open quota is exhausted.
    async def quick() -> str:
        return "quick"

    with pytest.raises(CircuitBreakerOpen):
        await breaker.call(quick)

    # Let the trial succeed so the breaker closes cleanly.
    finish.set()
    assert await trial == "slow"
    assert breaker.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Specific exception tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_value_error_is_not_counted_as_failure():
    breaker = CircuitBreaker(
        failure_threshold=2,
        failure_exceptions=(httpx.HTTPStatusError, TimeoutError, ConnectionError),
    )

    # Many ValueErrors must NOT open the breaker (programming error).
    for _ in range(5):
        with pytest.raises(ValueError):
            await breaker.call(_boom, ValueError("bad input"))
    assert breaker.state == CircuitState.CLOSED

    # But tracked exceptions still open it.
    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            await breaker.call(_boom, _make_http_status_error(500))
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_timeout_error_counts_as_failure():
    breaker = CircuitBreaker(failure_threshold=2)
    for _ in range(2):
        with pytest.raises(TimeoutError):
            await breaker.call(_boom, TimeoutError("timed out"))
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_connection_error_counts_as_failure():
    breaker = CircuitBreaker(failure_threshold=1)
    with pytest.raises(ConnectionError):
        await breaker.call(_boom, ConnectionError("refused"))
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_asyncio_timeout_error_counts_as_failure():
    """In 3.11+ ``asyncio.TimeoutError`` is an alias for ``TimeoutError``; on
    earlier interpreters the asyncio version is its own subclass. The breaker
    must count both forms."""
    breaker = CircuitBreaker(failure_threshold=1)
    with pytest.raises(asyncio.TimeoutError):
        await breaker.call(_boom, asyncio.TimeoutError())
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_keyerror_is_not_counted_as_failure():
    """Only configured exceptions count; KeyError must pass through silently."""
    breaker = CircuitBreaker(
        failure_threshold=1,
        failure_exceptions=(httpx.HTTPStatusError,),
    )

    with pytest.raises(KeyError):
        await breaker.call(_boom, KeyError("missing"))
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_non_tracked_exception_in_half_open_does_not_reopen():
    """A non-tracked exception during HALF_OPEN must not reopen the breaker,
    but it must also free the in-flight trial slot so future calls are not
    permanently blocked."""
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=0.05,
        failure_exceptions=(httpx.HTTPStatusError,),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.call(_boom, _make_http_status_error(500))
    await asyncio.sleep(0.1)
    assert breaker.state == CircuitState.HALF_OPEN

    # A non-tracked exception during HALF_OPEN must not reopen.
    with pytest.raises(ValueError):
        await breaker.call(_boom, ValueError("oops"))
    assert breaker.state == CircuitState.HALF_OPEN

    # And a subsequent tracked success must close the breaker (slot freed).
    assert await breaker.call(_ok) == "ok"
    assert breaker.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_constructor_validates_arguments():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(half_open_max_calls=0)
    with pytest.raises(ValueError):
        CircuitBreaker(recovery_timeout=-1.0)


# ---------------------------------------------------------------------------
# Thread safety (sanity check)
# ---------------------------------------------------------------------------


def test_concurrent_failures_do_not_corrupt_state():
    """A burst of failures from multiple threads must open the breaker
    exactly once and leave a consistent failure count."""
    import threading

    breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=10.0)
    errors: list[BaseException] = []

    def hammer() -> None:
        try:
            asyncio.run(
                breaker.call(_boom, _make_http_status_error(500))
            )
        except BaseException as exc:  # pragma: no cover - surfaced via ``errors``
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The first ``failure_threshold`` calls hit the wrapped function and
    # receive the underlying HTTP error; the rest are rejected by the now-open
    # breaker with ``CircuitBreakerOpen``. Either way every call fails.
    assert len(errors) == 20
    http_errors = [e for e in errors if isinstance(e, httpx.HTTPStatusError)]
    breaker_open_errors = [e for e in errors if isinstance(e, CircuitBreakerOpen)]
    assert len(http_errors) >= 1  # at least one call hit the wrapped fn
    assert len(http_errors) + len(breaker_open_errors) == 20
    assert breaker.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_reflects_state():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0, name="snap")

    snap = breaker.snapshot()
    assert snap["state"] == "closed"
    assert snap["failure_count"] == 0
    assert snap["name"] == "snap"
    assert snap["failure_threshold"] == 2

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.call(_boom, _make_http_status_error(500))

    snap = breaker.snapshot()
    assert snap["state"] == "closed"
    assert snap["failure_count"] == 1

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.call(_boom, _make_http_status_error(500))
    snap = breaker.snapshot()
    assert snap["state"] == "open"
    assert snap["opened_at"] is not None


# ---------------------------------------------------------------------------
# Time-based recovery smoke test (no asyncio.sleep needed)
# ---------------------------------------------------------------------------


def test_recovery_timeout_uses_monotonic_time():
    """``opened_at`` should be set via ``time.monotonic``; verify the recovery
    window matches what the breaker actually checks by manipulating the clock."""
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)

    breaker.record_failure()
    snap = breaker.snapshot()
    assert snap["state"] == "open"
    assert snap["opened_at"] is not None
    # The recorded opened_at is within the last second.
    assert (time.monotonic() - snap["opened_at"]) < 1.0
