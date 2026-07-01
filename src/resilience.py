"""Resilience primitives for external dependencies.

This module provides a simple three-state circuit breaker used to fail fast
when a downstream dependency (e.g. an LLM provider) is unhealthy. The
breaker is intentionally lightweight and dependency-free apart from
``httpx`` (which is already a project dependency).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Tuple, Type

import httpx

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Operating states for :class:`CircuitBreaker`."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when a call is rejected because the breaker is ``OPEN``.

    This is *not* counted as a breaker failure: it is a local refusal, not
    a remote error. Callers should treat it as a signal to back off / fall
    back rather than retry immediately.
    """


_DEFAULT_FAILURE_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    httpx.HTTPStatusError,
    httpx.TimeoutException,
    httpx.RequestError,
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
)


class CircuitBreaker:
    """A simple three-state circuit breaker.

    States:

    * ``CLOSED``     - calls flow through; tracked failures count toward
      ``failure_threshold``. Reaching the threshold transitions to ``OPEN``.
    * ``OPEN``       - calls are rejected immediately with
      :class:`CircuitBreakerOpen`. After ``recovery_timeout`` seconds the
      breaker transitions to ``HALF_OPEN``.
    * ``HALF_OPEN``  - at most ``half_open_max_calls`` trial calls are
      allowed concurrently. A trial success transitions to ``CLOSED``; a
      tracked failure transitions back to ``OPEN``.

    Only exceptions listed in ``failure_exceptions`` (or a caller-supplied
    tuple) count toward opening the breaker. Programming errors such as
    ``ValueError`` are re-raised untouched without affecting state.

    State updates are serialised with a :class:`threading.Lock`, which is
    safe to use from both threaded and ``asyncio`` code paths because the
    critical sections only mutate plain Python state (no ``await``).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        failure_exceptions: Optional[Tuple[Type[BaseException], ...]] = None,
        name: str = "CircuitBreaker",
    ):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        if recovery_timeout < 0:
            raise ValueError("recovery_timeout must be >= 0")
        if half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be > 0")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.failure_exceptions: Tuple[Type[BaseException], ...] = (
            failure_exceptions if failure_exceptions is not None else _DEFAULT_FAILURE_EXCEPTIONS
        )
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        self._half_open_in_flight = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Inspection helpers
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> CircuitState:
        """Current state, with a lazy OPEN -> HALF_OPEN transition."""
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            return self._state

    def snapshot(self) -> dict:
        """Return a JSON-serialisable view of the breaker (mostly for tests/logs)."""
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "half_open_in_flight": self._half_open_in_flight,
                "half_open_max_calls": self.half_open_max_calls,
                "opened_at": self._opened_at,
                "recovery_timeout": self.recovery_timeout,
            }

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    async def call(self, async_fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Invoke ``async_fn`` through the breaker.

        Raises:
            CircuitBreakerOpen: if the breaker is ``OPEN`` (or ``HALF_OPEN``
                and the in-flight trial quota is exhausted).
        """
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            if not self._admit_locked():
                raise CircuitBreakerOpen(
                    f"{self.name} is {self._state.value}; refusing call"
                )

        try:
            result = await async_fn(*args, **kwargs)
        except BaseException as exc:
            self._handle_exception(exc)
            raise
        else:
            with self._lock:
                self._on_success_locked()
            return result

    def record_failure(self) -> None:
        """Manually record a failure (e.g. when a callee swallows the exception).

        Useful when the wrapped function catches its own errors and returns
        a sentinel like ``None``; the caller can then explicitly inform the
        breaker that the call should count as a failure.
        """
        with self._lock:
            self._on_failure_locked()

    def record_success(self) -> None:
        """Manually record a success.

        Mostly useful in tests or when the caller wants to acknowledge
        success without going through :meth:`call` (e.g. for a swallowed
        exception that was actually benign).
        """
        with self._lock:
            self._on_success_locked()

    # ------------------------------------------------------------------ #
    # State machine (callers must hold ``self._lock``)
    # ------------------------------------------------------------------ #

    def _maybe_transition_to_half_open_locked(self) -> None:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if (time.monotonic() - self._opened_at) >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_in_flight = 0
                logger.info("%s transitioning OPEN -> HALF_OPEN", self.name)

    def _admit_locked(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_in_flight < self.half_open_max_calls:
                self._half_open_in_flight += 1
                return True
            return False
        return False  # OPEN

    def _on_success_locked(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at = None
            self._half_open_in_flight = 0
            logger.info("%s transitioning HALF_OPEN -> CLOSED", self.name)
        else:
            # CLOSED: reset the failure window after any success.
            self._failure_count = 0

    def _on_failure_locked(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            self._half_open_in_flight = 0
            logger.warning("%s transitioning HALF_OPEN -> OPEN", self.name)
            return

        # CLOSED
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "%s transitioning CLOSED -> OPEN (failures=%d threshold=%d)",
                self.name,
                self._failure_count,
                self.failure_threshold,
            )

    # ------------------------------------------------------------------ #
    # Exception classification
    # ------------------------------------------------------------------ #

    def _handle_exception(self, exc: BaseException) -> None:
        with self._lock:
            if isinstance(exc, self.failure_exceptions):
                self._on_failure_locked()
            elif self._state == CircuitState.HALF_OPEN:
                # Non-tracked exception during a trial: don't reopen, but
                # free the trial slot so future calls aren't permanently
                # blocked.
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
