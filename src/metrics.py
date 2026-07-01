"""Thread-safe metrics counters and gauges for Konsta.

This module exposes :class:`Metrics`, a small in-process metrics store that
the rest of the codebase can write to without worrying about contention.
The store tracks four primary metrics, exposed both as named properties
(``compression_ratio``, ``distillation_latency_ms``,
``distillation_failures_total``, ``requests_processed_total``) and via
:meth:`Metrics.snapshot` for one-shot reporting (e.g. from a ``doctor``
command).

A module-level shared singleton is also exposed as a convenience for
operator-facing diagnostics. The proxy is loaded by ``mitmdump`` as a
subprocess addon, so :mod:`src.proxy_core` cannot receive a
:class:`Metrics` instance directly through ``main``. ``main`` therefore
constructs the canonical instance and publishes it through
:func:`set_shared_metrics`; :func:`get_shared_metrics` returns it without
auto-creating one. Components inside the proxy should construct their own
:class:`Metrics` instance (and pass it explicitly through their
constructors) rather than relying on the shared singleton.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Shared singleton plumbing.
# ---------------------------------------------------------------------------

_shared_metrics: Optional["Metrics"] = None
_shared_metrics_lock = threading.Lock()


def get_shared_metrics() -> Optional["Metrics"]:
    """Return the process-wide shared :class:`Metrics` instance, if any.

    Returns the instance previously published via :func:`set_shared_metrics`,
    or ``None`` when nothing has been installed yet. Callers that need a
    :class:`Metrics` instance unconditionally should not rely on this
    helper to materialise one; construct a local :class:`Metrics` instead.
    ``main.py`` still calls :func:`set_shared_metrics` early in startup so
    the rest of the operator-facing codebase can observe the configured
    instance via this accessor.
    """
    return _shared_metrics


def set_shared_metrics(metrics: "Metrics") -> None:
    """Install ``metrics`` as the shared :class:`Metrics` instance.

    Intended to be called once by ``main.py`` during startup so that the
    mitmdump subprocess addon (``proxy_core``) can publish to the same
    store the operator inspects.
    """
    global _shared_metrics
    with _shared_metrics_lock:
        _shared_metrics = metrics


def reset_shared_metrics() -> None:
    """Drop the shared instance so the next caller gets a fresh one.

    Primarily useful in tests that need to isolate state between cases.
    """
    global _shared_metrics
    with _shared_metrics_lock:
        _shared_metrics = None


# ---------------------------------------------------------------------------
# Metrics implementation.
# ---------------------------------------------------------------------------


class Metrics:
    """Thread-safe counters/gauges for Konsta.

    The store is intentionally simple: a single :class:`threading.Lock`
    guards every read and write, so all public methods and the
    :meth:`snapshot` helper are safe to call from any thread. This matches
    the threading model of :class:`DistillationWorker` (mitmproxy handler
    thread + dedicated asyncio loop thread) and the
    :class:`ContextCompressionProxy` request path.

    Recorded values:

        * ``compression_ratio`` (gauge): total compressed bytes divided by
          total original bytes across every ``record_compression`` call.
          ``0.0`` when no compression has been recorded yet.
        * ``distillation_latency_ms`` (gauge): mean latency (in
          milliseconds) of every ``record_distillation`` call, regardless
          of success. ``0.0`` when nothing has been recorded yet.
        * ``distillation_failures_total`` (counter): number of
          ``record_distillation(..., success=False)`` calls.
        * ``requests_processed_total`` (counter): number of intercepted
          requests observed via :meth:`record_request`.
    """

    def __init__(self) -> None:
        # Single lock guarding every attribute on this instance. All
        # methods take it before reading or mutating state.
        self._lock = threading.Lock()

        # Running totals used to derive gauges on demand.
        self._compression_orig_bytes: int = 0
        self._compression_compressed_bytes: int = 0
        self._compression_count: int = 0

        self._distillation_latency_total_ms: float = 0.0
        self._distillation_count: int = 0

        # Pure monotonic counters.
        self._distillation_failures_total: int = 0
        self._requests_processed_total: int = 0

        # Unexpected-crash counter. ``_crashes`` maps an exception class
        # name (e.g. ``"RuntimeError"``) to the number of times it has
        # been observed via :meth:`record_crash`. Components that narrow
        # their ``except`` handlers call this helper when they encounter
        # an exception they cannot classify, so observability is preserved
        # without re-introducing a broad ``except Exception`` that would
        # swallow real bugs.
        self._crashes: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording API.
    # ------------------------------------------------------------------

    def record_compression(self, original_size: int, compressed_size: int) -> None:
        """Record one compression observation.

        Args:
            original_size: Byte length of the input payload before
                compression. Must be non-negative.
            compressed_size: Byte length of the payload after
                compression. Must be non-negative.

        Raises:
            TypeError: If either argument is not coercible to ``int``.
            ValueError: If either argument is negative.
        """
        orig = int(original_size)
        comp = int(compressed_size)
        if orig < 0 or comp < 0:
            raise ValueError(
                "record_compression sizes must be non-negative "
                f"(original_size={orig}, compressed_size={comp})"
            )
        with self._lock:
            self._compression_orig_bytes += orig
            self._compression_compressed_bytes += comp
            self._compression_count += 1

    def record_distillation(self, latency_ms: float, success: bool) -> None:
        """Record one distillation observation.

        Args:
            latency_ms: Wall-clock latency of the distillation call in
                milliseconds. Negative values are clamped to ``0.0`` to
                keep the running average meaningful (e.g. clock skew or a
                bug returning a negative duration).
            success: ``True`` if the LLM call succeeded, ``False``
                otherwise. Failures increment
                ``distillation_failures_total``.
        """
        latency = float(latency_ms)
        if latency < 0.0:
            latency = 0.0
        with self._lock:
            self._distillation_latency_total_ms += latency
            self._distillation_count += 1
            if not success:
                self._distillation_failures_total += 1

    def record_request(self) -> None:
        """Record one intercepted request."""
        with self._lock:
            self._requests_processed_total += 1

    def record_crash(self, error_type: str) -> None:
        """Record one unexpected (unclassified) exception.

        Narrowed exception handlers use this when they encounter an
        exception class they did not explicitly handle: the exception is
        still logged and the offending component keeps running, but the
        crash is counted so operators can spot regressions in the
        handler coverage.

        Args:
            error_type: Class name of the exception (typically
                ``type(exc).__name__``). An empty / non-string value is
                coerced to ``"Unknown"`` so the counter dict never
                accumulates junk keys.
        """
        if not isinstance(error_type, str) or not error_type:
            error_type = "Unknown"
        with self._lock:
            self._crashes[error_type] = self._crashes.get(error_type, 0) + 1

    # ------------------------------------------------------------------
    # Snapshot / gauges.
    # ------------------------------------------------------------------

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed bytes to original bytes.

        Returns ``0.0`` when no compression has been recorded so callers
        can safely use the value without a separate "no data" check.
        """
        with self._lock:
            if self._compression_orig_bytes == 0:
                return 0.0
            return self._compression_compressed_bytes / self._compression_orig_bytes

    @property
    def distillation_latency_ms(self) -> float:
        """Mean observed distillation latency in milliseconds.

        Returns ``0.0`` when no distillation has been recorded yet.
        """
        with self._lock:
            if self._distillation_count == 0:
                return 0.0
            return self._distillation_latency_total_ms / self._distillation_count

    @property
    def distillation_failures_total(self) -> int:
        """Cumulative number of failed distillation calls."""
        with self._lock:
            return self._distillation_failures_total

    @property
    def requests_processed_total(self) -> int:
        """Cumulative number of intercepted requests recorded."""
        with self._lock:
            return self._requests_processed_total

    @property
    def crashes(self) -> Dict[str, int]:
        """Mapping of ``error_type`` -> count for unexpected exceptions.

        Returns a shallow copy so callers can mutate the result without
        disturbing the store.
        """
        with self._lock:
            return dict(self._crashes)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of every metric.

        The snapshot is taken under the instance lock so it represents a
        consistent point-in-time view of all gauges and counters. This is
        the format consumed by diagnostic commands such as ``doctor``.
        """
        with self._lock:
            ratio = (
                self._compression_compressed_bytes / self._compression_orig_bytes
                if self._compression_orig_bytes > 0
                else 0.0
            )
            avg_latency = (
                self._distillation_latency_total_ms / self._distillation_count
                if self._distillation_count > 0
                else 0.0
            )
            return {
                "compression_ratio": ratio,
                "distillation_latency_ms": avg_latency,
                "distillation_failures_total": self._distillation_failures_total,
                "requests_processed_total": self._requests_processed_total,
                "crashes": dict(self._crashes),
                "crashes_total": sum(self._crashes.values()),
                # Bookkeeping fields that operators may also want to see.
                "compression_count": self._compression_count,
                "compression_orig_bytes": self._compression_orig_bytes,
                "compression_compressed_bytes": self._compression_compressed_bytes,
                "distillation_count": self._distillation_count,
            }
