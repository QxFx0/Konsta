"""
Background asyncio worker that decouples LLM context distillation from the
mitmproxy request path.

The mitmproxy ``request`` handler is synchronous and runs on a mitmproxy
worker thread, so it cannot ``await`` the LLM directly without blocking
other in-flight flows.  :class:`DistillationWorker` accepts requests from
any thread via :meth:`DistillationWorker.submit`, executes them on a
dedicated asyncio loop, and stores the results in a thread-safe LRU cache
keyed by ``request_id`` so that the handler can later retrieve them
through :meth:`DistillationWorker.get_result`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .llm_processor import LLMProcessor
from .metrics import Metrics

logger = logging.getLogger(__name__)


# A pending task is just the (request_id, compressed_text) pair handed to us
# by the mitmproxy handler thread.
PendingTask = Tuple[str, str]


class DistillationWorker:
    """Asyncio background worker that performs LLM distillation off the request path.

    Responsibilities:
        * Accept submissions from any thread (typically the mitmproxy handler).
        * Run ``processor.process_context`` (or ``processor.distill`` when
          available) inside the asyncio loop the worker was started in.
        * Cache the resulting text keyed by ``request_id`` in a bounded LRU
          so the handler can fetch it later via :meth:`get_result`.

    Threading model:
        * ``start`` and ``stop`` must be awaited from inside the asyncio loop
          that owns the worker.
        * ``submit`` and ``get_result`` are safe to invoke from any thread,
          including the synchronous mitmproxy handler thread.
    """

    def __init__(
        self,
        processor: LLMProcessor,
        cache_size: int = 128,
        metrics: Optional[Metrics] = None,
    ):
        """Store dependencies and prepare empty asyncio primitives.

        Args:
            processor: LLM processor exposing an async ``process_context``
                coroutine (or, optionally, an async ``distill`` coroutine).
            cache_size: Maximum number of ``request_id`` results kept in the
                LRU cache.
            metrics: Optional :class:`Metrics` instance used to record
                latency and success of every distillation call. When
                ``None``, the process-wide shared singleton is used so the
                rest of the codebase observes the same store.
        """
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.processor = processor
        self._cache_size = cache_size
        # Metrics is plumbed in explicitly. When ``None`` we fall back to
        # a fresh local ``Metrics`` so the worker always has somewhere to
        # record; callers that want to share state should pass the same
        # instance through.
        if metrics is None:
            metrics = Metrics()
        self.metrics = metrics

        # request_id -> distilled text; LRU bounded, thread-safe.
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._cache_lock = threading.Lock()

        # Buffer for tasks submitted from other threads.  ``submit`` appends
        # here under a lock, then schedules an asyncio-side drain.
        self._pending: List[PendingTask] = []
        self._pending_lock = threading.Lock()

        # Asyncio primitives created in ``start`` so they bind to the loop.
        self._queue: Optional[asyncio.Queue[PendingTask]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._stopping = False
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the asyncio queue and launch the worker loop.

        Must be awaited from inside the event loop that will own the worker.
        """
        if self._running:
            logger.warning("DistillationWorker.start() called while already running; ignoring.")
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._stopping = False
        self._running = True
        self._worker_task = asyncio.create_task(self.run(), name="konsta-distillation-worker")
        logger.info("DistillationWorker started.")

    async def stop(self) -> None:
        """Drain the queue and stop the worker loop cleanly.

        After ``stop`` returns, no further ``submit`` calls are accepted and
        the worker task has exited.  Any requests already in the queue are
        processed before the loop terminates.

        Shutdown uses a sentinel ``None`` task: ``stop`` first flushes the
        cross-thread ``_pending`` buffer, then puts the sentinel at the end
        of ``_queue``.  The :meth:`run` loop awaits ``_queue.get()`` with no
        timeout, so the loop wakes up as soon as either real tasks or the
        sentinel become available.
        """
        if not self._running:
            return
        # Set ``_stopping`` first so ``submit`` rejects new tasks before we
        # drain, minimising the window in which a producer thread could
        # append to ``_pending`` after the drain but before the sentinel.
        self._stopping = True
        self._drain_pending()
        queue = self._queue
        if queue is not None:
            await queue.put(None)
        task = self._worker_task
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        self._running = False
        logger.info("DistillationWorker stopped.")

    # ------------------------------------------------------------------
    # Producer / consumer API
    # ------------------------------------------------------------------

    def submit(self, request_id: str, compressed_text: str) -> None:
        """Enqueue a distillation request from any thread.

        The task is buffered under a lock and a drain callback is scheduled
        on the worker's asyncio loop via ``call_soon_threadsafe``.  This is
        safe to invoke from the mitmproxy handler thread and guarantees that
        each task is moved from ``_pending`` into ``_queue`` exactly once.

        Args:
            request_id: Caller-defined identifier used to retrieve the
                result via :meth:`get_result`.
            compressed_text: Already-compressed prompt body to distill.

        Raises:
            RuntimeError: If the worker has not been started yet, or has
                already been stopped. After ``stop`` begins no further
                submissions are accepted so the sentinel-based shutdown can
                observe the sentinel as the last item in the queue.
        """
        loop = self._loop
        if loop is None or not self._running:
            raise RuntimeError("DistillationWorker.submit() called before start()")
        task: PendingTask = (request_id, compressed_text)
        with self._pending_lock:
            if self._stopping:
                raise RuntimeError(
                    "DistillationWorker.submit() called after stop()"
                )
            self._pending.append(task)
        try:
            loop.call_soon_threadsafe(self._drain_pending)
        except RuntimeError:
            # Loop is closed; treat as a best-effort drop with a warning.
            logger.warning(
                "DistillationWorker.submit() could not reach event loop; dropping %s",
                request_id,
            )

    def get_result(self, request_id: str) -> Optional[str]:
        """Return the latest cached distillation result for ``request_id``.

        Returns ``None`` if the worker has not finished processing the
        request yet, the request was never submitted, or the cached entry
        has been evicted.
        """
        with self._cache_lock:
            value = self._cache.get(request_id)
            if value is not None:
                self._cache.move_to_end(request_id)
            return value

    # ------------------------------------------------------------------
    # Worker internals
    # ------------------------------------------------------------------

    def _drain_pending(self) -> None:
        """Move every buffered task from ``_pending`` into ``_queue`` once.

        This method is synchronous and safe to schedule with
        ``call_soon_threadsafe`` from any thread.  Because it pops the
        buffered list under ``_pending_lock`` and immediately enqueues the
        items with ``put_nowait``, a task can never be enqueued twice.

        The drain does not inspect ``_stopping``: :meth:`stop` itself
        relies on this method to flush the buffer before it inserts the
        ``None`` sentinel. New submissions are rejected by :meth:`submit`
        once ``_stopping`` is set, so a drain callback that fires after
        the sentinel cannot add new items because there are none in the
        buffer.
        """
        queue = self._queue
        if queue is None:
            return
        with self._pending_lock:
            buffered = self._pending
            self._pending = []
        for task in buffered:
            try:
                queue.put_nowait(task)
            except asyncio.QueueFull:
                logger.warning(
                    "DistillationWorker queue is full; dropping %s",
                    task[0],
                )

    async def _flush_pending(self) -> None:
        """Awaitable wrapper used by :meth:`stop` to flush cross-thread buffer."""
        self._drain_pending()

    async def _process_one(self, task: PendingTask) -> None:
        """Run distillation for a single task and store the result."""
        request_id, compressed_text = task

        # Time the LLM call so we can report distillation latency through
        # the metrics store. ``time.monotonic`` is monotonic and immune to
        # wall-clock adjustments, which is what latency needs.
        start = time.monotonic()
        success = False
        try:
            # Prefer a dedicated ``distill`` coroutine when the processor exposes
            # one; otherwise fall back to ``process_context`` with the compressed
            # text wrapped as a single user message.
            distill = getattr(self.processor, "distill", None)
            if callable(distill):
                raw = await distill(compressed_text)
                if isinstance(raw, str):
                    result: str = raw
                elif raw is None:
                    result = compressed_text
                else:
                    result = str(raw)
            else:
                messages: List[Dict[str, Any]] = [
                    {"role": "user", "content": compressed_text}
                ]
                processed = await self.processor.process_context(messages)
                result = self._flatten_messages(processed, fallback=compressed_text)
            success = True
        finally:
            # Record the observation exactly once per task, regardless of
            # outcome. Exceptions propagate to the outer ``run`` loop
            # which logs and continues with the next task.
            latency_ms = (time.monotonic() - start) * 1000.0
            self._record_distillation_metric(latency_ms, success=success)

        with self._cache_lock:
            if request_id in self._cache:
                self._cache.move_to_end(request_id)
            self._cache[request_id] = result
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        logger.debug("DistillationWorker stored result for %s", request_id)

    def _record_distillation_metric(self, latency_ms: float, success: bool) -> None:
        """Safely record one distillation observation.

        A misbehaving metrics store must never break the worker loop, so
        any exception from ``record_distillation`` is swallowed and
        logged. The latency is clamped to ``>= 0`` by the metrics class
        itself, so we just forward the value.
        """
        try:
            self.metrics.record_distillation(latency_ms, success)
        except Exception:
            logger.exception("Failed to record distillation metrics")

    @staticmethod
    def _flatten_messages(
        messages: Optional[List[Dict[str, Any]]],
        fallback: str,
    ) -> str:
        """Convert a list of message dicts back into a single content string."""
        if not messages:
            return fallback
        parts: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
        if not parts:
            return fallback
        return "\n".join(parts)

    async def run(self) -> None:
        """Worker loop.  Processes queued tasks until :meth:`stop` is called.

        The loop awaits :meth:`asyncio.Queue.get` with no timeout, so it
        blocks until either a real task or the ``None`` sentinel arrives.
        This removes the previous 50 ms polling in favour of a clean
        event-driven wakeup.
        """
        if self._queue is None:
            raise RuntimeError("DistillationWorker.run() called before start()")
        logger.debug("DistillationWorker loop running.")
        try:
            while True:
                task = await self._queue.get()
                if task is None:
                    # Sentinel from ``stop`` signals shutdown.
                    break
                try:
                    await self._process_one(task)
                except asyncio.CancelledError:
                    # Cooperative cancellation: do NOT swallow -- re-raise
                    # so the asyncio task terminates cleanly and
                    # ``run_until_complete`` (in ``stop``) observes the
                    # cancellation. The ``finally`` block still runs
                    # ``task_done`` so the queue bookkeeping stays
                    # consistent.
                    raise
                except (
                    RuntimeError,
                    ValueError,
                    TypeError,
                    OSError,
                    KeyError,
                    AttributeError,
                    IndexError,
                ) as exc:
                    # Worker-specific errors: log and keep the loop
                    # running so a single bad task can't tear down the
                    # distillation pipeline. We deliberately exclude
                    # broader ``Exception`` so genuinely unexpected
                    # failures surface in tests/operations instead of
                    # being silently logged.
                    logger.exception(
                        "DistillationWorker failed for %s: %s",
                        task[0],
                        exc,
                    )
                finally:
                    self._queue.task_done()
        finally:
            logger.debug("DistillationWorker loop exited.")
