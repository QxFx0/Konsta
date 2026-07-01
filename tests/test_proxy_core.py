import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.compression_engine import CompressionEngine
from src.distillation_worker import DistillationWorker
from src.engine import KonstaEngine
from src.llm_processor import LLMProcessor
from src.metrics import Metrics
from src.proxy_adapter import MitmproxyAdapter
from src.proxy_core import ContextCompressionProxy
from tests.conftest import FakeProcessor, _wait_for_cached


class _Request:
    """Mutable request stand-in. Exposes the attributes proxy_core touches."""

    def __init__(self, host, content=b""):
        self.host = host
        self.pretty_host = host
        self.content = content
        self.headers = {}
        self.pretty_url = f"https://{host}/"

    def get_text(self):
        return self.content.decode("utf-8") if self.content else ""

    def set_text(self, text):
        self.content = text.encode("utf-8") if isinstance(text, str) else text


class _Response:
    def __init__(self, content=b""):
        self.content = content
        self.headers = {}
        self.status_code = 200

    def get_text(self):
        return self.content.decode("utf-8") if self.content else ""


def _make_flow(host, content=b"", response_content=None):
    """Build a test flow whose type passes isinstance(flow, http.HTTPFlow)."""
    from tests.conftest import HTTPFlow

    flow = HTTPFlow()
    flow.request = _Request(host, content=content)
    if response_content is not None:
        flow.response = _Response(response_content)
    return flow


def test_is_target_request_exact_match():
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    flow = _make_flow("api.openai.com")
    assert proxy.is_target_request(flow) is True


def test_is_target_request_suffix_match():
    proxy = ContextCompressionProxy(target_hosts=["openai.com"])
    flow = _make_flow("api.openai.com")
    assert proxy.is_target_request(flow) is True


def test_is_target_request_false_positive():
    proxy = ContextCompressionProxy(target_hosts=["openai.com"])
    flow = _make_flow("fake-api.openai.com.evil.com")
    assert proxy.is_target_request(flow) is False


def test_is_target_request_no_match():
    proxy = ContextCompressionProxy(target_hosts=["openai.com"])
    flow = _make_flow("google.com")
    assert proxy.is_target_request(flow) is False


def test_request_interception_target():
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    flow = _make_flow("api.openai.com", content=b'{"messages":[],"model":"x"}')

    # Compression logic now lives on the engine; patch it there.
    with patch.object(proxy.engine, 'compress_context', return_value={"messages": []}) as mock_compress:
        proxy.request(flow)
        mock_compress.assert_called_once()
        assert flow.request.content == b'{"messages": []}'


def test_request_interception_non_target():
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    flow = _make_flow("google.com", content=b"test content")

    with patch.object(proxy.engine, 'compress_context') as mock_compress:
        proxy.request(flow)
        mock_compress.assert_not_called()
        assert flow.request.content == b"test content"


def test_response_interception_target():
    # The current response() handler records pre/post compression metrics on
    # the response headers and dumps the payload. Verify the observable side
    # effects.
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    flow = _make_flow(
        "api.openai.com",
        content=b"response request body",
        response_content=b"response content",
    )

    with patch.object(proxy.engine, '_dump_data') as mock_dump:
        proxy.response(flow)
        mock_dump.assert_called_once()
        assert "X-Konsta-Original-Size" in flow.response.headers
        assert "X-Konsta-Compressed-Size" in flow.response.headers


def test_response_interception_non_target():
    """Non-target responses must be untouched: no headers, no dumps."""
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    flow = _make_flow(
        "google.com",
        content=b"response request body",
        response_content=b"response content",
    )

    with patch.object(proxy.engine, '_dump_data') as mock_dump:
        proxy.response(flow)

    assert flow.response.content == b"response content"
    mock_dump.assert_not_called()
    assert "X-Konsta-Original-Size" not in flow.response.headers
    assert "X-Konsta-Compressed-Size" not in flow.response.headers


# ---------------------------------------------------------------------------
# Non-blocking behaviour: request() must not wait for LLM distillation.
# ---------------------------------------------------------------------------


def test_request_does_not_block_on_slow_distillation():
    """request() must return immediately even when the worker is slow."""
    # Processor takes 500ms per call; if request() were blocking on the
    # worker, this test would take >= 500ms.
    slow_processor = FakeProcessor(output="SLOW_DISTILLED", delay=0.5)
    worker = DistillationWorker(slow_processor, cache_size=4)

    proxy = ContextCompressionProxy(
        target_hosts=["api.openai.com"],
        distillation_worker=worker,
    )

    flow = _make_flow(
        "api.openai.com",
        content=b'{"messages":[{"role":"user","content":"hello"}],"model":"x"}',
    )

    with patch.object(
        proxy.engine,
        'compress_context',
        return_value={"messages": [{"role": "user", "content": "LOCAL"}]},
    ):
        start = time.monotonic()
        proxy.request(flow)
        elapsed = time.monotonic() - start

    # request() must finish much faster than the worker delay. Generous
    # bound to avoid flakiness on slow CI.
    assert elapsed < 0.2, (
        f"request() blocked for {elapsed:.3f}s while worker delay is 0.5s"
    )
    # The flow body is the locally compressed payload (cached not ready yet).
    assert flow.request.content == b'{"messages": [{"role": "user", "content": "LOCAL"}]}'
    # The task was submitted to the worker even though processing is slow.
    assert proxy.engine._last_distillation_request_id == "req-1"


def test_cached_distillation_applied_on_next_request():
    """A distillation result finished in the background is applied on the next request."""
    fast_processor = FakeProcessor(output="DISTILLED_VERSION", delay=0.01)
    worker = DistillationWorker(fast_processor, cache_size=4)

    proxy = ContextCompressionProxy(
        target_hosts=["api.openai.com"],
        distillation_worker=worker,
    )

    # First request: compresses locally and submits to the worker.
    flow1 = _make_flow(
        "api.openai.com",
        content=b'{"messages":[{"role":"user","content":"first"}],"model":"x"}',
    )
    with patch.object(
        proxy.engine,
        'compress_context',
        return_value={"messages": [{"role": "user", "content": "LOCAL"}]},
    ):
        proxy.request(flow1)

    # Wait for the worker to finish processing request #1.
    cached = _wait_for_cached(worker, "req-1", timeout=2.0)
    assert cached == "DISTILLED_VERSION"
    assert proxy.engine._last_distillation_request_id == "req-1"

    # Second request: should pick up the cached result from request #1 and
    # substitute it into flow.request.
    flow2 = _make_flow(
        "api.openai.com",
        content=b'{"messages":[{"role":"user","content":"second"}],"model":"x"}',
    )
    with patch.object(
        proxy.engine,
        'compress_context',
        return_value={"messages": [{"role": "user", "content": "LOCAL"}]},
    ) as mock_compress:
        proxy.request(flow2)

    # The cached distilled body must have been written into the second flow
    # as valid provider JSON containing the distilled content.
    parsed = json.loads(flow2.request.content.decode("utf-8"))
    assert parsed["messages"] == [{"role": "user", "content": "DISTILLED_VERSION"}]
    # compress_context still ran (used to build the JSON that was submitted
    # for distillation on this cycle).
    mock_compress.assert_called_once()
    # And a new task was submitted for the second request.
    assert proxy.engine._last_distillation_request_id == "req-2"


def test_request_submits_to_worker_when_distillation_enabled():
    """When a worker is present, request() must hand off work to it."""
    processor = FakeProcessor(output="X", delay=0.0)
    worker = DistillationWorker(processor, cache_size=4)

    proxy = ContextCompressionProxy(
        target_hosts=["api.openai.com"],
        distillation_worker=worker,
    )

    flow = _make_flow(
        "api.openai.com",
        content=b'{"messages":[{"role":"user","content":"hi"}],"model":"x"}',
    )
    with patch.object(
        proxy.engine,
        'compress_context',
        return_value={"messages": [{"role": "user", "content": "C"}]},
    ):
        proxy.request(flow)

    # Hand-off must have happened synchronously inside request(): the
    # pending buffer should already contain the task even before the loop
    # has had a chance to drain it.
    pending_snapshot = list(worker._pending)
    assert len(pending_snapshot) >= 1
    request_ids = {task[0] for task in pending_snapshot}
    assert "req-1" in request_ids
    # The submitted body is the locally compressed JSON.
    submitted_bodies = {task[1] for task in pending_snapshot}
    assert b'{"messages": [{"role": "user", "content": "C"}]}'.decode() in submitted_bodies


# ---------------------------------------------------------------------------
# Async loop startup: ``_init_async_loop`` must use ``threading.Event``
# (set by the background thread) instead of a busy-wait ``time.sleep``
# poll, and the Event must be observable as set once construction returns.
# ---------------------------------------------------------------------------


def test_loop_ready_event_is_initialized_in_ctor():
    """``_loop_ready`` must exist on the instance before the worker thread starts.

    This guards against the race where the background thread calls
    ``self._loop_ready.set()`` before the main thread has assigned the
    attribute. We verify that the Event is present and not yet set
    immediately after ``__init__`` is entered, by constructing a proxy
    and then checking the attribute exists.
    """
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    try:
        assert isinstance(proxy._loop_ready, threading.Event)
    finally:
        # Tear down: ask the loop to stop. We don't await it because the
        # loop is daemon and will be cleaned up at interpreter exit.
        if proxy._async_loop is not None and proxy._async_loop.is_running():
            proxy._async_loop.call_soon_threadsafe(proxy._async_loop.stop)


def test_loop_ready_is_set_after_construction():
    """After construction the Event must be set (loop is up)."""
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    try:
        assert proxy._loop_started is True
        assert proxy._loop_ready.is_set() is True
    finally:
        if proxy._async_loop is not None and proxy._async_loop.is_running():
            proxy._async_loop.call_soon_threadsafe(proxy._async_loop.stop)


def test_loop_ready_wait_returns_immediately_after_construction():
    """``_loop_ready.wait()`` must not block once the proxy has been built."""
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    try:
        # A generous timeout; in practice this returns essentially
        # instantly because the Event is already set.
        assert proxy._loop_ready.wait(timeout=1.0) is True
    finally:
        if proxy._async_loop is not None and proxy._async_loop.is_running():
            proxy._async_loop.call_soon_threadsafe(proxy._async_loop.stop)


def test_proxy_construction_does_not_busy_wait():
    """``__init__`` must not perform a polling ``time.sleep`` loop.

    We assert on the source of ``_init_async_loop`` to lock in the
    implementation choice: a ``threading.Event.wait`` (or equivalent
    non-busy primitive), not a ``while ... time.sleep`` poll.
    """
    import inspect

    from src.proxy_core import ContextCompressionProxy

    source = inspect.getsource(ContextCompressionProxy._init_async_loop)
    # The Event-based wait replaces the old busy-wait. We forbid the
    # legacy polling pattern explicitly so future edits cannot silently
    # reintroduce it.
    assert "time.sleep" not in source, (
        "_init_async_loop still contains a time.sleep busy-wait; "
        "use threading.Event.wait() instead"
    )
    assert "_loop_ready" in source


def test_proxy_starts_with_injected_distillation_worker():
    """Injected worker does not prevent the async loop from coming up."""
    processor = FakeProcessor(output="X", delay=0.0)
    worker = DistillationWorker(processor, cache_size=4)

    proxy = ContextCompressionProxy(
        target_hosts=["api.openai.com"],
        distillation_worker=worker,
    )
    try:
        assert proxy._loop_started is True
        assert proxy._loop_ready.is_set() is True
        # The loop should be running on its dedicated thread.
        assert proxy._async_thread is not None
        assert proxy._async_thread.is_alive() is True
        assert proxy._async_loop is not None
        assert proxy._async_loop.is_running() is True
    finally:
        if proxy._async_loop is not None and proxy._async_loop.is_running():
            proxy._async_loop.call_soon_threadsafe(proxy._async_loop.stop)


def test_loop_ready_is_event_instance():
    """The readiness primitive must be a ``threading.Event``."""
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    try:
        assert isinstance(proxy._loop_ready, threading.Event)
        # Event objects expose ``wait`` / ``set`` / ``is_set`` /
        # ``clear``. Confirm at least the API surface we rely on.
        assert hasattr(proxy._loop_ready, "wait")
        assert hasattr(proxy._loop_ready, "set")
        assert hasattr(proxy._loop_ready, "is_set")
        assert hasattr(proxy._loop_ready, "clear")
    finally:
        if proxy._async_loop is not None and proxy._async_loop.is_running():
            proxy._async_loop.call_soon_threadsafe(proxy._async_loop.stop)


# ---------------------------------------------------------------------------
# done() shutdown semantics.
# ---------------------------------------------------------------------------


def _build_mock_worker() -> MagicMock:
    """Return a :class:`MagicMock` that quacks like a DistillationWorker.

    The proxy's background loop invokes ``worker.start()`` and
    ``worker.stop()`` from inside ``run_until_complete`` calls, so both
    methods must return awaitables. ``AsyncMock`` produces awaitables by
    default, which is what we need here.
    """
    worker = MagicMock()
    worker.start = AsyncMock()
    worker.stop = AsyncMock()
    # ``ContextCompressionProxy.__init__`` reassigns ``worker.metrics`` to
    # the proxy's metrics instance. MagicMock auto-creates attributes on
    # access, so this is a no-op safety net rather than a requirement.
    worker.metrics = None
    return worker


def test_proxy_done_stops_async_loop_and_worker():
    """done() must tear down the async loop, join the worker thread, and
    drain the injected distillation worker."""
    mock_worker = _build_mock_worker()

    proxy = ContextCompressionProxy(distillation_worker=mock_worker)

    # The loop must be up before we ask it to shut down; the Event-based
    # wait replaces the old busy-wait and must succeed within the proxy's
    # 5s startup window.
    assert proxy._loop_ready.wait(timeout=5.0) is True
    assert proxy._async_thread is not None and proxy._async_thread.is_alive() is True
    # ``start`` was awaited once during loop bring-up.
    mock_worker.start.assert_awaited_once()

    proxy.done()

    # done() joins the thread with a 5s internal timeout. Poll briefly so
    # we don't race the join but also don't depend on the exact wake-up
    # latency of the background loop's finally block.
    deadline = time.monotonic() + 2.0
    while (
        time.monotonic() < deadline
        and proxy._async_thread is not None
        and proxy._async_thread.is_alive()
    ):
        time.sleep(0.01)

    assert proxy._async_thread is not None
    assert proxy._async_thread.is_alive() is False, (
        "Async thread should have exited after proxy.done() returned"
    )

    # The run_loop finally block invokes ``worker.stop()`` via
    # ``run_until_complete`` after ``run_forever`` returns; assert it was
    # awaited at least once.
    mock_worker.stop.assert_awaited()


# ---------------------------------------------------------------------------
# Component factories.
# ---------------------------------------------------------------------------


def test_proxy_create_adapter_returns_adapter():
    """_create_adapter must produce a MitmproxyAdapter bound to target hosts."""
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    try:
        assert proxy._loop_ready.wait(timeout=5.0) is True
        assert isinstance(proxy.adapter, MitmproxyAdapter)
        assert proxy.adapter._target_hosts == ["api.openai.com"]
    finally:
        proxy.done()


def test_proxy_create_compression_engine_returns_engine():
    """_create_compression_engine must produce a CompressionEngine."""
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    try:
        assert proxy._loop_ready.wait(timeout=5.0) is True
        assert isinstance(proxy.compression_engine, CompressionEngine)
    finally:
        proxy.done()


def test_proxy_create_llm_processor_returns_processor():
    """_create_llm_processor must produce an LLMProcessor."""
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    try:
        assert proxy._loop_ready.wait(timeout=5.0) is True
        assert isinstance(proxy.llm_processor, LLMProcessor)
    finally:
        proxy.done()


def test_proxy_create_distillation_worker_when_enabled():
    """When distillation is enabled and no worker is injected, a worker is created."""
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    try:
        assert proxy._loop_ready.wait(timeout=5.0) is True
        assert proxy.distillation_worker is not None
        assert isinstance(proxy.distillation_worker, DistillationWorker)
    finally:
        proxy.done()


def test_proxy_create_engine_returns_engine():
    """_create_engine must produce a KonstaEngine wired to the proxy's components."""
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    try:
        assert proxy._loop_ready.wait(timeout=5.0) is True
        assert isinstance(proxy.engine, KonstaEngine)
        assert proxy.engine.config is proxy.config
        assert proxy.engine.compression_engine is proxy.compression_engine
        assert proxy.engine.distillation_worker is proxy.distillation_worker
        assert proxy.engine.metrics is proxy.metrics
    finally:
        proxy.done()


def test_proxy_done_is_idempotent():
    """Calling done() twice must be a safe no-op on the second call."""
    mock_worker = _build_mock_worker()

    proxy = ContextCompressionProxy(distillation_worker=mock_worker)

    assert proxy._loop_ready.wait(timeout=5.0) is True

    # First call performs the actual shutdown.
    proxy.done()
    # Second call must not raise (the ``_shutting_down`` flag short-
    # circuits it) and must leave the thread in its already-exited state.
    proxy.done()

    assert proxy._async_thread is not None
    assert proxy._async_thread.is_alive() is False

    # ``stop`` is only awaited once: the second done() call returns
    # before touching the loop or the thread.
    assert mock_worker.stop.await_count == 1


# ---------------------------------------------------------------------------
# Factory methods (``__init__`` split): each ``_create_*`` helper must
# produce a component of the expected type with the expected wiring.
# ---------------------------------------------------------------------------


def test_proxy_creates_adapter_with_target_hosts():
    """``_create_adapter`` must return a :class:`MitmproxyAdapter` bound to target hosts."""
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com", "api.anthropic.com"])
    try:
        assert isinstance(proxy.adapter, MitmproxyAdapter)
        # The adapter is bound to the same target_hosts the proxy was
        # constructed with (caller-supplied values, not config defaults).
        assert proxy.adapter._target_hosts == [
            "api.openai.com",
            "api.anthropic.com",
        ]
        # Re-invoking the factory must yield a fresh adapter with the
        # same target_hosts wiring.
        adapter2 = proxy._create_adapter()
        assert isinstance(adapter2, MitmproxyAdapter)
        assert adapter2._target_hosts == [
            "api.openai.com",
            "api.anthropic.com",
        ]
    finally:
        if proxy._async_loop is not None and proxy._async_loop.is_running():
            proxy._async_loop.call_soon_threadsafe(proxy._async_loop.stop)


def test_proxy_creates_compression_engine():
    """``_create_compression_engine`` must return a :class:`CompressionEngine`
    wired to the proxy's config and metrics."""
    metrics = Metrics()
    proxy = ContextCompressionProxy(
        target_hosts=["api.openai.com"], metrics=metrics
    )
    try:
        assert isinstance(proxy.compression_engine, CompressionEngine)
        # The engine shares the injected metrics instance.
        assert proxy.compression_engine.metrics is metrics
        # Re-invoking the factory yields a fresh engine with the same
        # metrics reference (so writes flow into the same store).
        engine2 = proxy._create_compression_engine()
        assert isinstance(engine2, CompressionEngine)
        assert engine2.metrics is metrics
    finally:
        if proxy._async_loop is not None and proxy._async_loop.is_running():
            proxy._async_loop.call_soon_threadsafe(proxy._async_loop.stop)


def test_proxy_creates_llm_processor():
    """``_create_llm_processor`` must return an :class:`LLMProcessor`
    wired to the proxy's config and started on the proxy's async loop."""
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    try:
        assert isinstance(proxy.llm_processor, LLMProcessor)
        # The processor's queue manager was started on the proxy's
        # dedicated async loop by ``_init_async_loop``. Inspect the
        # queue's bound loop directly: calling ``_get_loop()`` from
        # the test thread would itself raise if the queue is bound to
        # a different running loop (which it is, intentionally).
        assert proxy.llm_processor.queue_manager._is_running is True
        assert proxy.llm_processor.queue_manager.queue._loop is proxy._async_loop
        # Re-invoking the factory yields a fresh, un-started processor;
        # it must still be an LLMProcessor instance and use the proxy's
        # config (api_key, model, endpoint).
        proc2 = proxy._create_llm_processor()
        assert isinstance(proc2, LLMProcessor)
        assert proc2.api_key == proxy.llm_processor.api_key
        assert proc2.model == proxy.llm_processor.model
        assert proc2.endpoint == proxy.llm_processor.endpoint
    finally:
        if proxy._async_loop is not None and proxy._async_loop.is_running():
            proxy._async_loop.call_soon_threadsafe(proxy._async_loop.stop)


def test_proxy_creates_distillation_worker_when_enabled():
    """``_create_distillation_worker`` must return a
    :class:`DistillationWorker` when distillation mode is enabled
    (default), and rebind ``metrics`` on an injected worker."""
    metrics = Metrics()
    # Default config has distillation_mode != "disabled", so the factory
    # must build a worker when none is injected.
    proxy = ContextCompressionProxy(
        target_hosts=["api.openai.com"], metrics=metrics
    )
    try:
        assert isinstance(proxy.distillation_worker, DistillationWorker)
        assert proxy.distillation_worker.metrics is metrics
        assert proxy.distillation_worker.processor is proxy.llm_processor
        # Re-invoking the factory yields another fresh worker.
        worker2 = proxy._create_distillation_worker(None)
        assert isinstance(worker2, DistillationWorker)
        assert worker2.metrics is metrics
        assert worker2.processor is proxy.llm_processor
    finally:
        if proxy._async_loop is not None and proxy._async_loop.is_running():
            proxy._async_loop.call_soon_threadsafe(proxy._async_loop.stop)

    # When a worker is injected, the factory must return it unchanged
    # (apart from rebinding ``metrics``) and not construct a new one.
    injected = _build_mock_worker()
    proxy2 = ContextCompressionProxy(
        target_hosts=["api.openai.com"],
        distillation_worker=injected,
        metrics=metrics,
    )
    try:
        assert proxy2.distillation_worker is injected
        # Injected worker.metrics must have been rebound to the proxy's
        # metrics instance so distillation latency flows into the same
        # store the operator inspects.
        assert injected.metrics is metrics
        # Re-invoking the factory with the same injected worker must
        # return it unchanged.
        again = proxy2._create_distillation_worker(injected)
        assert again is injected
        assert injected.metrics is metrics
    finally:
        if proxy2._async_loop is not None and proxy2._async_loop.is_running():
            proxy2._async_loop.call_soon_threadsafe(proxy2._async_loop.stop)


def test_proxy_creates_engine():
    """``_create_engine`` must return a :class:`KonstaEngine` wired to the
    proxy's config, compression engine, distillation worker, and metrics."""
    metrics = Metrics()
    proxy = ContextCompressionProxy(
        target_hosts=["api.openai.com"], metrics=metrics
    )
    try:
        assert isinstance(proxy.engine, KonstaEngine)
        assert proxy.engine.config is proxy.config
        assert proxy.engine.compression_engine is proxy.compression_engine
        assert proxy.engine.distillation_worker is proxy.distillation_worker
        assert proxy.engine.metrics is metrics
        # Re-invoking the factory yields a fresh engine with the same
        # wiring (same config, same component instances, same metrics).
        engine2 = proxy._create_engine()
        assert isinstance(engine2, KonstaEngine)
        assert engine2.config is proxy.config
        assert engine2.compression_engine is proxy.compression_engine
        assert engine2.distillation_worker is proxy.distillation_worker
        assert engine2.metrics is metrics
    finally:
        if proxy._async_loop is not None and proxy._async_loop.is_running():
            proxy._async_loop.call_soon_threadsafe(proxy._async_loop.stop)


