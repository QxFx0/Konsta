"""Tests for ``src.metrics.Metrics`` and the shared-singleton plumbing."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.metrics import (
    Metrics,
    get_shared_metrics,
    reset_shared_metrics,
    set_shared_metrics,
)

# ---------------------------------------------------------------------------
# record_compression
# ---------------------------------------------------------------------------


def test_record_compression_updates_running_totals():
    metrics = Metrics()
    metrics.record_compression(original_size=100, compressed_size=25)
    snap = metrics.snapshot()
    assert snap["compression_count"] == 1
    assert snap["compression_orig_bytes"] == 100
    assert snap["compression_compressed_bytes"] == 25


def test_compression_ratio_is_compressed_over_original():
    metrics = Metrics()
    metrics.record_compression(original_size=100, compressed_size=25)
    metrics.record_compression(original_size=200, compressed_size=100)
    # (25 + 100) / (100 + 200) = 125 / 300
    assert metrics.compression_ratio == pytest.approx(125 / 300)


def test_compression_ratio_with_no_data_is_zero():
    metrics = Metrics()
    # No observations yet -> a clean zero rather than a ZeroDivisionError
    # leaking out of the property getter.
    assert metrics.compression_ratio == 0.0


def test_record_compression_rejects_negative_sizes():
    metrics = Metrics()
    with pytest.raises(ValueError):
        metrics.record_compression(original_size=-1, compressed_size=10)
    with pytest.raises(ValueError):
        metrics.record_compression(original_size=10, compressed_size=-1)


def test_record_compression_coerces_numeric_arguments():
    """Floats that are integer-valued should be accepted."""
    metrics = Metrics()
    metrics.record_compression(original_size=10.0, compressed_size=5.0)
    snap = metrics.snapshot()
    assert snap["compression_orig_bytes"] == 10
    assert snap["compression_compressed_bytes"] == 5


# ---------------------------------------------------------------------------
# record_distillation
# ---------------------------------------------------------------------------


def test_record_distillation_updates_running_totals():
    metrics = Metrics()
    metrics.record_distillation(latency_ms=120.0, success=True)
    metrics.record_distillation(latency_ms=80.0, success=False)
    snap = metrics.snapshot()
    assert snap["distillation_count"] == 2
    assert snap["distillation_failures_total"] == 1
    # Mean of 120 and 80 is 100.
    assert metrics.distillation_latency_ms == pytest.approx(100.0)


def test_distillation_latency_with_no_data_is_zero():
    metrics = Metrics()
    assert metrics.distillation_latency_ms == 0.0


def test_distillation_failures_total_increments_only_on_failure():
    metrics = Metrics()
    metrics.record_distillation(latency_ms=10.0, success=True)
    assert metrics.distillation_failures_total == 0
    metrics.record_distillation(latency_ms=10.0, success=False)
    assert metrics.distillation_failures_total == 1
    metrics.record_distillation(latency_ms=10.0, success=False)
    assert metrics.distillation_failures_total == 2


def test_record_distillation_clamps_negative_latency():
    """Negative latencies (clock skew, bugs) are clamped to zero."""
    metrics = Metrics()
    metrics.record_distillation(latency_ms=-5.0, success=True)
    assert metrics.distillation_latency_ms == 0.0


def test_record_distillation_accepts_non_bool_success():
    """``success`` is interpreted by truthiness, not strict bool identity."""
    metrics = Metrics()
    metrics.record_distillation(latency_ms=10.0, success=0)  # falsy -> failure
    metrics.record_distillation(latency_ms=10.0, success=1)  # truthy -> success
    assert metrics.distillation_failures_total == 1


# ---------------------------------------------------------------------------
# record_request
# ---------------------------------------------------------------------------


def test_record_request_increments_counter():
    metrics = Metrics()
    assert metrics.requests_processed_total == 0
    metrics.record_request()
    metrics.record_request()
    metrics.record_request()
    assert metrics.requests_processed_total == 3
    assert metrics.snapshot()["requests_processed_total"] == 3


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


def test_snapshot_keys_include_all_advertised_metrics():
    metrics = Metrics()
    snap = metrics.snapshot()
    for key in (
        "compression_ratio",
        "distillation_latency_ms",
        "distillation_failures_total",
        "requests_processed_total",
    ):
        assert key in snap, f"missing key {key!r} in snapshot {snap!r}"


def test_snapshot_is_a_plain_dict_copy():
    """Mutating the snapshot must not mutate the metrics store."""
    metrics = Metrics()
    metrics.record_request()
    snap = metrics.snapshot()
    snap["requests_processed_total"] = 9999
    assert metrics.requests_processed_total == 1


def test_snapshot_is_consistent_point_in_time():
    """All fields are taken under a single lock."""
    metrics = Metrics()
    metrics.record_compression(100, 25)
    metrics.record_distillation(latency_ms=50.0, success=True)
    metrics.record_request()

    snap = metrics.snapshot()
    assert snap["compression_count"] == 1
    assert snap["compression_ratio"] == pytest.approx(0.25)
    assert snap["distillation_count"] == 1
    assert snap["distillation_latency_ms"] == pytest.approx(50.0)
    assert snap["distillation_failures_total"] == 0
    assert snap["requests_processed_total"] == 1


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


def test_record_compression_is_thread_safe():
    metrics = Metrics()
    n_threads = 16
    per_thread = 500

    def worker():
        for _ in range(per_thread):
            metrics.record_compression(original_size=10, compressed_size=5)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = metrics.snapshot()
    expected_calls = n_threads * per_thread
    assert snap["compression_count"] == expected_calls
    assert snap["compression_orig_bytes"] == expected_calls * 10
    assert snap["compression_compressed_bytes"] == expected_calls * 5


def test_record_distillation_is_thread_safe():
    metrics = Metrics()
    n_threads = 16
    per_thread = 500

    def worker(failure: bool):
        for _ in range(per_thread):
            metrics.record_distillation(latency_ms=2.0, success=failure)

    threads = [threading.Thread(target=worker, args=(i % 2 == 0,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = metrics.snapshot()
    expected_calls = n_threads * per_thread
    assert snap["distillation_count"] == expected_calls
    # Half the threads record failures -> half the total observations.
    assert snap["distillation_failures_total"] == expected_calls // 2
    assert metrics.distillation_latency_ms == pytest.approx(2.0)


def test_record_request_is_thread_safe():
    metrics = Metrics()
    n_threads = 16
    per_thread = 1_000

    def worker():
        for _ in range(per_thread):
            metrics.record_request()

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert metrics.requests_processed_total == n_threads * per_thread


def test_concurrent_writes_across_all_methods():
    """Mixed concurrent writes through every public method must not lose updates."""
    metrics = Metrics()
    n_threads = 12
    iterations = 200

    def compression_worker():
        for _ in range(iterations):
            metrics.record_compression(10, 5)

    def distillation_worker(failure: bool):
        for _ in range(iterations):
            metrics.record_distillation(3.0, failure)

    def request_worker():
        for _ in range(iterations):
            metrics.record_request()

    threads = []
    for i in range(n_threads):
        if i % 3 == 0:
            threads.append(threading.Thread(target=compression_worker))
        elif i % 3 == 1:
            threads.append(threading.Thread(target=distillation_worker, args=(i % 2 == 0,)))
        else:
            threads.append(threading.Thread(target=request_worker))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = metrics.snapshot()
    # Each method group runs 4 threads (n_threads // 3 == 4) doing
    # ``iterations`` ops.
    comp_threads = n_threads // 3
    assert snap["compression_count"] == comp_threads * iterations
    assert snap["compression_orig_bytes"] == comp_threads * iterations * 10
    assert snap["distillation_count"] == comp_threads * iterations
    assert snap["requests_processed_total"] == comp_threads * iterations


def test_snapshot_during_concurrent_writes_does_not_raise():
    """Reading the snapshot while writers are active must always succeed."""
    metrics = Metrics()
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            metrics.record_compression(10, 5)
            metrics.record_distillation(1.0, i % 2 == 0)
            metrics.record_request()
            i += 1

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()

    # Hammer the snapshot getter concurrently with the writers.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(metrics.snapshot) for _ in range(200)]
        for f in futures:
            snap = f.result()
            assert isinstance(snap, dict)
            assert "compression_ratio" in snap
            assert "distillation_latency_ms" in snap

    stop.set()
    for t in threads:
        t.join()


# ---------------------------------------------------------------------------
# Shared singleton plumbing
# ---------------------------------------------------------------------------


def test_get_shared_metrics_returns_none_when_unset():
    reset_shared_metrics()
    try:
        assert get_shared_metrics() is None
    finally:
        reset_shared_metrics()


def test_get_shared_metrics_returns_same_instance():
    reset_shared_metrics()
    try:
        first = Metrics()
        set_shared_metrics(first)
        a = get_shared_metrics()
        b = get_shared_metrics()
        assert a is first
        assert a is b
    finally:
        reset_shared_metrics()


def test_set_shared_metrics_replaces_default():
    reset_shared_metrics()
    try:
        first = Metrics()
        first.record_request()
        set_shared_metrics(first)
        custom = Metrics()
        custom.record_request()
        custom.record_request()
        set_shared_metrics(custom)
        current = get_shared_metrics()
        assert current is custom
        assert current is not first
        assert current.requests_processed_total == 2
    finally:
        reset_shared_metrics()


def test_reset_shared_metrics_clears_singleton():
    reset_shared_metrics()
    try:
        set_shared_metrics(Metrics())
        assert get_shared_metrics() is not None
        reset_shared_metrics()
        assert get_shared_metrics() is None
    finally:
        reset_shared_metrics()


def test_set_shared_metrics_is_thread_safe():
    """Concurrent ``set_shared_metrics`` calls must not corrupt state."""
    reset_shared_metrics()
    try:
        instances = [Metrics() for _ in range(8)]

        def setter(idx):
            set_shared_metrics(instances[idx])

        threads = [threading.Thread(target=setter, args=(i,)) for i in range(len(instances))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        current = get_shared_metrics()
        assert current in instances
    finally:
        reset_shared_metrics()


# ---------------------------------------------------------------------------
# Integration: CompressionEngine and DistillationWorker use the metrics store
# ---------------------------------------------------------------------------


def _metrics_for_isolated_engine():
    """Return a fresh ``Metrics`` for engine isolation.

    Components no longer default to the shared singleton, so a fresh
    instance is enough; no global reset is required.
    """
    return Metrics()


def test_compression_engine_records_metrics():
    from src.compression_engine import CompressionEngine

    metrics = _metrics_for_isolated_engine()
    # Bypass the SentenceTransformer model so the test doesn't need to
    # download the model on first run.
    engine = CompressionEngine(metrics=metrics)
    engine.model = None

    messages = [
        {"role": "user", "content": "hello world " * 20},
        {"role": "user", "content": "hello world " * 20},  # merge candidate
        {"role": "assistant", "content": "hi there"},
    ]
    out = engine.compress(messages)
    assert isinstance(out, list)

    snap = metrics.snapshot()
    assert snap["compression_count"] == 1
    assert snap["compression_orig_bytes"] > 0
    assert snap["compression_compressed_bytes"] >= 0


def test_compression_engine_with_metrics_failure_does_not_break_compress(monkeypatch):
    """A broken metrics store must never break the compression pipeline."""
    from src.compression_engine import CompressionEngine

    class BrokenMetrics:
        def record_compression(self, *args, **kwargs):
            raise RuntimeError("metrics broken")

    engine = CompressionEngine(metrics=BrokenMetrics())
    engine.model = None

    out = engine.compress([{"role": "user", "content": "hi"}])
    assert out  # compression must still produce a result


def test_distillation_worker_records_latency_and_success():
    import asyncio

    from src.distillation_worker import DistillationWorker

    class _Proc:
        def __init__(self, output: str):
            self.output = output
            self.calls = 0

        async def process_context(self, messages):
            self.calls += 1
            await asyncio.sleep(0.001)
            return [{"role": "system", "content": self.output}]

    metrics = Metrics()
    proc = _Proc("done")
    worker = DistillationWorker(proc, cache_size=4, metrics=metrics)

    async def run():
        await worker.start()
        worker.submit("r1", "body")
        # Wait for the worker to finish processing.
        for _ in range(200):
            if worker.get_result("r1") is not None:
                break
            await asyncio.sleep(0.005)
        await worker.stop()

    asyncio.run(run())

    snap = metrics.snapshot()
    assert snap["distillation_count"] == 1
    assert snap["distillation_failures_total"] == 0
    assert snap["distillation_latency_ms"] >= 0.0


def test_distillation_worker_records_failure_on_exception():
    import asyncio

    from src.distillation_worker import DistillationWorker

    class _BoomProc:
        async def process_context(self, messages):
            raise RuntimeError("upstream failure")

    metrics = Metrics()
    worker = DistillationWorker(_BoomProc(), cache_size=4, metrics=metrics)

    async def run():
        await worker.start()
        worker.submit("r1", "body")
        # Wait long enough for the worker to process and fail.
        for _ in range(200):
            await asyncio.sleep(0.005)
        await worker.stop()

    asyncio.run(run())

    snap = metrics.snapshot()
    assert snap["distillation_count"] == 1
    assert snap["distillation_failures_total"] == 1


def test_proxy_core_records_request_metric(monkeypatch):
    """``ContextCompressionProxy.request`` must call ``record_request``."""
    from src.proxy_core import ContextCompressionProxy
    from tests.test_proxy_core import _make_flow  # type: ignore

    metrics = Metrics()
    proxy = ContextCompressionProxy(
        target_hosts=["api.openai.com"],
        metrics=metrics,
        # Skip the auto-built worker to keep the test fast and
        # deterministic; the request path must still record metrics.
        distillation_worker=None,
    )
    # Force "disabled" mode so the proxy doesn't create one either.
    proxy.config.distillation_mode = "disabled"

    flow = _make_flow(
        "api.openai.com",
        content=b'{"messages":[{"role":"user","content":"hi"}],"model":"x"}',
    )

    from unittest.mock import patch

    with patch.object(proxy.engine, "compress_context", return_value={"messages": []}):
        proxy.request(flow)

    assert metrics.requests_processed_total == 1


def test_proxy_core_skips_metric_for_non_target_request(monkeypatch):
    """``record_request`` is only called for targeted hosts."""
    from src.proxy_core import ContextCompressionProxy
    from tests.test_proxy_core import _make_flow  # type: ignore

    metrics = Metrics()
    proxy = ContextCompressionProxy(
        target_hosts=["api.openai.com"],
        metrics=metrics,
        distillation_worker=None,
    )
    proxy.config.distillation_mode = "disabled"

    flow = _make_flow("google.com", content=b"hello")

    from unittest.mock import patch

    with patch.object(proxy.engine, "compress_context") as mock_compress:
        proxy.request(flow)
        mock_compress.assert_not_called()

    assert metrics.requests_processed_total == 0
