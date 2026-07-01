"""Tests for ``src.distillation_worker.DistillationWorker``."""

from __future__ import annotations

import threading

import pytest

from src.distillation_worker import DistillationWorker
from tests.conftest import FakeProcessor, _wait_for_result


@pytest.mark.asyncio
async def test_submit_enqueues_and_worker_processes():
    processor = FakeProcessor(output="hello", delay=0.0)
    worker = DistillationWorker(processor, cache_size=4)
    await worker.start()
    try:
        worker.submit("req-1", "compressed body")

        assert await _wait_for_result(lambda: worker.get_result("req-1") is not None)

        assert worker.get_result("req-1") == "hello"
        assert processor.call_count == 1
        assert processor.calls and processor.calls[0][0]["content"] == "compressed body"
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_result_caching_overwrites_same_request_id():
    processor = FakeProcessor(output="first", delay=0.0)
    worker = DistillationWorker(processor, cache_size=4)
    await worker.start()
    try:
        worker.submit("req-A", "body-X")
        assert await _wait_for_result(lambda: worker.get_result("req-A") == "first")

        # Re-submitting the same request id overwrites the cached entry.
        processor.output = "second"
        worker.submit("req-A", "body-X")
        assert await _wait_for_result(lambda: worker.get_result("req-A") == "second")
        assert processor.call_count == 2
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_lru_cache_evicts_oldest_entry():
    processor = FakeProcessor(delay=0.0)
    worker = DistillationWorker(processor, cache_size=2)
    await worker.start()
    try:
        worker.submit("r1", "body-a")
        assert await _wait_for_result(lambda: worker.get_result("r1") is not None)
        worker.submit("r2", "body-b")
        assert await _wait_for_result(lambda: worker.get_result("r2") is not None)
        # Inserting a third distinct request_id should evict the oldest.
        worker.submit("r3", "body-c")
        assert await _wait_for_result(lambda: worker.get_result("r3") is not None)
        # ``r1`` should have been evicted from the LRU cache.
        assert worker.get_result("r1") is None
        assert worker.get_result("r2") is not None
        assert worker.get_result("r3") is not None
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_stop_drains_queue():
    processor = FakeProcessor(delay=0.005)
    worker = DistillationWorker(processor, cache_size=16)
    await worker.start()

    for i in range(5):
        worker.submit(f"req-{i}", f"body-{i}")

    await worker.stop()

    # All five submitted requests should have been processed before stop()
    # returned, so their results must be available in the cache.
    for i in range(5):
        assert worker.get_result(f"req-{i}") == processor.output
    assert processor.call_count == 5


@pytest.mark.asyncio
async def test_submit_before_start_raises():
    processor = FakeProcessor()
    worker = DistillationWorker(processor)
    with pytest.raises(RuntimeError):
        worker.submit("req-1", "body")


@pytest.mark.asyncio
async def test_submit_after_stop_raises():
    """Once ``stop`` has been initiated, ``submit`` must reject new tasks.

    This guarantees the ``None`` sentinel can be observed as the last item
    in the queue and that the worker exits cleanly without losing the
    shutdown signal.
    """
    processor = FakeProcessor()
    worker = DistillationWorker(processor, cache_size=4)
    await worker.start()
    await worker.stop()
    with pytest.raises(RuntimeError):
        worker.submit("req-late", "body")


@pytest.mark.asyncio
async def test_submit_from_other_thread():
    processor = FakeProcessor(output="cross-thread", delay=0.0)
    worker = DistillationWorker(processor, cache_size=4)
    await worker.start()
    try:
        errors: list[BaseException] = []

        def submitter():
            try:
                worker.submit("req-cross", "body")
            except BaseException as exc:  # pragma: no cover - surfaced via ``errors``
                errors.append(exc)

        thread = threading.Thread(target=submitter)
        thread.start()
        thread.join()

        assert not errors
        assert await _wait_for_result(lambda: worker.get_result("req-cross") == "cross-thread")
        assert processor.call_count == 1
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_get_result_unknown_request_returns_none():
    processor = FakeProcessor()
    worker = DistillationWorker(processor, cache_size=4)
    await worker.start()
    try:
        assert worker.get_result("never-submitted") is None
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_fifo_processing_order():
    """Tasks are processed in the order they were submitted.

    The asyncio Queue preserves FIFO ordering for ``put_nowait`` /
    ``get`` pairs in a single consumer, so we verify that the processor
    observes the submitted bodies in the same order they were enqueued.
    """
    processor = FakeProcessor(delay=0.0)
    worker = DistillationWorker(processor, cache_size=16)
    await worker.start()
    try:
        submitted_bodies = [f"body-{i}" for i in range(6)]
        for i, body in enumerate(submitted_bodies):
            worker.submit(f"req-{i}", body)

        # Wait for every task to land in the cache.
        for i in range(len(submitted_bodies)):
            assert await _wait_for_result(lambda i=i: worker.get_result(f"req-{i}") is not None)

        # Processor must have seen them in submission order.  Each call wraps
        # the body in a single ``user`` message; check the content values.
        assert processor.call_count == len(submitted_bodies)
        seen_bodies = [calls[0]["content"] for calls in processor.calls]
        assert seen_bodies == submitted_bodies
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_multiple_submissions_from_other_thread_fifo():
    """Submissions from another thread are still processed in FIFO order.

    Exercises the cross-thread ``_pending`` buffer + ``call_soon_threadsafe``
    drain path: even when many tasks arrive from a different thread they
    must be enqueued in submission order onto the worker's asyncio queue.
    """
    processor = FakeProcessor(delay=0.0)
    worker = DistillationWorker(processor, cache_size=32)
    await worker.start()

    submitted_bodies = [f"cross-{i}" for i in range(8)]
    errors: list[BaseException] = []

    def submitter():
        try:
            for i, body in enumerate(submitted_bodies):
                worker.submit(f"req-{i}", body)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    thread = threading.Thread(target=submitter)
    thread.start()
    thread.join()
    assert not errors

    try:
        for i in range(len(submitted_bodies)):
            assert await _wait_for_result(lambda i=i: worker.get_result(f"req-{i}") is not None)

        assert processor.call_count == len(submitted_bodies)
        seen = [calls[0]["content"] for calls in processor.calls]
        assert seen == submitted_bodies
    finally:
        await worker.stop()
