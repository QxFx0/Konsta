import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Generic, Optional, TypeVar

# Type variables for task input and output
T_in = TypeVar("T_in")
T_out = TypeVar("T_out")

@dataclass
class DistillationTask(Generic[T_in, T_out]):
    """Represents a single distillation task in the queue."""
    task_id: str
    payload: T_in
    future: asyncio.Future

class AsyncRateLimiter:
    """
    A simple token-bucket rate limiter for asynchronous tasks.
    """
    def __init__(self, requests_per_second: float):
        self.rps = requests_per_second
        self.tokens = requests_per_second
        self.updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def wait(self):
        """
        Wait until a token is available to proceed.
        """
        if self.rps <= 0:
            return

        async with self._lock:
            while self.tokens < 1:
                now = time.monotonic()
                time_passed = now - self.updated_at
                self.tokens += time_passed * self.rps
                self.updated_at = now

                if self.tokens < 1:
                    # Sleep for the time needed to get at least one token
                    wait_time = (1 - self.tokens) / self.rps
                    await asyncio.sleep(wait_time)

            self.tokens -= 1

class QueueManager(Generic[T_in, T_out]):
    """
    Manages an asynchronous queue for LLM distillation tasks to reduce latency
    by decoupling task submission from execution, with integrated rate limiting.
    """
    def __init__(self, max_size: int = 100, num_workers: int = 1, requests_per_second: float = 10.0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_size = max_size
        self.num_workers = num_workers
        self.rps = requests_per_second

        # Initialize queue as None; will be created in start() to ensure it's tied to the correct loop
        self.queue: Optional[asyncio.Queue[DistillationTask[T_in, T_out]]] = None
        self.rate_limiter: Optional[AsyncRateLimiter] = None
        self.workers: list[asyncio.Task] = []
        self._is_running = False

    async def start(self, processor: Callable[[T_in], Coroutine[Any, Any, T_out]]):
        """
        Initializes the queue and starts the worker tasks.

        Args:
            processor: An async function that takes the task payload and returns a result.
        """
        if self._is_running:
            self.logger.warning("QueueManager is already running.")
            return

        # Initialize async primitives inside the running loop
        self.queue = asyncio.Queue(maxsize=self.max_size)
        self.rate_limiter = AsyncRateLimiter(self.rps)
        self._is_running = True

        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker_loop(i, processor))
            self.workers.append(worker)

        self.logger.info(f"Started {self.num_workers} distillation workers with rate limit {self.rps} RPS.")

    async def stop(self):
        """
        Gracefully shuts down the queue manager, ensuring all current tasks are processed
        or cancelled.
        """
        self._is_running = False

        # Signal workers to stop by cancelling them
        for worker in self.workers:
            worker.cancel()

        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        self.logger.info("QueueManager stopped.")

    async def enqueue(self, task_id: str, payload: T_in, wait_for_space: bool = True) -> T_out:
        """
        Adds a task to the queue and waits for its completion.

        Args:
            task_id: Unique identifier for the task.
            payload: The data to be processed by the distillation logic.
            wait_for_space: If True, blocks until space is available in the queue.

        Returns:
            The result of the distillation process.

        Raises:
            RuntimeError: If the queue is full and wait_for_space is False.
            ValueError: If the QueueManager has not been started.
        """
        if self.queue is None:
            raise ValueError("QueueManager must be started before enqueuing tasks.")

        future = asyncio.get_running_loop().create_future()
        task = DistillationTask(task_id=task_id, payload=payload, future=future)

        try:
            if wait_for_space:
                await self.queue.put(task)
            else:
                self.queue.put_nowait(task)
        except asyncio.QueueFull:
            self.logger.error(f"Queue full: cannot enqueue task {task_id}")
            raise RuntimeError(f"Distillation queue is full. Task {task_id} rejected.")

        # Wait for the worker to complete the task and set the result on the future
        return await future

    async def _worker_loop(self, worker_id: int, processor: Callable[[T_in], Coroutine[Any, Any, T_out]]):
        """
        Internal loop that pulls tasks from the queue and processes them with rate limiting.

        Cancellation semantics:
            ``asyncio.CancelledError`` is re-raised so that ``stop()`` can shut the
            worker down cleanly. It is NOT logged as a worker error.
        """
        self.logger.debug(f"Worker-{worker_id} started.")
        try:
            while self._is_running:
                # Get a task from the queue
                task = await self.queue.get()
                try:
                    # Apply rate limiting before processing
                    if self.rate_limiter:
                        await self.rate_limiter.wait()

                    self.logger.debug(f"Worker-{worker_id} processing task {task.task_id}")
                    result = await processor(task.payload)

                    if not task.future.done():
                        task.future.set_result(result)
                except asyncio.CancelledError:
                    # Cooperative cancellation: re-raise so the outer loop exits.
                    raise
                except Exception as e:
                    self.logger.exception(f"Worker-{worker_id} failed task {task.task_id}: {e}")
                    if not task.future.done():
                        task.future.set_exception(e)
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            # Expected during shutdown; not an error.
            self.logger.debug(f"Worker-{worker_id} cancelled cleanly.")
            raise
        except Exception:
            self.logger.exception(f"Worker-{worker_id} encountered a critical error")
