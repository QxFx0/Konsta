import asyncio
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional, Union

from mitmproxy import http

from src.ca_manager import CAManager
from src.compression_engine import CompressionEngine
from src.config import Config
from src.distillation_worker import DistillationWorker
from src.engine import KonstaEngine
from src.llm_processor import LLMProcessor
from src.metrics import Metrics
from src.proxy_adapter import MitmproxyAdapter

logger = logging.getLogger(__name__)


class ContextCompressionProxy:
    """
    mitmproxy addon for intercepting HTTP/HTTPS requests to specific target hosts
    and applying context compression to the request body using both semantic compression
    and LLM-based context processing.

    The proxy itself is a thin mitmproxy-specific shell. The compression and
    distillation logic live in :class:`src.engine.KonstaEngine`, and all
    proxy-library-specific access is funnelled through
    :class:`src.proxy_adapter.MitmproxyAdapter`.
    """

    def __init__(
        self,
        target_hosts=None,
        distillation_worker: Optional[DistillationWorker] = None,
        metrics: Optional[Metrics] = None,
        config: Optional[Config] = None,
    ):
        # Configuration: caller-supplied or constructed once at
        # proxy-init time. The default-constructed ``Config()`` here is
        # the *only* place the proxy lazily creates a configuration; no
        # module-level singleton is consulted. Pass an explicit
        # :class:`Config` to inject custom settings in tests or to
        # share the configuration created by :func:`src.main.main`.
        if config is None:
            config = Config()
        self.config = config
        self.target_hosts = target_hosts or self.config.target_hosts
        logger.info(f"Initializing ContextCompressionProxy with target hosts: {self.target_hosts}")

        # Metrics: explicit injection, otherwise a fresh local instance.
        if metrics is None:
            metrics = Metrics()
        self.metrics = metrics

        # Core components, each wired via a dedicated factory method.
        self.compression_engine = self._create_compression_engine()
        self.llm_processor = self._create_llm_processor()
        self.ca_manager = CAManager(self.config)
        self.distillation_worker = self._create_distillation_worker(distillation_worker)
        self.adapter = self._create_adapter()
        self.engine = self._create_engine()

        # Async-loop state. ``_loop_ready`` must exist *before* the
        # worker thread launches so the background thread can safely
        # ``set()`` it without racing attribute creation. ``_shutting_down``
        # guards ``done()`` against double-shutdown races.
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None
        self._loop_started = False
        self._loop_ready = threading.Event()
        self._shutting_down = False
        self._init_async_loop()

        # HTTPS interception CA setup.
        self._init_https_interception()

    # ------------------------------------------------------------------
    # Component factories
    # ------------------------------------------------------------------

    def _create_adapter(self) -> MitmproxyAdapter:
        """Build the mitmproxy adapter bound to this proxy's target hosts.

        The adapter funnels all mitmproxy-specific attribute access; the
        engine talks to it through the :class:`ProxyAdapter` interface.
        """
        return MitmproxyAdapter(target_hosts=self.target_hosts)

    def _create_compression_engine(self) -> CompressionEngine:
        """Build the semantic-compression engine with our config and metrics."""
        return CompressionEngine(self.config, metrics=self.metrics)

    def _create_llm_processor(self) -> LLMProcessor:
        """Build the LLM processor used for both in-line requests and distillation."""
        return LLMProcessor(self.config)

    def _create_distillation_worker(
        self, distillation_worker: Optional[DistillationWorker]
    ) -> Optional[DistillationWorker]:
        """Return the distillation worker to use, constructing one if needed.

        Distillation runs on a dedicated background worker so the
        synchronous mitmproxy handler does not block on the remote LLM
        call. A caller (typically ``main.py``) may inject a pre-built
        worker; if none is supplied and LLM distillation is enabled, we
        construct one lazily so tests and the default ``addons = [...]``
        entrypoint both work.

        When an injected worker is supplied we re-bind its ``metrics`` to
        the proxy's metrics instance so distillation latency and failures
        flow into the same store the operator inspects.
        """
        if distillation_worker is None and self.config.distillation_mode != "disabled":
            return DistillationWorker(
                processor=self.llm_processor,
                cache_size=self.config.distillation_cache_size,
                metrics=self.metrics,
            )
        if distillation_worker is not None:
            distillation_worker.metrics = self.metrics
        return distillation_worker

    def _create_engine(self) -> KonstaEngine:
        """Build the engine that owns compression, distillation lookup/submission, and dumping."""
        return KonstaEngine(
            config=self.config,
            compression_engine=self.compression_engine,
            distillation_worker=self.distillation_worker,
            metrics=self.metrics,
        )

    def _init_async_loop(self):
        """Initialize a dedicated event loop in a separate thread for async operations.

        The loop hosts both the :class:`LLMProcessor` (which needs an asyncio
        loop for its :class:`QueueManager`) and, when present, the
        :class:`DistillationWorker` that decouples LLM distillation from the
        synchronous mitmproxy handler. Both components are started inside the
        loop and stopped again during thread shutdown.
        """
        def run_loop():
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)

            try:
                # Start the LLM processor on this loop.
                self._async_loop.run_until_complete(self.llm_processor.start())

                # Start the distillation worker on the same loop. The worker's
                # ``start`` binds it to the currently running loop, so this must
                # be invoked from inside ``run_until_complete``.
                if self.distillation_worker is not None:
                    self._async_loop.run_until_complete(
                        self.distillation_worker.start()
                    )

                self._loop_started = True
                # Signal the main thread that the async loop is up and
                # running. ``_init_async_loop`` blocks on this Event to
                # avoid polling.
                self._loop_ready.set()

                # Keep the loop running until ``run_until_complete`` is
                # cancelled or the loop is stopped externally.
                self._async_loop.run_forever()
            except Exception:
                logger.exception("Async event loop crashed")
            finally:
                # Best-effort cleanup: stop the worker (drains pending tasks)
                # then the LLM processor, in reverse start order.
                if self.distillation_worker is not None:
                    try:
                        self._async_loop.run_until_complete(
                            self.distillation_worker.stop()
                        )
                    except Exception:
                        logger.exception(
                            "Failed to stop DistillationWorker during loop shutdown"
                        )
                try:
                    self._async_loop.run_until_complete(self.llm_processor.stop())
                except Exception:
                    logger.exception(
                        "Failed to stop LLMProcessor during loop shutdown"
                    )

        self._async_thread = threading.Thread(target=run_loop, daemon=True)
        self._async_thread.start()

        # Wait for the background loop to finish bringing up the LLM
        # processor (and the distillation worker, if any). The background
        # thread sets ``_loop_ready`` once it transitions into
        # ``run_forever``; we use an Event instead of polling to avoid
        # busy-waiting the main thread during construction.
        ready = self._loop_ready.wait(timeout=5.0)

        if not ready or not self._loop_started:
            logger.error("Failed to start async event loop")

    def _init_https_interception(self):
        """Initialize CA certificates for HTTPS interception.

        By default this only generates the local CA files (cert + key). It does NOT
        install the CA into the system trust store. Installing the CA system-wide is
        an explicit, opt-in operation that must be triggered by the caller via
        `install_ca_system_wide()`. Without that step, mitmproxy cannot transparently
        intercept HTTPS traffic for the local user; clients will see certificate
        warnings until the CA is installed.
        """
        try:
            # Ensure paths are Path objects to avoid AttributeError in ca_manager
            cert_path = Path(self.config.ca_cert_path)
            key_path = Path(self.config.ca_key_path)

            self.ca_manager.setup_ca(
                cert_path,
                key_path
            )
            logger.info("HTTPS interception CA files generated locally at %s", cert_path)
            logger.info(
                "CA is NOT installed system-wide. Call install_ca_system_wide() or "
                "run with --install-ca to enable transparent HTTPS interception."
            )
        except (OSError, subprocess.SubprocessError) as e:
            # Best-effort: filesystem or ``subprocess`` failures during CA
            # setup must not prevent the proxy from starting. ``OSError``
            # covers filesystem/permission errors and ``SubprocessError``
            # is the base class for every error raised by ``subprocess``
            # (including ``CalledProcessError``). Logged as a warning
            # because the proxy can still operate without transparent
            # HTTPS interception. Anything outside these categories is
            # genuinely unexpected and propagates so it cannot be
            # silently swallowed by a broad handler.
            logger.warning(
                f"Failed to initialize HTTPS interception (OS/subprocess error): {e}. "
                f"Continuing without transparent HTTPS interception."
            )

    def install_ca_system_wide(self) -> bool:
        """
        Explicitly install the Konsta CA into the system trust store.

        This is an opt-in operation. It allows the proxy to transparently intercept
        HTTPS traffic, but it also means Konsta (or any process that can use the
        installed CA) can issue certificates trusted by the local machine. Only run
        this on machines you control and trust.

        Returns:
            bool: True if installation succeeded (or already installed), False otherwise.
        """
        # Loud, unmistakable warning so this never runs silently.
        logger.warning(
            "======================================================================"
        )
        logger.warning(
            "WARNING: Installing Konsta self-signed CA into the SYSTEM trust store."
        )
        logger.warning(
            "This allows Konsta to decrypt HTTPS traffic on this machine."
        )
        logger.warning(
            "Only proceed if you trust this machine and the Konsta installation."
        )
        logger.warning(
            "======================================================================"
        )

        try:
            cert_path = Path(self.config.ca_cert_path)
            ok = self.ca_manager.install_ca_system_wide(cert_path)
            if ok:
                logger.warning("Konsta CA installed system-wide successfully.")
            else:
                logger.error("Konsta CA system-wide installation failed.")
            return ok
        except Exception as e:
            logger.error(f"Failed to install CA system-wide: {e}")
            return False

    def done(self) -> None:
        """Gracefully shut down the async event loop and worker thread.

        Idempotent: subsequent calls are no-ops. Signals the dedicated
        asyncio loop to stop via ``call_soon_threadsafe``, then joins the
        background thread with a bounded timeout. The ``run_loop``
        ``finally`` block takes care of draining the distillation worker
        and shutting down the :class:`LLMProcessor` once ``run_forever``
        returns, so :meth:`done` only needs to nudge the loop and wait
        for the thread to exit.

        Safe to call from any thread *except* the async thread itself
        (joining from inside the thread would deadlock); that case is
        guarded via the daemon-thread check.
        """
        if self._shutting_down:
            return
        self._shutting_down = True

        loop = self._async_loop
        thread = self._async_thread

        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                # Loop already stopped or not running; fall through to
                # the join which will simply observe the thread has
                # exited.
                logger.debug("Async loop already stopped; skipping call_soon_threadsafe")

        if thread is not None and thread.is_alive():
            # Guard against ``done`` being called from inside the async
            # thread itself (e.g. from a mitmproxy ``done`` hook); joining
            # the current thread from itself would deadlock forever.
            if thread is threading.current_thread():
                logger.debug(
                    "done() invoked from inside the async thread; "
                    "skipping join"
                )
            else:
                # ``run_loop``'s finally block calls ``run_until_complete`` to
                # drain the distillation worker and LLM processor. That
                # requires the loop to accept new tasks, so we give the
                # thread a bounded window to finish before reporting.
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(
                        "Async event loop thread did not exit within 5s; "
                        "leaving it as a daemon for interpreter shutdown"
                    )
        logger.info("ContextCompressionProxy shutdown complete")

    def is_target_request(self, host_or_flow: Union[str, http.HTTPFlow]) -> bool:
        """Check if the request is targeted at one of our configured hosts.

        Delegates to the mitmproxy adapter for flow objects, and applies the
        same suffix-matching rule for bare host strings so existing call
        sites that pass a hostname directly keep working.
        """
        if isinstance(host_or_flow, str):
            return any(
                host_or_flow.endswith(target) for target in self.target_hosts
            )
        return self.adapter.is_target_request(host_or_flow)

    def request(self, flow: http.HTTPFlow) -> None:
        """mitmproxy ``request`` hook.

        Filters by target host then delegates the compression/distillation
        pipeline to :class:`KonstaEngine` via :class:`MitmproxyAdapter`.
        """
        if not self.adapter.is_target_request(flow):
            return
        self.engine.process_request(self.adapter, flow)

    def response(self, flow: http.HTTPFlow) -> None:
        """mitmproxy ``response`` hook.

        Filters by target host (same rule as ``request``) then delegates to
        :class:`KonstaEngine` which injects compression metrics into
        response headers and dumps the payload for diagnostics.
        """
        if not self.adapter.is_target_request(flow):
            return
        self.engine.process_response(self.adapter, flow)

def make_addon() -> ContextCompressionProxy:
    """Factory function that creates a fresh :class:`ContextCompressionProxy`.

    Tests and other call sites use this to obtain a proxy instance without
    paying the import-time construction cost. ``proxy_core`` is imported by
    mitmdump as a script, so deferring instantiation until the addon is
    actually used keeps ``import proxy_core`` cheap.
    """
    return ContextCompressionProxy()


class _LazyAddons:
    """Lazy iterable wrapper used as mitmproxy's module-level ``addons``.

    mitmproxy's addon manager iterates over ``addons`` to discover the
    addons registered by a script. By yielding a freshly-built
    :class:`ContextCompressionProxy` only when iteration begins, we avoid
    constructing the proxy (which spins up an asyncio loop, CA setup, and
    an LLM processor) at module-import time.
    """

    def __iter__(self):
        yield make_addon()


# Register the addon with mitmproxy via a lazy iterable so the proxy is
# constructed only when mitmproxy actually iterates over ``addons``.
addons = _LazyAddons()
