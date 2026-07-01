"""Proxy subprocess management.

This module owns the *non-engine* responsibilities of launching the Konsta
proxy:

- :func:`install_ca_if_requested` -- opt-in CA system-wide installation.
- :func:`create_distillation_worker` -- build a wired
  :class:`DistillationWorker` for the proxy addon.
- :func:`_build_mitmdump_cmd` -- assemble the ``mitmdump`` argv list.
- :func:`launch_proxy` -- start ``mitmdump`` as a subprocess and stream its
  stdout back through the project logger until the user interrupts.

The actual request/response pipeline lives in
:class:`src.engine.KonstaEngine` and is loaded by ``mitmdump`` as a
``-s`` addon script.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING, List, Optional

from src.config import Config

if TYPE_CHECKING:
    from src.metrics import Metrics

logger = logging.getLogger("main.proxy_launcher")


def install_ca_if_requested(install_ca: bool, config: Config):
    """Install the Konsta CA system-wide if the user opted in via --install-ca.

    ``config`` is the process-wide :class:`Config` constructed by
    :func:`src.main.main`; it is forwarded to the temporary
    :class:`ContextCompressionProxy` so the CA install uses the same
    paths, dump directory, and target hosts as the real proxy.

    Returns the temporary :class:`ContextCompressionProxy` used to drive
    the CA install, or ``None`` when the user did not request CA
    installation. The caller (typically :func:`launch_proxy`) is
    responsible for invoking ``proxy.done()`` so the proxy's async
    background thread is torn down cleanly before the process exits.
    """
    if not install_ca:
        logger.info(
            "CA system-wide installation skipped (no --install-ca flag). "
            "The proxy will run but HTTPS interception will not be transparent."
        )
        return None

    # Loud, user-visible warning on stdout (in addition to the logger.warning inside
    # the proxy) so the user cannot miss it even if logs are silenced.
    print("=" * 72, file=sys.stderr)
    print(
        "WARNING: Installing Konsta self-signed CA into the system trust store.",
        file=sys.stderr,
    )
    print(
        "This allows Konsta to decrypt HTTPS traffic. "
        "Only proceed if you trust this machine.",
        file=sys.stderr,
    )
    print("=" * 72, file=sys.stderr)

    # Defer the heavy import so --help and non-CA paths don't pay the cost.
    from src.proxy_core import ContextCompressionProxy

    # Use a temporary ContextCompressionProxy instance purely to drive the CA install.
    # The actual proxy is loaded by mitmdump as an addon; this object is just a
    # convenient handle to the same CAManager/config. Pass a fresh ``Metrics``
    # so the temporary proxy has a private store; we throw the proxy away
    # after the install so any state recorded here is discarded.
    try:
        from src.metrics import Metrics
        tmp_proxy = ContextCompressionProxy(metrics=Metrics(), config=config)
    except Exception as e:
        logger.error(f"Failed to construct proxy for CA installation: {e}")
        sys.exit(1)

    ok = tmp_proxy.install_ca_system_wide()
    if not ok:
        logger.error(
            "CA installation failed. The proxy will still start, but HTTPS "
            "interception will not be trusted by clients."
        )
    else:
        logger.info("CA installed. Transparent HTTPS interception is now available.")

    return tmp_proxy


def create_distillation_worker(
    config: Config,
    metrics: Optional["Metrics"] = None,
):
    """Construct a :class:`DistillationWorker` wired to the configured LLM processor.

    ``config`` is the process-wide :class:`Config` constructed by
    :func:`src.main.main`; passing it explicitly replaces the previous
    implicit dependency on the ``from src.config import config``
    singleton, which was instantiated at import time and therefore
    forced every consumer of this module to have ``LLM_API_KEY`` set.

    ``metrics`` is an optional shared :class:`Metrics` instance. When
    provided, the worker records latency/success into the same store as
    the parent process and the mitmdump addon. When omitted, a fresh
    private store is created for backwards compatibility with direct
    callers/tests.

    ``main.py`` runs ``mitmdump`` as a subprocess and the actual proxy --
    including its :class:`DistillationWorker` -- lives inside that subprocess.
    This helper documents the wiring pattern used by :class:`ContextCompressionProxy`
    and gives the operator a single place to inspect or pre-validate the worker
    construction.  The function is safe to call from outside an asyncio loop:
    the worker only needs to be ``await start()``-ed once an event loop is
    available, which the proxy's ``_init_async_loop`` performs.

    Returns:
        The configured :class:`DistillationWorker` instance, or ``None`` if
        LLM distillation is disabled in the supplied config (in which case
        the proxy also skips background distillation entirely).
    """
    # Defer heavy imports so --help and other non-proxy paths don't pay the cost.
    from src.distillation_worker import DistillationWorker
    from src.llm_processor import LLMProcessor
    from src.metrics import Metrics

    if metrics is None:
        metrics = Metrics()

    if config.distillation_mode == "disabled":
        logger.debug(
            "LLM distillation disabled (mode=disabled); "
            "DistillationWorker will not be created."
        )
        return None

    # Defer the breaker import so --help and other non-proxy paths don't pay
    # the cost (and so the resilience module loads only when needed).
    from src.resilience import CircuitBreaker

    breaker = CircuitBreaker(
        failure_threshold=config.breaker_failure_threshold,
        recovery_timeout=config.breaker_recovery_timeout,
        half_open_max_calls=config.breaker_half_open_max_calls,
        name="LLMProcessor",
    )
    processor = LLMProcessor(config, circuit_breaker=breaker)
    worker = DistillationWorker(
        processor=processor,
        cache_size=config.distillation_cache_size,
        metrics=metrics,
    )
    logger.info(
        "DistillationWorker prepared for processor=%s mode=%s (circuit breaker enabled)",
        config.llm_model,
        config.distillation_mode,
    )
    return worker


def _build_mitmdump_cmd(addon_path: str, port: int) -> List[str]:
    """Assemble the argv list used to launch ``mitmdump``.

    Centralised so tests can assert on the exact command shape (in
    particular the ``-s`` addon path and the ``--set block_global=false``
    override) without depending on the surrounding subprocess plumbing.

    Args:
        addon_path: Filesystem path to the ``proxy_core.py`` addon module.
        port: TCP port ``mitmdump`` should listen on.

    Returns:
        A list of argv tokens suitable for :func:`subprocess.Popen`.
    """
    return [
        "mitmdump",
        "-s", addon_path,
        "-p", str(port),
        "--set", "block_global=false",
    ]


def launch_proxy(cmd=None, proxy=None, config: Optional[Config] = None) -> int:
    """Start the ``mitmdump`` subprocess and stream its stdout to the logger.

    Validates the configuration, verifies ``mitmdump`` is on ``PATH``, then
    spawns the subprocess and forwards its stdout line-by-line through the
    project logger until the process exits or the user interrupts.

    When ``proxy`` is supplied (typically a temporary proxy constructed by
    :func:`install_ca_if_requested` to drive CA installation), this
    function guarantees ``proxy.done()`` is invoked exactly once during
    shutdown via a ``finally`` block. That tears down the proxy's async
    loop and worker thread before the process exits, preventing the
    "RuntimeError: Event loop is closed" / orphaned-daemon-thread
    warnings that previously surfaced on Ctrl+C.

    Args:
        cmd: Optional pre-built mitmdump argv list. When ``None``, the
            command is assembled from ``config`` via
            :func:`_build_mitmdump_cmd`.
        proxy: Optional :class:`ContextCompressionProxy` whose ``done()``
            will be called on exit. ``None`` is fine; this is the case
            when the user did not pass ``--install-ca``.
        config: Process-wide :class:`Config` instance constructed by
            :func:`src.main.main`. When ``None`` (e.g. when callers use
            this helper directly without going through ``main``), the
            proxy port / target hosts fall back to a freshly-built
            default :class:`Config` so the launcher still has something
            sensible to print. Production callers always supply
            ``config``.

    Returns:
        The process exit code (0 on graceful shutdown).

    Raises:
        SystemExit: on configuration errors, missing ``mitmdump`` binary,
            or any unexpected runtime failure. Signal handlers in
            :mod:`src.main` translate ``KeyboardInterrupt`` into a clean
            shutdown.
    """
    if config is None:
        config = Config()
    if cmd is None:
        addon_path = os.path.join(os.path.dirname(__file__), "proxy_core.py")
        cmd = _build_mitmdump_cmd(addon_path, int(config.proxy_port))

    logger.info(f"Launching proxy on {config.proxy_host}:{config.proxy_port}")
    logger.info(f"Targeting hosts: {', '.join(config.target_hosts)}")
    logger.info(f"Command: {' '.join(cmd)}")

    process: Optional[subprocess.Popen] = None
    rc: Optional[int] = None
    try:
        # Validate configuration first
        config._validate()

        # Verify mitmdump is available before launching the long-running process
        subprocess.run(["mitmdump", "--version"], capture_output=True, check=True)

        # Start the proxy process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream logs from mitmdump to our logger
        assert process.stdout is not None  # PIPE was passed, so stdout is set
        for line in process.stdout:
            logger.info(f"[mitmproxy] {line.strip()}")

        rc = process.wait()

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        sys.exit(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(
            "mitmdump not found in PATH. "
            "Please install mitmproxy: pip install mitmproxy"
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down proxy...")
        if process is not None:
            process.terminate()
            try:
                rc = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                rc = process.wait()
        else:
            rc = 0
    except Exception as e:
        # Safety net: log unexpected errors and exit gracefully
        logger.exception(f"Unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        # Tear down any proxy we were handed. ``done`` is idempotent and
        # thread-safe, so double-invocation is harmless; the inner
        # try/except guards against ``done`` blowing up the parent
        # shutdown path on top of whatever else is unwinding.
        if proxy is not None:
            try:
                proxy.done()
            except Exception:
                logger.exception("Failed to shut down proxy cleanly")

    # ``rc`` is None only if an exception handler called ``sys.exit``;
    # fall back to a non-zero exit code so the function still returns an
    # int for callers that ignore ``SystemExit``.
    return rc if rc is not None else 1
