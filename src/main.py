"""Konsta Context Compression Proxy -- thin entry point.

The heavy lifting lives in dedicated modules:

- :mod:`src.cli` -- argparse wiring and config mutation helpers.
- :mod:`src.doctor` -- health checks for the ``doctor`` subcommand.
- :mod:`src.proxy_launcher` -- CA installation, distillation worker
  construction, and the ``mitmdump`` subprocess loop.
- :mod:`src.engine` -- the actual compression / distillation pipeline
  loaded by ``mitmdump`` as an addon script.

This file is intentionally short: it sets up logging, registers signal
handlers, parses arguments, and dispatches to either ``doctor`` or the
proxy launcher.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

from src.cli import apply_distillation_settings, parse_args
from src.config import Config, set_config
from src.doctor import _run_doctor_command
from src.proxy_launcher import install_ca_if_requested, launch_proxy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def main() -> None:
    """Entry point for the Konsta Context Compression Proxy.

    Handles argument parsing, optional ``doctor`` subcommand, applies CLI
    flags to the global config, publishes the shared Metrics instance,
    ensures the local CA exists, applies the optional ``--llm-model``
    override, and finally launches the mitmproxy engine.
    """
    args = parse_args()

    # Construct the process-wide ``Config`` exactly once at startup and
    # publish it via :func:`src.config.set_config`. Doing this *after*
    # ``parse_args`` is intentional: ``--help`` (and ``doctor --help``)
    # exit before this point so the ``Config()`` constructor -- which
    # runs env loading and validation -- is never invoked when the user
    # only wants help text. See
    # https://.../konsta-fourth-wave-audit for the original report.
    config = Config()
    set_config(config)

    # ``doctor`` is a standalone subcommand: it runs checks and exits without
    # touching the proxy / CA / model selection pipeline.
    if getattr(args, "command", None) == "doctor":
        sys.exit(_run_doctor_command(args, config))

    # Add project root to PYTHONPATH so mitmdump can find 'src' module
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.environ["PYTHONPATH"] = (
        project_root + os.pathsep + os.environ.get("PYTHONPATH", "")
    )

    logger.info("Welcome to Konsta Context Compression Proxy")

    # Apply distillation CLI flags to the global config before any proxy is
    # constructed (the temporary CA-install proxy and the mitmdump addon
    # both read from `config`).
    apply_distillation_settings(
        config,
        args.distillation_mode,
        args.enable_llm_distillation,
        args.llm_timeout,
    )

    # Publish the shared Metrics instance before any proxy/worker is
    # constructed. The mitmdump subprocess addon loads proxy_core.py, which
    # reads the shared singleton via get_shared_metrics(); doing this here
    # guarantees a single store is observed across the operator-facing
    # code and the addon itself.
    from src.metrics import Metrics, set_shared_metrics

    metrics = Metrics()
    set_shared_metrics(metrics)
    logger.debug("Shared Metrics instance installed for the proxy addon.")

    # 1. Resolve the CA passphrase and ensure the local CA key/cert exist before
    # the proxy launches. The passphrase is sourced from CA_KEY_PASSWORD if set,
    # otherwise from the OS keyring via CAKeyManager. If neither is available
    # AND stdin is non-interactive (e.g. CI, --help exits earlier), this is a
    # hard failure -- the proxy cannot transparently intercept HTTPS without a
    # valid CA.
    from src.ca_manager import CAManager

    try:
        password = config.resolve_ca_password()
    except ValueError as ve:
        logger.error(str(ve))
        sys.exit(1)

    ca_manager = CAManager()
    if not ca_manager.setup_ca(
        cert_path=Path(config.ca_cert_path),
        key_path=Path(config.ca_key_path),
        password=password,
    ):
        logger.error(
            "CA setup failed. The proxy cannot start without a valid CA "
            "key/cert pair."
        )
        sys.exit(1)

    # Apply the optional --llm-model flag to the global config. Env var
    # LLM_MODEL is already consumed by Config.__post_init__; the CLI flag
    # wins over the env var so operators can override defaults per-run.
    if getattr(args, "llm_model", None):
        config.llm_model = args.llm_model
        logger.info(f"LLM model overridden via --llm-model: {config.llm_model}")
    else:
        logger.info(f"Using LLM model from configuration: {config.llm_model}")

    logger.info("Starting system...")

    # 1. Optional CA system-wide installation (opt-in via --install-ca).
    # ``install_ca_if_requested`` returns a temporary proxy instance when
    # it had to construct one to drive the CA install, or ``None`` when
    # CA installation was skipped. ``launch_proxy`` then guarantees
    # ``proxy.done()`` is invoked so the proxy's async loop and worker
    # thread are torn down cleanly before the process exits.
    proxy = install_ca_if_requested(args.install_ca, config=config)

    # 2. Launch mitmdump as a subprocess; stream its output until the user
    # interrupts or the process exits.
    #
    # Before launching, we ensure that the proxy is configured with the required
    # auth middleware and settings.
    launch_proxy(proxy=proxy, config=config)


if __name__ == "__main__":
    # Handle termination signals for clean shutdown
    def signal_handler(sig, frame):
        logger.info("Received termination signal, exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main()
