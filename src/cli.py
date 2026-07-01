"""Command-line interface helpers for the Konsta entrypoint.

This module is intentionally side-effect free at import time: it only
defines :func:`parse_args` (argparse wiring) and
:func:`apply_distillation_settings` (config mutation). Logging
configuration and the actual proxy / doctor dispatch live in
:mod:`src.main`, :mod:`src.doctor` and :mod:`src.proxy_launcher`.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from src.config import Config

logger = logging.getLogger("main.cli")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the Konsta entrypoint.

    This function deliberately reads only class-level constants on
    :class:`Config` (e.g. :attr:`Config.DISTILLATION_MODES`) so it can be
    invoked during ``--help`` without instantiating a :class:`Config`
    and therefore without requiring ``LLM_API_KEY`` or any other env
    variable to be set.
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m src.main",
        description=(
            "Konsta Context Compression Proxy. By default, Konsta generates a local "
            "CA but does NOT install it into the system trust store. Pass --install-ca "
            "to opt in to system-wide CA installation (required for transparent HTTPS "
            "interception)."
        ),
    )
    parser.add_argument(
        "--install-ca",
        action="store_true",
        help=(
            "Opt in to installing the Konsta self-signed CA into the system trust "
            "store. Required for transparent HTTPS interception. Will prompt for "
            "sudo / administrator privileges. Without this flag, Konsta only "
            "generates the local CA files and does not modify system trust stores."
        ),
    )
    parser.add_argument(
        "--distillation-mode",
        choices=list(Config.DISTILLATION_MODES),
        default="background",
        help=(
            "How LLM distillation is invoked from the request path. "
            "'background' (default) runs distillation asynchronously in "
            "DistillationWorker so the handler never blocks on the LLM call. "
            "'disabled' skips the worker entirely. Use --enable-llm-distillation "
            "for the legacy boolean control."
        ),
    )
    parser.add_argument(
        "--enable-llm-distillation",
        action="store_true",
        help=(
            "Deprecated alias. Passes through to --distillation-mode: when set, "
            "distillation_mode becomes 'background'; otherwise it stays at the "
            "value provided to --distillation-mode (default 'background'). Use "
            "--distillation-mode directly in new code."
        ),
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=2.0,
        help=(
            "Maximum number of seconds to wait for the LLM distillation call "
            "before falling back to the locally compressed result. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Override the LLM model used for context distillation. When set, "
            "this overrides the LLM_MODEL environment variable. When neither "
            "this flag nor LLM_MODEL is provided, the Config default applies. "
            "Konsta does not prompt interactively for model selection."
        ),
    )

    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run health checks for the Konsta deployment.",
        description=(
            "Inspect CA files, passphrase, LLM credentials, endpoint "
            "reachability, embedding engine imports, and the shared Metrics "
            "singleton. Exits 0 on success, 1 when a critical check fails."
        ),
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        dest="doctor_json",
        help="Emit machine-readable JSON instead of formatted text.",
    )

    return parser.parse_args(argv)


def apply_distillation_settings(
    config: Config,
    distillation_mode: str,
    enable_llm_distillation: bool,
    llm_timeout: float,
) -> None:
    """
    Push CLI flags into ``config`` before any proxy is constructed.

    ``config`` is the process-wide :class:`Config` instance built by
    :func:`src.main.main`; passing it explicitly removes the historical
    dependency on a module-level ``from src.config import config``
    singleton, which used to force ``Config.__post_init__`` to run at
    import time.

    CLI flags take precedence over environment variables so the operator can
    override defaults from the command line. ``enable_llm_distillation`` is
    honoured for backwards compatibility: when the operator passes the legacy
    flag it overrides the explicit ``distillation_mode`` selection (since the
    older boolean form could not express "background vs. something else").
    """
    chosen_mode = distillation_mode
    if enable_llm_distillation:
        chosen_mode = "background"
    config.distillation_mode = chosen_mode
    config.llm_timeout = float(llm_timeout)
    logger.info(
        "LLM distillation: mode=%s (timeout=%.2fs)",
        config.distillation_mode,
        config.llm_timeout,
    )
