"""Health-check helpers for the ``doctor`` CLI subcommand.

Each ``_check_*`` function returns a dict shaped as::

    {
        "name": <str>,
        "status": "ok" | "warn" | "fail",
        "message": <str>,
        # optional:
        "details": {...},
    }

:func:`_run_doctor` aggregates them into a report dict with summary counts
and an exit code (1 if any check ``status == "fail"`` else 0). Warnings do
not affect the exit code so operators can run ``doctor`` from cron without
spurious failures.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("main.doctor")


def _check_ca_files(cfg) -> Dict[str, Any]:
    """Verify the configured CA cert/key paths exist and are readable."""
    cert_path = Path(getattr(cfg, "ca_cert_path", ""))
    key_path = Path(getattr(cfg, "ca_key_path", ""))
    cert_ok = bool(cert_path.is_file()) and os.access(cert_path, os.R_OK)
    key_ok = bool(key_path.is_file()) and os.access(key_path, os.R_OK)
    if cert_ok and key_ok:
        return {
            "name": "ca_files",
            "status": "ok",
            "message": f"CA cert and key present at {cert_path.parent}",
            "details": {"cert_path": str(cert_path), "key_path": str(key_path)},
        }
    missing = []
    if not cert_ok:
        missing.append(f"cert={cert_path}")
    if not key_ok:
        missing.append(f"key={key_path}")
    return {
        "name": "ca_files",
        "status": "warn",
        "message": (
            "CA files missing or unreadable (" + ", ".join(missing) + "). "
            "They will be generated on first run."
        ),
    }


def _check_ca_passphrase() -> Dict[str, Any]:
    """Verify a CA key passphrase is available without prompting."""
    env_password = os.getenv("CA_KEY_PASSWORD", "").strip()
    if env_password:
        return {
            "name": "ca_passphrase",
            "status": "ok",
            "message": "CA_KEY_PASSWORD environment variable is set.",
        }
    try:
        import keyring  # local import: avoid cost on --help / proxy paths

        from src.ca_manager import CAKeyManager

        manager = CAKeyManager()
        existing = keyring.get_password(manager.service_name, manager.username)
        if existing:
            return {
                "name": "ca_passphrase",
                "status": "ok",
                "message": "CA key passphrase present in OS keyring.",
            }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "name": "ca_passphrase",
            "status": "warn",
            "message": f"Keyring lookup failed ({type(exc).__name__}): {exc}",
        }
    return {
        "name": "ca_passphrase",
        "status": "warn",
        "message": (
            "No CA_KEY_PASSWORD env var and no keyring entry. Non-interactive "
            "use requires CA_KEY_PASSWORD or an unlocked OS keyring."
        ),
    }


def _check_llm_api_key(cfg) -> Dict[str, Any]:
    """Verify an LLM API key is configured."""
    api_key = (getattr(cfg, "llm_api_key", "") or "").strip()
    if api_key:
        # Don't print the key itself; just report its presence and length.
        return {
            "name": "llm_api_key",
            "status": "ok",
            "message": f"LLM API key configured (length={len(api_key)}).",
        }
    return {
        "name": "llm_api_key",
        "status": "fail",
        "message": (
            "LLM_API_KEY is not set. Set the LLM_API_KEY environment variable "
            "(CEREBRAS_API_KEY is also accepted)."
        ),
    }


def _check_llm_endpoint(
    cfg,
    timeout: float = 3.0,
    httpx_client=None,
) -> Dict[str, Any]:
    """Lightly probe the configured LLM endpoint for reachability.

    ``httpx_client`` is an optional pre-built httpx client. Production callers
    pass a real :class:`httpx.Client`; tests pass a stub that implements
    ``head()`` and ``stream()`` so the check is deterministic and does not
    perform real network I/O.
    """
    endpoint = getattr(cfg, "llm_endpoint", "")
    if not endpoint:
        return {
            "name": "llm_endpoint",
            "status": "warn",
            "message": "No llm_endpoint configured.",
        }

    if httpx_client is None:
        try:
            import httpx  # local import: keep --help / offline doctor cheap
        except ImportError:
            return {
                "name": "llm_endpoint",
                "status": "warn",
                "message": "httpx not installed; skipping reachability check.",
            }
        head = httpx.head
        stream = httpx.stream
    else:
        head = httpx_client.head
        stream = httpx_client.stream

    try:
        response = head(endpoint, timeout=timeout)
    except Exception:
        # Some endpoints reject HEAD with a transport-level error; fall back
        # to a streaming GET so we only confirm reachability, not content.
        try:
            with stream("GET", endpoint, timeout=timeout) as resp_stream:
                status_code = resp_stream.status_code
        except Exception as exc2:
            return {
                "name": "llm_endpoint",
                "status": "warn",
                "message": (
                    f"LLM endpoint unreachable: {endpoint} "
                    f"({type(exc2).__name__}: {exc2})"
                ),
            }
        else:
            return {
                "name": "llm_endpoint",
                "status": "ok",
                "message": (
                    f"LLM endpoint reachable: {endpoint} (HTTP {status_code})"
                ),
                "details": {"status_code": status_code, "url": endpoint},
            }
    # Any HTTP response (including 4xx/5xx) implies the network path is open.
    return {
        "name": "llm_endpoint",
        "status": "ok",
        "message": (
            f"LLM endpoint reachable: {endpoint} (HTTP {response.status_code})"
        ),
        "details": {"status_code": response.status_code, "url": endpoint},
    }


def _check_embedding_engine() -> Dict[str, Any]:
    """Confirm semantic compression (CompressionEngine + SentenceTransformer) is available.

    This check is intentionally *import-only*: constructing :class:`CompressionEngine`
    triggers a SentenceTransformer model load, which on first run downloads
    weights from the HuggingFace Hub and can easily exceed 30 seconds. We
    defer the actual model load to first request and rely on the runtime
    error path (logged by :class:`CompressionEngine`) to surface a bad
    installation. ``doctor`` should stay fast and offline-friendly.
    """
    try:
        from src.compression_engine import HAS_SENTENCE_TRANSFORMERS, CompressionEngine
    except Exception as exc:
        return {
            "name": "embedding_engine",
            "status": "warn",
            "message": f"Failed to import CompressionEngine: {exc}",
        }
    if not HAS_SENTENCE_TRANSFORMERS or getattr(
        CompressionEngine, "compress", None
    ) is None:
        return {
            "name": "embedding_engine",
            "status": "warn",
            "message": (
                "sentence-transformers is not installed; semantic dedup "
                "will be unavailable."
            ),
        }
    return {
        "name": "embedding_engine",
        "status": "ok",
        "message": "CompressionEngine with SentenceTransformer importable.",
    }


def _check_metrics() -> Dict[str, Any]:
    """Capture a snapshot from the shared Metrics singleton if available."""
    try:
        from src.metrics import get_shared_metrics
        metrics = get_shared_metrics()
        if metrics is None:
            return {
                "name": "metrics",
                "status": "warn",
                "message": "Shared Metrics singleton has not been installed yet.",
            }
        snapshot = metrics.snapshot()
    except Exception as exc:
        return {
            "name": "metrics",
            "status": "warn",
            "message": f"Failed to capture metrics snapshot: {exc}",
        }
    return {
        "name": "metrics",
        "status": "ok",
        "message": "Shared Metrics snapshot captured.",
        "details": {"snapshot": snapshot},
    }


def _run_doctor(cfg, httpx_client=None) -> Dict[str, Any]:
    """Run every doctor check against ``cfg`` and return a structured report.

    The returned dict has the shape::

        {
            "checks": [{"name": ..., "status": ..., "message": ...}, ...],
            "summary": {"ok": int, "warn": int, "fail": int},
            "exit_code": 0 or 1,
        }

    ``exit_code`` is ``1`` if any check has ``status == "fail"`` (critical),
    otherwise ``0``. Warnings do not affect the exit code so operators can
    run ``doctor`` from cron without spurious failures.
    """
    checks = [
        _check_ca_files(cfg),
        _check_ca_passphrase(),
        _check_llm_api_key(cfg),
        _check_llm_endpoint(cfg, httpx_client=httpx_client),
        _check_embedding_engine(),
        _check_metrics(),
    ]
    summary = {"ok": 0, "warn": 0, "fail": 0}
    for check in checks:
        summary[check["status"]] = summary.get(check["status"], 0) + 1
    has_fail = summary.get("fail", 0) > 0
    return {
        "checks": checks,
        "summary": summary,
        "exit_code": 1 if has_fail else 0,
    }


def _format_doctor_report(report: Dict[str, Any]) -> str:
    """Render a doctor report as a human-readable text block."""
    markers = {"ok": "[OK]   ", "warn": "[WARN] ", "fail": "[FAIL] "}
    lines = ["Konsta Doctor Report", "=" * 40]
    for check in report.get("checks", []):
        marker = markers.get(check["status"], "[?]    ")
        lines.append(f"{marker}{check['name']}: {check['message']}")
    summary = report.get("summary", {})
    lines.append("")
    lines.append(
        "Summary: "
        f"{summary.get('ok', 0)} OK, "
        f"{summary.get('warn', 0)} WARN, "
        f"{summary.get('fail', 0)} FAIL"
    )
    return "\n".join(lines)


def _run_doctor_command(args, cfg) -> int:
    """Entry point for the ``doctor`` subcommand.

    Prints the report (formatted text or JSON) and returns the process exit
    code so :func:`main` can ``sys.exit`` with the right value.
    """
    try:
        import httpx
    except ImportError:
        report = _run_doctor(cfg, httpx_client=None)
    else:
        with httpx.Client() as client:
            report = _run_doctor(cfg, httpx_client=client)
    if getattr(args, "doctor_json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_doctor_report(report))
    return report["exit_code"]
