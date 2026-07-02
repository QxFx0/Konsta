"""Engine that encapsulates Konsta's compression/distillation logic.

:class:`KonstaEngine` owns the request/response processing pipeline that
used to live directly inside :class:`src.proxy_core.ContextCompressionProxy`.
By moving that logic behind a ``ProxyAdapter`` boundary, the engine can be
unit-tested with a mock adapter and, in principle, retargeted at a
different proxy library by writing a new adapter subclass.

The engine deliberately avoids importing mitmproxy types. All access to
proxy-specific request/response attributes goes through the supplied
:class:`ProxyAdapter`.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src import api_parsers
from src.api_parsers import Message, ParsedRequest
from src.compression_engine import CompressionEngine
from src.config import Config
from src.distillation_worker import DistillationWorker
from src.metrics import Metrics
from src.proxy_adapter import ProxyAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dump sanitisation
# ---------------------------------------------------------------------------

# Keys whose values must always be redacted in dump output. Compared
# case-insensitively against dict keys.
_SENSITIVE_KEYS = frozenset({
    "authorization",
    "api_key",
    "apikey",
    "token",
    "password",
    "ca_key_password",
    "secret",
    "private_key",
    "cookie",
    "set-cookie",
    "session",
    "x-api-key",
    "api-key",
    "access_token",
    "id_token",
    "refresh_token",
    "bearer",
    "privatekey",
})

# ``key`` is too generic to always redact (it can show up in benign
# config payloads). Only redact it when the value is a string that
# "looks like a secret" -- heuristically, a non-trivial string length.
_MIN_KEY_SECRET_LENGTH = 8

# Value substituted for any redacted field.
_REDACTED = "[REDACTED]"


def _sanitize_for_dump(data: Any) -> Any:
    """Recursively redact sensitive values from ``data`` for safe dumping.

    Walks dicts and lists in place-shape, replacing values for keys in
    :data:`_SENSITIVE_KEYS` (matched case-insensitively) with
    ``"[REDACTED]"``. The generic key ``key`` is only redacted when its
    value is a string of at least :data:`_MIN_KEY_SECRET_LENGTH`
    characters, to avoid false positives on benign short values like
    ``"color"``.

    Returns a deep-enough copy that the caller's input is not mutated.
    """
    if isinstance(data, dict):
        sanitized: Dict[str, Any] = {}
        for k, v in data.items():
            if _should_redact_key(k, v):
                sanitized[k] = _REDACTED
            else:
                sanitized[k] = _sanitize_for_dump(v)
        return sanitized
    if isinstance(data, list):
        return [_sanitize_for_dump(item) for item in data]
    return data


def _should_redact_key(key: Any, value: Any) -> bool:
    """Return True if the ``(key, value)`` pair should be redacted."""
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    if lowered in _SENSITIVE_KEYS:
        return True
    if lowered == "key":
        # Only redact ``key`` when the value looks like a real secret.
        return (
            isinstance(value, str)
            and len(value) >= _MIN_KEY_SECRET_LENGTH
        )
    return False


class KonstaEngine:
    """Compression / distillation engine decoupled from any proxy library.

    Args:
        config: Application configuration.
        compression_engine: Optional pre-built :class:`CompressionEngine`.
            If ``None``, the engine constructs one from ``config``.
        distillation_worker: Optional pre-built :class:`DistillationWorker`.
            When ``None`` and ``config.distillation_mode != "disabled"``,
            the engine still works but submits no background tasks.
        metrics: Optional :class:`Metrics` instance. Falls back to the
            process-wide shared singleton when omitted.
    """

    def __init__(
        self,
        config: Config,
        compression_engine: Optional[CompressionEngine] = None,
        distillation_worker: Optional[DistillationWorker] = None,
        metrics: Optional[Metrics] = None,
    ):
        self.config = config
        if metrics is None:
            metrics = Metrics()
        self.metrics = metrics
        self.compression_engine = compression_engine or CompressionEngine(
            self.config, metrics=self.metrics
        )
        self.distillation_worker = distillation_worker

        # Track the most recently submitted distillation request so we can
        # apply its result on the *next* request. Background-mode distillation
        # lags one request behind: by the time the next flow arrives the
        # worker has usually finished processing the previous body.
        self._last_distillation_request_id: Optional[str] = None
        self._distillation_request_counter = 0

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def process_request(self, adapter: ProxyAdapter, request: Any) -> None:
        """Apply context compression to ``request`` via ``adapter``.

        Performs the same pipeline as the legacy
        :meth:`ContextCompressionProxy.request` handler:

        1. Record metrics.
        2. Snapshot the original body size on the request object so the
           response phase can report compression ratios.
        3. Skip Cerebras hosts to avoid recursion.
        4. Look up a cached distillation result from the previous request.
        5. Parse, compress, and rewrite the request body.
        6. Enqueue the locally-compressed body for background distillation.

        Any exception is caught and logged so a misbehaving flow can never
        break the proxy. The body of this method is intentionally thin;
        each step lives in a dedicated private helper below.
        """
        host = adapter.get_request_host(request)

        try:
            self.metrics.record_request()
        except Exception:
            logger.exception("Failed to record request metrics")

        try:
            if not self._snapshot_original_size(adapter, request):
                # Helper short-circuited (e.g. host is Cerebras).
                return

            # DUMP: Original request
            self._dump_data(f"{host}_original_req.json", {
                "url": adapter.get_request_url(request),
                "headers": dict(adapter.get_request_headers(request)),
                "content": (
                    adapter.get_request_text(request)
                    if self.config.dump_include_bodies
                    else "[OMITTED]"
                ),
            })

            compressed_json = await self._compress_and_rewrite(adapter, request, host)
            if compressed_json is None:
                # Either the body was unsuitable for compression or the
                # compression pipeline returned nothing. In both cases
                # there is nothing to distill.
                return

            self._submit_for_distillation(compressed_json)

        except json.JSONDecodeError as e:
            logger.exception(f"Failed to parse request body for {host}: {e}")
        except (ValueError, TypeError) as e:
            logger.exception(f"Failed to compress/serialize request for {host}: {e}")
        except OSError as e:
            logger.exception(f"I/O error processing request for {host}: {e}")
        except Exception as exc:
            # Unclassified exception: log, record a crash metric for
            # observability, then re-raise so a real bug surfaces instead
            # of being silently swallowed by a broad handler. Callers in
            # the request path (mitmproxy) must never let an exception
            # escape, so any external wrapper that invokes this method is
            # responsible for catching it. Tests assert that the metric is
            # incremented and the exception propagates.
            try:
                self.metrics.record_crash(type(exc).__name__)
            except Exception:
                logger.exception("Failed to record crash metric")
            logger.exception(
                f"Unexpected error processing request for {host}: {exc}"
            )
            raise

    # ------------------------------------------------------------------
    # process_request pipeline helpers
    # ------------------------------------------------------------------

    def _snapshot_original_size(
        self, adapter: ProxyAdapter, request: Any
    ) -> bool:
        """Capture pre-compression body size and skip Cerebras hosts.

        Returns ``False`` to signal the caller should short-circuit (e.g.
        the host is Cerebras and the request must be passed through
        untouched). Otherwise the size is recorded on ``request`` and
        ``True`` is returned so processing continues.
        """
        host = adapter.get_request_host(request)

        # Capture the original request size BEFORE any compression. The
        # response() handler reads this off the flow to populate
        # X-Konsta-Original-Size; after compression, the body is the
        # compressed one, so we must snapshot the size first.
        original_size = len(adapter.get_request_body(request))
        try:
            request.konsta_orig_size = original_size
        except Exception:
            # If the request object doesn't accept attribute writes
            # (e.g. a mock), skip the snapshot; response() will fall
            # back to the current body size.
            pass

        # CRITICAL: Skip Cerebras requests to avoid recursion/403
        if "cerebras.ai" in host:
            logger.debug("Skipping Cerebras request to avoid recursion/403")
            return False

        return True

    async def _compress_and_rewrite(
        self, adapter: ProxyAdapter, request: Any, host: str
    ) -> Optional[str]:
        """Parse, compress, and rewrite ``request`` body in place.

        Returns the serialised compressed payload (so the caller can hand
        it off to the distillation worker) or ``None`` when nothing was
        rewritten (oversized / empty body, unparseable JSON, or
        compression returned no payload).
        """
        # If the previous request's distillation finished in the
        # background while we were idle, apply it now. This is the
        # "lags one request behind" semantic: the body that hits the
        # upstream server for request N+1 is the *distilled* version
        # of request N.
        cached_distilled_body = self._get_cached_distillation()

        content_bytes = adapter.get_request_body(request)
        if content_bytes and len(content_bytes) > self.config.max_payload_bytes:
            logger.warning(
                f"Request payload too large ({len(content_bytes)} bytes). "
                f"Skipping compression."
            )
            return None

        request_body = adapter.get_request_text(request)
        if not request_body:
            return None

        parsed_request = self._parse_request(request_body)
        if not parsed_request:
            return None

        compressed_request = await self.compress_context(parsed_request)
        if not compressed_request:
            return None

        # DUMP: Compressed request
        self._dump_data(f"{host}_compressed_req.json", {
            "content": (
                compressed_request
                if self.config.dump_include_bodies
                else "[OMITTED]"
            ),
        })

        # Update the request body with compressed context.
        compressed_json = json.dumps(compressed_request)

        if cached_distilled_body is not None:
            # Inject the distilled text back into the compressed payload so
            # the upstream receives valid provider JSON instead of a raw
            # flattened summary string.
            distilled_request = dict(compressed_request)
            distilled_request["messages"] = [
                {"role": "user", "content": cached_distilled_body}
            ]
            compressed_json = json.dumps(distilled_request)
            adapter.set_request_body(request, compressed_json.encode("utf-8"))
            logger.info(
                "Applied cached LLM distillation from prior request "
                f"(id={self._last_distillation_request_id})"
            )
        else:
            adapter.set_request_body(request, compressed_json.encode("utf-8"))
            logger.info(f"Compressed request for {host}")

        logger.debug(f"Compressed payload size: {len(compressed_json)} bytes")
        logger.debug(f"Compressed payload keys: {list(compressed_request.keys())}")

        return compressed_json

    def _submit_for_distillation(self, compressed_json: str) -> None:
        """Hand ``compressed_json`` off to the background distillation worker."""
        self._submit_distillation(compressed_json)

    def process_response(self, adapter: ProxyAdapter, response: Any) -> None:
        """Inject compression metrics and dump diagnostics for ``response``.

        Writes ``X-Konsta-Original-Size`` and ``X-Konsta-Compressed-Size``
        headers so the demo client can display the compression ratio, and
        dumps the raw response body for offline inspection.

        Any exception is caught and logged so a misbehaving flow can never
        break the proxy.
        """
        host = ""
        try:
            host = adapter.get_request_host(response)
            # konsta_orig_size is set by process_request() before any
            # compression runs, so it reflects the true pre-compression
            # request size. After process_request() rewrites the body,
            # adapter.get_request_body() returns the compressed body
            # whose length is the post-compression size. If compression
            # was skipped or failed, both values will be equal (the
            # original body).
            original_size = getattr(response, "konsta_orig_size", len(adapter.get_request_body(response)))
            compressed_size = len(adapter.get_request_body(response))

            adapter.set_response_header(response, "X-Konsta-Original-Size", str(original_size))
            adapter.set_response_header(response, "X-Konsta-Compressed-Size", str(compressed_size))

            # DUMP: Raw response
            self._dump_data(f"{host}_response.json", {
                "url": adapter.get_request_url(response),
                "status": adapter.get_response_status(response),
                "headers": dict(adapter.get_response_headers(response)),
                "content": (
                    adapter.get_response_text(response)
                    if self.config.dump_include_bodies
                    else "[OMITTED]"
                ),
            })
        except (OSError, TypeError) as e:
            logger.exception(f"Error dumping response for {host}: {e}")
        except Exception:
            logger.exception(f"Unexpected error in process_response for {host}")

    # ------------------------------------------------------------------
    # Compression helpers
    # ------------------------------------------------------------------

    async def compress_context(
        self,
        request_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Apply semantic context compression to ``request_data``.

        LLM-based distillation is *not* performed here. Distillation runs
        asynchronously on :class:`DistillationWorker` and is wired into the
        request path by :meth:`process_request`. This keeps the proxy
        handler non-blocking: it never awaits the remote LLM call.

        Args:
            request_data: The request data to compress.
        """
        try:
            messages = request_data.get("messages", [])
            provider = request_data.get("provider", "generic")
            original_payload = request_data.get("original_payload", {})

            # Step 1: Apply semantic compression (local, fast, safe).
            compressed_messages = self.compression_engine.compress(messages)

            # Step 2: Reconstruct the original payload with compressed
            # messages. LLM-based distillation runs asynchronously on
            # DistillationWorker; see process_request() for the
            # enqueue/lookup logic.
            standardized_msgs = [
                Message(role=m['role'], content=m['content'])
                for m in compressed_messages
            ]

            parsed_obj = ParsedRequest(
                payload=original_payload,
                messages=standardized_msgs,
                provider=provider,
            )

            final_payload = api_parsers.serialize_request(parsed_obj)
            logger.info(f"Final compressed payload prepared for provider: {provider}")
            return final_payload

        except (KeyError, TypeError, ValueError) as e:
            logger.exception(f"Error in compress_context (message reconstruction): {e}")
            return None
        except Exception as exc:
            # Unclassified exception in the compression pipeline. Log and
            # record a crash metric so operators see the new failure mode
            # without re-introducing a bare ``except Exception`` that
            # would mask real bugs. We still return ``None`` rather than
            # re-raising: the upstream caller (``_compress_and_rewrite``)
            # treats ``None`` as "no rewrite possible" and lets the proxy
            # fall through with the original body. This preserves the
            # best-effort semantics for misbehaving payloads.
            try:
                self.metrics.record_crash(type(exc).__name__)
            except Exception:
                logger.exception("Failed to record crash metric")
            logger.exception(f"Unexpected error in compress_context: {exc}")
            return None

    # ------------------------------------------------------------------
    # Distillation plumbing
    # ------------------------------------------------------------------

    def _get_cached_distillation(self) -> Optional[str]:
        """Return the most recently submitted distillation result, if any.

        Returns ``None`` if no worker is configured, no request has been
        submitted yet, or the worker raised while looking up the result.
        """
        if (
            self.distillation_worker is None
            or self._last_distillation_request_id is None
        ):
            return None
        try:
            return self.distillation_worker.get_result(
                self._last_distillation_request_id
            )
        except Exception:
            logger.exception(
                "DistillationWorker.get_result() raised; "
                "falling back to local compression"
            )
            return None

    def _submit_distillation(self, compressed_json: str) -> None:
        """Enqueue ``compressed_json`` for background LLM distillation.

        Best-effort: a misbehaving worker must never break the request
        path, so any ``RuntimeError`` from :meth:`DistillationWorker.submit`
        is logged and swallowed.
        """
        if (
            self.distillation_worker is None
            or self.config.distillation_mode == "disabled"
        ):
            return
        self._distillation_request_counter += 1
        new_request_id = f"req-{self._distillation_request_counter}"
        try:
            self.distillation_worker.submit(new_request_id, compressed_json)
            self._last_distillation_request_id = new_request_id
        except RuntimeError as submit_err:
            logger.warning(f"Could not submit distillation task: {submit_err}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_request(self, request_body: str) -> Optional[Dict[str, Any]]:
        """Parse the request body based on the API format."""
        try:
            data = json.loads(request_body)
            parsed_request = api_parsers.parse_request(data, self.config.proxy_host)
            if not parsed_request:
                return None
            return {
                "messages": [msg._asdict() for msg in parsed_request.messages],
                "provider": parsed_request.provider,
                "original_payload": parsed_request.payload,
            }
        except json.JSONDecodeError:
            logger.warning("Failed to parse request body as JSON")
            return None

    def _dump_data(self, filename: str, data: Any) -> None:
        """Helper to dump raw data to the dumps directory.

        Best-effort: any file system or serialization failure is logged
        but never raised so that diagnostics dumping cannot break the
        proxy.

        Before writing, the payload is passed through
        :func:`_sanitize_for_dump` so that credentials (API keys,
        authorization headers, etc.) never end up on disk.
        """
        try:
            dump_dir = Path(self.config.dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = dump_dir / f"{timestamp}_{filename}"

            sanitized = _sanitize_for_dump(data)

            with open(file_path, "w", encoding="utf-8") as f:
                if isinstance(sanitized, (dict, list)):
                    json.dump(sanitized, f, indent=2, ensure_ascii=False)
                else:
                    f.write(str(sanitized))
        except (OSError, TypeError) as e:
            logger.exception(f"Failed to dump data to {filename}: {e}")
