"""Adapter abstraction decoupling Konsta from a specific proxy library.

This module defines :class:`ProxyAdapter`, an abstract base class that maps
proxy-specific request/response objects (e.g. ``mitmproxy.http.HTTPFlow``)
to a uniform interface used by :class:`src.engine.KonstaEngine`.

The intent is to keep Konsta's compression/distillation logic free of any
proxy-library imports so it can be unit-tested with a mock adapter and, in
principle, retargeted at a different proxy library by writing a new
adapter subclass.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping


class ProxyAdapter(ABC):
    """Abstract adapter between a proxy library and :class:`KonstaEngine`.

    All methods take an opaque ``request`` / ``response`` object whose
    concrete type is defined by the subclass (typically the proxy library's
    flow object). Implementations are responsible for translating between
    that concrete type and the bytes / metadata this interface exposes.
    """

    @abstractmethod
    def is_target_request(self, request: Any) -> bool:
        """Return ``True`` if the request should be intercepted.

        For a host-based filter this typically checks both that the
        request is an HTTP/HTTPS flow the adapter understands and that
        its host matches one of the configured target hosts.
        """

    @abstractmethod
    def get_request_body(self, request: Any) -> bytes:
        """Return the raw request body as bytes (possibly empty)."""

    @abstractmethod
    def set_request_body(self, request: Any, body: bytes) -> None:
        """Replace the request body with ``body``."""

    @abstractmethod
    def get_response_body(self, response: Any) -> bytes:
        """Return the raw response body as bytes (possibly empty)."""

    @abstractmethod
    def set_response_body(self, response: Any, body: bytes) -> None:
        """Replace the response body with ``body``."""

    @abstractmethod
    def get_request_id(self, request: Any) -> str:
        """Stable identifier for the request.

        Implementations should derive the id from properties that
        uniquely identify the request, e.g. a hash of host + path + method.
        """

    # ------------------------------------------------------------------
    # Optional helpers (non-abstract; subclasses may override as needed
    # for diagnostics like URL/header capture in dumps).
    # ------------------------------------------------------------------

    def get_request_host(self, request: Any) -> str:
        """Return the request's host string, or empty string if unavailable."""
        return ""

    def get_request_url(self, request: Any) -> str:
        """Return a human-readable URL for the request."""
        return ""

    def get_request_headers(self, request: Any) -> Mapping[str, str]:
        """Return the request's headers as a plain mapping."""
        return {}

    def get_request_text(self, request: Any) -> str:
        """Return the request body decoded as text (best-effort)."""
        return ""

    def get_response_status(self, response: Any) -> int:
        """Return the response's HTTP status code."""
        return 0

    def get_response_headers(self, response: Any) -> Mapping[str, str]:
        """Return the response's headers as a plain mapping."""
        return {}

    def set_response_header(self, response: Any, name: str, value: str) -> None:
        """Set a single response header."""
        return None

    def get_response_text(self, response: Any) -> str:
        """Return the response body decoded as text (best-effort)."""
        return ""


class MitmproxyAdapter(ProxyAdapter):
    """Adapter that operates on :class:`mitmproxy.http.HTTPFlow`.

    The adapter is constructed with the list of target hosts so it can
    implement :meth:`is_target_request` with a host-suffix match without
    needing access to the proxy's config object.
    """

    def __init__(self, target_hosts: Iterable[str]):
        self._target_hosts = [str(h) for h in target_hosts]

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def is_target_request(self, flow: Any) -> bool:
        # Imported lazily so the module is usable even when mitmproxy is
        # not installed (e.g. for unit tests of other modules).
        from mitmproxy import http

        if not isinstance(flow, http.HTTPFlow):
            return False
        host = getattr(flow.request, "host", "") or ""
        return any(
            host == target or host.endswith("." + target)
            for target in self._target_hosts
        )

    def get_request_body(self, flow: Any) -> bytes:
        return flow.request.content or b""

    def set_request_body(self, flow: Any, body: bytes) -> None:
        flow.request.content = body

    def get_response_body(self, flow: Any) -> bytes:
        return flow.response.content or b""

    def set_response_body(self, flow: Any, body: bytes) -> None:
        flow.response.content = body

    def get_request_id(self, flow: Any) -> str:
        request = getattr(flow, "request", None)
        host = getattr(request, "host", "") or ""
        method = getattr(request, "method", "") or ""
        path = self._strip_query_from_path(getattr(request, "path", "") or "")
        query = self._get_request_query(flow)
        body_hash = hashlib.sha256(
            (getattr(request, "content", b"") or b"")
        ).hexdigest()
        raw = f"{method}://{host}{path}?{query}#{body_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Helper overrides
    # ------------------------------------------------------------------

    def _get_request_query(self, flow: Any) -> str:
        """Return a canonicalised query string for ``flow.request``.

        Mitmproxy exposes the parsed query as a ``MultiDictView``; we render
        it in deterministic ``key=value`` order joined by ``&``. The leading
        ``?`` is intentionally not added so callers can build the URL
        themselves. Empty / missing queries collapse to an empty string so
        the resulting id is stable for requests without a query string.
        """
        request = getattr(flow, "request", None)
        if request is None:
            return ""
        query = getattr(request, "query", None)
        if query is None:
            return ""
        try:
            items = list(query.items(multi=True))
        except Exception:
            # Some objects expose ``query`` as a plain mapping rather than a
            # MultiDictView. Fall back to a plain items() iteration in that
            # case so we still return a deterministic string.
            try:
                items = [(str(k), str(v)) for k, v in dict(query).items()]
            except Exception:
                return ""
        if not items:
            return ""
        return "&".join(f"{k}={v}" for k, v in items)

    @staticmethod
    def _strip_query_from_path(path: str) -> str:
        """Strip an embedded query string from ``path``.

        Mitmproxy's ``request.path`` includes the query string
        (``/v1/chat?foo=bar``). For request-id derivation we want to keep the
        query separate so it can be serialised deterministically by
        :meth:`_get_request_query`.
        """
        if "?" not in path:
            return path
        return path.split("?", 1)[0]

    def get_request_host(self, flow: Any) -> str:
        return getattr(flow.request, "host", "") or ""

    def get_request_url(self, flow: Any) -> str:
        return getattr(flow.request, "pretty_url", "") or ""

    def get_request_headers(self, flow: Any) -> Mapping[str, str]:
        return dict(getattr(flow.request, "headers", {}) or {})

    def get_request_text(self, flow: Any) -> str:
        get_text = getattr(flow.request, "get_text", None)
        if callable(get_text):
            return get_text() or ""
        return ""

    def get_response_status(self, flow: Any) -> int:
        return getattr(flow.response, "status_code", 0) or 0

    def get_response_headers(self, flow: Any) -> Mapping[str, str]:
        return dict(getattr(flow.response, "headers", {}) or {})

    def set_response_header(self, flow: Any, name: str, value: str) -> None:
        flow.response.headers[name] = value

    def get_response_text(self, flow: Any) -> str:
        get_text = getattr(flow.response, "get_text", None)
        if callable(get_text):
            return get_text() or ""
        return ""
