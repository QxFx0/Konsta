"""Tests for ``src.proxy_adapter``."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from src.proxy_adapter import MitmproxyAdapter, ProxyAdapter

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal request stand-in used to exercise adapter helpers."""

    def __init__(
        self,
        host: str = "api.openai.com",
        content: bytes = b"",
        path: str = "/v1/chat",
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
    ):
        self.host = host
        self.content = content
        self.path = path
        self.method = method
        self.pretty_url = f"https://{host}{path}"
        self.headers = dict(headers or {"content-type": "application/json"})

    def get_text(self) -> str:
        return self.content.decode("utf-8") if self.content else ""


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        content: bytes = b"",
        headers: Mapping[str, str] | None = None,
    ):
        self.status_code = status_code
        self.content = content
        self.headers = dict(headers or {})

    def get_text(self) -> str:
        return self.content.decode("utf-8") if self.content else ""


def _make_http_flow(host: str = "api.openai.com", content: bytes = b"") -> Any:
    """Build a flow whose isinstance check passes for mitmproxy.http.HTTPFlow."""
    from tests.conftest import HTTPFlow

    flow = HTTPFlow()
    flow.request = _FakeRequest(host=host, content=content)
    return flow


def _make_http_flow_with_response(
    host: str = "api.openai.com",
    request_content: bytes = b"",
    response_content: bytes = b"ok",
    status_code: int = 200,
) -> Any:
    from tests.conftest import HTTPFlow

    flow = HTTPFlow()
    flow.request = _FakeRequest(host=host, content=request_content)
    flow.response = _FakeResponse(content=response_content, status_code=status_code)
    return flow


# ---------------------------------------------------------------------------
# Abstract base class behaviour
# ---------------------------------------------------------------------------


def test_proxy_adapter_is_abstract():
    """ProxyAdapter cannot be instantiated directly because of abstract methods."""
    with pytest.raises(TypeError):
        ProxyAdapter()  # type: ignore[abstract]


def test_subclass_must_implement_all_abstract_methods():
    """A subclass missing any abstract method is still abstract."""

    class IncompleteAdapter(ProxyAdapter):
        def is_target_request(self, request):
            return True

        # Other abstract methods intentionally omitted.

    with pytest.raises(TypeError):
        IncompleteAdapter()  # type: ignore[abstract]


def test_subclass_with_all_methods_is_instantiable():
    class FullAdapter(ProxyAdapter):
        def is_target_request(self, request):
            return True

        def get_request_body(self, request):
            return b""

        def set_request_body(self, request, body):
            return None

        def get_response_body(self, response):
            return b""

        def set_response_body(self, response, body):
            return None

        def get_request_id(self, request):
            return "id"

    adapter = FullAdapter()
    assert adapter.is_target_request(object()) is True
    assert adapter.get_request_body(None) == b""
    assert adapter.get_request_id(None) == "id"


# ---------------------------------------------------------------------------
# MitmproxyAdapter
# ---------------------------------------------------------------------------


def test_is_target_request_exact_match():
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow = _make_http_flow("api.openai.com")
    assert adapter.is_target_request(flow) is True


def test_is_target_request_suffix_match():
    adapter = MitmproxyAdapter(target_hosts=["openai.com"])
    flow = _make_http_flow("api.openai.com")
    assert adapter.is_target_request(flow) is True


def test_is_target_request_exact_match_allows_target():
    """The bare target host itself must match exactly."""
    adapter = MitmproxyAdapter(target_hosts=["openai.com"])
    flow = _make_http_flow("openai.com")
    assert adapter.is_target_request(flow) is True


def test_is_target_request_allows_direct_subdomain():
    """A direct subdomain of the target must match."""
    adapter = MitmproxyAdapter(target_hosts=["openai.com"])
    flow = _make_http_flow("api.openai.com")
    assert adapter.is_target_request(flow) is True


def test_is_target_request_allows_nested_subdomain():
    """Nested subdomains of the target must match."""
    adapter = MitmproxyAdapter(target_hosts=["openai.com"])
    flow = _make_http_flow("v1.api.openai.com")
    assert adapter.is_target_request(flow) is True


def test_is_target_request_rejects_superdomain():
    """A superdomain (parent) of the target must not match."""
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow = _make_http_flow("openai.com")
    assert adapter.is_target_request(flow) is False


def test_is_target_request_rejects_appended_label():
    """A host that appends a label after the target must not match."""
    adapter = MitmproxyAdapter(target_hosts=["openai.com"])
    flow = _make_http_flow("openai.com.evil.com")
    assert adapter.is_target_request(flow) is False


def test_is_target_request_rejects_prefix_suffix():
    """A host that merely ends with the target string as part of a longer
    label must not match (e.g. ``myopenai.com``)."""
    adapter = MitmproxyAdapter(target_hosts=["openai.com"])
    flow = _make_http_flow("myopenai.com")
    assert adapter.is_target_request(flow) is False


def test_is_target_request_rejects_false_positive():
    """A host that *contains* the target as a label must not match."""
    adapter = MitmproxyAdapter(target_hosts=["openai.com"])
    flow = _make_http_flow("fake-api.openai.com.evil.com")
    assert adapter.is_target_request(flow) is False


def test_is_target_request_no_match():
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow = _make_http_flow("google.com")
    assert adapter.is_target_request(flow) is False


def test_is_target_request_rejects_non_flow():
    """Non-flow objects must not be considered target requests."""
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    assert adapter.is_target_request("api.openai.com") is False
    assert adapter.is_target_request(object()) is False
    assert adapter.is_target_request(None) is False


def test_get_and_set_request_body_roundtrip():
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow = _make_http_flow("api.openai.com", content=b"orig")
    assert adapter.get_request_body(flow) == b"orig"
    adapter.set_request_body(flow, b"new body")
    assert flow.request.content == b"new body"
    assert adapter.get_request_body(flow) == b"new body"


def test_get_request_body_handles_none_content():
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow = _make_http_flow("api.openai.com", content=None)  # type: ignore[arg-type]
    assert adapter.get_request_body(flow) == b""


def test_get_and_set_response_body_roundtrip():
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow = _make_http_flow_with_response(
        "api.openai.com",
        request_content=b"",
        response_content=b"orig-response",
    )
    assert adapter.get_response_body(flow) == b"orig-response"
    adapter.set_response_body(flow, b"new-response")
    assert flow.response.content == b"new-response"
    assert adapter.get_response_body(flow) == b"new-response"


def test_get_response_body_handles_none_content():
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow = _make_http_flow_with_response("api.openai.com", response_content=None)  # type: ignore[arg-type]
    assert adapter.get_response_body(flow) == b""


def test_get_request_id_is_deterministic_and_distinguishes_requests():
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow1 = _make_http_flow("api.openai.com")
    flow1.request.path = "/v1/chat"
    flow1.request.method = "POST"
    flow2 = _make_http_flow("api.openai.com")
    flow2.request.path = "/v1/other"
    flow2.request.method = "POST"

    id1 = adapter.get_request_id(flow1)
    id2 = adapter.get_request_id(flow2)
    assert id1 != id2
    # Same inputs -> same hash (deterministic).
    flow3 = _make_http_flow("api.openai.com")
    flow3.request.path = "/v1/chat"
    flow3.request.method = "POST"
    assert adapter.get_request_id(flow3) == id1


def test_get_request_id_is_sha256_hex():
    import hashlib

    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow = _make_http_flow("api.openai.com")
    flow.request.path = "/v1/chat"
    flow.request.method = "POST"
    # New format includes query string and sha256 of body (empty here).
    body_hash = hashlib.sha256(b"").hexdigest()
    expected = hashlib.sha256(
        f"POST://api.openai.com/v1/chat?#{body_hash}".encode("utf-8")
    ).hexdigest()
    assert adapter.get_request_id(flow) == expected


def test_get_request_id_handles_missing_attributes():
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])

    class Bare:
        pass

    bare = Bare()
    # Missing all of host/path/method -> empty hash, but must not raise.
    rid = adapter.get_request_id(bare)
    assert isinstance(rid, str)
    assert len(rid) == 64  # SHA-256 hex length.


# ---------------------------------------------------------------------------
# Helper overrides
# ---------------------------------------------------------------------------


def test_helper_overrides_expose_request_metadata():
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow = _make_http_flow("api.openai.com", content=b'{"k":1}')
    flow.request.headers = {"x-test": "1"}
    assert adapter.get_request_host(flow) == "api.openai.com"
    assert adapter.get_request_url(flow) == "https://api.openai.com/v1/chat"
    assert adapter.get_request_headers(flow) == {"x-test": "1"}
    assert adapter.get_request_text(flow) == '{"k":1}'


def test_helper_overrides_expose_response_metadata():
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow = _make_http_flow_with_response(
        "api.openai.com",
        response_content=b"hi",
        status_code=201,
    )
    flow.response.headers = {"content-type": "application/json"}
    assert adapter.get_response_status(flow) == 201
    assert adapter.get_response_headers(flow) == {"content-type": "application/json"}
    assert adapter.get_response_text(flow) == "hi"


def test_set_response_header_writes_through():
    adapter = MitmproxyAdapter(target_hosts=["api.openai.com"])
    flow = _make_http_flow_with_response("api.openai.com")
    adapter.set_response_header(flow, "X-Konsta-Test", "yes")
    assert flow.response.headers["X-Konsta-Test"] == "yes"


def test_default_adapter_helpers_return_neutral_values():
    """The base ProxyAdapter defaults are safe to call without override."""

    class MinimalAdapter(ProxyAdapter):
        def is_target_request(self, request):
            return False

        def get_request_body(self, request):
            return b""

        def set_request_body(self, request, body):
            return None

        def get_response_body(self, response):
            return b""

        def set_response_body(self, response, body):
            return None

        def get_request_id(self, request):
            return "id"

    a = MinimalAdapter()
    assert a.get_request_host(None) == ""
    assert a.get_request_url(None) == ""
    assert a.get_request_headers(None) == {}
    assert a.get_request_text(None) == ""
    assert a.get_response_status(None) == 0
    assert a.get_response_headers(None) == {}
    assert a.get_response_text(None) == ""
    # set_response_header default is a no-op (returns None).
    assert a.set_response_header(None, "X", "Y") is None
