"""Shared pytest fixtures and configuration for the Konsta test suite."""

import asyncio
import inspect
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import MagicMock

import keyring
import pytest

# Ensure the project root (where the `src` package lives) is importable when
# pytest is invoked without an explicit PYTHONPATH=. This matches the layout
# pytest would otherwise only see if PYTHONPATH were set.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Provide required env vars before any src.* import triggers Config() instantiation.
# `LLM_API_KEY` is set as a default so test collection never trips Config's
# required-field validation; tests can still override it via
# `monkeypatch.setenv()`. CA paths are deliberately *not* defaulted here:
# the legacy ``/tmp/konsta_test_ca.{pem,key}`` paths leaked state between
# test runs and made the suite order-dependent. Tests that need a CA file
# pair should depend on the ``ca_paths`` fixture below instead, which writes
# dummy files into ``tmp_path``.
os.environ.setdefault("LLM_API_KEY", "test-llm-api-key")

# Fix CI failure where PYTHON_KEYRING_BACKEND is set to a missing module (keyring.alt).
# We unset it to allow keyring to use its default backend detection.
if os.environ.get("PYTHON_KEYRING_BACKEND") == "keyring.alt.file.PlaintextKeyring":
    os.environ.pop("PYTHON_KEYRING_BACKEND", None)

# Install a null keyring backend globally so keyring.get_password() never
# raises NoKeyringError in test environments (CI often has no backend).
# Individual tests that need a real fake can still override via monkeypatch
# or the fake_keyring fixture below.
import keyring as _keyring
from keyring.backends.null import Keyring as _NullKeyring
_keyring.set_keyring(_NullKeyring())


class HTTPFlow:
    """Stand-in for :class:`mitmproxy.http.HTTPFlow` used in unit tests.

    ``proxy_core.is_target_request()`` does ``isinstance(flow, http.HTTPFlow)``,
    so test flows must be instances of a class that compares equal to the real
    mitmproxy class. Using a small dedicated class (rather than relying on a
    ``MagicMock`` for the stub) keeps ``isinstance`` checks straightforward.
    """


def _install_mitmproxy_stub() -> tuple[MagicMock, object | None, object | None]:
    """Install a ``mitmproxy`` / ``mitmproxy.http`` stub in ``sys.modules``.

    Returns the new stub and the prior ``sys.modules`` entries so the caller
    can later restore them. The stub exposes ``HTTPFlow`` as the
    :class:`HTTPFlow` class defined above so ``isinstance(flow, http.HTTPFlow)``
    checks succeed against the test-only class.

    This must run *before* ``src.proxy_core`` is imported: production code
    does ``from mitmproxy import http`` at module load and binds the result
    into its own namespace. If the stub is not in place at that point, the
    real mitmproxy is cached on the importing module and our stub never
    takes effect for ``isinstance`` checks. We therefore invoke the install
    logic at conftest import time as well as from the session-scoped fixture
    below, so test collection sees the stub before any ``src.proxy_core``
    import happens.
    """
    saved_mitmproxy = sys.modules.get("mitmproxy")
    saved_mitmproxy_http = sys.modules.get("mitmproxy.http")

    mitmproxy_stub = MagicMock()
    sys.modules["mitmproxy"] = mitmproxy_stub
    sys.modules["mitmproxy.http"] = mitmproxy_stub.http
    mitmproxy_stub.http.HTTPFlow = HTTPFlow
    return mitmproxy_stub, saved_mitmproxy, saved_mitmproxy_http


def _restore_mitmproxy_stub(
    saved_mitmproxy: object | None,
    saved_mitmproxy_http: object | None,
) -> None:
    """Undo :func:`_install_mitmproxy_stub` and remove the stub from sys.modules."""
    if saved_mitmproxy is None:
        sys.modules.pop("mitmproxy", None)
    else:
        sys.modules["mitmproxy"] = saved_mitmproxy
    if saved_mitmproxy_http is None:
        sys.modules.pop("mitmproxy.http", None)
    else:
        sys.modules["mitmproxy.http"] = saved_mitmproxy_http


# Install the stub eagerly so any `from mitmproxy import http` executed during
# test collection binds to our stub. The companion ``mitmproxy_mock`` fixture
# below yields the installed stub for the test session and tears it down afterwards so
# the stub does not leak into the host interpreter.
(_MITMPROXY_STUB, _SAVED_MITMPROXY, _SAVED_MITMPROXY_HTTP) = _install_mitmproxy_stub()


# ---------------------------------------------------------------------------
# Keyring fake
# ---------------------------------------------------------------------------


class FakeKeyringError(keyring.errors.KeyringError):
    """Stand-in for ``keyring.errors.PasswordDeleteError`` used by ``delete_password``."""


class FakeKeyring:
    """Minimal in-memory replacement for the OS keyring backend.

    Tests use this by patching ``keyring.get_password``, ``keyring.set_password``,
    ``keyring.delete_password``, and ``keyring.errors.PasswordDeleteError`` so
    that nothing leaks into the real OS keyring during the test run.
    """

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.store:
            raise FakeKeyringError("not found")
        del self.store[(service, username)]


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> FakeKeyring:
    """Install a :class:`FakeKeyring` for the duration of a test.

    Returns the fake instance so tests can inspect ``fake.store`` to assert
    what was written or pre-populate it with existing entries.
    """
    fake = FakeKeyring()
    monkeypatch.setattr(keyring, "get_password", fake.get_password)
    monkeypatch.setattr(keyring, "set_password", fake.set_password)
    monkeypatch.setattr(keyring, "delete_password", fake.delete_password)
    monkeypatch.setattr(keyring.errors, "PasswordDeleteError", FakeKeyringError)
    return fake


# ---------------------------------------------------------------------------
# Distillation worker processor fake
# ---------------------------------------------------------------------------


class FakeProcessor:
    """Stand-in for :class:`src.llm_processor.LLMProcessor`.

    Exposes the same async ``process_context`` coroutine the real processor
    uses and records every call so tests can assert on behaviour. The
    worker flattens the returned message list into a cached string; for
    tests that need the raw distilled text, ``worker.get_result(...)``
    surfaces it after the worker drains the queue.
    """

    def __init__(self, output: str = "distilled", delay: float = 0.0) -> None:
        self.output = output
        self.delay = delay
        self.calls: list[list[dict]] = []
        self.call_count = 0
        self._lock = threading.Lock()

    async def process_context(self, messages: list[dict]) -> list[dict]:
        with self._lock:
            self.calls.append(list(messages))
            self.call_count += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return [{"role": "system", "content": self.output}]


@pytest.fixture
def fake_processor() -> FakeProcessor:
    """Default :class:`FakeProcessor` instance with ``output='distilled'``."""
    return FakeProcessor()


# ---------------------------------------------------------------------------
# ProxyAdapter fake
# ---------------------------------------------------------------------------


class FakeAdapter:
    """Minimal :class:`src.proxy_adapter.ProxyAdapter` for unit tests.

    Implements the abstract interface with an in-memory backing store and
    records every ``set_*`` call so tests can assert what the engine wrote.
    The adapter is configured via keyword arguments so each test can
    customise the host, body, and headers it presents to the engine.
    """

    def __init__(
        self,
        *,
        target_hosts: list[str] | None = None,
        request_body: bytes = b"",
        response_body: bytes = b"",
        request_host: str = "api.openai.com",
        request_url: str = "https://api.openai.com/v1/chat",
        request_headers: Mapping[str, str] | None = None,
        response_status: int = 200,
        response_headers: Mapping[str, str] | None = None,
        target_match: bool = True,
    ) -> None:
        self._target_hosts = list(target_hosts or [])
        self._request_body = request_body
        self._response_body = response_body
        self._request_host = request_host
        self._request_url = request_url
        self._request_headers: dict[str, str] = dict(request_headers or {})
        self._response_status = response_status
        self._response_headers: dict[str, str] = dict(response_headers or {})
        self._target_match = target_match

        # Recorded calls for assertions.
        self.set_request_body_calls: list[bytes] = []
        self.set_response_body_calls: list[bytes] = []
        self.set_response_header_calls: list[tuple[str, str]] = []
        self.get_request_id_calls = 0

    # ---- Abstract methods -------------------------------------------------

    def is_target_request(self, request: Any) -> bool:
        return self._target_match

    def get_request_body(self, request: Any) -> bytes:
        return self._request_body

    def set_request_body(self, request: Any, body: bytes) -> None:
        self.set_request_body_calls.append(body)
        self._request_body = body

    def get_response_body(self, request: Any) -> bytes:
        return self._response_body

    def set_response_body(self, request: Any, body: bytes) -> None:
        self.set_response_body_calls.append(body)
        self._response_body = body

    def get_request_id(self, request: Any) -> str:
        self.get_request_id_calls += 1
        return f"id-{self.get_request_id_calls}"

    # ---- Helper overrides -------------------------------------------------

    def get_request_host(self, request: Any) -> str:
        return self._request_host

    def get_request_url(self, request: Any) -> str:
        return self._request_url

    def get_request_headers(self, request: Any) -> Mapping[str, str]:
        return dict(self._request_headers)

    def get_request_text(self, request: Any) -> str:
        return self._request_body.decode("utf-8") if self._request_body else ""

    def get_response_status(self, response: Any) -> int:
        return self._response_status

    def get_response_headers(self, response: Any) -> Mapping[str, str]:
        return dict(self._response_headers)

    def set_response_header(self, response: Any, name: str, value: str) -> None:
        self.set_response_header_calls.append((name, value))
        self._response_headers[name] = value

    def get_response_text(self, response: Any) -> str:
        return self._response_body.decode("utf-8") if self._response_body else ""


# Register ``FakeAdapter`` as a virtual subclass of :class:`ProxyAdapter`
# so engine code that takes a ``ProxyAdapter`` accepts it. We do this
# after the class body so the import is deferred until ``ProxyAdapter`` is
# needed (the conftest may load before ``src.proxy_adapter`` is on the
# import path in some test layouts).
def _register_fake_adapter() -> None:
    from src.proxy_adapter import ProxyAdapter

    ProxyAdapter.register(FakeAdapter)


_register_fake_adapter()


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    """Default :class:`FakeAdapter` instance for engine tests."""
    return FakeAdapter()


# ---------------------------------------------------------------------------
# Async / sync wait helpers
# ---------------------------------------------------------------------------


async def _wait_for_result(
    async_fn: Any,
    timeout: float = 1.0,
    interval: float = 0.01,
) -> Any:
    """Poll ``async_fn()`` until it returns a truthy value or ``timeout`` elapses.

    ``async_fn`` may be either a sync or async callable; its result is
    awaited (if needed) and then checked for truthiness. Returns the
    first truthy result, or the final result if the timeout elapses.
    Designed for awaiting background work in async tests without
    resorting to busy-wait sleeps inline.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    is_async = inspect.iscoroutinefunction(async_fn)
    while loop.time() < deadline:
        result = await async_fn() if is_async else async_fn()
        if result:
            return result
        await asyncio.sleep(interval)
    return await async_fn() if is_async else async_fn()


def _wait_for_cached(
    worker: Any,
    request_id: str,
    timeout: float = 2.0,
    interval: float = 0.01,
) -> Any:
    """Poll a :class:`DistillationWorker` for a cached result.

    Sync helper for use in non-async tests. Returns the cached value if it
    becomes available before ``timeout``, otherwise returns the final
    ``worker.get_result(request_id)`` (which may be ``None``).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = worker.get_result(request_id)
        if result is not None:
            return result
        time.sleep(interval)
    return worker.get_result(request_id)


# ---------------------------------------------------------------------------
# Existing fixtures: CA paths and mitmproxy stub
# ---------------------------------------------------------------------------


@pytest.fixture
def ca_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Create a pair of dummy CA files inside ``tmp_path``.

    Returns ``(cert_path, key_path)`` as :class:`pathlib.Path` objects and
    sets ``CA_CERT_PATH`` / ``CA_KEY_PATH`` in the environment to point at
    those files so :class:`src.config.Config` picks them up automatically
    when constructed inside the test. The fixture is the canonical
    replacement for the previous ``os.environ.setdefault`` defaults that
    leaked ``/tmp/konsta_test_ca.*`` into every test run.
    """
    cert_path = tmp_path / "konsta-ca-cert.pem"
    key_path = tmp_path / "konsta-ca-key.pem"
    cert_path.write_text("dummy-cert")
    key_path.write_text("dummy-key")
    monkeypatch.setenv("CA_CERT_PATH", str(cert_path))
    monkeypatch.setenv("CA_KEY_PATH", str(key_path))
    return cert_path, key_path


@pytest.fixture(autouse=True, scope="session")
def mitmproxy_mock():
    """Session-scoped autouse fixture that exposes the mitmproxy stub.

    The actual ``sys.modules`` swap is performed at conftest import time
    (see :func:`_install_mitmproxy_stub`) so test-collection-time imports of
    ``src.proxy_core`` bind to the stub rather than to the real mitmproxy.
    This fixture's job is to yield the installed stub to tests that want it
    and to restore ``sys.modules`` once the pytest session finishes.
    """
    yield _MITMPROXY_STUB
    _restore_mitmproxy_stub(_SAVED_MITMPROXY, _SAVED_MITMPROXY_HTTP)
