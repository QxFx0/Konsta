"""Tests for ``src.engine.KonstaEngine``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.compression_engine import CompressionEngine
from src.config import Config
from src.distillation_worker import DistillationWorker
from src.engine import KonstaEngine
from tests.conftest import FakeAdapter, FakeProcessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Config instance with deterministic settings."""
    return Config(
        target_hosts={"api.openai.com"},
        proxy_host="127.0.0.1",
        llm_api_key="test-key",
        distillation_mode="background",
    )


@pytest.fixture
def metrics():
    from src.metrics import Metrics

    return Metrics()


@pytest.fixture
def compression_engine(config, metrics):
    # Avoid loading sentence-transformers in tests by passing a stubbed model.
    engine = CompressionEngine(config=config, metrics=metrics)
    engine.model = None  # Disable semantic dedup to keep tests deterministic.
    return engine


def _build_request_payload(messages=None, model="x"):
    return json.dumps({
        "model": model,
        "messages": messages or [
            {"role": "user", "content": "hello"},
        ],
    }).encode("utf-8")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructs_with_defaults(config, metrics):
    engine = KonstaEngine(config=config, metrics=metrics)
    assert engine.config is config
    assert engine.metrics is metrics
    assert engine.compression_engine is not None
    assert engine.distillation_worker is None
    assert engine._last_distillation_request_id is None
    assert engine._distillation_request_counter == 0


def test_constructs_with_injected_components(config, compression_engine, metrics):
    worker = DistillationWorker(FakeProcessor(), cache_size=4, metrics=metrics)
    engine = KonstaEngine(
        config=config,
        compression_engine=compression_engine,
        distillation_worker=worker,
        metrics=metrics,
    )
    assert engine.compression_engine is compression_engine
    # When the model is intentionally disabled (test fixture), the engine
    # must still report itself as healthy.
    assert engine.compression_engine.is_healthy() is True


def test_compression_engine_is_unhealthy_after_model_load_failure(config, metrics):
    """A bad model name should mark the compression engine as unhealthy."""
    from src.compression_engine import HAS_SENTENCE_TRANSFORMERS, CompressionEngine

    if not HAS_SENTENCE_TRANSFORMERS:
        pytest.skip("sentence-transformers not installed")

    bad_config = Config(
        target_hosts=config.target_hosts,
        proxy_host=config.proxy_host,
        llm_api_key="test-key",
        embedding_model="definitely-not-a-real-model-12345",
    )
    engine = CompressionEngine(config=bad_config, metrics=metrics)
    assert engine.is_healthy() is False


# ---------------------------------------------------------------------------
# process_request
# ---------------------------------------------------------------------------


def test_process_request_skips_non_target(config, metrics):
    """The engine itself does not filter by target host -- that's the
    proxy's job (``proxy.request()`` calls ``adapter.is_target_request``
    before invoking the engine). Verify the engine processes a valid
    request end-to-end and that the FakeAdapter surfaces what was written.
    """
    engine = KonstaEngine(config=config, metrics=metrics)
    payload = _build_request_payload()
    adapter = FakeAdapter(request_body=payload, target_match=False)
    flow = MagicMock()

    engine.process_request(adapter, flow)

    # The engine processed the request regardless of target_match -- the
    # adapter's target flag is only consulted by the proxy.
    assert len(adapter.set_request_body_calls) == 1


def test_process_request_writes_compressed_body(config, compression_engine, metrics):
    engine = KonstaEngine(
        config=config,
        compression_engine=compression_engine,
        metrics=metrics,
    )
    payload = _build_request_payload()
    adapter = FakeAdapter(request_body=payload)
    flow = MagicMock()

    engine.process_request(adapter, flow)

    # Exactly one body write happened, containing JSON.
    assert len(adapter.set_request_body_calls) == 1
    written = adapter.set_request_body_calls[0]
    decoded = json.loads(written.decode("utf-8"))
    assert "messages" in decoded


def test_process_request_snapshots_original_size_on_flow(config, metrics):
    engine = KonstaEngine(config=config, metrics=metrics)
    payload = _build_request_payload()
    adapter = FakeAdapter(request_body=payload)
    flow = MagicMock()

    engine.process_request(adapter, flow)

    # konsta_orig_size recorded the pre-compression byte count.
    assert flow.konsta_orig_size == len(payload)


def test_process_request_skips_oversized_payload(config, metrics):
    engine = KonstaEngine(config=config, metrics=metrics)
    # max_payload_bytes + 1 of content.
    payload = b"x" * (config.max_payload_bytes + 1)
    adapter = FakeAdapter(request_body=payload)
    flow = MagicMock()

    engine.process_request(adapter, flow)

    assert adapter.set_request_body_calls == []


def test_process_request_skips_empty_body(config, metrics):
    engine = KonstaEngine(config=config, metrics=metrics)
    adapter = FakeAdapter(request_body=b"")
    flow = MagicMock()

    engine.process_request(adapter, flow)

    assert adapter.set_request_body_calls == []


def test_process_request_handles_invalid_json(config, metrics):
    engine = KonstaEngine(config=config, metrics=metrics)
    adapter = FakeAdapter(request_body=b"not json {{{")
    flow = MagicMock()

    # Must not raise.
    engine.process_request(adapter, flow)

    # No body rewrite.
    assert adapter.set_request_body_calls == []


def test_process_request_submits_to_worker(config, compression_engine, metrics):
    import asyncio

    async def _run():
        processor = FakeProcessor()
        worker = DistillationWorker(processor, cache_size=4, metrics=metrics)
        await worker.start()
        try:
            engine = KonstaEngine(
                config=config,
                compression_engine=compression_engine,
                distillation_worker=worker,
                metrics=metrics,
            )
            payload = _build_request_payload()
            adapter = FakeAdapter(request_body=payload)
            flow = MagicMock()

            engine.process_request(adapter, flow)

            # Hand-off happened synchronously inside process_request(): the
            # pending buffer already contains the task.
            pending = list(worker._pending)
            assert any(task[0] == "req-1" for task in pending)
        finally:
            await worker.stop()

    asyncio.run(_run())


def test_process_request_does_not_submit_when_worker_missing(
    config, compression_engine, metrics
):
    engine = KonstaEngine(
        config=config,
        compression_engine=compression_engine,
        distillation_worker=None,
        metrics=metrics,
    )
    payload = _build_request_payload()
    adapter = FakeAdapter(request_body=payload)
    flow = MagicMock()

    engine.process_request(adapter, flow)

    # No crash, no submission, but the body was still rewritten.
    assert len(adapter.set_request_body_calls) == 1
    assert engine._distillation_request_counter == 0


def test_process_request_does_not_submit_when_distillation_disabled(
    config, compression_engine, metrics
):
    config.distillation_mode = "disabled"
    processor = FakeProcessor()
    worker = DistillationWorker(processor, cache_size=4, metrics=metrics)
    engine = KonstaEngine(
        config=config,
        compression_engine=compression_engine,
        distillation_worker=worker,
        metrics=metrics,
    )
    payload = _build_request_payload()
    adapter = FakeAdapter(request_body=payload)
    flow = MagicMock()

    engine.process_request(adapter, flow)

    # Worker exists but distillation disabled -> nothing submitted.
    assert list(worker._pending) == []
    assert engine._distillation_request_counter == 0


# ---------------------------------------------------------------------------
# process_response
# ---------------------------------------------------------------------------


def test_process_response_injects_metrics_headers(config, metrics):
    engine = KonstaEngine(config=config, metrics=metrics)
    adapter = FakeAdapter(request_body=b"hello", response_body=b"world")
    flow = MagicMock(spec=["konsta_orig_size", "request", "response"])
    flow.konsta_orig_size = 100

    engine.process_response(adapter, flow)

    # Both headers written.
    names = [name for name, _ in adapter.set_response_header_calls]
    assert "X-Konsta-Original-Size" in names
    assert "X-Konsta-Compressed-Size" in names
    values = dict(adapter.set_response_header_calls)
    assert values["X-Konsta-Original-Size"] == "100"
    # Compressed size reflects the current (post-compression) request body length.
    assert values["X-Konsta-Compressed-Size"] == str(len(b"hello"))


def test_process_response_falls_back_when_snapshot_missing(config, metrics):
    engine = KonstaEngine(config=config, metrics=metrics)
    adapter = FakeAdapter(request_body=b"abc", response_body=b"ok")
    flow = MagicMock(spec=["request", "response"])
    # No konsta_orig_size attribute set -> defaults to current body size.

    engine.process_response(adapter, flow)

    values = dict(adapter.set_response_header_calls)
    assert values["X-Konsta-Original-Size"] == "3"
    assert values["X-Konsta-Compressed-Size"] == "3"


def test_process_response_does_not_crash_on_adapter_error(config, metrics):
    engine = KonstaEngine(config=config, metrics=metrics)

    class BrokenAdapter(FakeAdapter):
        def get_request_host(self, request):
            raise RuntimeError("boom")

    engine.process_response(BrokenAdapter(), MagicMock())


# ---------------------------------------------------------------------------
# compress_context
# ---------------------------------------------------------------------------


def test_compress_context_uses_compression_engine(config, metrics):
    engine = KonstaEngine(config=config, metrics=metrics)
    out = engine.compress_context({
        "messages": [{"role": "user", "content": "hello"}],
        "provider": "openai",
        "original_payload": {"model": "x"},
    })
    assert out is not None
    assert out["model"] == "x"
    assert "messages" in out


def test_compress_context_handles_bad_input(config, metrics):
    engine = KonstaEngine(config=config, metrics=metrics)
    # Malformed request_data (not a dict) -> engine returns None rather than raising.
    assert engine.compress_context(None) is None  # type: ignore[arg-type]
    # Empty dict is a valid input: engine returns a payload with the
    # default (empty) messages.
    out = engine.compress_context({})
    assert out is not None
    assert out.get("messages") == []


# ---------------------------------------------------------------------------
# _dump_data sanitisation
# ---------------------------------------------------------------------------


def _read_single_dump(dump_dir):
    """Return the parsed JSON payload of the (only) ``*_*.json`` file in ``dump_dir``."""
    files = list(Path(dump_dir).glob("*_*.json"))
    assert len(files) == 1, f"expected exactly one dump file, got {files!r}"
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_dump_data_redacts_authorization_header(tmp_path, config, metrics):
    """Authorization headers are redacted in the written dump, but other
    headers in the same dict are preserved verbatim."""
    config.dump_dir = str(tmp_path)
    engine = KonstaEngine(config=config, metrics=metrics)

    payload = {
        "url": "https://api.openai.com/v1/chat",
        "headers": {
            "Authorization": "Bearer super-secret-token",
            "authorization": "Basic dXNlcjpwYXNz",  # lowercase variant
            "Content-Type": "application/json",
            "User-Agent": "konsta/1.0",
        },
        "content": "hello",
    }

    engine._dump_data("auth_test.json", payload)

    written = _read_single_dump(tmp_path)
    assert written["headers"]["Authorization"] == "[REDACTED]"
    assert written["headers"]["authorization"] == "[REDACTED]"
    # Non-sensitive headers in the same dict must survive untouched.
    assert written["headers"]["Content-Type"] == "application/json"
    assert written["headers"]["User-Agent"] == "konsta/1.0"
    # Top-level non-sensitive fields must not be touched either.
    assert written["url"] == "https://api.openai.com/v1/chat"
    assert written["content"] == "hello"


def test_dump_data_redacts_api_key(tmp_path, config, metrics):
    """Sensitive keys (api_key, apikey, token, password, secret, private_key,
    ca_key_password) are redacted at every nesting level, and case-insensitive
    variants of each are also redacted."""
    config.dump_dir = str(tmp_path)
    engine = KonstaEngine(config=config, metrics=metrics)

    payload = {
        "headers": {
            "api_key": "sk-1234567890abcdef",
            "apikey": "sk-abcdef1234567890",
            "ApiKey": "sk-mixedCase12345678",
            "token": "tk_1234567890",
            "Token": "tk_CamelCase12345",
            "password": "hunter2-long-enough",
        },
        "tls": {
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIE...",
            "PRIVATE_KEY": "second secret",
            "ca_key_password": "unlock-the-ca",
            "CA_KEY_PASSWORD": "again-unlock",
            "secret": "shh",
        },
        "nested_list": [
            {"apiKey": "list-item-secret"},
            {"safe": "kept"},
        ],
    }

    engine._dump_data("redact_test.json", payload)

    written = _read_single_dump(tmp_path)
    assert written["headers"]["api_key"] == "[REDACTED]"
    assert written["headers"]["apikey"] == "[REDACTED]"
    assert written["headers"]["ApiKey"] == "[REDACTED]"
    assert written["headers"]["token"] == "[REDACTED]"
    assert written["headers"]["Token"] == "[REDACTED]"
    assert written["headers"]["password"] == "[REDACTED]"
    assert written["tls"]["private_key"] == "[REDACTED]"
    assert written["tls"]["PRIVATE_KEY"] == "[REDACTED]"
    assert written["tls"]["ca_key_password"] == "[REDACTED]"
    assert written["tls"]["CA_KEY_PASSWORD"] == "[REDACTED]"
    assert written["tls"]["secret"] == "[REDACTED]"
    # Nested list items are sanitised recursively.
    assert written["nested_list"][0]["apiKey"] == "[REDACTED]"
    assert written["nested_list"][1]["safe"] == "kept"


def test_dump_data_redacts_common_secret_keys(tmp_path, config, metrics):
    """The expanded secret vocabulary covers cookies, tokens, and API keys."""
    config.dump_dir = str(tmp_path)
    engine = KonstaEngine(config=config, metrics=metrics)

    payload = {
        "headers": {
            "cookie": "session=abc123",
            "set-cookie": "session=def456",
            "x-api-key": "sk-xxx",
            "api-key": "sk-yyy",
            "access_token": "at-xxx",
            "id_token": "id-yyy",
            "refresh_token": "rt-zzz",
            "bearer": "tk-bearer",
            "session": "session-secret",
            "privatekey": "-----BEGIN PRIVATE KEY-----",
        }
    }

    engine._dump_data("secret_keys_test.json", payload)

    written = _read_single_dump(tmp_path)
    for key in payload["headers"]:
        assert written["headers"][key] == "[REDACTED]", f"{key} was not redacted"


def test_dump_data_presents_nonsensitive_data(tmp_path, config, metrics):
    """Non-sensitive keys, and ``key`` with short non-secret values, survive
    sanitisation untouched. ``key`` is only redacted when its value looks
    like a real secret (string of length >= 8 in this implementation)."""
    config.dump_dir = str(tmp_path)
    engine = KonstaEngine(config=config, metrics=metrics)

    payload = {
        "url": "https://api.openai.com/v1/chat",
        "status": 200,
        "headers": {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-Id": "abc-123",
        },
        "config": {
            "max_tokens": 256,
            "temperature": 0.7,
            "key": "color",            # short, NOT a secret-looking string
            "model": "gpt-4",
        },
        "items": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }

    engine._dump_data("safe_test.json", payload)

    written = _read_single_dump(tmp_path)
    assert written["url"] == "https://api.openai.com/v1/chat"
    assert written["status"] == 200
    assert written["headers"]["Content-Type"] == "application/json"
    assert written["headers"]["Accept"] == "application/json"
    assert written["headers"]["X-Request-Id"] == "abc-123"
    assert written["config"]["max_tokens"] == 256
    assert written["config"]["temperature"] == 0.7
    assert written["config"]["key"] == "color"
    assert written["config"]["model"] == "gpt-4"
    assert written["items"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


# ---------------------------------------------------------------------------
# Cached distillation application
# ---------------------------------------------------------------------------


def test_cached_distillation_applied_as_valid_json_body(config, metrics):
    """A cached distillation result must be injected into valid provider JSON."""
    config.distillation_mode = "background"
    compression_engine = CompressionEngine(config=config, metrics=metrics)
    compression_engine.model = None  # disable semantic model loading
    processor = FakeProcessor(output="Distilled: hello world")
    worker = DistillationWorker(processor, cache_size=4, metrics=metrics)

    async def _run():
        await worker.start()
        engine = KonstaEngine(
            config=config,
            compression_engine=compression_engine,
            distillation_worker=worker,
            metrics=metrics,
        )

        body = _build_request_payload(messages=[{"role": "user", "content": "hello"}])
        adapter = FakeAdapter(request_body=body)
        request = object()
        engine.process_request(adapter, request)

        # Wait for the background worker to finish distillation.
        for _ in range(100):
            result = worker.get_result("req-1")
            if result is not None:
                break
            await asyncio.sleep(0.05)
        assert worker.get_result("req-1") is not None

        # Second request should consume the cached distillation.
        adapter2 = FakeAdapter(request_body=body)
        request2 = object()
        engine.process_request(adapter2, request2)

        final_body = adapter2.get_request_body(request2)
        await worker.stop()
        return final_body

    final_body = asyncio.run(_run())
    parsed = json.loads(final_body.decode("utf-8"))
    assert parsed["messages"] == [{"role": "user", "content": "Distilled: hello world"}]


# ---------------------------------------------------------------------------
# dump_include_bodies
# ---------------------------------------------------------------------------


def test_dump_request_includes_body_when_config_enabled(tmp_path, config, metrics):
    """When dump_include_bodies is True, the request dump contains the body."""
    config.dump_dir = str(tmp_path)
    config.dump_include_bodies = True
    engine = KonstaEngine(config=config, metrics=metrics)

    body = _build_request_payload(messages=[{"role": "user", "content": "hello"}])
    adapter = FakeAdapter(request_body=body)
    request = object()

    engine.process_request(adapter, request)

    dump_file = next(Path(tmp_path).glob("*_original_req.json"))
    written = json.loads(dump_file.read_text(encoding="utf-8"))
    assert written["content"] == body.decode("utf-8")


def test_dump_request_omits_body_when_config_disabled(tmp_path, config, metrics):
    """When dump_include_bodies is False, the request dump redacts the body."""
    config.dump_dir = str(tmp_path)
    config.dump_include_bodies = False
    engine = KonstaEngine(config=config, metrics=metrics)

    body = _build_request_payload(messages=[{"role": "user", "content": "hello"}])
    adapter = FakeAdapter(request_body=body)
    request = object()

    engine.process_request(adapter, request)

    dump_file = next(Path(tmp_path).glob("*_original_req.json"))
    written = json.loads(dump_file.read_text(encoding="utf-8"))
    assert written["content"] == "[OMITTED]"


def test_dump_response_omits_body_when_config_disabled(tmp_path, config, compression_engine, metrics):
    """When dump_include_bodies is False, the response dump redacts the body."""
    config.dump_dir = str(tmp_path)
    config.dump_include_bodies = False
    engine = KonstaEngine(
        config=config,
        compression_engine=compression_engine,
        metrics=metrics,
    )

    body = _build_request_payload(messages=[{"role": "user", "content": "hello"}])
    adapter = FakeAdapter(request_body=body, response_body=b'{"ok": true}')
    request = object()
    response = object()

    engine.process_request(adapter, request)
    engine.process_response(adapter, response)

    dump_file = next(Path(tmp_path).glob("*_response.json"))
    written = json.loads(dump_file.read_text(encoding="utf-8"))
    assert written["content"] == "[OMITTED]"
