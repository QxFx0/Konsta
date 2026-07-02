"""Tests for ``src.llm_processor.LLMProcessor``.

Covers the new circuit-breaker integration:

* ``CircuitBreakerOpen`` raised inside the queue worker propagates through
  :func:`process_context` and is translated into graceful degradation
  (original messages are returned).
* When no breaker is configured, behaviour is unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm_processor import LLMProcessor
from src.resilience import CircuitBreaker, CircuitBreakerOpen, CircuitState


class _StubConfig:
    """Minimal config object exposing the attributes LLMProcessor reads."""

    llm_api_key = "test-key"
    llm_model = "test-model"
    llm_endpoint = "https://example.test/chat"


@pytest.fixture
def config():
    return _StubConfig()


@pytest.mark.asyncio
async def test_process_context_without_breaker_returns_distilled(config):
    """No breaker -> normal distilled output is returned."""
    processor = LLMProcessor(config)

    async def fake_enqueue(task_id, messages):
        return "distilled body"

    processor.queue_manager.enqueue = fake_enqueue  # type: ignore[assignment]

    messages = [{"role": "user", "content": "hi"}]
    out = await processor.process_context(messages)

    assert out == [{"role": "system", "content": "Distilled Context: distilled body"}]


@pytest.mark.asyncio
async def test_process_context_returns_original_on_circuit_breaker_open(config, caplog):
    """When the queue raises CircuitBreakerOpen, ``process_context`` falls
    back to the original messages and logs a warning."""
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    processor = LLMProcessor(config, circuit_breaker=breaker)

    async def fake_enqueue(task_id, messages):
        raise CircuitBreakerOpen("LLMProcessor is open; refusing call")

    processor.queue_manager.enqueue = fake_enqueue  # type: ignore[assignment]

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level("WARNING"):
        out = await processor.process_context(messages)

    # Graceful degradation: original messages returned unchanged.
    assert out is messages


@pytest.mark.asyncio
async def test_process_context_falls_back_when_llm_returns_none(config):
    """``None`` from the queue (LLM call failed) still returns originals."""
    processor = LLMProcessor(config, circuit_breaker=CircuitBreaker())

    async def fake_enqueue(task_id, messages):
        return None

    processor.queue_manager.enqueue = fake_enqueue  # type: ignore[assignment]

    messages = [{"role": "user", "content": "hi"}]
    out = await processor.process_context(messages)
    assert out is messages


@pytest.mark.asyncio
async def test_process_context_handles_runtime_error_from_queue(config):
    """RuntimeError from the queue still falls back to originals."""
    processor = LLMProcessor(config, circuit_breaker=CircuitBreaker())

    async def fake_enqueue(task_id, messages):
        raise RuntimeError("queue failure")

    processor.queue_manager.enqueue = fake_enqueue  # type: ignore[assignment]

    messages = [{"role": "user", "content": "hi"}]
    out = await processor.process_context(messages)
    assert out is messages


def test_breaker_stored_on_instance(config):
    """The constructor stores the breaker; ``None`` is a valid default."""
    assert LLMProcessor(config).circuit_breaker is None
    breaker = CircuitBreaker(name="x")
    assert LLMProcessor(config, circuit_breaker=breaker).circuit_breaker is breaker


@pytest.mark.asyncio
async def test_custom_config_is_used_when_provided():
    """``LLMProcessor(custom_cfg)`` must read credentials from the injected
    config, not from the process-wide singleton.

    Regression guard for the ``cfg = config if config else config`` bug where
    the global singleton was always used regardless of the argument.

    Also verifies the Authorization header carries the custom key (not the
    global one) on the outgoing HTTP request, ensuring the API key is
    threaded through per-call headers rather than cached on the instance.
    """
    from src.config import get_config

    class _CustomConfig:
        llm_api_key = "custom-key"
        llm_model = "custom-model"
        llm_endpoint = "https://custom.example.test/v1/chat/completions"

    custom = _CustomConfig()
    # Sanity: the custom config must differ from the singleton so the
    # assertion below would actually fail if the bug regressed.
    global_config = get_config()
    assert (custom.llm_api_key, custom.llm_model, custom.llm_endpoint) != (
        global_config.llm_api_key,
        global_config.llm_model,
        global_config.llm_endpoint,
    )

    processor = LLMProcessor(custom)

    assert processor.api_key == "custom-key"
    assert processor.model == "custom-model"
    assert processor.endpoint == "https://custom.example.test/v1/chat/completions"
    # The processor must not cache the Authorization header on the instance.
    assert not hasattr(processor, "headers")

    # Issue a real call through the queue's worker path so headers are built
    # locally and forwarded to ``_client.post``.
    fake_client = AsyncMock()
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    fake_client.post = AsyncMock(return_value=fake_response)

    with patch.object(processor, "_client", new=fake_client):
        await processor._call_llm_async([{"role": "user", "content": "x"}])

    fake_client.post.assert_called_once()
    call_kwargs = fake_client.post.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer custom-key"
    assert call_kwargs["headers"]["Content-Type"] == "application/json"


def test_config_is_required_argument():
    """``LLMProcessor.__init__`` must reject a missing config.

    The historical ``LLMProcessor()`` form silently fell back to the
    module-level singleton, which was constructed at import time and
    required ``LLM_API_KEY``. New code must always pass an explicit
    ``Config`` (or stub) so test isolation does not depend on
    environment state.
    """
    with pytest.raises(TypeError):
        LLMProcessor()  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_call_llm_async_uses_breaker_for_http_call(config):
    """``_call_llm_async`` routes the HTTP POST through ``breaker.call``
    when a breaker is configured; the breaker counts HTTP failures."""
    import httpx

    breaker = CircuitBreaker(failure_threshold=1)
    processor = LLMProcessor(config, circuit_breaker=breaker)

    # Stub the httpx client so the HTTP call raises an httpx.HTTPStatusError,
    # which the breaker must count as a failure and then open on.
    fake_client = AsyncMock()

    async def fake_post(*args, **kwargs):
        request = httpx.Request("POST", kwargs.get("url", "https://example.test"))
        response = httpx.Response(status_code=500, request=request)
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    fake_client.post = fake_post

    with patch.object(processor, "_client", new=fake_client):
        result = await processor._call_llm_async([{"role": "user", "content": "x"}])

    assert result is None
    # One tracked failure -> breaker should be OPEN.
    assert breaker.state == CircuitState.OPEN
