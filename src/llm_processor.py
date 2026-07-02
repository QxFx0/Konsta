import asyncio
import json
import logging
import random
import uuid
from typing import Any, Dict, List, Optional

import httpx

from src.config import Config

from .queue_manager import QueueManager
from .resilience import CircuitBreaker, CircuitBreakerOpen

logger = logging.getLogger(__name__)

class LLMProcessor:
    """
    Handles context processing using a cheaper LLM (gpt-oss-120b from cerebras.ai).
    Integrates QueueManager for asynchronous processing to handle rate limits and concurrency.
    """

    def __init__(self, config: Config, circuit_breaker: Optional[CircuitBreaker] = None):
        """
        Initializes the LLMProcessor with configuration.

        Args:
            config: Application configuration. **Required** -- call sites must
                construct or inject a :class:`src.config.Config` instance
                explicitly. There is no implicit fallback to a process-wide
                singleton, so importing this module never triggers
                ``Config.__post_init__`` (env load + validation) and the
                processor cannot accidentally read stale or unset credentials
                from a default-constructed :class:`Config`.
            circuit_breaker: Optional :class:`CircuitBreaker` used to fail fast when the
                downstream LLM provider is unhealthy. When the breaker is ``OPEN`` the
                processor logs a warning and returns the original (un-distilled)
                messages instead of issuing a new request.
        """
        # Store the config on the instance so non-``__init__`` code paths
        # (e.g. ``_call_cerebras_async``) can read settings without
        # relying on a module-level singleton.
        self.config = config
        self.api_key = config.llm_api_key
        self.model = config.llm_model
        self.endpoint = config.llm_endpoint
        self.circuit_breaker = circuit_breaker

        if not self.endpoint:
            logger.warning("LLM endpoint not configured. Using default cerebras.ai endpoint.")
            self.endpoint = "https://api.cerebras.ai/v1/chat/completions"

        if not self.api_key:
            logger.warning("LLM API key not configured. LLM processing will be disabled.")

        # Initialize QueueManager for async processing
        self.queue_manager = QueueManager(max_size=100, num_workers=2)
        self._client = None

    async def start(self):
        """
        Starts the LLM processor and its underlying queue workers.
        """
        if self._client is None:
            # Enforce TLS validation and set a secure timeout.
            # verify=True is default for httpx, but explicit setting
            # ensures consistency across environments.
            self._client = httpx.AsyncClient(
                timeout=30.0,
                verify=True
            )

        await self.queue_manager.start(self._call_cerebras_async)
        logger.info("LLMProcessor started with async queue workers.")

    async def stop(self):
        """
        Gracefully shuts down the LLM processor and its client.
        """
        await self.queue_manager.stop()
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("LLMProcessor stopped.")

    async def process_context(self, messages: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Processes the given messages using the LLM to distill context asynchronously via the queue.

        Args:
            messages: List of message dicts (e.g., [{"role": "user", "content": "..."}]).

        Returns:
            List of processed message dicts, or original messages if processing fails.
        """
        if not self.api_key:
            logger.warning("LLM API key not set. Skipping LLM processing.")
            return messages

        if not messages:
            return messages

        try:
            task_id = str(uuid.uuid4())
            # Enqueue the request for processing by workers. The queue's worker
            # invokes ``_call_cerebras_async`` which wraps the HTTP call in the
            # circuit breaker (when one is configured), so a CircuitBreakerOpen
            # raised inside the worker propagates back through ``enqueue``.
            processed_content = await self.queue_manager.enqueue(task_id, messages)

            if not processed_content:
                logger.warning("LLM returned empty response. Using original context.")
                return messages

            # Return the distilled context as a single message to maintain conversation flow
            return [{"role": "system", "content": f"Distilled Context: {processed_content}"}]

        except CircuitBreakerOpen as e:
            # Graceful degradation: skip distillation while the breaker is open
            # and return the original compressed text.
            logger.warning(f"Circuit breaker open; returning original context: {e}")
            return messages
        except asyncio.TimeoutError as e:
            logger.error(f"LLM processing timed out: {e}")
            return messages
        except RuntimeError as e:
            logger.error(f"Queue runtime error during LLM processing: {e}")
            return messages
        except Exception as e:
            # Safety net: log and fall back to original messages
            logger.error(f"Unexpected error during async LLM processing: {e}")
            return messages

    def _build_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build the request payload: the distillation system prompt prepended to
        the user-supplied ``messages`` plus the model parameters.
        """
        # Prepend a distillation prompt to ensure the LLM compresses the context
        # instead of just continuing the conversation.
        distillation_prompt = {
            "role": "system",
            "content": (
                "You are a context distillation expert. Your task is to compress the following "
                "conversation history into a concise summary. Preserve all critical technical "
                "details, key decisions, and the user's ultimate intent. Remove redundancies "
                "and filler text. Output only the distilled summary."
            )
        }
        full_messages = [distillation_prompt] + messages
        return {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": 4096,
            "temperature": 0.3,
        }

    def _handle_response(self, response: httpx.Response) -> str:
        """
        Parse the OpenAI-compatible JSON response and return the assistant
        message ``content``. May raise :class:`json.JSONDecodeError`,
        :class:`KeyError`, or :class:`IndexError` for malformed payloads;
        callers are expected to translate those into graceful degradation.
        """
        try:
            response_data = response.json()
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # P1 Fix: Sanitize logs. Never log the raw content of the distillation.
            # Only log the length to maintain observability without leaking data.
            logger.debug(f"LLM returned content. Length: {len(content)} chars")

            return content
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return ""

    def _calculate_backoff_delay(
        self,
        attempt: int,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> float:
        """
        Exponential backoff with full jitter, capped at ``max_delay``.

        Formula: ``min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)``.
        """
        return min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)

    async def _call_cerebras_async(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Asynchronous API call to Cerebras AI to distill the context
        with exponential backoff for HTTP 429 errors.

        Args:
            messages: The conversation history to be distilled.

        Returns:
            The distilled content string, or None if the call fails.
        """
        if not self._client:
            logger.error("HTTP client not initialized. Call start() first.")
            return None

        max_retries = 5
        payload = self._build_payload(messages)

        for attempt in range(max_retries):
            try:
                # Build headers locally per request so the API key is never
                # cached on the instance. Only request metadata is logged
                # below — never the Authorization header itself.
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                logger.debug(
                    "Sending LLM request (attempt %d/%d): endpoint=%s model=%s messages=%d",
                    attempt + 1,
                    max_retries,
                    self.endpoint,
                    self.model,
                    len(messages),
                )

                if self.circuit_breaker is not None:
                    response = await self.circuit_breaker.call(
                        self._client.post,
                        self.endpoint,
                        headers=headers,
                        json=payload,
                    )
                else:
                    response = await self._client.post(
                        self.endpoint,
                        headers=headers,
                        json=payload,
                    )

                if response.status_code == 429:
                    # Exponential backoff with jitter, capped at ``max_delay``
                    # so very late attempts never sleep for an unbounded time.
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"HTTP 429 Too Many Requests. Retrying in {delay:.2f}s... "
                        f"(Attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return self._handle_response(response)

            except CircuitBreakerOpen:
                # Fail fast: don't retry while the breaker is open. The
                # queue worker will surface this as a task failure, which
                # ``process_context`` translates into graceful degradation.
                raise
            except httpx.TimeoutException as e:
                logger.error(f"LLM API request timed out: {e}")
                if self.circuit_breaker is not None:
                    self.circuit_breaker.record_failure()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 429:
                    logger.error(f"LLM API HTTP error: {e}")
                    if self.circuit_breaker is not None:
                        self.circuit_breaker.record_failure()
                    break
            except httpx.RequestError as e:
                logger.error(f"LLM API request failed: {e}")
                if self.circuit_breaker is not None:
                    self.circuit_breaker.record_failure()
                break
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.error(f"Failed to parse LLM API response: {e}")
                break
            except Exception as e:
                # Safety net: log and stop retrying
                logger.error(f"Unexpected error during _call_cerebras_async: {e}")
                break

        logger.error(f"Failed to get response from LLM after {max_retries} attempts.")
        return None
