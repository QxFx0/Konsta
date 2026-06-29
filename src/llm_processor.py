import logging
import httpx
import asyncio
import json
import random
import uuid
from typing import List, Dict, Any, Optional

from .config import config
from .queue_manager import QueueManager

logger = logging.getLogger(__name__)

class LLMProcessor:
    """
    Handles context processing using a cheaper LLM (gpt-oss-120b from cerebras.ai).
    Integrates QueueManager for asynchronous processing to handle rate limits and concurrency.
    """

    def __init__(self, config=None):
        """
        Initializes the LLMProcessor with configuration.
        """
        # Use provided config or fall back to global config
        cfg = config if config else config
        self.api_key = cfg.llm_api_key
        self.model = cfg.llm_model
        self.endpoint = cfg.llm_endpoint
        
        if not self.endpoint:
            logger.warning("LLM endpoint not configured. Using default cerebras.ai endpoint.")
            self.endpoint = "https://api.cerebras.ai/v1/chat/completions"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
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
            self._client = httpx.AsyncClient(timeout=30.0)
        
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
            # Enqueue the request for processing by workers
            processed_content = await self.queue_manager.enqueue(task_id, messages)
            
            if not processed_content:
                logger.warning("LLM returned empty response. Using original context.")
                return messages
                
            # Return the distilled context as a single message to maintain conversation flow
            return [{"role": "system", "content": f"Distilled Context: {processed_content}"}]
            
        except Exception as e:
            logger.error(f"Unexpected error during async LLM processing: {e}")
            return messages

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
        
        # Combine prompt with original messages
        full_messages = [distillation_prompt] + messages

        max_retries = 5
        base_delay = 1.0  # seconds
        
        payload = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": 4096,
            "temperature": 0.3
        }
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Sending context to LLM for processing (attempt {attempt + 1}): {len(messages)} messages")
                
                response = await self._client.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 429:
                    # Exponential backoff with jitter
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    logger.warning(f"HTTP 429 Too Many Requests. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                
                response.raise_for_status()
                
                # Parse the response
                response_data = response.json()
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                return content

            except httpx.HTTPStatusError as e:
                if e.response.status_code != 429:
                    logger.error(f"LLM API HTTP error: {e}")
                    break
            except httpx.RequestError as e:
                logger.error(f"LLM API request failed: {e}")
                break
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.error(f"Failed to parse LLM API response: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error during _call_cerebras_async: {e}")
                break
        
        logger.error(f"Failed to get response from LLM after {max_retries} attempts.")
        return None
