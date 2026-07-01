import json
import logging
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from src.metrics import Metrics

logger = logging.getLogger(__name__)

class CompressionEngine:
    """
    Coordinator for semantic deduplication and context filtering.
    Reduces token usage by removing redundant information while preserving semantic meaning.
    """

    def __init__(
        self,
        config=None,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.85,
        max_messages: int = 50,
        metrics: Optional[Metrics] = None,
    ):
        # If config object is provided, extract values from it. The caller
        # can still override ``model_name`` explicitly; otherwise we fall
        # back to ``config.embedding_model`` and finally the default.
        if config:
            self.similarity_threshold = config.similarity_threshold
            self.max_messages = max_messages  # Keep this as is or add to config
            if model_name != 'all-MiniLM-L6-v2':
                actual_model_name = model_name
            else:
                actual_model_name = getattr(config, "embedding_model", model_name)
        else:
            self.similarity_threshold = similarity_threshold
            self.max_messages = max_messages
            actual_model_name = model_name

        # Metrics is optional to keep existing call sites working; when None
        # we construct a fresh local instance rather than relying on the
        # process-wide singleton, so callers get a private store by default.
        if metrics is None:
            metrics = Metrics()
        self.metrics = metrics

        self.model = None
        self._model_load_error: Optional[str] = None

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Forcing local load to avoid HF 429 errors and speed up startup
                try:
                    self.model = SentenceTransformer(actual_model_name, local_files_only=True)
                    logger.info(f"CompressionEngine initialized with model {actual_model_name} (offline mode)")
                except (OSError, ConnectionError, ValueError):
                    # Local-only load can fail when the model is not cached yet
                    # (OSError for missing files, ConnectionError for Hub calls
                    # that slipped through, ValueError for invalid identifiers).
                    # Fall back to a normal (potentially online) load.
                    self.model = SentenceTransformer(actual_model_name)
                    logger.info(f"CompressionEngine initialized with model {actual_model_name} (online mode)")
            except Exception as e:
                self._model_load_error = str(e)
                logger.error(f"Failed to load SentenceTransformer model: {e}")
        else:
            self._model_load_error = "sentence-transformers not installed"
            logger.warning("sentence-transformers not installed. Semantic deduplication will be disabled.")

    def is_healthy(self) -> bool:
        """Return whether the compression engine is in a usable state.

        Healthy means there was no model load failure. ``model`` may be
        ``None`` because semantic deduplication was intentionally disabled
        (e.g. tests) or because the optional dependency is missing; both
        are acceptable. A load failure (bad model name, missing files,
        etc.) makes the engine unhealthy because the operator may be
        expecting semantic deduplication that can never succeed.
        """
        if self._model_load_error is not None:
            return "not installed" in self._model_load_error
        return True

    def compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for context compression.
        Applies a pipeline of compression strategies.
        """
        if not messages:
            return []

        # Snapshot the byte size of the input BEFORE any pipeline step so
        # the recorded compression ratio reflects the original payload.
        original_size = self._messages_byte_size(messages)

        # 1. Merge consecutive messages from the same role
        compressed = self._merge_consecutive_messages(messages)

        # 2. Semantic deduplication (if model is available)
        if self.model:
            compressed = self._deduplicate_semantically(compressed)

        # 3. Truncate old context to fit within limits
        compressed = self._truncate_old_context(compressed)

        # Record the compression observation. We measure size as the JSON
        # byte length of the message lists so the gauge is meaningful
        # regardless of message content shape (text vs multimodal).
        compressed_size = self._messages_byte_size(compressed)
        try:
            self.metrics.record_compression(original_size, compressed_size)
        except Exception:
            # A misbehaving metrics store must never break the request path.
            logger.exception("Failed to record compression metrics")

        return compressed

    @staticmethod
    def _messages_byte_size(messages: List[Dict[str, Any]]) -> int:
        """Return the JSON byte length of ``messages`` for metrics tracking.

        ``ensure_ascii=False`` matches the encoding the proxy uses when
        rewriting request bodies, so the recorded sizes line up with the
        bytes actually sent upstream.
        """
        try:
            return len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        except (TypeError, ValueError):
            # Non-serialisable content (e.g. raw bytes, custom objects).
            # Fall back to a coarse string length so we still record a
            # non-zero observation rather than silently dropping it.
            total = 0
            for msg in messages:
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, str):
                    total += len(content)
            return total

    def _merge_consecutive_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merges consecutive messages that have the same role (e.g., multiple 'user' messages in a row).
        """
        if not messages:
            return []

        merged = []
        current_msg = messages[0].copy()

        for next_msg in messages[1:]:
            if next_msg.get('role') == current_msg.get('role'):
                # Merge content
                current_content = current_msg.get('content', '')
                next_content = next_msg.get('content', '')
                current_msg['content'] = f"{current_content}\n{next_content}".strip()
            else:
                merged.append(current_msg)
                current_msg = next_msg.copy()

        merged.append(current_msg)
        return merged

    def _deduplicate_semantically(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes messages that are semantically too similar to more recent messages.
        Prioritizes keeping the most recent information.
        Protects tool-call chains to prevent API deserialization errors (e.g., DeepSeek tool_call_id).
        """
        if len(messages) <= 1:
            return messages

        # Identify tool-related messages that MUST be kept together
        # A tool chain is: Assistant (with tool_calls) -> Tool (response)
        protected_indices = set()
        for i in range(len(messages)):
            msg = messages[i]
            # Keep tool responses
            if msg.get('role') == 'tool':
                protected_indices.add(i)
                # Also protect the preceding assistant message that called this tool
                if i > 0 and messages[i-1].get('role') == 'assistant':
                    protected_indices.add(i-1)
            # Keep assistant messages that initiate tool calls
            if 'tool_calls' in msg:
                protected_indices.add(i)
                # Also protect the following tool response
                if i < len(messages) - 1 and messages[i+1].get('role') == 'tool':
                    protected_indices.add(i+1)

        # Extract content for embedding only for non-protected messages
        actual_embeddings = []
        indices = []
        for i, m in enumerate(messages):
            if i in protected_indices:
                continue

            content = m.get('content', '')

            # Handle multimodal content (list of dicts with text/image)
            if isinstance(content, list):
                # Extract only text parts from multimodal content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                content = ' '.join(text_parts)

            if content and isinstance(content, str):
                actual_embeddings.append(content)
                indices.append(i)

        if not actual_embeddings:
            return messages

        try:
            vecs = self.model.encode(actual_embeddings)
            similarity_matrix = cosine_similarity(vecs)
        except (ValueError, TypeError) as e:
            logger.error(f"Semantic deduplication encoding/similarity failed: {e}")
            return messages

        keep_indices = set(range(len(messages)))

        # Iterate through the non-protected messages backwards
        for i_idx in range(len(indices) - 1, -1, -1):
            i = indices[i_idx]
            if i not in keep_indices:
                continue

            for j_idx in range(i_idx - 1, -1, -1):
                j = indices[j_idx]
                if j not in keep_indices:
                    continue

                if similarity_matrix[i_idx, j_idx] > self.similarity_threshold:
                    keep_indices.remove(j)

        return [messages[i] for i in sorted(list(keep_indices))]


    def _truncate_old_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensures the number of messages does not exceed max_messages.
        Keeps the most recent messages.
        """
        if len(messages) <= self.max_messages:
            return messages

        # Keep the last N messages
        return messages[-self.max_messages:]
