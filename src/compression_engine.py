import logging
from typing import List, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)

class CompressionEngine:
    """
    Coordinator for semantic deduplication and context filtering.
    Reduces token usage by removing redundant information while preserving semantic meaning.
    """
    
    def __init__(self, config=None, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.85, max_messages: int = 50):
        # If config object is provided, extract values from it
        if config:
            self.similarity_threshold = config.similarity_threshold
            self.max_messages = max_messages # Keep this as is or add to config
            # For model_name, prioritize the passed model_name, then config, then default
            actual_model_name = model_name
        else:
            self.similarity_threshold = similarity_threshold
            self.max_messages = max_messages
            actual_model_name = model_name

        self.model = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Forcing local load to avoid HF 429 errors and speed up startup
                try:
                    self.model = SentenceTransformer(actual_model_name, local_files_only=True)
                    logger.info(f"CompressionEngine initialized with model {actual_model_name} (offline mode)")
                except Exception:
                    self.model = SentenceTransformer(actual_model_name)
                    logger.info(f"CompressionEngine initialized with model {actual_model_name} (online mode)")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
        else:
            logger.warning("sentence-transformers not installed. Semantic deduplication will be disabled.")

    def compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for context compression.
        Applies a pipeline of compression strategies.
        """
        if not messages:
            return []

        # 1. Merge consecutive messages from the same role
        compressed = self._merge_consecutive_messages(messages)
        
        # 2. Semantic deduplication (if model is available)
        if self.model:
            compressed = self._deduplicate_semantically(compressed)
        
        # 3. Truncate old context to fit within limits
        compressed = self._truncate_old_context(compressed)
        
        return compressed

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
            
        vecs = self.model.encode(actual_embeddings)
        similarity_matrix = cosine_similarity(vecs)
        
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
