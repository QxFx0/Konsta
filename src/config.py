import os
import logging
from dataclasses import dataclass, field
from typing import Set, Type, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar('T')

@dataclass
class Config:
    """
    Configuration for the context compression proxy.
    
    Attributes:
        similarity_threshold (float): Cosine similarity threshold for semantic deduplication.
            Values > 0.85 are typically used to identify redundant context.
        exclusion_patterns (Set[str]): Set of file/directory patterns to exclude from context.
        proxy_port (int): Port on which the local proxy server will run.
        proxy_host (str): Host address for the local proxy server.
        llm_api_key (str): API key for the LLM provider (e.g., cerebras.ai).
        llm_model (str): Model name for the LLM provider (e.g., 'gpt-oss-120b').
        llm_endpoint (str): API endpoint for the LLM provider (e.g., 'https://api.cerebras.ai/v1/chat/completions').
        ca_cert_path (str): Path to the CA certificate file.
        ca_key_path (str): Path to the CA private key file.
    """
    target_hosts: Set[str] = field(default_factory=lambda: {
        "api.openai.com",
        "api.anthropic.com",
        "api.cerebras.ai",
        "models.dev",
        "api.mistral.ai"
    })
    similarity_threshold: float = 0.85
    exclusion_patterns: Set[str] = field(default_factory=lambda: {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
        "dist",
        "build"
    })
    proxy_port: int = 8080
    proxy_host: str = "127.0.0.1"
    llm_api_key: str = ""
    llm_model: str = "gpt-oss-120b"
    llm_endpoint: str = "https://api.cerebras.ai/v1/chat/completions"
    ca_cert_path: str = os.path.expanduser("~/.mitmproxy/konsta-ca-cert.pem")
    ca_key_path: str = os.path.expanduser("~/.mitmproxy/konsta-ca-key.pem")
    
    # Available models for selection
    SUPPORTED_MODELS = {
        "1": "gpt-oss-120b",
        "2": "zai-glm-4.7",
        "3": "gemma-4-31b"
    }

    def _safe_cast(self, value: str, cast_type: Type[T], attr_name: str, default: T) -> T:
        """
        Safely casts a string value to the specified type, falling back to default on failure.
        """
        try:
            return cast_type(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid value '{value}' for {attr_name}. Using default: {default}")
            return default

    def __post_init__(self):
        """
        Overrides default values with environment variables if present and validates them.
        """
        # Support both generic and prefixed environment variables
        sim_threshold = os.getenv("PROXY_SIMILARITY_THRESHOLD") or os.getenv("SIMILARITY_THRESHOLD")
        if sim_threshold is not None:
            self.similarity_threshold = self._safe_cast(
                sim_threshold, float, "similarity_threshold", self.similarity_threshold
            )

        env_exclusions = os.getenv("PROXY_EXCLUSION_PATTERNS") or os.getenv("EXCLUSION_PATTERNS")
        if env_exclusions:
            self.exclusion_patterns = {p.strip() for p in env_exclusions.split(",")}

        port = os.getenv("PROXY_PORT")
        if port:
            self.proxy_port = self._safe_cast(
                port, int, "proxy_port", self.proxy_port
            )

        host = os.getenv("PROXY_HOST")
        if host:
            self.proxy_host = host
            
        # LLM Configuration
        llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("CEREBRAS_API_KEY")
        if llm_api_key:
            self.llm_api_key = llm_api_key
            
        llm_model = os.getenv("LLM_MODEL")
        if llm_model:
            self.llm_model = llm_model
            
        llm_endpoint = os.getenv("LLM_ENDPOINT")
        if llm_endpoint:
            self.llm_endpoint = llm_endpoint
            
        # CA Configuration
        ca_cert_path = os.getenv("CA_CERT_PATH")
        if ca_cert_path:
            self.ca_cert_path = ca_cert_path
            
        ca_key_path = os.getenv("CA_KEY_PATH")
        if ca_key_path:
            self.ca_key_path = ca_key_path

        self._validate()

    def _validate(self):
        """
        Validates configuration parameters to ensure they are within acceptable bounds.
        """
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY is required. Please set it via environment variable or config.")
        
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}")
            
        if not (1 <= self.proxy_port <= 65535):
            raise ValueError(f"proxy_port must be between 1 and 65535, got {self.proxy_port}")

    @classmethod
    def load_from_env(cls) -> 'Config':
        """
        Creates a Config instance. Environment variable loading is handled in __post_init__.
        """
        return cls()

# Global configuration instance
config = Config()