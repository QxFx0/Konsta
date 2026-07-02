import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Set, Type, TypeVar

import keyring

from src.ca.crypto import CAKeyManager

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
        llm_api_key (str): API key for the LLM provider.
        llm_model (str): Model name for the LLM provider (e.g., 'llama3.1-8b').
            Override with the ``LLM_MODEL`` environment variable, or choose from
            ``SUPPORTED_MODELS`` at startup.
        llm_endpoint (str): API endpoint for the LLM provider.
        llm_timeout (float): Maximum number of seconds to wait for the remote
            distillation call before falling back to the locally compressed
            result. Defaults to 2.0 seconds.
        distillation_mode (str): How LLM distillation is invoked from the
            request path. ``"background"`` (default) means distillation runs
            asynchronously in :class:`DistillationWorker`; the request handler
            does not block on the LLM call. The result is applied on the
            *next* request when it becomes available (distillation lags one
            request behind). ``"disabled"`` skips the worker entirely. Other
            values may be added in the future.
        ca_cert_path (str): Path to the CA certificate file.
        ca_key_path (str): Path to the CA private key file.
        dump_include_bodies (bool): Whether diagnostic dumps should include
            raw request/response bodies. Defaults to ``False``; set to ``True``
            to include bodies in dumps.
        embedding_model (str): SentenceTransformer model name used for semantic
            deduplication. Override with the ``EMBEDDING_MODEL`` environment
            variable.
    """
    target_hosts: Set[str] = field(default_factory=lambda: {
        "api.openai.com",
        "api.anthropic.com",
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
    proxy_auth_token: Optional[str] = None
    llm_api_key: str = ""
    llm_model: str = ""
    llm_endpoint: str = ""
    llm_timeout: float = 2.0
    distillation_mode: str = "background"
    ca_cert_path: str = os.path.expanduser("~/.mitmproxy/konsta-ca-cert.pem")
    ca_key_path: str = os.path.expanduser("~/.mitmproxy/konsta-ca-key.pem")
    distillation_cache_size: int = 128
    dump_dir: str = os.path.expanduser("~/.konsta/dumps")
    max_payload_bytes: int = 10 * 1024 * 1024
    dump_include_bodies: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"

    # Circuit breaker parameters for the LLM distillation pipeline.
    # Defaults match the historical hardcoded values used by proxy_launcher.
    breaker_failure_threshold: int = 5
    breaker_recovery_timeout: float = 30.0
    breaker_half_open_max_calls: int = 1

    # Allowed values for ``distillation_mode``. Kept as a class-level tuple so
    # tests and CLI choices share a single source of truth.
    DISTILLATION_MODES: tuple[str, ...] = ("background", "disabled")

    # Available models for selection. These are real Cerebras-hosted model
    # IDs. The user can also override ``llm_model`` directly with the
    # ``LLM_MODEL`` environment variable.
    SUPPORTED_MODELS = {
        "1": "llama3.1-8b",
        "2": "llama-3.3-70b",
    }

    @property
    def enable_llm_distillation(self) -> bool:
        """Deprecated alias for ``distillation_mode != "disabled"``.

        Retained for backwards compatibility with existing callers and the
        legacy ``ENABLE_LLM_DISTILLATION`` / ``--enable-llm-distillation``
        controls. New code should read ``distillation_mode`` directly.
        """
        return self.distillation_mode != "disabled"

    def _safe_cast(self, value: str, cast_type: Type[T], attr_name: str, default: T) -> T:
        """
        Safely casts a string value to the specified type, falling back to default on failure.
        """
        try:
            return cast_type(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid value '{value}' for {attr_name}. Using default: {default}")
            return default

    def _parse_bool(self, value: str, attr_name: str, default: bool) -> bool:
        """
        Parse a string into a boolean using a small set of truthy/falsy tokens.

        Note: bool("false") is True because the string is non-empty, so we
        cannot rely on plain bool() for env var parsing.
        """
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off", ""):
            return False
        logger.warning(
            f"Invalid boolean value '{value}' for {attr_name}. Using default: {default}"
        )
        return default

    # ------------------------------------------------------------------
    # Field-level env var loaders
    #
    # These helpers keep ``_load_from_env`` declarative: each helper
    # reads a single env var (with optional legacy aliases) and writes
    # the parsed value onto the named attribute. Range constraints are
    # documented via ``min``/``max`` kwargs but deliberately not
    # enforced here — strict validation lives in :meth:`_validate` so
    # every env-var issue surfaces through one consistent error path.
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_env_value(name: str, aliases=()):
        """
        Return the first non-empty env value among ``name`` and ``aliases``.

        Mirrors the historical ``A or B`` pattern used in the previous
        inline implementation: empty strings are treated as "not set" so
        legacy fallbacks (``CEREBRAS_API_KEY`` → ``LLM_API_KEY`` etc.)
        still take effect when the modern variable is set but blank.
        Returns ``None`` if nothing is set or every candidate is empty.
        """
        value = os.getenv(name)
        if not value:
            for alias in aliases:
                value = os.getenv(alias)
                if value:
                    break
        return value or None

    def _load_str_env(self, name: str, attr: str, *, aliases=()):
        """
        Load a string-typed field from the environment.

        If ``name`` (or any of ``aliases``) is set to a non-empty value,
        write it to ``self.<attr>``. Empty / unset values leave the
        field unchanged.
        """
        raw = self._resolve_env_value(name, aliases)
        if raw is not None:
            setattr(self, attr, raw)

    def _load_int_env(self, name: str, attr: str, *, min=None, max=None, aliases=()):
        """
        Load an int-typed field from the environment.

        The optional ``min`` / ``max`` bounds document the expected
        range — strict range enforcement happens in :meth:`_validate`
        so all env-var issues surface through the same error path.
        Malformed values fall back to the field's current value via
        :meth:`_safe_cast`.
        """
        raw = self._resolve_env_value(name, aliases)
        if raw is None:
            return
        current = getattr(self, attr)
        value = self._safe_cast(raw, int, attr, current)
        setattr(self, attr, value)

    def _load_float_env(self, name: str, attr: str, *, min=None, aliases=()):
        """
        Load a float-typed field from the environment.

        The optional ``min`` bound documents the expected range — strict
        range enforcement happens in :meth:`_validate`. Malformed values
        fall back to the field's current value via :meth:`_safe_cast`.
        """
        raw = self._resolve_env_value(name, aliases)
        if raw is None:
            return
        current = getattr(self, attr)
        value = self._safe_cast(raw, float, attr, current)
        setattr(self, attr, value)

    def _load_bool_env(self, name: str, attr: str, *, aliases=()):
        """
        Load a boolean-typed field from the environment using
        :meth:`_parse_bool`. Malformed values fall back to the field's
        current value.
        """
        raw = self._resolve_env_value(name, aliases)
        if raw is None:
            return
        current = getattr(self, attr)
        setattr(self, attr, self._parse_bool(raw, attr, current))

    def _load_set_env(self, name: str, attr: str, *, aliases=()):
        """
        Load a comma-separated set-typed field from the environment.

        The result is a set of stripped strings, replacing any previous
        value when the env var is non-empty.
        """
        raw = self._resolve_env_value(name, aliases)
        if raw is not None:
            setattr(self, attr, {p.strip() for p in raw.split(",")})

    def __post_init__(self):
        """
        Orchestrate environment loading, legacy alias application, and
        validation. The actual work lives in the private helpers below;
        keeping them split makes it easy to test each concern in isolation.
        """
        self._load_from_env()
        self._apply_legacy_aliases()
        self._validate()

    def _load_from_env(self):
        """
        Apply environment variable overrides to every field that supports
        one. The actual env parsing lives in the field-level helpers
        (``_load_str_env``, ``_load_int_env``, ``_load_float_env``,
        ``_load_bool_env``, ``_load_set_env``); this method stays a
        single readable list of "which env var drives which field".

        Unknown / malformed values fall back to the field's existing
        value (via :meth:`_safe_cast` and :meth:`_parse_bool`).
        Strict range validation is deferred to :meth:`_validate`.
        """
        # Similarity threshold — legacy SIMILARITY_THRESHOLD alias supported.
        self._load_float_env(
            "PROXY_SIMILARITY_THRESHOLD",
            "similarity_threshold",
            min=0.0,
            aliases=("SIMILARITY_THRESHOLD",),
        )

        # Exclusion patterns — legacy EXCLUSION_PATTERNS alias supported.
        self._load_set_env(
            "PROXY_EXCLUSION_PATTERNS",
            "exclusion_patterns",
            aliases=("EXCLUSION_PATTERNS",),
        )

        self._load_int_env("PROXY_PORT", "proxy_port", min=1, max=65535)
        self._load_str_env("PROXY_HOST", "proxy_host")
        self._load_str_env("KONSTA_AUTH_TOKEN", "proxy_auth_token")

        # LLM configuration — LLM_API_KEY falls back to CEREBRAS_API_KEY.
        self._load_str_env(
            "LLM_API_KEY", "llm_api_key", aliases=("CEREBRAS_API_KEY",)
        )
        self._load_str_env("LLM_MODEL", "llm_model")
        self._load_str_env("LLM_ENDPOINT", "llm_endpoint")
        self._load_float_env("LLM_TIMEOUT", "llm_timeout", min=0.0)

        # DISTILLATION_MODE has bespoke behaviour: whitespace-only values
        # fall back to ``"background"`` rather than the field's existing
        # value. Handled inline because ``_load_str_env`` strips/preserves
        # raw text as-is.
        raw_mode = os.getenv("DISTILLATION_MODE")
        if raw_mode:
            self.distillation_mode = raw_mode.strip() or "background"

        # CA configuration.
        self._load_str_env("CA_CERT_PATH", "ca_cert_path")
        self._load_str_env("CA_KEY_PATH", "ca_key_path")

        # Distillation cache size / dump directory / max payload bytes.
        self._load_int_env("DISTILLATION_CACHE_SIZE", "distillation_cache_size", min=1)
        self._load_str_env("DUMP_DIR", "dump_dir")
        self._load_int_env("MAX_PAYLOAD_BYTES", "max_payload_bytes", min=1)
        self._load_bool_env("DUMP_INCLUDE_BODIES", "dump_include_bodies")
        self._load_str_env("EMBEDDING_MODEL", "embedding_model")

        # Circuit breaker parameters.
        self._load_int_env("BREAKER_FAILURE_THRESHOLD", "breaker_failure_threshold", min=1)
        self._load_float_env("BREAKER_RECOVERY_TIMEOUT", "breaker_recovery_timeout", min=0.0)
        self._load_int_env("BREAKER_HALF_OPEN_MAX_CALLS", "breaker_half_open_max_calls", min=1)

    def _apply_legacy_aliases(self):
        """
        Honour deprecated environment variables as aliases for their modern
        counterparts. Keeping legacy aliases in a dedicated method keeps
        ``_load_from_env`` focused on the canonical variables and makes the
        deprecation surface easy to find.

        Currently honoured:
          * ``ENABLE_LLM_DISTILLATION`` -> ``distillation_mode`` (only when
            ``DISTILLATION_MODE`` was not also set).
        """
        enable_distill = os.getenv("ENABLE_LLM_DISTILLATION")
        if enable_distill is not None and os.getenv("DISTILLATION_MODE") is None:
            self.distillation_mode = (
                "background" if self._parse_bool(
                    enable_distill, "ENABLE_LLM_DISTILLATION", False
                ) else "disabled"
            )

    def _validate(self):
        """
        Validates configuration parameters to ensure they are within acceptable bounds.
        """
        if not self.llm_api_key:
            raise ValueError(
                "LLM_API_KEY is missing. The system cannot operate without a valid "
                "API key for the distillation provider. Please set it via environment "
                "variable LLM_API_KEY or CEREBRAS_API_KEY."
            )

        # Basic format validation for API keys (fail-fast startup check).
        # We log a warning instead of raising an error to avoid breaking test mocks.
        if len(self.llm_api_key) < 16:
            logger.warning(
                f"LLM_API_KEY looks too short ({len(self.llm_api_key)} chars). "
                "Please check your configuration."
            )

        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}")

        if not (1 <= self.proxy_port <= 65535):
            raise ValueError(f"proxy_port must be between 1 and 65535, got {self.proxy_port}")

        if self.llm_timeout <= 0:
            raise ValueError(f"llm_timeout must be positive, got {self.llm_timeout}")

        if self.distillation_mode not in self.DISTILLATION_MODES:
            raise ValueError(
                f"distillation_mode must be one of {self.DISTILLATION_MODES}, "
                f"got {self.distillation_mode!r}"
            )

        if self.distillation_cache_size <= 0:
            raise ValueError(
                f"distillation_cache_size must be positive, got {self.distillation_cache_size}"
            )

        if not self.dump_dir:
            raise ValueError("dump_dir must be a non-empty path string")

        if self.max_payload_bytes <= 0:
            raise ValueError(
                f"max_payload_bytes must be positive, got {self.max_payload_bytes}"
            )

        if self.breaker_failure_threshold <= 0:
            raise ValueError(
                f"breaker_failure_threshold must be positive, got {self.breaker_failure_threshold}"
            )

        if self.breaker_recovery_timeout < 0:
            raise ValueError(
                f"breaker_recovery_timeout must be non-negative, got {self.breaker_recovery_timeout}"
            )

        if self.breaker_half_open_max_calls <= 0:
            raise ValueError(
                f"breaker_half_open_max_calls must be positive, got {self.breaker_half_open_max_calls}"
            )

    def resolve_ca_password(self) -> str:
        """
        Resolve the passphrase used to encrypt the CA private key.

        Resolution order:

        1. The ``CA_KEY_PASSWORD`` environment variable. If set to a non-empty
           value (whitespace stripped), that value is returned without touching
           the keyring or the user.
        2. The OS keyring via :meth:`CAKeyManager.get_or_create_passphrase`.
           If a passphrase has previously been stored for the Konsta CA, it is
           returned without prompting.
        3. Interactive prompt via :meth:`CAKeyManager.prompt_new_passphrase`,
           which stores the new passphrase in the keyring for subsequent runs.

        Raises:
            ValueError: when neither ``CA_KEY_PASSWORD`` nor a keyring entry is
                available AND the current process has no interactive terminal
                (``sys.stdin.isatty()`` returns ``False``) so prompting would
                block or fail. The message tells the operator exactly how to
                unblock startup.
        """
        env_password = os.getenv("CA_KEY_PASSWORD", "").strip()
        if env_password:
            return env_password

        manager = CAKeyManager()
        # Inspect the keyring directly so we can return a clean error before
        # getpass() would try to read from a non-interactive stdin.
        existing = keyring.get_password(manager.service_name, manager.username)
        if existing:
            return existing

        if not sys.stdin.isatty():
            raise ValueError(
                "CA key passphrase is required. Set CA_KEY_PASSWORD environment "
                "variable or ensure an interactive terminal so the passphrase "
                "can be prompted and stored in the OS keyring."
            )

        # Interactive path: prompt the user and store in keyring.
        return manager.get_or_create_passphrase()

    @classmethod
    def load_from_env(cls) -> 'Config':
        """
        Creates a Config instance. Environment variable loading is handled in __post_init__.
        """
        return cls()

    # ------------------------------------------------------------------
    # repr / str: redact secrets so a logged Config never leaks API keys.
    # ------------------------------------------------------------------

    # Field names whose values must never appear in ``repr()`` or
    # ``str()`` output. ``ca_key_password`` is listed as a forward-
    # compatibility marker: it is not currently a dataclass field (the
    # passphrase is resolved on demand via ``resolve_ca_password``), but
    # if a future iteration stores it on the instance, it will be
    # redacted automatically.
    _SENSITIVE_FIELDS: tuple[str, ...] = ("llm_api_key", "ca_key_password", "proxy_auth_token")

    def _redacted_repr(self) -> str:
        """
        Build a ``repr``-style string with sensitive fields masked as
        ``"***"``. Used by :meth:`__repr__` and :meth:`__str__` so the
        two are guaranteed to stay in sync.
        """
        parts: list[str] = []
        for name in self.__dataclass_fields__:
            if name in self._SENSITIVE_FIELDS:
                value: object = "***"
            else:
                value = getattr(self, name)
            parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __repr__(self) -> str:
        """Return a repr with sensitive fields (e.g. ``llm_api_key``) redacted."""
        return self._redacted_repr()

    def __str__(self) -> str:
        """Return a string representation with sensitive fields redacted."""
        return self._redacted_repr()


# ---------------------------------------------------------------------------
# Process-wide singleton accessor
# ---------------------------------------------------------------------------
#
# Historically this module exposed a module-level ``config = Config()``
# instance, which meant simply *importing* :mod:`src.config` triggered
# ``__post_init__`` and therefore env loading, legacy alias resolution,
# and :meth:`_validate`. That made ``python3 -m src.main --help`` fail
# in any environment where ``LLM_API_KEY`` was unset (or any other
# validator tripped), because ``from src.config import config`` at the
# top of :mod:`src.main` raised :class:`ValueError` during argparse's
# help path.
#
# The instance is now lazy: nothing happens at import time, and the
# singleton is created on the first call to :func:`get_config`. Callers
# that want to override the singleton (tests, multi-tenant setups) use
# :func:`set_config`; tests use :func:`reset_config` to clear the
# singleton between cases. Prefer constructing a :class:`Config`
# explicitly and passing it to components in new code.
_config_instance: Optional["Config"] = None


def get_config() -> "Config":
    """Return the process-wide :class:`Config` singleton, creating it on first use.

    The singleton is constructed lazily so that simply importing
    :mod:`src.config` (e.g. for ``--help`` or type inspection) does not
    trigger environment validation or require ``LLM_API_KEY`` to be
    set.

    Returns:
        The cached :class:`Config` instance, freshly built on first
        call. Subsequent calls return the same object until
        :func:`set_config` or :func:`reset_config` is invoked.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def set_config(config: "Config") -> None:
    """Install ``config`` as the process-wide singleton.

    Call this once at process startup (typically from :func:`src.main.main`)
    with a :class:`Config` instance the operator has already configured,
    so downstream call sites that read the singleton see the same values
    the rest of the application is using.

    Args:
        config: The :class:`Config` instance to expose via
            :func:`get_config`. The previous singleton (if any) is
            replaced; this is the supported way to inject a custom
            configuration in tests.
    """
    global _config_instance
    _config_instance = config


def reset_config() -> None:
    """Forget the current singleton so the next :func:`get_config` rebuilds it.

    Intended for test teardown: clear the cached instance between test
    cases so each one gets a fresh, env-driven :class:`Config` without
    leaking state from the previous case.
    """
    global _config_instance
    _config_instance = None

