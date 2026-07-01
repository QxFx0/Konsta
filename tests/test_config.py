"""Unit tests for src.config."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Config
from src.main import parse_args

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def no_env_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CA_KEY_PASSWORD is unset/empty for the test."""
    monkeypatch.delenv("CA_KEY_PASSWORD", raising=False)


# ---------------------------------------------------------------------------
# resolve_ca_password: env var path
# ---------------------------------------------------------------------------


def test_resolve_ca_password_returns_env_var_when_set(monkeypatch, fake_keyring):
    """A non-empty CA_KEY_PASSWORD must short-circuit and bypass the keyring."""
    monkeypatch.setenv("CA_KEY_PASSWORD", "env-pass")

    config = Config()

    with patch("src.config.CAKeyManager") as mock_manager:
        result = config.resolve_ca_password()

    assert result == "env-pass"
    # The keyring manager must not be touched when the env var is set.
    mock_manager.assert_not_called()
    # Nothing should have been written to the keyring.
    assert fake_keyring.store == {}


def test_resolve_ca_password_strips_whitespace_env_var(monkeypatch, fake_keyring):
    """Leading/trailing whitespace in CA_KEY_PASSWORD is stripped."""
    monkeypatch.setenv("CA_KEY_PASSWORD", "   padded-pass   ")

    config = Config()

    result = config.resolve_ca_password()

    assert result == "padded-pass"


def test_resolve_ca_password_empty_env_falls_through(monkeypatch, fake_keyring, no_env_password):
    """Empty CA_KEY_PASSWORD must not be treated as a valid passphrase."""
    monkeypatch.setenv("CA_KEY_PASSWORD", "")
    fake_keyring.store[("Konsta", "ca-key")] = "from-keyring"

    config = Config()
    result = config.resolve_ca_password()

    assert result == "from-keyring"


# ---------------------------------------------------------------------------
# resolve_ca_password: keyring fallback path
# ---------------------------------------------------------------------------


def test_resolve_ca_password_falls_back_to_keyring(
    monkeypatch, fake_keyring, no_env_password
):
    """When env is empty, the keyring entry is returned without prompting."""
    fake_keyring.store[("Konsta", "ca-key")] = "stored-pass"

    config = Config()

    with patch("src.ca.crypto.getpass.getpass") as mock_getpass:
        result = config.resolve_ca_password()

    assert result == "stored-pass"
    mock_getpass.assert_not_called()


def test_resolve_ca_password_prompts_when_keyring_empty_and_interactive(
    monkeypatch, fake_keyring, no_env_password
):
    """Interactive terminal + empty keyring => prompt the user and store."""
    # Pretend we have a TTY.
    class FakeStdin:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", FakeStdin())

    config = Config()

    responses = iter(["fresh-pass", "fresh-pass"])
    with patch(
        "src.ca.crypto.getpass.getpass", side_effect=lambda _prompt="": next(responses)
    ):
        result = config.resolve_ca_password()

    assert result == "fresh-pass"
    assert fake_keyring.store[("Konsta", "ca-key")] == "fresh-pass"


# ---------------------------------------------------------------------------
# resolve_ca_password: non-interactive ValueError path
# ---------------------------------------------------------------------------


def test_resolve_ca_password_raises_when_no_env_no_keyring_non_interactive(
    monkeypatch, fake_keyring, no_env_password
):
    """Missing env + empty keyring + non-tty => ValueError with actionable message."""

    class FakeStdin:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", FakeStdin())

    config = Config()

    with pytest.raises(ValueError) as exc_info:
        config.resolve_ca_password()

    message = str(exc_info.value)
    assert "CA key passphrase is required" in message
    assert "CA_KEY_PASSWORD" in message
    assert "keyring" in message


def test_resolve_ca_password_does_not_prompt_when_non_interactive(
    monkeypatch, fake_keyring, no_env_password
):
    """A non-interactive run must never call getpass (would block on EOF)."""

    class FakeStdin:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", FakeStdin())

    config = Config()

    with patch("src.ca.crypto.getpass.getpass") as mock_getpass:
        with pytest.raises(ValueError):
            config.resolve_ca_password()
    mock_getpass.assert_not_called()


# ---------------------------------------------------------------------------
# parse_args + --help behaviour (no passphrase required)
# ---------------------------------------------------------------------------


def test_parse_args_help_exits_without_passphrase(monkeypatch, no_env_password):
    """`parse_args(['--help'])` must exit cleanly before any CA logic runs."""
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--help"])
    assert exc_info.value.code == 0


def test_main_help_works_without_passphrase(monkeypatch, no_env_password, tmp_path):
    """End-to-end: `python -m src.main --help` exits 0 with no passphrase set."""
    project_root = Path(__file__).resolve().parent.parent
    env = {
        # Subprocess env: only the minimum required to let Config() instantiate.
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(project_root),
        "LLM_API_KEY": "dummy-key-for-help",
        "HOME": str(tmp_path),
    }
    # CA_KEY_PASSWORD is intentionally absent.

    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--help"],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=15,
    )

    assert result.returncode == 0, (
        f"--help exited with {result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "Konsta Context Compression Proxy" in result.stdout
    # The passphrase-required error must NOT appear when --help is used.
    assert "CA key passphrase is required" not in result.stderr
    assert "CA key passphrase is required" not in result.stdout


def test_main_help_works_when_env_password_unset_and_stdin_closed(
    monkeypatch, no_env_password, tmp_path
):
    """
    Defence-in-depth: even if the subprocess inherits a non-TTY stdin (e.g.
    pipes via CI), --help must still succeed because argparse exits before
    resolve_ca_password() runs.
    """
    project_root = Path(__file__).resolve().parent.parent
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(project_root),
        "LLM_API_KEY": "dummy-key-for-help",
        "HOME": str(tmp_path),
    }

    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--help"],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=15,
    )

    assert result.returncode == 0, (
        f"--help exited with {result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# distillation_mode field + deprecated enable_llm_distillation alias
# ---------------------------------------------------------------------------


def test_distillation_mode_default_is_background(monkeypatch):
    """A fresh Config picks ``background`` as the default mode."""
    monkeypatch.delenv("DISTILLATION_MODE", raising=False)
    monkeypatch.delenv("ENABLE_LLM_DISTILLATION", raising=False)

    config = Config()

    assert config.distillation_mode == "background"


def test_distillation_modes_class_constant_is_source_of_truth():
    """The DISTILLATION_MODES tuple enumerates every supported mode."""
    assert Config.DISTILLATION_MODES == ("background", "disabled")
    assert "background" in Config.DISTILLATION_MODES
    assert "disabled" in Config.DISTILLATION_MODES


def test_enable_llm_distillation_is_alias_for_mode(monkeypatch):
    """``enable_llm_distillation`` is a property derived from ``distillation_mode``."""
    monkeypatch.delenv("DISTILLATION_MODE", raising=False)
    monkeypatch.delenv("ENABLE_LLM_DISTILLATION", raising=False)

    config = Config()
    assert config.enable_llm_distillation is True  # default "background"

    config.distillation_mode = "disabled"
    assert config.enable_llm_distillation is False

    config.distillation_mode = "background"
    assert config.enable_llm_distillation is True


def test_distillation_mode_overrides_enable_llm_distillation_env(monkeypatch):
    """Explicit DISTILLATION_MODE env var wins over ENABLE_LLM_DISTILLATION."""
    monkeypatch.setenv("ENABLE_LLM_DISTILLATION", "false")
    monkeypatch.setenv("DISTILLATION_MODE", "background")

    config = Config()

    assert config.distillation_mode == "background"
    assert config.enable_llm_distillation is True


def test_enable_llm_distillation_env_true_sets_background(monkeypatch):
    """Legacy ENABLE_LLM_DISTILLATION=true maps to distillation_mode=background."""
    monkeypatch.delenv("DISTILLATION_MODE", raising=False)
    monkeypatch.setenv("ENABLE_LLM_DISTILLATION", "true")

    config = Config()

    assert config.distillation_mode == "background"
    assert config.enable_llm_distillation is True


def test_enable_llm_distillation_env_false_sets_disabled(monkeypatch):
    """Legacy ENABLE_LLM_DISTILLATION=false maps to distillation_mode=disabled."""
    monkeypatch.delenv("DISTILLATION_MODE", raising=False)
    monkeypatch.setenv("ENABLE_LLM_DISTILLATION", "false")

    config = Config()

    assert config.distillation_mode == "disabled"
    assert config.enable_llm_distillation is False


def test_distillation_mode_env_explicit(monkeypatch):
    """DISTILLATION_MODE env var is honoured verbatim (whitespace stripped)."""
    monkeypatch.setenv("DISTILLATION_MODE", "  disabled  ")

    config = Config()

    assert config.distillation_mode == "disabled"
    assert config.enable_llm_distillation is False


def test_distillation_mode_invalid_value_raises(monkeypatch):
    """Unknown modes are rejected at construction time."""
    monkeypatch.setenv("DISTILLATION_MODE", "synchronous")

    with pytest.raises(ValueError) as exc_info:
        Config()
    assert "distillation_mode" in str(exc_info.value)


def test_enable_llm_distillation_is_not_assignable(monkeypatch):
    """The deprecated alias is read-only; assigning to it must raise."""
    config = Config()

    with pytest.raises(AttributeError):
        config.enable_llm_distillation = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Circuit breaker parameter fields + env var overrides + validation
# ---------------------------------------------------------------------------


def test_breaker_params_default_values(monkeypatch):
    """Default breaker params match the historical hardcoded values."""
    for var in (
        "BREAKER_FAILURE_THRESHOLD",
        "BREAKER_RECOVERY_TIMEOUT",
        "BREAKER_HALF_OPEN_MAX_CALLS",
    ):
        monkeypatch.delenv(var, raising=False)

    config = Config()

    assert config.breaker_failure_threshold == 5
    assert config.breaker_recovery_timeout == 30.0
    assert config.breaker_half_open_max_calls == 1


def test_breaker_failure_threshold_env_overrides(monkeypatch):
    """``BREAKER_FAILURE_THRESHOLD`` env var sets ``breaker_failure_threshold``."""
    monkeypatch.setenv("BREAKER_FAILURE_THRESHOLD", "11")

    config = Config()

    assert config.breaker_failure_threshold == 11


def test_breaker_recovery_timeout_env_overrides(monkeypatch):
    """``BREAKER_RECOVERY_TIMEOUT`` env var sets ``breaker_recovery_timeout`` (float)."""
    monkeypatch.setenv("BREAKER_RECOVERY_TIMEOUT", "12.5")

    config = Config()

    assert config.breaker_recovery_timeout == 12.5


def test_breaker_half_open_max_calls_env_overrides(monkeypatch):
    """``BREAKER_HALF_OPEN_MAX_CALLS`` env var sets ``breaker_half_open_max_calls``."""
    monkeypatch.setenv("BREAKER_HALF_OPEN_MAX_CALLS", "3")

    config = Config()

    assert config.breaker_half_open_max_calls == 3


def test_breaker_params_all_env_overrides_together(monkeypatch):
    """All three breaker env vars are honoured simultaneously."""
    monkeypatch.setenv("BREAKER_FAILURE_THRESHOLD", "7")
    monkeypatch.setenv("BREAKER_RECOVERY_TIMEOUT", "5.5")
    monkeypatch.setenv("BREAKER_HALF_OPEN_MAX_CALLS", "2")

    config = Config()

    assert config.breaker_failure_threshold == 7
    assert config.breaker_recovery_timeout == 5.5
    assert config.breaker_half_open_max_calls == 2


def test_breaker_failure_threshold_zero_raises(monkeypatch):
    """``breaker_failure_threshold`` must be > 0; zero is rejected."""
    monkeypatch.setenv("BREAKER_FAILURE_THRESHOLD", "0")

    with pytest.raises(ValueError) as exc_info:
        Config()
    assert "breaker_failure_threshold" in str(exc_info.value)


def test_breaker_recovery_timeout_negative_raises(monkeypatch):
    """``breaker_recovery_timeout`` must be >= 0; negative values are rejected."""
    monkeypatch.setenv("BREAKER_RECOVERY_TIMEOUT", "-1.0")

    with pytest.raises(ValueError) as exc_info:
        Config()
    assert "breaker_recovery_timeout" in str(exc_info.value)


def test_breaker_recovery_timeout_zero_is_allowed(monkeypatch):
    """``breaker_recovery_timeout=0`` is a valid edge case (instant recovery)."""
    monkeypatch.setenv("BREAKER_RECOVERY_TIMEOUT", "0")

    config = Config()

    assert config.breaker_recovery_timeout == 0.0


def test_breaker_half_open_max_calls_zero_raises(monkeypatch):
    """``breaker_half_open_max_calls`` must be > 0; zero is rejected."""
    monkeypatch.setenv("BREAKER_HALF_OPEN_MAX_CALLS", "0")

    with pytest.raises(ValueError) as exc_info:
        Config()
    assert "breaker_half_open_max_calls" in str(exc_info.value)


# ---------------------------------------------------------------------------
# dump body inclusion
# ---------------------------------------------------------------------------


def test_dump_include_bodies_defaults_to_false():
    """By default, diagnostic dumps do NOT include raw bodies."""
    config = Config()
    assert config.dump_include_bodies is False


def test_dump_include_bodies_env_overrides(monkeypatch):
    """``DUMP_INCLUDE_BODIES`` toggles the diagnostic body inclusion flag."""
    monkeypatch.setenv("DUMP_INCLUDE_BODIES", "false")
    config = Config()
    assert config.dump_include_bodies is False


def test_embedding_model_defaults_to_mini_lm():
    """By default, the semantic dedup model is the offline MiniLM variant."""
    config = Config()
    assert config.embedding_model == "all-MiniLM-L6-v2"


def test_embedding_model_env_overrides(monkeypatch):
    """``EMBEDDING_MODEL`` overrides the default SentenceTransformer model."""
    monkeypatch.setenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
    config = Config()
    assert config.embedding_model == "all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# repr / str redaction of secret fields
# ---------------------------------------------------------------------------


def test_config_repr_does_not_leak_api_key(monkeypatch):
    """repr(config) must never contain the actual llm_api_key value.

    The key is set via the modern env var; repr must show it as ``***``.
    """
    secret = "super-secret-llm-key-abc123XYZ"
    monkeypatch.setenv("LLM_API_KEY", secret)

    config = Config()

    rendered = repr(config)
    assert secret not in rendered, (
        f"repr leaked llm_api_key value: {rendered!r}"
    )
    assert "***" in rendered, "expected redacted placeholder '***' in repr"
    assert "llm_api_key='***'" in rendered


def test_config_str_does_not_leak_api_key(monkeypatch):
    """str(config) must never contain the actual llm_api_key value."""
    secret = "another-top-secret-llm-key-9876"
    monkeypatch.setenv("LLM_API_KEY", secret)

    config = Config()

    rendered = str(config)
    assert secret not in rendered, (
        f"str leaked llm_api_key value: {rendered!r}"
    )
    assert "***" in rendered, "expected redacted placeholder '***' in str"
    assert "llm_api_key='***'" in rendered


def test_config_repr_redacts_legacy_cerebras_api_key(monkeypatch):
    """Setting the legacy CEREBRAS_API_KEY alias must also be redacted."""
    secret = "cerebras-fallback-secret-key"
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("CEREBRAS_API_KEY", secret)

    config = Config()

    # Sanity: the legacy alias actually populated llm_api_key.
    assert config.llm_api_key == secret
    assert secret not in repr(config)
    assert secret not in str(config)


def test_config_repr_keeps_non_secret_fields_visible(monkeypatch):
    """Non-secret fields remain readable in repr() for debugging."""
    monkeypatch.setenv("LLM_API_KEY", "shhh-secret")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b")
    monkeypatch.setenv("PROXY_PORT", "9090")

    config = Config()

    rendered = repr(config)
    assert "llm_api_key='***'" in rendered
    assert "llm_model='llama-3.3-70b'" in rendered
    assert "proxy_port=9090" in rendered
    # The actual secret string must still be absent.
    assert "shhh-secret" not in rendered


def test_config_repr_and_str_are_redacted_identically(monkeypatch):
    """repr and str share the same redaction logic."""
    monkeypatch.setenv("LLM_API_KEY", "shhh-shared-secret")

    config = Config()

    assert repr(config) == str(config)
    assert "shhh-shared-secret" not in repr(config)
    assert "shhh-shared-secret" not in str(config)

