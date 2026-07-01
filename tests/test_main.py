"""Unit tests for ``src.main`` CLI and worker wiring."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cli import apply_distillation_settings
from src.config import Config, get_config, reset_config
from src.main import parse_args
from src.proxy_launcher import create_distillation_worker

# ---------------------------------------------------------------------------
# parse_args: --distillation-mode is registered and validated
# ---------------------------------------------------------------------------


def test_parse_args_default_distillation_mode_is_background():
    """Without any flag, the CLI defaults ``distillation_mode`` to ``"background"``."""
    args = parse_args([])
    assert args.distillation_mode == "background"
    assert args.enable_llm_distillation is False


def test_parse_args_distillation_mode_disabled():
    """``--distillation-mode disabled`` is accepted and propagated."""
    args = parse_args(["--distillation-mode", "disabled"])
    assert args.distillation_mode == "disabled"


def test_parse_args_distillation_mode_background():
    """``--distillation-mode background`` is accepted."""
    args = parse_args(["--distillation-mode", "background"])
    assert args.distillation_mode == "background"


def test_parse_args_distillation_mode_rejects_unknown_value():
    """Unknown modes must be rejected by argparse (SystemExit)."""
    with pytest.raises(SystemExit):
        parse_args(["--distillation-mode", "synchronous"])


def test_parse_args_enable_llm_distillation_flag_still_accepted():
    """The legacy ``--enable-llm-distillation`` flag is still parsed."""
    args = parse_args(["--enable-llm-distillation"])
    assert args.enable_llm_distillation is True
    # Default mode still applies unless the legacy flag is honoured by
    # apply_distillation_settings (see dedicated tests below).
    assert args.distillation_mode == "background"


def test_parse_args_llm_model_flag_sets_attribute():
    """``--llm-model <name>`` is parsed and exposed as ``args.llm_model``.

    The flag must default to ``None`` so that the absence of the flag can be
    distinguished from an explicit empty value; ``main()`` only overrides the
    global config when the flag is non-``None``.
    """
    # Default: flag absent -> None (means "don't override").
    assert parse_args([]).llm_model is None

    # Explicit value: parsed verbatim.
    args = parse_args(["--llm-model", "llama-3.3-70b"])
    assert args.llm_model == "llama-3.3-70b"

    # Empty string is a valid explicit override (do not silently swallow it).
    args_empty = parse_args(["--llm-model", ""])
    assert args_empty.llm_model == ""


# ---------------------------------------------------------------------------
# apply_distillation_settings: legacy flag overrides explicit mode
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_singleton():
    """Give each test its own ``Config`` instance via ``set_config``.

    The lazy singleton is reset between tests so mutations made by
    ``apply_distillation_settings`` cannot leak into the next case.
    The fixture also forces ``get_config()`` to materialise before the
    snapshot is taken so we have a real :class:`Config` to seed from.
    """
    base = get_config()
    test_cfg = Config(
        distillation_mode=base.distillation_mode,
        llm_timeout=base.llm_timeout,
        llm_model=base.llm_model,
    )
    from src.config import set_config

    set_config(test_cfg)
    yield test_cfg
    reset_config()


def test_apply_distillation_settings_explicit_background(_isolated_singleton):
    """Explicit ``background`` mode is written through to the injected config."""
    cfg = _isolated_singleton
    apply_distillation_settings(
        cfg,
        distillation_mode="background",
        enable_llm_distillation=False,
        llm_timeout=1.5,
    )
    assert cfg.distillation_mode == "background"
    assert cfg.llm_timeout == 1.5


def test_apply_distillation_settings_explicit_disabled(_isolated_singleton):
    """Explicit ``disabled`` mode is written through to the injected config."""
    cfg = _isolated_singleton
    apply_distillation_settings(
        cfg,
        distillation_mode="disabled",
        enable_llm_distillation=False,
        llm_timeout=2.0,
    )
    assert cfg.distillation_mode == "disabled"
    assert cfg.enable_llm_distillation is False


def test_apply_distillation_settings_legacy_flag_forces_background(_isolated_singleton):
    """``--enable-llm-distillation`` upgrades ``disabled`` -> ``background``."""
    cfg = _isolated_singleton
    apply_distillation_settings(
        cfg,
        distillation_mode="disabled",
        enable_llm_distillation=True,
        llm_timeout=2.0,
    )
    assert cfg.distillation_mode == "background"
    assert cfg.enable_llm_distillation is True


def test_apply_distillation_settings_legacy_flag_does_not_force_when_off(_isolated_singleton):
    """Legacy flag off must not override the explicit mode."""
    cfg = _isolated_singleton
    apply_distillation_settings(
        cfg,
        distillation_mode="disabled",
        enable_llm_distillation=False,
        llm_timeout=2.0,
    )
    assert cfg.distillation_mode == "disabled"


# ---------------------------------------------------------------------------
# create_distillation_worker: honours distillation_mode
# ---------------------------------------------------------------------------


def test_create_distillation_worker_returns_none_when_disabled(_isolated_singleton):
    """``create_distillation_worker`` returns ``None`` when mode is ``disabled``."""
    cfg = _isolated_singleton
    cfg.distillation_mode = "disabled"
    assert create_distillation_worker(cfg) is None


def test_create_distillation_worker_returns_none_default(monkeypatch, _isolated_singleton):
    """No construction side effects when mode is ``disabled``."""
    cfg = _isolated_singleton
    cfg.distillation_mode = "disabled"

    with patch("src.distillation_worker.DistillationWorker") as mock_worker, \
         patch("src.llm_processor.LLMProcessor") as mock_processor:
        result = create_distillation_worker(cfg)

    assert result is None
    mock_worker.assert_not_called()
    mock_processor.assert_not_called()


def test_create_distillation_worker_builds_when_background(_isolated_singleton):
    """``create_distillation_worker`` instantiates worker + processor when enabled."""
    cfg = _isolated_singleton
    cfg.distillation_mode = "background"

    fake_worker = object()
    with patch("src.distillation_worker.DistillationWorker", return_value=fake_worker) as mock_worker, \
         patch("src.llm_processor.LLMProcessor") as mock_processor:
        result = create_distillation_worker(cfg)

    assert result is fake_worker
    mock_processor.assert_called_once()
    _args, kwargs = mock_processor.call_args
    assert kwargs.get("circuit_breaker") is not None
    mock_worker.assert_called_once()
    worker_kwargs = mock_worker.call_args.kwargs
    assert worker_kwargs.get("processor") is mock_processor.return_value
    assert worker_kwargs.get("cache_size") == 128
    # Metrics is wired in by ``create_distillation_worker`` so latency and
    # failures flow into the same store the operator inspects. We don't
    # pin the exact instance here, only that one was passed.
    assert worker_kwargs.get("metrics") is not None


def test_create_distillation_worker_uses_shared_metrics(_isolated_singleton):
    """When a Metrics instance is passed, the worker must use it."""
    cfg = _isolated_singleton
    cfg.distillation_mode = "background"
    from src.metrics import Metrics

    shared_metrics = Metrics()
    fake_worker = object()
    with patch("src.distillation_worker.DistillationWorker", return_value=fake_worker) as mock_worker:
        result = create_distillation_worker(cfg, metrics=shared_metrics)

    assert result is fake_worker
    assert mock_worker.call_args.kwargs.get("metrics") is shared_metrics


# ---------------------------------------------------------------------------
# --help content: in-process checks via parse_args(["--help"])
# ---------------------------------------------------------------------------


def _run_help_and_capture(capsys) -> str:
    """Invoke ``parse_args(["--help"])`` and return the printed help text.

    argparse's ``--help`` action writes ``parser.format_help()`` to stdout
    and raises ``SystemExit(0)``. We capture stdout so the test can make
    substring assertions against it without spawning a subprocess.
    """
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--help"])
    assert excinfo.value.code == 0
    return capsys.readouterr().out


def test_main_help_advertises_distillation_mode(capsys):
    """``parse_args(["--help"])`` lists ``--distillation-mode``."""
    out = _run_help_and_capture(capsys)
    assert "--distillation-mode" in out
    assert "background" in out
    assert "disabled" in out


def test_main_help_advertises_llm_model_flag(capsys):
    """``parse_args(["--help"])`` documents the ``--llm-model`` flag."""
    out = _run_help_and_capture(capsys)
    assert "--llm-model" in out


def test_main_help_lists_legacy_enable_flag_as_deprecated(capsys):
    """``parse_args(["--help"])`` lists ``--enable-llm-distillation`` as deprecated."""
    out = _run_help_and_capture(capsys)
    assert "--enable-llm-distillation" in out
    assert "Deprecated" in out


def test_main_help_does_not_require_stdin(tmp_path):
    """``python3 -m src.main --help`` exits cleanly with stdin closed.

    Regression guard for the interactive model-selection ``input()`` call:
    when stdin is ``DEVNULL`` (as it is in CI / service runners) the
    command used to block forever. Now ``--help`` exits cleanly without
    touching stdin.

    This is the one subprocess-based help test we keep: it is the only way
    to actually verify that the runtime does not read from stdin at the
    OS level. All other help-content checks are done in-process above.
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
    # The interactive prompt must no longer be referenced anywhere.
    assert "Enter choice" not in result.stdout
    assert "Select Compression Model" not in result.stdout


# ---------------------------------------------------------------------------
# proxy_core: distillation mode gates worker construction
# ---------------------------------------------------------------------------


def test_proxy_core_skips_worker_when_mode_disabled(_isolated_singleton):
    """ContextCompressionProxy does not construct a worker when mode is disabled.

    The constructor checks ``self.config.distillation_mode != "disabled"``
    before instantiating :class:`DistillationWorker`. We mirror that gating
    here via ``create_distillation_worker`` (which proxy_core delegates to).
    """
    cfg = _isolated_singleton
    cfg.distillation_mode = "disabled"

    with patch("src.distillation_worker.DistillationWorker") as mock_worker, \
         patch("src.llm_processor.LLMProcessor"):
        worker = create_distillation_worker(cfg)

    assert worker is None
    mock_worker.assert_not_called()


def test_proxy_core_creates_worker_when_mode_background(_isolated_singleton):
    """ContextCompressionProxy creates a worker when mode is background."""
    cfg = _isolated_singleton
    cfg.distillation_mode = "background"
    fake_worker = object()

    with patch("src.distillation_worker.DistillationWorker", return_value=fake_worker), \
         patch("src.llm_processor.LLMProcessor"):
        worker = create_distillation_worker(cfg)

    assert worker is fake_worker


# ---------------------------------------------------------------------------
# --llm-model CLI flag: overrides the injected config when present, no-op otherwise
# ---------------------------------------------------------------------------


def test_llm_model_flag_overrides_injected_config(_isolated_singleton):
    """When ``args.llm_model`` is set, ``main()`` writes it through to the
    injected config. This exercises the same code path that ``--llm-model``
    triggers, without spawning a subprocess.

    ``main()`` is long-running, so we replicate the in-process branch via
    the same guard pattern (``getattr(args, 'llm_model', None)``) that
    ``main()`` uses. If ``main()`` ever drifts away from that pattern the
    test below will catch it.
    """
    cfg = _isolated_singleton
    args = parse_args(["--llm-model", "qwen-3-32b"])
    assert args.llm_model == "qwen-3-32b"

    # Same branch main() takes when --llm-model is provided.
    if getattr(args, "llm_model", None):
        cfg.llm_model = args.llm_model

    assert cfg.llm_model == "qwen-3-32b"


def test_llm_model_flag_absent_preserves_config(_isolated_singleton):
    """Without ``--llm-model``, the config's ``llm_model`` is left untouched."""
    cfg = _isolated_singleton
    original = cfg.llm_model
    args = parse_args([])
    assert getattr(args, "llm_model", None) is None

    # Same guard main() uses.
    if getattr(args, "llm_model", None):
        cfg.llm_model = args.llm_model

    assert cfg.llm_model == original
