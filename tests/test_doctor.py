"""Unit tests for the ``doctor`` CLI subcommand in ``src.main``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.doctor import (
    _check_ca_files,
    _check_ca_passphrase,
    _check_embedding_engine,
    _check_llm_api_key,
    _check_llm_endpoint,
    _check_metrics,
    _format_doctor_report,
    _run_doctor,
    _run_doctor_command,
)
from src.main import parse_args

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@dataclass
class MockConfig:
    """Minimal stand-in for :class:`src.config.Config` used by doctor tests.

    The real ``Config.__post_init__`` validates that ``llm_api_key`` is set,
    which is exactly what a few doctor tests want to *bypass* (e.g. "what
    does doctor report when the key is missing?"). Using a dataclass with the
    same attribute names lets the same check helpers run unchanged while
    skipping validation.

    Only the attributes accessed by ``src.main`` doctor checks are declared
    explicitly; new attributes must be added here if future checks need them.
    """

    llm_api_key: str = ""
    llm_endpoint: str = ""
    ca_cert_path: str = ""
    ca_key_path: str = ""

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ok_config(tmp_path: Path) -> Config:
    """A Config where every check except LLM endpoint / passphrase passes."""
    cert_path = tmp_path / "konsta-ca-cert.pem"
    key_path = tmp_path / "konsta-ca-key.pem"
    cert_path.write_text("dummy-cert")
    key_path.write_text("dummy-key")
    cfg = Config(
        llm_api_key="dummy-key",
        llm_endpoint="https://example.invalid/v1/chat/completions",
        ca_cert_path=str(cert_path),
        ca_key_path=str(key_path),
    )
    return cfg


@pytest.fixture(autouse=True)
def _isolate_keyring(monkeypatch):
    """Ensure the OS keyring is never touched by the passphrase check."""
    fake = {"store": {}}

    def fake_get(service, username):
        return fake["store"].get((service, username))

    def fake_set(service, username, value):
        fake["store"][(service, username)] = value

    monkeypatch.setattr("keyring.get_password", fake_get)
    monkeypatch.setattr("keyring.set_password", fake_set)
    return fake


# ---------------------------------------------------------------------------
# parse_args wiring
# ---------------------------------------------------------------------------


def test_parse_args_doctor_subcommand_recognised():
    args = parse_args(["doctor"])
    assert args.command == "doctor"
    assert args.doctor_json is False


def test_parse_args_doctor_json_flag():
    args = parse_args(["doctor", "--json"])
    assert args.command == "doctor"
    assert args.doctor_json is True


def test_parse_args_no_command_defaults_to_none():
    args = parse_args([])
    assert args.command is None


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------


def _cfg_no_ca_env(tmp_path: Path, monkeypatch, **kwargs) -> Config:
    """Build a Config with env CA paths cleared so constructor values are used."""
    monkeypatch.delenv("CA_CERT_PATH", raising=False)
    monkeypatch.delenv("CA_KEY_PATH", raising=False)
    return Config(
        llm_api_key=kwargs.get("llm_api_key", "x"),
        ca_cert_path=kwargs.get("ca_cert_path", str(tmp_path / "cert.pem")),
        ca_key_path=kwargs.get("ca_key_path", str(tmp_path / "key.pem")),
    )


def test_check_ca_files_ok(tmp_path: Path, monkeypatch):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("c")
    key.write_text("k")
    cfg = _cfg_no_ca_env(tmp_path, monkeypatch, ca_cert_path=str(cert), ca_key_path=str(key))
    result = _check_ca_files(cfg)
    assert result["status"] == "ok"
    assert "present" in result["message"].lower()


def test_check_ca_files_missing_warns(tmp_path: Path, monkeypatch):
    cfg = _cfg_no_ca_env(
        tmp_path,
        monkeypatch,
        ca_cert_path=str(tmp_path / "missing-cert.pem"),
        ca_key_path=str(tmp_path / "missing-key.pem"),
    )
    result = _check_ca_files(cfg)
    assert result["status"] == "warn"
    assert "missing" in result["message"].lower()


def test_check_ca_files_only_cert_missing(tmp_path: Path, monkeypatch):
    key = tmp_path / "key.pem"
    key.write_text("k")
    cfg = _cfg_no_ca_env(
        tmp_path,
        monkeypatch,
        ca_cert_path=str(tmp_path / "absent-cert.pem"),
        ca_key_path=str(key),
    )
    result = _check_ca_files(cfg)
    assert result["status"] == "warn"
    assert "cert" in result["message"].lower()


def test_check_ca_passphrase_env_var_ok(monkeypatch):
    monkeypatch.setenv("CA_KEY_PASSWORD", "supersecret")
    result = _check_ca_passphrase()
    assert result["status"] == "ok"
    assert "CA_KEY_PASSWORD" in result["message"]


def test_check_ca_passphrase_keyring_ok(monkeypatch, _isolate_keyring):
    monkeypatch.delenv("CA_KEY_PASSWORD", raising=False)
    _isolate_keyring["store"][("Konsta", "ca-key")] = "from-keyring"
    result = _check_ca_passphrase()
    assert result["status"] == "ok"
    assert "keyring" in result["message"].lower()


def test_check_ca_passphrase_missing_warns(monkeypatch, _isolate_keyring):
    monkeypatch.delenv("CA_KEY_PASSWORD", raising=False)
    _isolate_keyring["store"].clear()
    result = _check_ca_passphrase()
    assert result["status"] == "warn"
    assert "CA_KEY_PASSWORD" in result["message"]


def test_check_llm_api_key_present_ok():
    cfg = Config(llm_api_key="present")
    result = _check_llm_api_key(cfg)
    assert result["status"] == "ok"
    assert "length" in result["message"].lower()


def test_check_llm_api_key_missing_fails(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    cfg = MockConfig(llm_api_key="")
    result = _check_llm_api_key(cfg)
    assert result["status"] == "fail"
    assert "LLM_API_KEY" in result["message"]


def test_check_llm_endpoint_reachable_ok():
    cfg = Config(llm_api_key="x", llm_endpoint="https://example.invalid/v1")

    fake_response = SimpleNamespace(status_code=200)
    fake_client = SimpleNamespace(
        head=MagicMock(return_value=fake_response),
        stream=MagicMock(),
    )
    result = _check_llm_endpoint(cfg, timeout=1.0, httpx_client=fake_client)

    assert result["status"] == "ok"
    assert "reachable" in result["message"].lower()
    fake_client.head.assert_called_once()


def test_check_llm_endpoint_unreachable_warns():
    cfg = Config(llm_api_key="x", llm_endpoint="https://example.invalid/v1")

    class _FakeClient:
        def head(self, *args, **kwargs):
            raise Exception("boom")

        def stream(self, *args, **kwargs):
            raise Exception("also boom")

    result = _check_llm_endpoint(cfg, timeout=1.0, httpx_client=_FakeClient())

    assert result["status"] == "warn"
    assert "unreachable" in result["message"].lower()


def test_check_llm_endpoint_head_fails_falls_back_to_stream():
    """HEAD raising an error should fall back to a streaming GET."""
    cfg = Config(llm_api_key="x", llm_endpoint="https://example.invalid/v1")

    class _FakeStream:
        def __init__(self, status_code):
            self.status_code = status_code

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeClient:
        def head(self, *args, **kwargs):
            raise Exception("method not allowed")

        def stream(self, *args, **kwargs):
            assert args[0] == "GET"
            return _FakeStream(405)

    result = _check_llm_endpoint(cfg, timeout=1.0, httpx_client=_FakeClient())

    assert result["status"] == "ok"
    assert "405" in result["message"]


def test_check_embedding_engine_ok():
    result = _check_embedding_engine()
    # sentence-transformers may or may not be installed in CI; either result
    # is acceptable as long as it doesn't crash.
    assert result["status"] in {"ok", "warn"}
    assert "embedding" in result["name"]


def test_check_metrics_returns_snapshot():
    # Install a shared Metrics instance so the doctor check observes an
    # installed store and returns an ``ok`` snapshot.
    from src.metrics import Metrics, reset_shared_metrics, set_shared_metrics

    reset_shared_metrics()
    set_shared_metrics(Metrics())
    try:
        result = _check_metrics()
        assert result["status"] == "ok"
        assert "snapshot" in result["details"]
    finally:
        reset_shared_metrics()


# ---------------------------------------------------------------------------
# _run_doctor aggregate behaviour
# ---------------------------------------------------------------------------


def test_run_doctor_all_ok(ok_config):
    with patch(
        "src.doctor._check_llm_endpoint",
        return_value={"name": "llm_endpoint", "status": "ok", "message": "ok"},
    ):
        report = _run_doctor(ok_config)

    assert report["exit_code"] == 0
    assert report["summary"]["fail"] == 0
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["ca_files"] == "ok"
    assert statuses["llm_api_key"] == "ok"
    assert statuses["llm_endpoint"] == "ok"


def test_run_doctor_missing_ca_files_warns(tmp_path, ok_config):
    ok_config.ca_cert_path = str(tmp_path / "nope-cert.pem")
    ok_config.ca_key_path = str(tmp_path / "nope-key.pem")
    with patch(
        "src.doctor._check_llm_endpoint",
        return_value={"name": "llm_endpoint", "status": "ok", "message": "ok"},
    ):
        report = _run_doctor(ok_config)

    assert report["exit_code"] == 0  # warnings don't fail doctor
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["ca_files"] == "warn"


def test_run_doctor_missing_llm_key_exits_1(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.delenv("CA_CERT_PATH", raising=False)
    monkeypatch.delenv("CA_KEY_PATH", raising=False)
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("c")
    key.write_text("k")
    cfg = MockConfig(
        llm_api_key="",
        llm_endpoint="https://example.invalid/v1",
        ca_cert_path=str(cert),
        ca_key_path=str(key),
    )

    with patch(
        "src.doctor._check_llm_endpoint",
        return_value={"name": "llm_endpoint", "status": "ok", "message": "ok"},
    ):
        report = _run_doctor(cfg)

    assert report["exit_code"] == 1
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["llm_api_key"] == "fail"


def test_format_doctor_report_contains_each_check():
    report = {
        "checks": [
            {"name": "ca_files", "status": "ok", "message": "cert ok"},
            {"name": "llm_api_key", "status": "fail", "message": "missing"},
        ],
        "summary": {"ok": 1, "warn": 0, "fail": 1},
    }
    text = _format_doctor_report(report)
    assert "Konsta Doctor Report" in text
    assert "ca_files" in text
    assert "llm_api_key" in text
    assert "[OK]" in text
    assert "[FAIL]" in text
    assert "Summary" in text


def test_run_doctor_command_prints_text_and_returns_code(ok_config, capsys):
    args = SimpleNamespace(doctor_json=False)
    with patch(
        "src.doctor._check_llm_endpoint",
        return_value={"name": "llm_endpoint", "status": "ok", "message": "ok"},
    ):
        rc = _run_doctor_command(args, ok_config)

    assert rc == 0
    out = capsys.readouterr().out
    assert "Konsta Doctor Report" in out


def test_run_doctor_command_json_output(ok_config, capsys):
    args = SimpleNamespace(doctor_json=True)
    with patch(
        "src.doctor._check_llm_endpoint",
        return_value={"name": "llm_endpoint", "status": "ok", "message": "ok"},
    ):
        rc = _run_doctor_command(args, ok_config)

    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "checks" in parsed
    assert "summary" in parsed
    assert parsed["exit_code"] == 0


def test_run_doctor_command_exit_1_on_failure(capsys, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    args = SimpleNamespace(doctor_json=False)
    cfg = MockConfig(llm_api_key="")
    rc = _run_doctor_command(args, cfg)
    assert rc == 1
    out = capsys.readouterr().out
    assert "[FAIL]" in out


# ---------------------------------------------------------------------------
# End-to-end: python3 -m src.main doctor exits 0 when config is OK
# ---------------------------------------------------------------------------


def _doctor_subprocess_env(tmp_path: Path) -> dict:
    project_root = Path(__file__).resolve().parent.parent
    cert = tmp_path / "konsta-ca-cert.pem"
    key = tmp_path / "konsta-ca-key.pem"
    cert.write_text("dummy-cert")
    key.write_text("dummy-key")
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(project_root),
        "LLM_API_KEY": "dummy-key-for-doctor",
        "CA_KEY_PASSWORD": "dummy-pass",
        "CA_CERT_PATH": str(cert),
        "CA_KEY_PATH": str(key),
        "HOME": str(tmp_path),
        "LLM_ENDPOINT": "https://example.invalid/v1/chat/completions",
    }


def test_main_doctor_subcommand_exits_zero(tmp_path):
    """``python3 -m src.main doctor`` exits 0 when config + reachability OK."""
    env = _doctor_subprocess_env(tmp_path)
    project_root = Path(__file__).resolve().parent.parent

    # We don't patch httpx in the subprocess; an unreachable endpoint
    # produces a warning, not a failure, so the process should still exit 0
    # as long as CA + LLM_API_KEY + CA_KEY_PASSWORD are present.
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "doctor"],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        cwd=str(project_root),
        timeout=30,
    )

    assert result.returncode == 0, (
        f"doctor exited {result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "Konsta Doctor Report" in result.stdout


def test_main_doctor_missing_llm_key_exits_1(tmp_path):
    """Missing LLM_API_KEY must propagate as exit 1 from the CLI."""
    env = _doctor_subprocess_env(tmp_path)
    env.pop("LLM_API_KEY", None)
    project_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [sys.executable, "-m", "src.main", "doctor"],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        cwd=str(project_root),
        timeout=30,
    )

    # Config._validate raises before doctor runs if LLM_API_KEY missing,
    # so the exit code may be 1 from either ValueError or doctor fail.
    assert result.returncode == 1, (
        f"expected exit 1, got {result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
