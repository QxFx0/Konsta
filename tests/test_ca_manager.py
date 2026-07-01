"""Unit tests for src.ca_manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization

from src.ca_manager import (
    CACertificateGenerator,
    CAKeyManager,
    CAManager,
)
from tests.conftest import FakeKeyring

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> str:
    return "KonstaTest"


# ---------------------------------------------------------------------------
# CAKeyManager: passphrase prompt flow and keyring storage/retrieval
# ---------------------------------------------------------------------------


def test_get_or_create_prompts_when_no_passphrase_stored(
    fake_keyring: FakeKeyring,
    service: str,
) -> None:
    """First call should prompt (with confirmation) and store the passphrase."""
    manager = CAKeyManager(service_name=service, username="ca-key")
    prompts: list[str] = []

    def fake_getpass(prompt: str = "") -> str:
        prompts.append(prompt)
        return "sup3r-s3cret" if prompts.index(prompt) == 0 else "sup3r-s3cret"

    with patch("src.ca.crypto.getpass.getpass", side_effect=fake_getpass):
        passphrase = manager.get_or_create_passphrase()

    assert passphrase == "sup3r-s3cret"
    # First prompt for the passphrase, second to confirm it.
    assert len(prompts) == 2
    assert fake_keyring.store[(service, "ca-key")] == "sup3r-s3cret"


def test_get_or_create_returns_existing_passphrase_without_prompting(
    fake_keyring: FakeKeyring,
    service: str,
) -> None:
    """Subsequent calls should reuse the stored passphrase and not prompt."""
    fake_keyring.store[(service, "ca-key")] = "already-here"
    manager = CAKeyManager(service_name=service, username="ca-key")

    with patch("src.ca.crypto.getpass.getpass") as mock_getpass:
        passphrase = manager.get_or_create_passphrase()

    assert passphrase == "already-here"
    mock_getpass.assert_not_called()


def test_prompt_new_passphrase_rejects_mismatch_then_accepts(
    fake_keyring: FakeKeyring,
    service: str,
) -> None:
    """A mismatched confirmation should be retried; the final value is stored."""
    manager = CAKeyManager(service_name=service, username="ca-key")
    responses = iter(["first-attempt", "wrong", "good-pass", "good-pass"])

    with patch("src.ca.crypto.getpass.getpass", side_effect=lambda _prompt="": next(responses)):
        passphrase = manager.prompt_new_passphrase()

    assert passphrase == "good-pass"
    assert fake_keyring.store[(service, "ca-key")] == "good-pass"


def test_prompt_new_passphrase_rejects_empty(
    fake_keyring: FakeKeyring,
    service: str,
) -> None:
    """Empty passphrases should be rejected and the user re-prompted."""
    manager = CAKeyManager(service_name=service, username="ca-key")
    responses = iter(["", "", "valid", "valid"])

    with patch("src.ca.crypto.getpass.getpass", side_effect=lambda _prompt="": next(responses)):
        passphrase = manager.prompt_new_passphrase()

    assert passphrase == "valid"
    assert fake_keyring.store[(service, "ca-key")] == "valid"


def test_delete_passphrase_removes_from_keyring(
    fake_keyring: FakeKeyring,
    service: str,
) -> None:
    fake_keyring.store[(service, "ca-key")] = "to-be-deleted"
    manager = CAKeyManager(service_name=service, username="ca-key")

    manager.delete_passphrase()

    assert (service, "ca-key") not in fake_keyring.store


def test_delete_passphrase_is_safe_when_missing(
    fake_keyring: FakeKeyring,
    service: str,
) -> None:
    """Deleting a passphrase that does not exist must not raise."""
    manager = CAKeyManager(service_name=service, username="ca-key")

    manager.delete_passphrase()  # should not raise

    assert (service, "ca-key") not in fake_keyring.store


# ---------------------------------------------------------------------------
# CACertificateGenerator: encrypted key generation and round-trip
# ---------------------------------------------------------------------------


def test_generate_ca_writes_encrypted_key(
    tmp_path: Path,
) -> None:
    """generate_ca must write a key that is encrypted with the supplied passphrase."""
    cert_path = tmp_path / "ca.pem"
    key_path = tmp_path / "ca.key"
    password = "encryption-test"

    CACertificateGenerator().generate_ca(cert_path, key_path, password=password)

    # The on-disk file must contain PEM encryption headers (legacy OpenSSL
    # format used by BestAvailableEncryption on PKCS#1 RSA keys).
    raw_key = key_path.read_bytes()
    assert b"Proc-Type: 4,ENCRYPTED" in raw_key
    assert b"DEK-Info:" in raw_key

    # Loading without a password should fail.
    with pytest.raises(TypeError):
        serialization.load_pem_private_key(raw_key, password=None)

    # Loading with the correct password should succeed.
    loaded = serialization.load_pem_private_key(raw_key, password=password.encode())
    assert loaded is not None

    # Permissions should be 0600.
    mode = key_path.stat().st_mode & 0o777
    assert mode == 0o600


def test_generate_ca_then_decrypt_round_trip(tmp_path: Path) -> None:
    """A generated key must be decryptable and produce the matching public key."""
    cert_path = tmp_path / "ca.pem"
    key_path = tmp_path / "ca.key"
    password = "round-trip-pass"

    generator = CACertificateGenerator(common_name="RoundTrip CA")
    generator.generate_ca(cert_path, key_path, password=password)

    cert_pem = cert_path.read_bytes()
    assert b"BEGIN CERTIFICATE" in cert_pem

    loaded_key = serialization.load_pem_private_key(
        key_path.read_bytes(),
        password=password.encode(),
    )
    assert loaded_key.key_size == 4096


# ---------------------------------------------------------------------------
# CAManager: keyring integration, decryption verification, error on mismatch
# ---------------------------------------------------------------------------


def test_camanager_setup_ca_uses_keyring_passphrase(
    tmp_path: Path,
    fake_keyring: FakeKeyring,
) -> None:
    """When no password is provided, setup_ca must source it from the keyring."""
    service = "KonstaManager"
    fake_keyring.store[(service, "ca-key")] = "from-keyring"
    manager = CAManager(key_manager=CAKeyManager(service_name=service, username="ca-key"))

    cert_path = tmp_path / "ca.pem"
    key_path = tmp_path / "ca.key"

    ok = manager.setup_ca(cert_path, key_path)
    assert ok is True
    assert cert_path.exists()
    assert key_path.exists()

    # The freshly-written key must decrypt with the keyring passphrase.
    serialization.load_pem_private_key(
        key_path.read_bytes(),
        password="from-keyring".encode(),
    )


def test_camanager_setup_ca_prompts_when_keyring_empty(
    tmp_path: Path,
    fake_keyring: FakeKeyring,
) -> None:
    """With nothing in the keyring, setup_ca must prompt and then store."""
    service = "KonstaPrompt"
    manager = CAManager(key_manager=CAKeyManager(service_name=service, username="ca-key"))

    cert_path = tmp_path / "ca.pem"
    key_path = tmp_path / "ca.key"

    responses = iter(["prompted-pass", "prompted-pass"])
    with patch("src.ca.crypto.getpass.getpass", side_effect=lambda _prompt="": next(responses)):
        ok = manager.setup_ca(cert_path, key_path)

    assert ok is True
    assert fake_keyring.store[(service, "ca-key")] == "prompted-pass"
    serialization.load_pem_private_key(
        key_path.read_bytes(),
        password="prompted-pass".encode(),
    )


def test_camanager_setup_ca_accepts_explicit_password(
    tmp_path: Path,
    fake_keyring: FakeKeyring,
) -> None:
    """An explicit password should bypass the keyring entirely."""
    service = "KonstaExplicit"
    fake_keyring.store[(service, "ca-key")] = "should-not-be-used"
    manager = CAManager(key_manager=CAKeyManager(service_name=service, username="ca-key"))

    cert_path = tmp_path / "ca.pem"
    key_path = tmp_path / "ca.key"

    ok = manager.setup_ca(cert_path, key_path, password="explicit-pass")
    assert ok is True

    # Key must decrypt with the explicit passphrase, not the keyring value.
    serialization.load_pem_private_key(
        key_path.read_bytes(),
        password="explicit-pass".encode(),
    )
    with pytest.raises(ValueError):
        serialization.load_pem_private_key(
            key_path.read_bytes(),
            password="should-not-be-used".encode(),
        )


def test_camanager_setup_ca_raises_on_wrong_passphrase(
    tmp_path: Path,
    fake_keyring: FakeKeyring,
) -> None:
    """If the on-disk key cannot be decrypted with the keyring passphrase, fail."""
    service = "KonstaWrong"
    generator = CACertificateGenerator()
    cert_path = tmp_path / "ca.pem"
    key_path = tmp_path / "ca.key"

    # Pre-populate the filesystem with a key encrypted under a *different* passphrase.
    generator.generate_ca(cert_path, key_path, password="original-pass")
    # Store a *different* passphrase in the keyring.
    fake_keyring.store[(service, "ca-key")] = "wrong-pass"

    manager = CAManager(key_manager=CAKeyManager(service_name=service, username="ca-key"))

    with pytest.raises(RuntimeError, match="cannot be decrypted"):
        manager.setup_ca(cert_path, key_path)


def test_camanager_setup_ca_succeeds_when_existing_key_decrypts(
    tmp_path: Path,
    fake_keyring: FakeKeyring,
) -> None:
    """An existing key that matches the keyring passphrase must be accepted."""
    service = "KonstaExisting"
    generator = CACertificateGenerator()
    cert_path = tmp_path / "ca.pem"
    key_path = tmp_path / "ca.key"

    generator.generate_ca(cert_path, key_path, password="matching-pass")
    fake_keyring.store[(service, "ca-key")] = "matching-pass"

    manager = CAManager(key_manager=CAKeyManager(service_name=service, username="ca-key"))
    assert manager.setup_ca(cert_path, key_path) is True
