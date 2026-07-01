"""Cryptographic primitives for the Konsta CA.

This module owns the parts of the CA pipeline that depend only on
``keyring`` and ``cryptography``:

* :class:`CAKeyManager` -- passphrase storage in the OS keyring.
* :class:`CACertificateGenerator` -- private-key + self-signed certificate
  generation.

It is deliberately leaf-level: it does **not** import from :mod:`src.config`,
:mod:`src.llm_processor`, or any other Konsta application module. That
isolation is what allows :mod:`src.config` to import :class:`CAKeyManager`
without dragging the rest of the CA / LLM pipeline along (which used to
cause a circular import in :mod:`src.llm_processor`).
"""

from __future__ import annotations

import getpass
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import keyring
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)


class CAKeyManager:
    """
    Manages the passphrase that protects the Konsta CA private key.

    The passphrase is stored in the OS-level keyring (via the `keyring`
    library) so the user does not have to retype it every time Konsta starts.
    """

    def __init__(self, service_name: str = "Konsta", username: str = "ca-key") -> None:
        self.service_name = service_name
        self.username = username

    def get_or_create_passphrase(self) -> str:
        """
        Return the stored passphrase, or prompt the user to create one if
        nothing is stored yet.

        The newly created passphrase is written to the keyring so subsequent
        calls do not require user interaction.
        """
        existing = keyring.get_password(self.service_name, self.username)
        if existing:
            return existing
        return self.prompt_new_passphrase()

    def prompt_new_passphrase(self) -> str:
        """
        Prompt the user (with confirmation) for a new passphrase and store it
        in the keyring.
        """
        prompt = f"Enter passphrase for {self.service_name} CA key: "
        while True:
            passphrase = getpass.getpass(prompt)
            confirm = getpass.getpass("Confirm passphrase: ")
            if passphrase != confirm:
                print("Passphrases do not match. Please try again.")
                continue
            if not passphrase:
                print("Passphrase must not be empty.")
                continue
            break
        keyring.set_password(self.service_name, self.username, passphrase)
        return passphrase

    def delete_passphrase(self) -> None:
        """
        Remove the stored passphrase from the keyring, if any.
        """
        try:
            keyring.delete_password(self.service_name, self.username)
        except keyring.errors.PasswordDeleteError:
            # Nothing to delete; treat as success.
            logger.debug("No passphrase stored in keyring to delete.")


class CACertificateGenerator:
    """
    Handles the cryptographic generation of CA certificates and private keys.
    """

    def __init__(self, common_name: str = "Konsta Root CA"):
        # Ensure common_name is a string and respects the X.509 length limit (max 64 chars)
        self.common_name = str(common_name)[:64]

    def generate_ca(self, cert_path: Path, key_path: Path, password: str, days_valid: int = 3650) -> None:
        """
        Generates a self-signed CA certificate and a private key.

        The private key is always written encrypted with `password`. The caller
        is responsible for sourcing the passphrase (typically from
        `CAKeyManager`).
        """
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )

        # Create CA certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, self.common_name),
        ])

        now = datetime.now(timezone.utc)
        cert = x509.CertificateBuilder()
        cert = cert.subject_name(subject)
        cert = cert.issuer_name(issuer)
        cert = cert.public_key(private_key.public_key())
        cert = cert.serial_number(x509.random_serial_number())
        cert = cert.not_valid_before(now)
        cert = cert.not_valid_after(now + timedelta(days=days_valid))

        # Basic constraints for CA
        cert = cert.add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True,
        )

        ca_cert = cert.sign(private_key, hashes.SHA256())

        # Always encrypt the private key with the supplied passphrase.
        encryption_algorithm = serialization.BestAvailableEncryption(password.encode())

        # Write private key with restricted permissions (chmod 600)
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=encryption_algorithm,
        )

        # Ensure the parent directory exists
        key_path.parent.mkdir(parents=True, exist_ok=True)

        # Use os.open to ensure the file is created with 0600 permissions immediately
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(key_bytes)

        # Write certificate
        with open(cert_path, 'wb') as f:
            f.write(ca_cert.public_bytes(serialization.Encoding.PEM))


__all__ = ["CAKeyManager", "CACertificateGenerator"]
