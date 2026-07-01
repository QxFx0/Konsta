"""CA manager facade.

This module historically owned every CA-related concern: key management,
certificate generation, OS trust-store installation, and a top-level
:class:`CAManager` facade. The cryptographic primitives were extracted into
:mod:`src.ca.crypto` so that :mod:`src.config` can depend on the leaf-level
:class:`~src.ca.crypto.CAKeyManager` without pulling in :class:`CAManager` or
the system installer (and, transitively, anything that the application core
loads on startup).

To keep existing imports (``from src.ca_manager import CAKeyManager``,
``CAManager``, etc.) working unchanged we re-export the moved classes here.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization

from src.ca.crypto import CACertificateGenerator, CAKeyManager

logger = logging.getLogger(__name__)

__all__ = [
    "CAKeyManager",
    "CACertificateGenerator",
    "CASystemInstaller",
    "CAManager",
]


class CASystemInstaller:
    """
    Handles the OS-level installation of CA certificates into the system trust store.
    """

    def install_certificate(self, cert_path: Path) -> bool:
        """
        Installs the certificate into the system trust store based on the OS.
        """
        system = platform.system().lower()

        if system == 'linux':
            return self._install_linux(cert_path)
        elif system == 'darwin':
            return self._install_macos(cert_path)
        elif system == 'windows':
            return self._install_windows(cert_path)
        else:
            logger.error(f"Unsupported operating system: {system}")
            return False

    def _run_as_sudo(self, command: list[str]) -> subprocess.CompletedProcess:
        """
        Runs a command using sudo to obtain root privileges on POSIX systems.
        """
        if shutil.which("sudo") is None:
            if os.name == 'posix' and os.geteuid() == 0:
                return subprocess.run(command, check=True, capture_output=True)
            else:
                raise RuntimeError("Sudo is not installed and current user is not root. Cannot install certificate.")

        if os.name == 'posix' and os.geteuid() == 0:
            return subprocess.run(command, check=True, capture_output=True)

        return subprocess.run(["sudo"] + command, check=True, capture_output=True)

    def _install_linux(self, cert_path: Path) -> bool:
        """
        Linux-specific installation logic supporting multiple distributions.

        Distribution family is determined (when possible) from
        ``/etc/os-release`` by inspecting ``ID`` and ``ID_LIKE`` so the
        correct CA update tool is selected without having to probe every
        package that might be installed. If ``/etc/os-release`` is
        unavailable or does not name a recognised family the code falls
        back to the original file-existence probes.
        """
        try:
            family = self._detect_linux_family()
            logger.debug("Detected Linux family for CA install: %s", family)

            if family == "debian":
                return self._install_linux_debian(cert_path)
            if family == "rhel":
                return self._install_linux_rhel(cert_path)

            # Fallback: probe the filesystem when /etc/os-release was
            # missing or did not name a known family.
            if Path("/usr/sbin/update-ca-certificates").exists():
                return self._install_linux_debian(cert_path)
            if Path("/usr/bin/update-ca-trust").exists():
                return self._install_linux_rhel(cert_path)

            raise RuntimeError(
                "Supported CA update tool (update-ca-certificates or "
                "update-ca-trust) not found on this system."
            )

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Command failed during Linux certificate installation: "
                f"{e.cmd}, error: {e.stderr.decode() if e.stderr else e}"
            )
            return False
        except Exception:
            logger.exception("Unexpected error occurred during Linux certificate installation")
            return False

    @staticmethod
    def _detect_linux_family() -> Optional[str]:
        """Return ``"debian"``, ``"rhel"``, or ``None`` for an unknown family.

        Reads ``/etc/os-release`` (the standard os-release spec field
        ``ID`` and the optional ``ID_LIKE``). Returns ``None`` when the
        file is missing or no recognised identifier is found so callers
        can fall back to filesystem-based detection.
        """
        os_release = Path("/etc/os-release")
        if not os_release.is_file():
            return None

        ids: set[str] = set()
        try:
            for line in os_release.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or "=" not in line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                if key != "ID" and key != "ID_LIKE":
                    continue
                # Values are quoted per the os-release spec.
                value = value.strip().strip('"').strip("'")
                for entry in value.split():
                    ids.add(entry.lower())
        except OSError as exc:
            logger.debug("Failed to read /etc/os-release: %s", exc)
            return None

        if not ids:
            return None

        # Debian-family: Debian, Ubuntu, Mint, Pop!_OS, elementary OS,
        # Kali, Raspbian, etc.
        debian_ids = {
            "debian", "ubuntu", "linuxmint", "pop", "elementary",
            "kali", "raspbian", "deepin", "zorin",
        }
        # RHEL-family: RHEL, CentOS, Fedora, Rocky, AlmaLinux, Nobara,
        # openSUSE (uses update-ca-trust on Tumbleweed/Leap), Arch,
        # Manjaro, etc.
        rhel_ids = {
            "rhel", "centos", "fedora", "rocky", "almalinux", "ol",
            "nobara", "opensuse", "sles", "arch", "manjaro",
            "arcolinux", "endeavouros",
        }

        if ids & debian_ids:
            return "debian"
        if ids & rhel_ids:
            return "rhel"
        return None

    def _install_linux_debian(self, cert_path: Path) -> bool:
        """Install ``cert_path`` into the Debian/Ubuntu trust store."""
        dest_dir = Path("/usr/local/share/ca-certificates")
        self._run_as_sudo(["mkdir", "-p", str(dest_dir)])
        cert_filename = (
            cert_path.name if cert_path.suffix == ".crt" else f"{cert_path.stem}.crt"
        )
        dest_path = dest_dir / cert_filename
        self._run_as_sudo(["cp", str(cert_path), str(dest_path)])
        self._run_as_sudo(["update-ca-certificates"])
        return True

    def _install_linux_rhel(self, cert_path: Path) -> bool:
        """Install ``cert_path`` into the RHEL/CentOS/Fedora/Arch trust store."""
        dest_dir = Path("/etc/pki/ca-trust/source/anchors")
        self._run_as_sudo(["mkdir", "-p", str(dest_dir)])
        dest_path = dest_dir / cert_path.name
        self._run_as_sudo(["cp", str(cert_path), str(dest_path)])
        self._run_as_sudo(["update-ca-trust"])
        return True

    def _install_macos(self, cert_path: Path) -> bool:
        """
        macOS-specific installation logic using the security tool.
        """
        try:
            # security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain <cert_path>
            # -d: add to admin trust settings
            # -r trustRoot: trust as a root CA
            # -k: specify the keychain
            command = ["security", "add-trusted-cert", "-d", "-r", "trustRoot", "-k", "/Library/Keychains/System.keychain", str(cert_path)]
            self._run_as_sudo(command)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed during macOS certificate installation: {e.cmd}, error: {e.stderr.decode() if e.stderr else e}")
            return False
        except Exception:
            logger.exception("Unexpected error occurred during macOS certificate installation")
            return False

    def _install_windows(self, cert_path: Path) -> bool:
        """
        Windows-specific installation logic using certutil.
        """
        try:
            # certutil -addstore Root <cert_path>
            # Root is the Trusted Root Certification Authorities store
            command = ["certutil", "-addstore", "Root", str(cert_path)]
            # On Windows, we don't have sudo. We assume the process is running as Admin or certutil will trigger UAC.
            subprocess.run(command, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed during Windows certificate installation: {e.cmd}, error: {e.stderr.decode() if e.stderr else e}")
            return False
        except Exception:
            logger.exception("Unexpected error occurred during Windows certificate installation")
            return False


class CAManager:
    """
    Facade that coordinates certificate generation and installation.
    """
    def __init__(self, common_name: str = "Konsta Root CA", key_manager: Optional[CAKeyManager] = None):
        self.generator = CACertificateGenerator(common_name)
        self.installer = CASystemInstaller()
        self.key_manager = key_manager or CAKeyManager()

    def setup_ca(self, cert_path: Path, key_path: Path, install: bool = False, password: Optional[str] = None) -> bool:
        """
        Generates the CA and optionally installs it to the system.

        The CA private key is encrypted with a passphrase sourced from the
        keyring (via ``CAKeyManager``). If ``password`` is provided it is used
        directly; otherwise the keyring is consulted. A new passphrase is
        prompted for if none is stored yet.

        If an existing key is found on disk we verify it can be decrypted with
        the supplied passphrase; otherwise a clear error is raised.
        """
        if password is None:
            password = self.key_manager.get_or_create_passphrase()

        try:
            if key_path.exists():
                # Verify the existing key can be loaded with the supplied passphrase.
                try:
                    serialization.load_pem_private_key(
                        key_path.read_bytes(),
                        password=password.encode(),
                    )
                except (ValueError, TypeError) as e:
                    raise RuntimeError(
                        "Existing CA private key cannot be decrypted with the "
                        "passphrase from the keyring. Update or delete the stored "
                        "passphrase before re-running setup."
                    ) from e
            else:
                self.generator.generate_ca(cert_path, key_path, password=password)

            if install:
                return self.install_ca_system_wide(cert_path)
            return True
        except RuntimeError:
            # Bubble up the clear "wrong passphrase" error to the caller.
            raise
        except Exception:
            logger.exception("Failed to setup CA")
            return False

    def install_ca_system_wide(self, cert_path: Path) -> bool:
        """
        Explicitly installs the CA certificate to the system trust store.
        """
        try:
            return self.installer.install_certificate(cert_path)
        except Exception:
            logger.exception("Failed to install CA system-wide")
            return False
