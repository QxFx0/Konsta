import os
import subprocess
import shutil
import logging
import platform
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

class CACertificateGenerator:
    """
    Handles the cryptographic generation of CA certificates and private keys.
    """
    
    def __init__(self, common_name: str = "Konsta Root CA"):
        # Ensure common_name is a string and respects the X.509 length limit (max 64 chars)
        self.common_name = str(common_name)[:64]

    def generate_ca(self, cert_path: Path, key_path: Path, days_valid: int = 3650, password: Optional[str] = None) -> None:
        """
        Generates a self-signed CA certificate and a private key.
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

        # Determine encryption algorithm
        encryption_algorithm = (
            serialization.BestAvailableEncryption(password.encode()) 
            if password else serialization.NoEncryption()
        )

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
        """
        try:
            # Debian/Ubuntu
            if Path("/usr/sbin/update-ca-certificates").exists():
                dest_dir = Path("/usr/local/share/ca-certificates")
                self._run_as_sudo(["mkdir", "-p", str(dest_dir)])
                cert_filename = cert_path.name if cert_path.suffix == ".crt" else f"{cert_path.stem}.crt"
                dest_path = dest_dir / cert_filename
                self._run_as_sudo(["cp", str(cert_path), str(dest_path)])
                self._run_as_sudo(["update-ca-certificates"])
                return True

            # RHEL/CentOS/Fedora/Arch
            if Path("/usr/bin/update-ca-trust").exists():
                dest_dir = Path("/etc/pki/ca-trust/source/anchors")
                self._run_as_sudo(["mkdir", "-p", str(dest_dir)])
                dest_path = dest_dir / cert_path.name
                self._run_as_sudo(["cp", str(cert_path), str(dest_path)])
                self._run_as_sudo(["update-ca-trust"])
                return True

            raise RuntimeError("Supported CA update tool (update-ca-certificates or update-ca-trust) not found on this system.")

        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed during Linux certificate installation: {e.cmd}, error: {e.stderr.decode() if e.stderr else e}")
            return False
        except Exception:
            logger.exception("Unexpected error occurred during Linux certificate installation")
            return False

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
    def __init__(self, common_name: str = "Konsta Root CA"):
        self.generator = CACertificateGenerator(common_name)
        self.installer = CASystemInstaller()

    def setup_ca(self, cert_path: Path, key_path: Path, install: bool = False) -> bool:
        """
        Generates the CA and optionally installs it to the system.
        """
        try:
            self.generator.generate_ca(cert_path, key_path)
            if install:
                return self.install_ca_system_wide(cert_path)
            return True
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
