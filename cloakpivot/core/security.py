"""Security utilities for CloakMap cryptographic operations."""

import base64
import hashlib
import hmac
import json
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

if TYPE_CHECKING:
    from .cloakmap import CloakMap

# Constants for cryptographic operations
DEFAULT_HMAC_ALGORITHM = "sha256"
DEFAULT_PBKDF2_ITERATIONS = 100000
DEFAULT_SALT_LENGTH = 32
DEFAULT_ENCRYPTION_ALGORITHM = "AES-GCM-256"
DEFAULT_NONCE_LENGTH = 12  # 96 bits for AES-GCM
DEFAULT_KEY_LENGTH = 32  # 256 bits for AES-256
SUPPORTED_HMAC_ALGORITHMS = {"sha256", "sha512", "sha384"}
SUPPORTED_ENCRYPTION_ALGORITHMS = {"AES-GCM-256", "AES-GCM-192", "AES-GCM-128"}


@dataclass(frozen=True)
class SecurityConfig:
    """Configuration for security operations."""

    hmac_algorithm: str = DEFAULT_HMAC_ALGORITHM
    pbkdf2_iterations: int = DEFAULT_PBKDF2_ITERATIONS
    salt_length: int = DEFAULT_SALT_LENGTH
    encryption_algorithm: str = DEFAULT_ENCRYPTION_ALGORITHM
    nonce_length: int = DEFAULT_NONCE_LENGTH
    key_length: int = DEFAULT_KEY_LENGTH
    key_derivation_enabled: bool = True
    constant_time_verification: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.hmac_algorithm not in SUPPORTED_HMAC_ALGORITHMS:
            raise ValueError(f"Unsupported HMAC algorithm: {self.hmac_algorithm}")

        if self.encryption_algorithm not in SUPPORTED_ENCRYPTION_ALGORITHMS:
            raise ValueError(
                f"Unsupported encryption algorithm: {self.encryption_algorithm}"
            )

        if self.pbkdf2_iterations < 10000:
            raise ValueError("PBKDF2 iterations must be at least 10,000 for security")

        if self.salt_length < 16:
            raise ValueError("Salt length must be at least 16 bytes")

        if self.nonce_length < 12:
            raise ValueError("Nonce length must be at least 12 bytes for AES-GCM")

        if self.key_length not in {16, 24, 32}:
            raise ValueError("Key length must be 16, 24, or 32 bytes for AES")


class KeyManager(ABC):
    """Abstract base class for key management implementations."""

    @abstractmethod
    def get_key(self, key_id: str, version: Optional[str] = None) -> bytes:
        """
        Retrieve a key by ID and optional version.

        Args:
            key_id: Unique identifier for the key
            version: Optional key version (defaults to latest)

        Returns:
            The raw key bytes

        Raises:
            KeyError: If key is not found
            ValueError: If key format is invalid
        """
        pass

    @abstractmethod
    def list_keys(self) -> list[str]:
        """
        List all available key IDs.

        Returns:
            List of key IDs
        """
        pass

    def get_encryption_key(self, key_id: str, version: Optional[str] = None) -> bytes:
        """
        Retrieve an encryption key by ID and optional version.

        Default implementation delegates to get_key() but can be overridden
        for different encryption key handling.

        Args:
            key_id: Unique identifier for the encryption key
            version: Optional key version (defaults to latest)

        Returns:
            The raw encryption key bytes

        Raises:
            KeyError: If key is not found
            ValueError: If key format is invalid
        """
        return self.get_key(key_id, version)

    def derive_key(
        self, base_key: bytes, salt: bytes, info: str, config: SecurityConfig
    ) -> bytes:
        """
        Derive a key using PBKDF2.

        Args:
            base_key: Base key material
            salt: Salt for key derivation
            info: Context information for derivation
            config: Security configuration

        Returns:
            Derived key bytes (32 bytes)
        """
        if not config.key_derivation_enabled:
            return base_key

        # Use PBKDF2 for key derivation
        derived_key = hashlib.pbkdf2_hmac(
            config.hmac_algorithm,
            base_key + info.encode("utf-8"),
            salt,
            config.pbkdf2_iterations,
            dklen=32,  # 256-bit key
        )

        return derived_key


class EnvironmentKeyManager(KeyManager):
    """Key manager that reads keys from environment variables."""

    def __init__(self, prefix: str = "CLOAKPIVOT_KEY_"):
        """
        Initialize environment key manager.

        Args:
            prefix: Prefix for environment variable names
        """
        self.prefix = prefix

    def get_key(self, key_id: str, version: Optional[str] = None) -> bytes:
        """Get key from environment variable."""
        env_name = f"{self.prefix}{key_id.upper()}"
        if version:
            env_name += f"_V{version}"

        key_value = os.environ.get(env_name)
        if not key_value:
            raise KeyError(f"Key not found in environment: {env_name}")

        try:
            # Support both hex and base64 encoded keys
            if key_value.startswith("hex:"):
                return bytes.fromhex(key_value[4:])
            elif key_value.startswith("b64:"):
                return base64.b64decode(key_value[4:])
            else:
                # Default to UTF-8 encoding
                return key_value.encode("utf-8")
        except Exception as e:
            raise ValueError(f"Invalid key format in {env_name}: {e}") from e

    def list_keys(self) -> list[str]:
        """List all keys with the configured prefix."""
        keys = []
        for env_name in os.environ:
            if env_name.startswith(self.prefix):
                # Extract key_id from environment name
                key_part = env_name[len(self.prefix) :]
                # Remove version suffix if present
                if "_V" in key_part:
                    key_part = key_part.split("_V")[0]
                keys.append(key_part.lower())

        return sorted(set(keys))


class FileKeyManager(KeyManager):
    """Key manager that reads keys from files."""

    def __init__(self, key_directory: Union[str, Path]):
        """
        Initialize file key manager.

        Args:
            key_directory: Directory containing key files
        """
        self.key_directory = Path(key_directory)
        if not self.key_directory.exists():
            raise ValueError(f"Key directory does not exist: {key_directory}")

    def get_key(self, key_id: str, version: Optional[str] = None) -> bytes:
        """Get key from file."""
        filename = f"{key_id}"
        if version:
            filename += f".v{version}"
        filename += ".key"

        key_path = self.key_directory / filename
        if not key_path.exists():
            raise KeyError(f"Key file not found: {key_path}")

        try:
            key_content = key_path.read_text().strip()

            # Support various key formats
            if key_content.startswith("hex:"):
                return bytes.fromhex(key_content[4:])
            elif key_content.startswith("b64:"):
                return base64.b64decode(key_content[4:])
            else:
                return key_content.encode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read key from {key_path}: {e}") from e

    def list_keys(self) -> list[str]:
        """List all key files in the directory."""
        keys = []
        for key_file in self.key_directory.glob("*.key"):
            key_name = key_file.stem
            # Remove version suffix if present
            if ".v" in key_name:
                key_name = key_name.split(".v")[0]
            keys.append(key_name)

        return sorted(set(keys))


class CompositeKeyManager(KeyManager):
    """Key manager that tries multiple sources in order."""

    def __init__(self, managers: list[KeyManager]):
        """
        Initialize composite key manager.

        Args:
            managers: List of key managers to try in order
        """
        if not managers:
            raise ValueError("At least one key manager is required")

        self.managers = managers

    def get_key(self, key_id: str, version: Optional[str] = None) -> bytes:
        """Try to get key from managers in order."""
        errors = []

        for manager in self.managers:
            try:
                return manager.get_key(key_id, version)
            except (KeyError, ValueError) as e:
                errors.append(f"{manager.__class__.__name__}: {e}")
                continue

        raise KeyError(f"Key '{key_id}' not found in any manager: {'; '.join(errors)}")

    def list_keys(self) -> list[str]:
        """List keys from all managers."""
        all_keys = set()
        for manager in self.managers:
            try:
                all_keys.update(manager.list_keys())
            except Exception:
                continue  # Skip managers that fail

        return sorted(all_keys)


class CryptoUtils:
    """Utility class for cryptographic operations."""

    @staticmethod
    def generate_salt(length: int = DEFAULT_SALT_LENGTH) -> bytes:
        """Generate a cryptographically secure random salt."""
        return secrets.token_bytes(length)

    @staticmethod
    def compute_salted_checksum(data: str, salt: bytes, config: SecurityConfig) -> str:
        """
        Compute a salted checksum using PBKDF2.

        Args:
            data: Data to checksum
            salt: Salt for the checksum
            config: Security configuration

        Returns:
            Hex-encoded checksum
        """
        data_bytes = data.encode("utf-8")

        checksum = hashlib.pbkdf2_hmac(
            config.hmac_algorithm, data_bytes, salt, config.pbkdf2_iterations, dklen=32
        )

        return checksum.hex()

    @staticmethod
    def verify_salted_checksum(
        data: str, salt: bytes, expected_checksum: str, config: SecurityConfig
    ) -> bool:
        """
        Verify a salted checksum.

        Args:
            data: Original data
            salt: Salt used for checksum
            expected_checksum: Expected checksum (hex)
            config: Security configuration

        Returns:
            True if checksum matches
        """
        computed_checksum = CryptoUtils.compute_salted_checksum(data, salt, config)

        if config.constant_time_verification:
            return hmac.compare_digest(computed_checksum, expected_checksum)
        else:
            return computed_checksum == expected_checksum

    @staticmethod
    def compute_hmac(
        data: bytes, key: bytes, algorithm: str = DEFAULT_HMAC_ALGORITHM
    ) -> str:
        """
        Compute HMAC for data.

        Args:
            data: Data to sign
            key: HMAC key
            algorithm: Hash algorithm to use

        Returns:
            Hex-encoded HMAC
        """
        if algorithm not in SUPPORTED_HMAC_ALGORITHMS:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")

        hash_func = getattr(hashlib, algorithm)
        mac = hmac.new(key, data, hash_func)
        return mac.hexdigest()

    @staticmethod
    def verify_hmac(
        data: bytes,
        key: bytes,
        expected_mac: str,
        algorithm: str = DEFAULT_HMAC_ALGORITHM,
        constant_time: bool = True,
    ) -> bool:
        """
        Verify HMAC signature.

        Args:
            data: Original data
            key: HMAC key
            expected_mac: Expected MAC (hex)
            algorithm: Hash algorithm used
            constant_time: Use constant-time comparison

        Returns:
            True if MAC is valid
        """
        computed_mac = CryptoUtils.compute_hmac(data, key, algorithm)

        if constant_time:
            return hmac.compare_digest(computed_mac, expected_mac)
        else:
            return computed_mac == expected_mac

    @staticmethod
    def generate_nonce(length: int = DEFAULT_NONCE_LENGTH) -> bytes:
        """
        Generate a cryptographically secure random nonce.

        Args:
            length: Length of nonce in bytes (defaults to 12 for AES-GCM)

        Returns:
            Random nonce bytes
        """
        return secrets.token_bytes(length)

    @staticmethod
    def derive_encryption_key(
        base_key: bytes,
        salt: bytes,
        key_length: int = DEFAULT_KEY_LENGTH,
        config: Optional[SecurityConfig] = None,
    ) -> bytes:
        """
        Derive an encryption key from base key material using PBKDF2.

        Args:
            base_key: Base key material
            salt: Salt for key derivation
            key_length: Desired key length in bytes
            config: Security configuration

        Returns:
            Derived encryption key
        """
        if config is None:
            config = SecurityConfig()

        return hashlib.pbkdf2_hmac(
            config.hmac_algorithm,
            base_key,
            salt,
            config.pbkdf2_iterations,
            dklen=key_length,
        )

    @staticmethod
    def encrypt_data(
        data: bytes,
        key: bytes,
        nonce: Optional[bytes] = None,
        associated_data: Optional[bytes] = None,
    ) -> tuple[bytes, bytes]:
        """
        Encrypt data using AES-GCM.

        Args:
            data: Data to encrypt
            key: Encryption key (32 bytes for AES-256)
            nonce: Optional nonce (generates random if None)
            associated_data: Optional additional authenticated data

        Returns:
            Tuple of (ciphertext, nonce)

        Raises:
            ValueError: If key length is invalid
        """
        if len(key) not in {16, 24, 32}:
            raise ValueError("Key must be 16, 24, or 32 bytes for AES")

        if nonce is None:
            nonce = CryptoUtils.generate_nonce()
        elif len(nonce) != DEFAULT_NONCE_LENGTH:
            raise ValueError(f"Nonce must be {DEFAULT_NONCE_LENGTH} bytes for AES-GCM")

        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, data, associated_data)

        return ciphertext, nonce

    @staticmethod
    def decrypt_data(
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        associated_data: Optional[bytes] = None,
    ) -> bytes:
        """
        Decrypt data using AES-GCM.

        Args:
            ciphertext: Encrypted data
            key: Decryption key (32 bytes for AES-256)
            nonce: Nonce used during encryption
            associated_data: Optional additional authenticated data

        Returns:
            Decrypted plaintext data

        Raises:
            ValueError: If key/nonce length is invalid
            cryptography.exceptions.InvalidTag: If authentication fails
        """
        if len(key) not in {16, 24, 32}:
            raise ValueError("Key must be 16, 24, or 32 bytes for AES")

        if len(nonce) != DEFAULT_NONCE_LENGTH:
            raise ValueError(f"Nonce must be {DEFAULT_NONCE_LENGTH} bytes for AES-GCM")

        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, associated_data)


@dataclass(frozen=True)
class SecurityMetadata:
    """Metadata for security operations."""

    algorithm: str
    key_id: str
    key_version: Optional[str] = None
    salt: Optional[bytes] = None
    iterations: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
        }

        if self.key_version:
            result["key_version"] = self.key_version

        if self.salt:
            result["salt"] = base64.b64encode(self.salt).decode("ascii")

        if self.iterations:
            result["iterations"] = self.iterations

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityMetadata":
        """Create from dictionary."""
        salt = None
        if data.get("salt"):
            salt = base64.b64decode(data["salt"])

        return cls(
            algorithm=data["algorithm"],
            key_id=data["key_id"],
            key_version=data.get("key_version"),
            salt=salt,
            iterations=data.get("iterations"),
        )


@dataclass(frozen=True)
class EncryptedCloakMap:
    """
    Encrypted wrapper for CloakMap data with metadata preserved for indexing.

    This class represents an encrypted CloakMap where sensitive anchor data
    is encrypted but document metadata remains in cleartext for indexing.
    """

    # Cleartext metadata for indexing
    version: str
    doc_id: str
    doc_hash: str
    created_at: Optional[str] = None

    # Encryption metadata
    algorithm: str = DEFAULT_ENCRYPTION_ALGORITHM
    key_id: str = "default"
    key_version: Optional[str] = None
    nonce: str = ""  # Base64-encoded nonce

    # Encrypted content (Base64-encoded)
    encrypted_anchors: str = ""
    encrypted_policy: str = ""
    encrypted_metadata: str = ""

    # Optional signature for integrity
    signature: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "doc_id": self.doc_id,
            "doc_hash": self.doc_hash,
            "created_at": self.created_at,
            "crypto": {
                "algorithm": self.algorithm,
                "key_id": self.key_id,
                "key_version": self.key_version,
                "nonce": self.nonce,
            },
            "encrypted_content": {
                "anchors": self.encrypted_anchors,
                "policy_snapshot": self.encrypted_policy,
                "metadata": self.encrypted_metadata,
            },
            "signature": self.signature,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EncryptedCloakMap":
        """Create from dictionary representation."""
        crypto = data.get("crypto", {})
        encrypted_content = data.get("encrypted_content", {})

        return cls(
            version=data.get("version", "1.0"),
            doc_id=data.get("doc_id", ""),
            doc_hash=data.get("doc_hash", ""),
            created_at=data.get("created_at"),
            algorithm=crypto.get("algorithm", DEFAULT_ENCRYPTION_ALGORITHM),
            key_id=crypto.get("key_id", "default"),
            key_version=crypto.get("key_version"),
            nonce=crypto.get("nonce", ""),
            encrypted_anchors=encrypted_content.get("anchors", ""),
            encrypted_policy=encrypted_content.get("policy_snapshot", ""),
            encrypted_metadata=encrypted_content.get("metadata", ""),
            signature=data.get("signature"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EncryptedCloakMap":
        """Create from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from e


class CloakMapEncryption:
    """Utility class for CloakMap encryption and decryption operations."""

    def __init__(
        self, key_manager: KeyManager, config: Optional[SecurityConfig] = None
    ):
        """
        Initialize encryption utility.

        Args:
            key_manager: Key manager for retrieving encryption keys
            config: Security configuration
        """
        self.key_manager = key_manager
        self.config = config or SecurityConfig()

    def encrypt_cloakmap(
        self,
        cloakmap: "CloakMap",
        key_id: str = "default",
        key_version: Optional[str] = None,
    ) -> EncryptedCloakMap:
        """
        Encrypt a CloakMap, keeping metadata in cleartext for indexing.

        Args:
            cloakmap: CloakMap to encrypt
            key_id: Encryption key identifier
            key_version: Optional key version

        Returns:
            EncryptedCloakMap with encrypted sensitive data

        Raises:
            KeyError: If encryption key is not found
            ValueError: If encryption fails
        """
        try:
            # Get encryption key
            encryption_key = self.key_manager.get_encryption_key(key_id, key_version)
            if len(encryption_key) != self.config.key_length:
                # Derive proper length key if needed
                salt = CryptoUtils.generate_salt()
                encryption_key = CryptoUtils.derive_encryption_key(
                    encryption_key, salt, self.config.key_length, self.config
                )

            # Generate nonce
            nonce = CryptoUtils.generate_nonce(self.config.nonce_length)

            # Use doc_id as associated data for authentication
            associated_data = cloakmap.doc_id.encode("utf-8")

            # Encrypt sensitive data sections
            anchors_json = json.dumps([anchor.to_dict() for anchor in cloakmap.anchors])
            anchors_data = anchors_json.encode("utf-8")
            encrypted_anchors, _ = CryptoUtils.encrypt_data(
                anchors_data, encryption_key, nonce, associated_data
            )

            policy_json = json.dumps(cloakmap.policy_snapshot)
            policy_data = policy_json.encode("utf-8")
            encrypted_policy, _ = CryptoUtils.encrypt_data(
                policy_data, encryption_key, nonce, associated_data
            )

            metadata_json = json.dumps(cloakmap.metadata)
            metadata_data = metadata_json.encode("utf-8")
            encrypted_metadata, _ = CryptoUtils.encrypt_data(
                metadata_data, encryption_key, nonce, associated_data
            )

            # Create encrypted CloakMap
            return EncryptedCloakMap(
                version=cloakmap.version,
                doc_id=cloakmap.doc_id,
                doc_hash=cloakmap.doc_hash,
                created_at=(
                    cloakmap.created_at.isoformat() if cloakmap.created_at else None
                ),
                algorithm=self.config.encryption_algorithm,
                key_id=key_id,
                key_version=key_version,
                nonce=base64.b64encode(nonce).decode("ascii"),
                encrypted_anchors=base64.b64encode(encrypted_anchors).decode("ascii"),
                encrypted_policy=base64.b64encode(encrypted_policy).decode("ascii"),
                encrypted_metadata=base64.b64encode(encrypted_metadata).decode("ascii"),
                signature=cloakmap.signature,
            )

        except Exception as e:
            raise ValueError(f"Failed to encrypt CloakMap: {e}") from e

    def decrypt_cloakmap(self, encrypted_cloakmap: EncryptedCloakMap) -> "CloakMap":
        """
        Decrypt an EncryptedCloakMap back to a standard CloakMap.

        Args:
            encrypted_cloakmap: EncryptedCloakMap to decrypt

        Returns:
            Decrypted CloakMap

        Raises:
            KeyError: If decryption key is not found
            ValueError: If decryption fails
            cryptography.exceptions.InvalidTag: If authentication fails
        """
        try:
            # Get decryption key
            decryption_key = self.key_manager.get_encryption_key(
                encrypted_cloakmap.key_id, encrypted_cloakmap.key_version
            )
            if len(decryption_key) != self.config.key_length:
                # Derive proper length key if needed
                salt = CryptoUtils.generate_salt()
                decryption_key = CryptoUtils.derive_encryption_key(
                    decryption_key, salt, self.config.key_length, self.config
                )

            # Decode nonce
            nonce = base64.b64decode(encrypted_cloakmap.nonce)

            # Use doc_id as associated data for authentication
            associated_data = encrypted_cloakmap.doc_id.encode("utf-8")

            # Decrypt sections
            encrypted_anchors_data = base64.b64decode(
                encrypted_cloakmap.encrypted_anchors
            )
            anchors_data = CryptoUtils.decrypt_data(
                encrypted_anchors_data, decryption_key, nonce, associated_data
            )

            encrypted_policy_data = base64.b64decode(
                encrypted_cloakmap.encrypted_policy
            )
            policy_data = CryptoUtils.decrypt_data(
                encrypted_policy_data, decryption_key, nonce, associated_data
            )

            encrypted_metadata_data = base64.b64decode(
                encrypted_cloakmap.encrypted_metadata
            )
            metadata_data = CryptoUtils.decrypt_data(
                encrypted_metadata_data, decryption_key, nonce, associated_data
            )

            # Parse decrypted JSON data
            anchors_dict = json.loads(anchors_data.decode("utf-8"))
            policy_snapshot = json.loads(policy_data.decode("utf-8"))
            metadata = json.loads(metadata_data.decode("utf-8"))

            # Reconstruct anchors
            from .anchors import AnchorEntry

            anchors = [
                AnchorEntry.from_dict(anchor_data) for anchor_data in anchors_dict
            ]

            # Parse timestamp
            created_at = None
            if encrypted_cloakmap.created_at:
                from datetime import datetime

                created_at = datetime.fromisoformat(encrypted_cloakmap.created_at)

            # Import CloakMap class
            from .cloakmap import CloakMap

            return CloakMap(
                version=encrypted_cloakmap.version,
                doc_id=encrypted_cloakmap.doc_id,
                doc_hash=encrypted_cloakmap.doc_hash,
                anchors=anchors,
                policy_snapshot=policy_snapshot,
                crypto={
                    "algorithm": encrypted_cloakmap.algorithm,
                    "key_id": encrypted_cloakmap.key_id,
                    "key_version": encrypted_cloakmap.key_version,
                },
                signature=encrypted_cloakmap.signature,
                created_at=created_at,
                metadata=metadata,
            )

        except Exception as e:
            raise ValueError(f"Failed to decrypt CloakMap: {e}") from e

    def rotate_encryption_key(
        self,
        encrypted_cloakmap: EncryptedCloakMap,
        new_key_id: str,
        new_key_version: Optional[str] = None,
    ) -> EncryptedCloakMap:
        """
        Rotate encryption key by decrypting with old key and re-encrypting with new key.

        Args:
            encrypted_cloakmap: EncryptedCloakMap with old key
            new_key_id: New encryption key identifier
            new_key_version: Optional new key version

        Returns:
            EncryptedCloakMap encrypted with new key

        Raises:
            KeyError: If old or new key is not found
            ValueError: If rotation fails
        """
        try:
            # Decrypt with old key
            decrypted_map = self.decrypt_cloakmap(encrypted_cloakmap)

            # Re-encrypt with new key
            return self.encrypt_cloakmap(decrypted_map, new_key_id, new_key_version)

        except Exception as e:
            raise ValueError(f"Failed to rotate encryption key: {e}") from e


class KeyRotationManager:
    """Utility for bulk key rotation operations."""

    def __init__(
        self, key_manager: KeyManager, config: Optional[SecurityConfig] = None
    ):
        """
        Initialize key rotation manager.

        Args:
            key_manager: Key manager with both old and new keys
            config: Security configuration
        """
        self.key_manager = key_manager
        self.config = config or SecurityConfig()
        self.encryption = CloakMapEncryption(key_manager, config)

    def rotate_directory(
        self,
        directory_path: Union[str, Path],
        old_key_id: str,
        new_key_id: str,
        old_key_version: Optional[str] = None,
        new_key_version: Optional[str] = None,
        pattern: str = "*.json",
        backup: bool = True,
    ) -> dict[str, Any]:
        """
        Rotate encryption keys for all CloakMap files in a directory.

        Args:
            directory_path: Directory containing CloakMap files
            old_key_id: Current key identifier
            new_key_id: New key identifier
            old_key_version: Current key version
            new_key_version: New key version
            pattern: File pattern to match
            backup: Whether to create backup files

        Returns:
            Dictionary with rotation statistics

        Raises:
            ValueError: If directory doesn't exist or rotation fails
        """
        from glob import glob

        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        # Find files matching pattern
        file_pattern = str(directory / pattern)
        files = glob(file_pattern)

        results = {
            "total_files": len(files),
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "errors": [],
            "backup_created": backup,
        }

        for file_path in files:
            try:
                results["processed"] += 1

                # Check if file is encrypted CloakMap
                with open(file_path, encoding="utf-8") as f:
                    data = json.loads(f.read())

                if "encrypted_content" not in data:
                    # Skip unencrypted files
                    continue

                # Load encrypted CloakMap
                encrypted_map = EncryptedCloakMap.from_dict(data)

                # Skip if already using new key
                if (
                    encrypted_map.key_id == new_key_id
                    and encrypted_map.key_version == new_key_version
                ):
                    continue

                # Create backup if requested
                if backup:
                    backup_path = Path(file_path).with_suffix(".bak")
                    import shutil

                    shutil.copy2(file_path, backup_path)

                # Rotate key
                rotated_map = self.encryption.rotate_encryption_key(
                    encrypted_map, new_key_id, new_key_version
                )

                # Save rotated file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(rotated_map.to_json(indent=2))

                results["succeeded"] += 1

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{file_path}: {e}")

        return results

    def verify_key_availability(
        self, key_ids: list[str], versions: Optional[list[Optional[str]]] = None
    ) -> dict[str, bool]:
        """
        Verify that required keys are available in the key manager.

        Args:
            key_ids: List of key identifiers to check
            versions: Optional list of key versions to check

        Returns:
            Dictionary mapping key_id to availability status
        """
        if versions is None:
            versions = [None] * len(key_ids)

        availability = {}

        for key_id, version in zip(key_ids, versions):
            try:
                self.key_manager.get_encryption_key(key_id, version)
                availability[key_id] = True
            except (KeyError, ValueError):
                availability[key_id] = False

        return availability


class SecurityValidator:
    """Comprehensive security validator for CloakMaps."""

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        key_manager: Optional[KeyManager] = None,
    ):
        """
        Initialize security validator.

        Args:
            config: Security configuration
            key_manager: Key manager for signature verification
        """
        self.config = config or SecurityConfig()
        self.key_manager = key_manager or create_default_key_manager()

    def validate_cloakmap(self, cloakmap: "CloakMap") -> dict[str, Any]:
        """
        Perform comprehensive security validation of a CloakMap.

        Args:
            cloakmap: CloakMap to validate

        Returns:
            Detailed validation results
        """

        results = {
            "valid": True,
            "security_level": "none",
            "errors": [],
            "warnings": [],
            "checks": {
                "structure": False,
                "anchors": False,
                "signature": False,
                "encryption": False,
                "tampering": False,
                "key_availability": False,
            },
            "performance": {
                "validation_time_ms": 0,
                "anchor_count": len(cloakmap.anchors),
                "crypto_operations": 0,
            },
        }

        start_time = self._get_time_ms()

        try:
            # Structure validation
            self._validate_structure(cloakmap, results)

            # Anchor validation with security checks
            self._validate_anchors(cloakmap, results)

            # Signature validation
            self._validate_signature(cloakmap, results)

            # Encryption validation
            self._validate_encryption(cloakmap, results)

            # Tampering detection
            self._detect_tampering(cloakmap, results)

            # Key availability check
            self._check_key_availability(cloakmap, results)

            # Determine overall security level
            self._assess_security_level(results)

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation failed: {e}")

        results["performance"]["validation_time_ms"] = max(
            1, int(self._get_time_ms() - start_time)
        )

        return results

    def _validate_structure(self, cloakmap: "CloakMap", results: dict) -> None:
        """Validate CloakMap structure."""
        try:
            # Check version compatibility
            version_parts = cloakmap.version.split(".")
            if len(version_parts) < 2:
                results["errors"].append("Invalid version format")
                return

            # Check required fields
            if not cloakmap.doc_id:
                results["errors"].append("Missing document ID")

            if not cloakmap.doc_hash:
                results["warnings"].append("Missing document hash")

            results["checks"]["structure"] = True

        except Exception as e:
            results["errors"].append(f"Structure validation error: {e}")

    def _validate_anchors(self, cloakmap: "CloakMap", results: dict) -> None:
        """Validate anchors with security considerations."""
        try:
            from .anchors import AnchorIndex

            # Basic anchor validation
            anchor_index = AnchorIndex(cloakmap.anchors)

            # Check for overlapping anchors (security risk)
            overlaps_found = False
            for _i, anchor in enumerate(cloakmap.anchors):
                overlapping = anchor_index.find_overlapping_anchors(anchor)
                if overlapping:
                    overlaps_found = True
                    results["errors"].append(
                        f"Overlapping anchors detected: {anchor.replacement_id}"
                    )

            # Check checksum salt presence (for new format)
            missing_salts = 0
            for anchor in cloakmap.anchors:
                if not hasattr(anchor, "checksum_salt") or not anchor.checksum_salt:
                    missing_salts += 1

            if missing_salts > 0:
                results["warnings"].append(
                    f"{missing_salts} anchors missing checksum salts (legacy format)"
                )

            if not overlaps_found:
                results["checks"]["anchors"] = True

        except Exception as e:
            results["errors"].append(f"Anchor validation error: {e}")

    def _validate_signature(self, cloakmap: "CloakMap", results: dict) -> None:
        """Validate CloakMap signature."""
        if not cloakmap.is_signed:
            results["warnings"].append("CloakMap is not signed")
            results["checks"]["signature"] = True  # Not an error if not signed
            return

        try:
            # Try to verify signature with available keys
            key_id = (
                cloakmap.crypto.get("key_id", "default")
                if cloakmap.crypto
                else "default"
            )

            try:
                self.key_manager.get_key(key_id)
                is_valid = cloakmap.verify_signature(
                    self.key_manager, config=self.config
                )

                if is_valid:
                    results["checks"]["signature"] = True
                    results["performance"]["crypto_operations"] += 1
                else:
                    results["errors"].append("Signature verification failed")

            except (KeyError, ValueError) as e:
                results["errors"].append(f"Signature key not available: {e}")

        except Exception as e:
            results["errors"].append(f"Signature validation error: {e}")

    def _validate_encryption(self, cloakmap: "CloakMap", results: dict) -> None:
        """Validate encryption metadata."""
        if not cloakmap.is_encrypted:
            results["warnings"].append("CloakMap is not encrypted")
            results["checks"]["encryption"] = True  # Not required
            return

        try:
            crypto = cloakmap.crypto
            if not crypto or not crypto.get("algorithm"):
                results["errors"].append("Invalid encryption metadata")
                return

            # Check for supported algorithms
            algorithm = crypto.get("algorithm", "")
            if "AES" not in algorithm and "ChaCha20" not in algorithm:
                results["warnings"].append(f"Unusual encryption algorithm: {algorithm}")

            results["checks"]["encryption"] = True

        except Exception as e:
            results["errors"].append(f"Encryption validation error: {e}")

    def _detect_tampering(self, cloakmap: "CloakMap", results: dict) -> None:
        """Detect signs of tampering."""
        try:
            # Check for timestamp anomalies
            if cloakmap.created_at:
                from datetime import datetime, timezone

                now = datetime.now(timezone.utc).replace(tzinfo=None)

                # Check for future timestamps (suspicious)
                if cloakmap.created_at > now:
                    results["warnings"].append("CloakMap timestamp is in the future")

            # Check anchor timestamp consistency
            if cloakmap.anchors and cloakmap.created_at:
                for anchor in cloakmap.anchors:
                    if anchor.timestamp and anchor.timestamp > cloakmap.created_at:
                        results["warnings"].append(
                            f"Anchor {anchor.replacement_id} timestamp is after CloakMap creation"
                        )

            # Check for suspicious patterns in replacement IDs
            replacement_ids = [a.replacement_id for a in cloakmap.anchors]
            if len(set(replacement_ids)) != len(replacement_ids):
                results["errors"].append("Duplicate replacement IDs detected")
                return

            results["checks"]["tampering"] = True

        except Exception as e:
            results["errors"].append(f"Tampering detection error: {e}")

    def _check_key_availability(
        self, cloakmap: "CloakMap", results: dict[str, Any]
    ) -> None:
        """Check if required keys are available."""
        try:
            required_keys = set()

            # Check signature key
            if cloakmap.is_signed and cloakmap.crypto:
                key_id = cloakmap.crypto.get("key_id", "default")
                required_keys.add(key_id)

            # Check encryption key
            if cloakmap.is_encrypted and cloakmap.crypto:
                key_id = cloakmap.crypto.get("encryption_key_id")
                if key_id:
                    required_keys.add(key_id)

            # Verify key availability
            available_keys = set(self.key_manager.list_keys())
            missing_keys = required_keys - available_keys

            if missing_keys:
                results["warnings"].append(f"Missing keys: {', '.join(missing_keys)}")
            else:
                results["checks"]["key_availability"] = True

        except Exception as e:
            results["warnings"].append(f"Key availability check failed: {e}")

    def _assess_security_level(self, results: dict[str, Any]) -> None:
        """Assess overall security level."""
        if results["errors"]:
            results["security_level"] = "compromised"
        elif results["checks"]["signature"] and results["checks"]["encryption"]:
            results["security_level"] = "high"
        elif results["checks"]["signature"] or results["checks"]["encryption"]:
            results["security_level"] = "medium"
        else:
            results["security_level"] = "low"

    def _get_time_ms(self) -> float:
        """Get current time in milliseconds using high precision counter."""
        import time

        return time.perf_counter() * 1000


def create_default_key_manager() -> KeyManager:
    """
    Create a default key manager with common sources.

    Returns:
        CompositeKeyManager with environment and file sources
    """
    managers: list[KeyManager] = [EnvironmentKeyManager()]

    # Try to add file manager if key directory exists
    possible_key_dirs = [
        Path.home() / ".cloakpivot" / "keys",
        Path("/etc/cloakpivot/keys"),
        Path("./keys"),
    ]

    for key_dir in possible_key_dirs:
        if key_dir.exists() and key_dir.is_dir():
            managers.append(FileKeyManager(key_dir))
            break

    return CompositeKeyManager(managers)
