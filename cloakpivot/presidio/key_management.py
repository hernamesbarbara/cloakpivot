"""Key management integration for encryption/decryption workflows."""

import base64
import json
import os
import secrets
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Optional import for Vault integration
try:
    import hvac
    HVAC_AVAILABLE = True
except ImportError:
    hvac = None
    HVAC_AVAILABLE = False


class KeyProvider(ABC):
    """Abstract base for encryption key providers."""

    @abstractmethod
    def get_key(self, key_id: str) -> bytes:
        """Get an encryption key by ID.

        Args:
            key_id: Identifier for the key

        Returns:
            Key as bytes

        Raises:
            KeyError: If key not found
        """
        raise NotImplementedError

    @abstractmethod
    def create_key(self, key_type: str = "AES-256") -> str:
        """Create a new encryption key.

        Args:
            key_type: Type of key to create

        Returns:
            Key identifier
        """
        raise NotImplementedError

    @abstractmethod
    def rotate_key(self, old_key_id: str) -> str:
        """Rotate an encryption key.

        Args:
            old_key_id: ID of key to rotate

        Returns:
            New key identifier
        """
        raise NotImplementedError

    def list_keys(self) -> list:
        """List available key IDs.

        Returns:
            List of key identifiers
        """
        return []

    def delete_key(self, key_id: str) -> bool:
        """Delete a key.

        Args:
            key_id: ID of key to delete

        Returns:
            True if deleted successfully
        """
        return False


class EnvironmentKeyProvider(KeyProvider):
    """Load encryption keys from environment variables."""

    def __init__(self, prefix: str = "CLOAKPIVOT_KEY"):
        """Initialize environment key provider.

        Args:
            prefix: Prefix for environment variable names
        """
        self.prefix = prefix
        self._key_cache: dict[str, bytes] = {}

    def get_key(self, key_id: str) -> bytes:
        """Get key from environment variable.

        Args:
            key_id: Key identifier

        Returns:
            Key as bytes

        Raises:
            KeyError: If key not found in environment
        """
        # Check cache first
        if key_id in self._key_cache:
            return self._key_cache[key_id]  # Type: bytes

        # Build environment variable name
        key_env_var = f"{self.prefix}_{key_id.upper()}"
        key_str = os.getenv(key_env_var)

        if not key_str:
            raise KeyError(f"Key {key_id} not found in environment variable {key_env_var}")

        # Decode from base64
        try:
            key_bytes = base64.b64decode(key_str)
        except Exception as e:
            raise ValueError(f"Invalid key format in {key_env_var}: {e}") from e

        # Cache the key
        self._key_cache[key_id] = key_bytes

        return key_bytes

    def create_key(self, key_type: str = "AES-256") -> str:
        """Create a new key (not supported for environment provider).

        Args:
            key_type: Type of key to create

        Returns:
            Key identifier

        Raises:
            NotImplementedError: Environment provider doesn't create keys
        """
        # Generate a new key
        if key_type in ["AES-256", "Fernet"]:
            key = Fernet.generate_key()
        else:
            # Generate random bytes
            key = secrets.token_bytes(32)

        # Generate key ID
        key_id = f"key_{secrets.token_hex(8)}"

        # Return key_id and instruction without exposing the key
        # User should retrieve the key securely
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Created key with ID: {key_id}. Store securely in environment variable {self.prefix}_{key_id.upper()}")

        # Store the encoded key in environment for this session only
        encoded_key = base64.b64encode(key).decode()
        env_var_name = f"{self.prefix}_{key_id.upper()}"
        os.environ[env_var_name] = encoded_key

        # Log instruction for user (security: do not log the actual key)
        logger.info(f"Created key with ID: {key_id}")
        logger.info(f"To use this key, set the environment variable: {env_var_name}")
        # Note: Not logging the actual key value for security reasons

        return key_id

    def rotate_key(self, old_key_id: str) -> str:
        """Rotate a key (not directly supported).

        Args:
            old_key_id: ID of key to rotate

        Returns:
            New key identifier
        """
        # Create a new key
        new_key_id = self.create_key()

        # Clear cache for old key
        if old_key_id in self._key_cache:
            del self._key_cache[old_key_id]

        return new_key_id

    def list_keys(self) -> list:
        """List keys available in environment.

        Returns:
            List of key identifiers
        """
        keys = []
        prefix_len = len(self.prefix) + 1  # +1 for underscore

        for env_var in os.environ:
            if env_var.startswith(f"{self.prefix}_"):
                key_id = env_var[prefix_len:].lower()
                keys.append(key_id)

        return keys


class FileKeyProvider(KeyProvider):
    """File-based key storage provider."""

    def __init__(self, key_directory: str = ".keys", password: Optional[str] = None):
        """Initialize file key provider.

        Args:
            key_directory: Directory to store key files
            password: Optional password for key derivation
        """
        self.key_directory = Path(key_directory)
        self.key_directory.mkdir(exist_ok=True, mode=0o700)
        # Require password to be explicitly set, no default value for security
        if password is None:
            password = os.getenv("CLOAKPIVOT_KEY_PASSWORD")
            if password is None:
                raise ValueError(
                    "Password is required for FileKeyProvider. "
                    "Set via parameter or CLOAKPIVOT_KEY_PASSWORD environment variable."
                )
        self.password = password
        self._key_cache: dict[str, bytes] = {}
        self._metadata_file = self.key_directory / "metadata.json"
        self._metadata: dict[str, Any] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load key metadata from file."""
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {
                "keys": {},
                "rotation_history": []
            }
            # Save initial metadata
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Save key metadata to file."""
        with open(self._metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key from password.

        Args:
            salt: Salt for key derivation

        Returns:
            Derived key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self.password.encode()))

    def get_key(self, key_id: str) -> bytes:
        """Get key from file storage.

        Args:
            key_id: Key identifier

        Returns:
            Key as bytes

        Raises:
            KeyError: If key not found
        """
        # Check cache
        if key_id in self._key_cache:
            return self._key_cache[key_id]  # Type: bytes

        # Check if key exists
        key_file = self.key_directory / f"{key_id}.key"
        if not key_file.exists():
            raise KeyError(f"Key {key_id} not found")

        # Read encrypted key
        with open(key_file, "rb") as f:
            data = f.read()

        # Decrypt if password is set and not empty
        if self.password and self.password.strip():
            salt = data[:16]
            encrypted_key = data[16:]
            fernet_key = self._derive_key(salt)
            fernet = Fernet(fernet_key)
            key = fernet.decrypt(encrypted_key)
        else:
            key = data

        # Cache the key
        self._key_cache[key_id] = key

        return key

    def create_key(self, key_type: str = "AES-256") -> str:
        """Create a new encryption key.

        Args:
            key_type: Type of key to create

        Returns:
            Key identifier
        """
        # Generate key
        if key_type == "Fernet":
            key = Fernet.generate_key()
        else:
            key = secrets.token_bytes(32)

        # Generate key ID
        key_id = f"key_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"

        # Encrypt key if password is set and not empty
        if self.password and self.password.strip():
            salt = secrets.token_bytes(16)
            fernet_key = self._derive_key(salt)
            fernet = Fernet(fernet_key)
            encrypted_key = fernet.encrypt(key)
            data = salt + encrypted_key
        else:
            data = key

        # Save to file
        key_file = self.key_directory / f"{key_id}.key"
        with open(key_file, "wb") as f:
            f.write(data)

        # Update metadata
        self._metadata["keys"][key_id] = {
            "created": datetime.now().isoformat(),
            "type": key_type,
            "active": True
        }
        self._save_metadata()

        return key_id

    def rotate_key(self, old_key_id: str) -> str:
        """Rotate an encryption key.

        Args:
            old_key_id: ID of key to rotate

        Returns:
            New key identifier
        """
        # Verify old key exists
        if old_key_id not in self._metadata["keys"]:
            raise KeyError(f"Key {old_key_id} not found")

        # Create new key of same type
        key_type = self._metadata["keys"][old_key_id].get("type", "AES-256")
        new_key_id = self.create_key(key_type)

        # Mark old key as rotated
        self._metadata["keys"][old_key_id]["active"] = False
        self._metadata["keys"][old_key_id]["rotated"] = datetime.now().isoformat()
        self._metadata["keys"][old_key_id]["replaced_by"] = new_key_id

        # Add to rotation history
        self._metadata["rotation_history"].append({
            "old_key": old_key_id,
            "new_key": new_key_id,
            "timestamp": datetime.now().isoformat()
        })

        self._save_metadata()

        # Clear cache
        if old_key_id in self._key_cache:
            del self._key_cache[old_key_id]

        return new_key_id

    def list_keys(self) -> list:
        """List available key IDs.

        Returns:
            List of key identifiers
        """
        return list(self._metadata["keys"].keys())

    def delete_key(self, key_id: str) -> bool:
        """Delete a key.

        Args:
            key_id: ID of key to delete

        Returns:
            True if deleted successfully
        """
        # Check if key exists
        if key_id not in self._metadata["keys"]:
            return False

        # Delete key file
        key_file = self.key_directory / f"{key_id}.key"
        if key_file.exists():
            key_file.unlink()

        # Remove from metadata
        del self._metadata["keys"][key_id]
        self._save_metadata()

        # Clear cache
        if key_id in self._key_cache:
            del self._key_cache[key_id]

        return True


class VaultKeyProvider(KeyProvider):
    """Integration with HashiCorp Vault for key management."""

    def __init__(self, vault_url: str, vault_token: Optional[str] = None, secret_path: str = "cloakpivot/keys"):
        """Initialize Vault key provider.

        Args:
            vault_url: URL of Vault server
            vault_token: Vault authentication token
            secret_path: Path in Vault for keys
        """
        if not HVAC_AVAILABLE:
            raise ImportError("hvac library required for Vault integration. Install with: pip install hvac")

        self.vault_url = vault_url
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self.secret_path = secret_path

        if not self.vault_token:
            raise ValueError("Vault token required. Set VAULT_TOKEN environment variable or pass token parameter.")

        self.vault_client = hvac.Client(url=vault_url, token=self.vault_token)

        if not self.vault_client.is_authenticated():
            raise ValueError("Failed to authenticate with Vault")

    def get_key(self, key_id: str) -> bytes:
        """Get key from Vault.

        Args:
            key_id: Key identifier

        Returns:
            Key as bytes

        Raises:
            KeyError: If key not found in Vault
        """
        try:
            response = self.vault_client.secrets.kv.v2.read_secret_version(
                path=f"{self.secret_path}/{key_id}"
            )
            key_data = response["data"]["data"]["key"]
            return base64.b64decode(key_data)
        except Exception as e:
            raise KeyError(f"Failed to retrieve key {key_id} from Vault: {e}") from e

    def create_key(self, key_type: str = "AES-256") -> str:
        """Create a new key in Vault.

        Args:
            key_type: Type of key to create

        Returns:
            Key identifier
        """
        # Generate key
        if key_type == "Fernet":
            key = Fernet.generate_key()
        else:
            key = secrets.token_bytes(32)

        # Generate key ID
        key_id = f"key_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"

        # Store in Vault
        try:
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=f"{self.secret_path}/{key_id}",
                secret={
                    "key": base64.b64encode(key).decode(),
                    "type": key_type,
                    "created": datetime.now().isoformat()
                }
            )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create key in Vault: {e}")
            raise ValueError(f"Failed to create key in Vault: {e}") from e

        return key_id

    def rotate_key(self, old_key_id: str) -> str:
        """Rotate a key in Vault.

        Args:
            old_key_id: ID of key to rotate

        Returns:
            New key identifier
        """
        # Get old key metadata
        try:
            response = self.vault_client.secrets.kv.v2.read_secret_version(
                path=f"{self.secret_path}/{old_key_id}"
            )
            old_key_data = response["data"]["data"]
            key_type = old_key_data.get("type", "AES-256")
        except Exception:
            key_type = "AES-256"

        # Create new key
        new_key_id = self.create_key(key_type)

        # Update old key metadata
        try:
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=f"{self.secret_path}/{old_key_id}",
                secret={
                    **old_key_data,
                    "rotated": datetime.now().isoformat(),
                    "replaced_by": new_key_id,
                    "active": False
                }
            )
        except Exception as e:
            # Log the error but continue - key rotation succeeded, metadata update is optional
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to update metadata for rotated key {old_key_id}: {e}")

        return new_key_id

    def list_keys(self) -> list:
        """List keys in Vault.

        Returns:
            List of key identifiers
        """
        try:
            response = self.vault_client.secrets.kv.v2.list_secrets(
                path=self.secret_path
            )
            keys = response.get("data", {}).get("keys", [])
            return list(keys) if keys else []
        except Exception:
            return []

    def delete_key(self, key_id: str) -> bool:
        """Delete a key from Vault.

        Args:
            key_id: ID of key to delete

        Returns:
            True if deleted successfully
        """
        try:
            self.vault_client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=f"{self.secret_path}/{key_id}"
            )
            return True
        except Exception:
            return False
