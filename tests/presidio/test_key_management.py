"""Tests for key management providers."""

import base64
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from cloakpivot.presidio.key_management import (
    EnvironmentKeyProvider,
    FileKeyProvider,
    KeyProvider,
    VaultKeyProvider,
)


class TestKeyProviderBase:
    """Test the abstract KeyProvider base class."""

    def test_abstract_methods(self):
        """Test that abstract methods cannot be instantiated."""
        # Cannot instantiate abstract class
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            KeyProvider()

    def test_default_methods(self):
        """Test default implementations using a concrete subclass."""
        # Create a minimal concrete implementation
        class ConcreteKeyProvider(KeyProvider):
            def get_key(self, key_id: str) -> bytes:
                return b"test"

            def create_key(self, key_type: str = "AES-256") -> str:
                return "test_key"

            def rotate_key(self, old_key_id: str) -> str:
                return "new_key"

        provider = ConcreteKeyProvider()

        # List keys returns empty list by default
        assert provider.list_keys() == []

        # Delete key returns False by default
        assert provider.delete_key("any_key") is False


class TestEnvironmentKeyProvider:
    """Test environment-based key provider."""

    @pytest.fixture
    def setup_env_keys(self):
        """Set up test environment variables."""
        # Create test keys
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()

        # Set environment variables
        os.environ["CLOAKPIVOT_KEY_TEST1"] = base64.b64encode(key1).decode()
        os.environ["CLOAKPIVOT_KEY_TEST2"] = base64.b64encode(key2).decode()

        yield key1, key2

        # Cleanup
        del os.environ["CLOAKPIVOT_KEY_TEST1"]
        del os.environ["CLOAKPIVOT_KEY_TEST2"]

    def test_initialization(self):
        """Test provider initialization."""
        provider = EnvironmentKeyProvider()
        assert provider.prefix == "CLOAKPIVOT_KEY"
        assert provider._key_cache == {}

        # Test with custom prefix
        provider = EnvironmentKeyProvider(prefix="CUSTOM_PREFIX")
        assert provider.prefix == "CUSTOM_PREFIX"

    def test_get_key_success(self, setup_env_keys):
        """Test successful key retrieval."""
        provider = EnvironmentKeyProvider()
        key1_expected, key2_expected = setup_env_keys

        # Get first key
        key1 = provider.get_key("test1")
        assert key1 == key1_expected

        # Get second key
        key2 = provider.get_key("test2")
        assert key2 == key2_expected

        # Verify caching
        assert "test1" in provider._key_cache
        assert "test2" in provider._key_cache

    def test_get_key_not_found(self):
        """Test key not found error."""
        provider = EnvironmentKeyProvider()

        with pytest.raises(KeyError) as exc_info:
            provider.get_key("nonexistent")

        assert "not found in environment" in str(exc_info.value)

    def test_get_key_invalid_format(self):
        """Test invalid key format error."""
        provider = EnvironmentKeyProvider()

        # Set invalid base64
        os.environ["CLOAKPIVOT_KEY_INVALID"] = "not-valid-base64!"

        try:
            with pytest.raises(ValueError) as exc_info:
                provider.get_key("invalid")

            assert "Invalid key format" in str(exc_info.value)
        finally:
            del os.environ["CLOAKPIVOT_KEY_INVALID"]

    def test_create_key(self, capsys):
        """Test key creation (prints instruction)."""
        provider = EnvironmentKeyProvider()

        # Create key
        key_id = provider.create_key("AES-256")

        # Verify key ID generated
        assert key_id.startswith("key_")

        # Check that the key was set in environment
        env_var_name = f"CLOAKPIVOT_KEY_{key_id.upper()}"
        import os
        assert env_var_name in os.environ
        # The key should be base64 encoded in the environment
        assert len(os.environ[env_var_name]) > 0

    def test_rotate_key(self, setup_env_keys, capsys):
        """Test key rotation."""
        provider = EnvironmentKeyProvider()
        key1, _ = setup_env_keys

        # Cache the old key
        old_key = provider.get_key("test1")
        assert old_key == key1

        # Rotate
        new_key_id = provider.rotate_key("test1")

        # Verify new key ID generated
        assert new_key_id != "test1"
        assert new_key_id.startswith("key_")

        # Verify old key removed from cache
        assert "test1" not in provider._key_cache

        # Check that the new key was set in environment
        env_var_name = f"CLOAKPIVOT_KEY_{new_key_id.upper()}"
        import os
        assert env_var_name in os.environ
        # The key should be base64 encoded in the environment
        assert len(os.environ[env_var_name]) > 0

    def test_list_keys(self, setup_env_keys):
        """Test listing available keys."""
        provider = EnvironmentKeyProvider()

        keys = provider.list_keys()

        # Should find our test keys
        assert "test1" in keys
        assert "test2" in keys

    def test_key_caching(self, setup_env_keys):
        """Test that keys are cached after first retrieval."""
        provider = EnvironmentKeyProvider()
        key1_expected, _ = setup_env_keys

        # First retrieval
        key1 = provider.get_key("test1")
        assert key1 == key1_expected

        # Remove from environment
        del os.environ["CLOAKPIVOT_KEY_TEST1"]

        # Should still work from cache
        key1_cached = provider.get_key("test1")
        assert key1_cached == key1_expected

        # Restore for cleanup
        os.environ["CLOAKPIVOT_KEY_TEST1"] = base64.b64encode(key1_expected).decode()


class TestFileKeyProvider:
    """Test file-based key provider."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_initialization(self, temp_dir):
        """Test provider initialization."""
        provider = FileKeyProvider(key_directory=temp_dir, password="test_password")

        assert provider.key_directory == Path(temp_dir)
        assert provider.password == "test_password"
        assert provider._metadata == {"keys": {}, "rotation_history": []}

        # Check metadata file created
        metadata_file = Path(temp_dir) / "metadata.json"
        assert metadata_file.exists()

    def test_initialization_with_password(self, temp_dir):
        """Test initialization with custom password."""
        provider = FileKeyProvider(key_directory=temp_dir, password="secret123")
        assert provider.password == "secret123"

    def test_create_and_get_key(self, temp_dir):
        """Test creating and retrieving a key."""
        provider = FileKeyProvider(key_directory=temp_dir, password="test_password")

        # Create key
        key_id = provider.create_key("AES-256")
        assert key_id.startswith("key_")

        # Verify key file created
        key_file = Path(temp_dir) / f"{key_id}.key"
        assert key_file.exists()

        # Get key
        key = provider.get_key(key_id)
        assert len(key) == 32  # 32 bytes for AES-256

        # Verify metadata updated
        assert key_id in provider._metadata["keys"]
        assert provider._metadata["keys"][key_id]["type"] == "AES-256"
        assert provider._metadata["keys"][key_id]["active"] is True

    def test_create_fernet_key(self, temp_dir):
        """Test creating a Fernet key."""
        provider = FileKeyProvider(key_directory=temp_dir, password="test_password")

        key_id = provider.create_key("Fernet")
        key = provider.get_key(key_id)

        # Fernet keys are URL-safe base64 encoded 32-byte keys (44 chars when encoded)
        # The key returned should be the encoded form
        assert len(key) == 44 or len(base64.urlsafe_b64decode(key)) == 32

    def test_encrypted_key_storage(self, temp_dir):
        """Test key encryption with password."""
        provider = FileKeyProvider(key_directory=temp_dir, password="test_password")

        # Create key
        key_id = provider.create_key("AES-256")
        original_key = provider.get_key(key_id)

        # Read raw file content
        key_file = Path(temp_dir) / f"{key_id}.key"
        with open(key_file, "rb") as f:
            raw_content = f.read()

        # Should be encrypted (salt + encrypted data)
        assert len(raw_content) > 32  # More than just the key
        assert raw_content[:16] != original_key[:16]  # Not plaintext

    def test_rotate_key(self, temp_dir):
        """Test key rotation."""
        provider = FileKeyProvider(key_directory=temp_dir, password="test_password")

        # Create initial key
        old_key_id = provider.create_key("AES-256")
        old_key = provider.get_key(old_key_id)

        # Rotate
        new_key_id = provider.rotate_key(old_key_id)
        new_key = provider.get_key(new_key_id)

        # Verify different keys
        assert new_key_id != old_key_id
        assert new_key != old_key

        # Check metadata
        assert provider._metadata["keys"][old_key_id]["active"] is False
        assert provider._metadata["keys"][old_key_id]["replaced_by"] == new_key_id
        assert provider._metadata["keys"][new_key_id]["active"] is True

        # Check rotation history
        assert len(provider._metadata["rotation_history"]) == 1
        assert provider._metadata["rotation_history"][0]["old_key"] == old_key_id
        assert provider._metadata["rotation_history"][0]["new_key"] == new_key_id

    def test_rotate_nonexistent_key(self, temp_dir):
        """Test rotating a key that doesn't exist."""
        provider = FileKeyProvider(key_directory=temp_dir, password="test_password")

        with pytest.raises(KeyError) as exc_info:
            provider.rotate_key("nonexistent")

        assert "not found" in str(exc_info.value)

    def test_list_keys(self, temp_dir):
        """Test listing keys."""
        provider = FileKeyProvider(key_directory=temp_dir, password="test_password")

        # Create multiple keys
        key_ids = []
        for _i in range(3):
            key_id = provider.create_key("AES-256")
            key_ids.append(key_id)

        # List keys
        listed_keys = provider.list_keys()

        # Verify all keys listed
        for key_id in key_ids:
            assert key_id in listed_keys

    def test_delete_key(self, temp_dir):
        """Test deleting a key."""
        provider = FileKeyProvider(key_directory=temp_dir, password="test_password")

        # Create key
        key_id = provider.create_key("AES-256")
        key_file = Path(temp_dir) / f"{key_id}.key"

        # Verify key exists
        assert key_file.exists()
        assert key_id in provider._metadata["keys"]

        # Delete key
        result = provider.delete_key(key_id)
        assert result is True

        # Verify key deleted
        assert not key_file.exists()
        assert key_id not in provider._metadata["keys"]

        # Try to get deleted key
        with pytest.raises(KeyError):
            provider.get_key(key_id)

    def test_delete_nonexistent_key(self, temp_dir):
        """Test deleting a key that doesn't exist."""
        provider = FileKeyProvider(key_directory=temp_dir, password="test_password")

        result = provider.delete_key("nonexistent")
        assert result is False

    def test_metadata_persistence(self, temp_dir):
        """Test that metadata persists across provider instances."""
        # Create provider and add keys
        provider1 = FileKeyProvider(key_directory=temp_dir, password="test_password")
        key_id1 = provider1.create_key("AES-256")
        key_id2 = provider1.create_key("Fernet")

        # Create new provider instance
        provider2 = FileKeyProvider(key_directory=temp_dir, password="test_password")

        # Verify metadata loaded
        assert key_id1 in provider2._metadata["keys"]
        assert key_id2 in provider2._metadata["keys"]

        # Verify can get keys
        key1 = provider2.get_key(key_id1)
        key2 = provider2.get_key(key_id2)
        assert key1 is not None
        assert key2 is not None


class TestVaultKeyProvider:
    """Test HashiCorp Vault key provider."""

    @patch('cloakpivot.presidio.key_management.HVAC_AVAILABLE', True)
    @patch('cloakpivot.presidio.key_management.hvac')
    def test_initialization_success(self, mock_hvac):
        """Test successful Vault provider initialization."""
        # Mock Vault client
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client

        provider = VaultKeyProvider(
            vault_url="https://vault.example.com",
            vault_token="test_token",
            secret_path="test/keys"
        )

        assert provider.vault_url == "https://vault.example.com"
        assert provider.vault_token == "test_token"
        assert provider.secret_path == "test/keys"

        # Verify client created
        mock_hvac.Client.assert_called_once_with(
            url="https://vault.example.com",
            token="test_token"
        )

    @patch('cloakpivot.presidio.key_management.HVAC_AVAILABLE', True)
    @patch('cloakpivot.presidio.key_management.hvac')
    def test_initialization_from_env(self, mock_hvac):
        """Test initialization with token from environment."""
        # Set environment variable
        os.environ["VAULT_TOKEN"] = "env_token"

        try:
            # Mock Vault client
            mock_client = MagicMock()
            mock_client.is_authenticated.return_value = True
            mock_hvac.Client.return_value = mock_client

            provider = VaultKeyProvider(
                vault_url="https://vault.example.com"
            )

            assert provider.vault_token == "env_token"
        finally:
            del os.environ["VAULT_TOKEN"]

    def test_initialization_without_hvac(self):
        """Test error when hvac is not installed."""
        with patch('builtins.__import__', side_effect=ImportError()):
            with pytest.raises(ImportError) as exc_info:
                VaultKeyProvider(
                    vault_url="https://vault.example.com",
                    vault_token="test_token"
                )

            assert "hvac library required" in str(exc_info.value)

    @patch('cloakpivot.presidio.key_management.HVAC_AVAILABLE', True)
    @patch('cloakpivot.presidio.key_management.hvac')
    def test_initialization_auth_failure(self, mock_hvac):
        """Test error when Vault authentication fails."""
        # Mock failed authentication
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = False
        mock_hvac.Client.return_value = mock_client

        with pytest.raises(ValueError) as exc_info:
            VaultKeyProvider(
                vault_url="https://vault.example.com",
                vault_token="test_token"
            )

        assert "Failed to authenticate" in str(exc_info.value)

    @patch('cloakpivot.presidio.key_management.HVAC_AVAILABLE', True)
    @patch('cloakpivot.presidio.key_management.hvac')
    def test_get_key(self, mock_hvac):
        """Test retrieving key from Vault."""
        # Mock Vault client
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True

        # Mock key retrieval
        test_key = Fernet.generate_key()
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {
                "data": {
                    "key": base64.b64encode(test_key).decode()
                }
            }
        }
        mock_hvac.Client.return_value = mock_client

        provider = VaultKeyProvider(
            vault_url="https://vault.example.com",
            vault_token="test_token"
        )

        # Get key
        key = provider.get_key("test_key_id")

        assert key == test_key

        # Verify Vault call
        mock_client.secrets.kv.v2.read_secret_version.assert_called_once_with(
            path="cloakpivot/keys/test_key_id"
        )

    @patch('cloakpivot.presidio.key_management.HVAC_AVAILABLE', True)
    @patch('cloakpivot.presidio.key_management.hvac')
    def test_create_key(self, mock_hvac):
        """Test creating key in Vault."""
        # Mock Vault client
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client

        provider = VaultKeyProvider(
            vault_url="https://vault.example.com",
            vault_token="test_token"
        )

        # Create key
        key_id = provider.create_key("AES-256")

        # Verify key ID generated
        assert key_id.startswith("key_")

        # Verify Vault call
        mock_client.secrets.kv.v2.create_or_update_secret.assert_called_once()
        call_args = mock_client.secrets.kv.v2.create_or_update_secret.call_args

        assert f"cloakpivot/keys/{key_id}" in call_args[1]["path"]
        assert "key" in call_args[1]["secret"]
        assert call_args[1]["secret"]["type"] == "AES-256"
