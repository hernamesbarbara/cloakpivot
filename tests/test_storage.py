"""
Tests for CloakMap storage backend system.

Comprehensive tests covering all storage backends, configuration management,
and registry functionality.
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.storage import (
    DatabaseStorage,
    GCSStorage,
    LocalStorage,
    S3Storage,
    StorageBackend,
    StorageConfig,
    StorageRegistry,
)
from cloakpivot.storage.backends.base import StorageMetadata


class TestStorageMetadata:
    """Test StorageMetadata functionality."""

    def test_from_cloakmap(self):
        """Test creating metadata from CloakMap."""
        # Create a simple CloakMap
        anchors = [
            AnchorEntry.create_from_detection(
                node_id="p1",
                start=0,
                end=10,
                entity_type="PERSON",
                confidence=0.9,
                original_text="John Doe",
                masked_value="[PERSON]",
                strategy_used="template",
            )
        ]

        cloakmap = CloakMap.create(
            doc_id="test_doc", doc_hash="abc123", anchors=anchors
        )

        content_bytes = b'{"test": "content"}'

        metadata = StorageMetadata.from_cloakmap(
            key="test_key",
            cloakmap=cloakmap,
            backend_type="test_backend",
            content_bytes=content_bytes,
        )

        assert metadata.key == "test_key"
        assert metadata.doc_id == "test_doc"
        assert metadata.version == "1.0"
        assert metadata.anchor_count == 1
        assert metadata.backend_type == "test_backend"
        assert metadata.size_bytes == len(content_bytes)

    def test_to_from_dict_roundtrip(self):
        """Test metadata serialization roundtrip."""
        metadata = StorageMetadata(
            key="test_key",
            size_bytes=1024,
            content_hash="hash123",
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
            doc_id="test_doc",
            version="1.0",
            anchor_count=5,
            is_encrypted=False,
            backend_type="local",
        )

        # Convert to dict and back
        data = metadata.to_dict()
        restored = StorageMetadata.from_dict(data)

        assert restored.key == metadata.key
        assert restored.doc_id == metadata.doc_id
        assert restored.backend_type == metadata.backend_type


class TestLocalStorage:
    """Test local filesystem storage backend."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(self.temp_dir)

        # Create test CloakMap
        anchors = [
            AnchorEntry.create_from_detection(
                node_id="p1",
                start=0,
                end=8,
                entity_type="PERSON",
                confidence=0.9,
                original_text="John Doe",
                masked_value="[PERSON]",
                strategy_used="template",
            )
        ]

        self.test_cloakmap = CloakMap.create(
            doc_id="test_document", doc_hash="abc123def456", anchors=anchors
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_backend_type(self):
        """Test backend type identifier."""
        assert self.storage.backend_type == "local_filesystem"

    def test_save_and_load(self):
        """Test basic save and load operations."""
        key = "test_document"

        # Save CloakMap
        metadata = self.storage.save(key, self.test_cloakmap)
        assert isinstance(metadata, StorageMetadata)
        assert metadata.key == key
        assert metadata.doc_id == self.test_cloakmap.doc_id

        # Load CloakMap
        loaded_cloakmap = self.storage.load(key)
        assert isinstance(loaded_cloakmap, CloakMap)
        assert loaded_cloakmap.doc_id == self.test_cloakmap.doc_id
        assert loaded_cloakmap.doc_hash == self.test_cloakmap.doc_hash
        assert len(loaded_cloakmap.anchors) == len(self.test_cloakmap.anchors)

    def test_exists(self):
        """Test existence checking."""
        key = "test_document"

        # Should not exist initially
        assert not self.storage.exists(key)

        # Save and check existence
        self.storage.save(key, self.test_cloakmap)
        assert self.storage.exists(key)

    def test_delete(self):
        """Test deletion."""
        key = "test_document"

        # Save first
        self.storage.save(key, self.test_cloakmap)
        assert self.storage.exists(key)

        # Delete
        deleted = self.storage.delete(key)
        assert deleted is True
        assert not self.storage.exists(key)

        # Delete non-existent
        deleted = self.storage.delete("nonexistent")
        assert deleted is False

    def test_list_keys(self):
        """Test key listing."""
        keys = ["doc1", "doc2", "folder/doc3"]

        # Save multiple CloakMaps
        for key in keys:
            self.storage.save(key, self.test_cloakmap)

        # List all keys
        all_keys = self.storage.list_keys()
        assert len(all_keys) == len(keys)
        for key in keys:
            assert key in all_keys

        # List with prefix
        folder_keys = self.storage.list_keys(prefix="folder/")
        assert len(folder_keys) == 1
        assert "folder/doc3" in folder_keys

        # List with limit
        limited_keys = self.storage.list_keys(limit=2)
        assert len(limited_keys) == 2

    def test_get_metadata(self):
        """Test metadata retrieval."""
        key = "test_document"

        # Save first
        save_metadata = self.storage.save(key, self.test_cloakmap)

        # Get metadata
        metadata = self.storage.get_metadata(key)
        assert isinstance(metadata, StorageMetadata)
        assert metadata.key == key
        assert metadata.doc_id == self.test_cloakmap.doc_id
        assert metadata.content_hash == save_metadata.content_hash

    def test_nested_directories(self):
        """Test handling of nested directory structures."""
        key = "deep/nested/path/document"

        # Save to nested path
        metadata = self.storage.save(key, self.test_cloakmap)
        assert metadata.key == key

        # Verify file was created in correct location
        file_path = Path(self.temp_dir) / f"{key}.cmap"
        assert file_path.exists()

        # Load from nested path
        loaded = self.storage.load(key)
        assert loaded.doc_id == self.test_cloakmap.doc_id

    def test_health_check(self):
        """Test health check functionality."""
        health = self.storage.health_check()

        assert health["status"] == "healthy"
        assert health["backend_type"] == "local_filesystem"
        assert "base_path" in health
        assert health["directory_accessible"] is True
        assert health["can_write"] is True

    def test_invalid_key_validation(self):
        """Test key validation."""
        invalid_keys = ["", "   ", "\x00invalid", "key\nwith\nnewlines"]

        for invalid_key in invalid_keys:
            with pytest.raises(ValueError):
                self.storage.save(invalid_key, self.test_cloakmap)


class TestStorageRegistry:
    """Test storage registry functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = StorageRegistry()

    def test_builtin_backends_registered(self):
        """Test that built-in backends are registered."""
        backend_types = self.registry.list_backend_types()

        assert "local_filesystem" in backend_types
        assert "aws_s3" in backend_types
        assert "google_cloud_storage" in backend_types
        assert "database" in backend_types

    def test_aliases(self):
        """Test backend type aliases."""
        aliases = self.registry.list_aliases()

        assert "local" in aliases
        assert aliases["local"] == "local_filesystem"
        assert "s3" in aliases
        assert aliases["s3"] == "aws_s3"
        assert "gcs" in aliases
        assert aliases["gcs"] == "google_cloud_storage"

    def test_resolve_backend_type(self):
        """Test backend type resolution."""
        # Direct types
        assert (
            self.registry.resolve_backend_type("local_filesystem") == "local_filesystem"
        )

        # Aliases
        assert self.registry.resolve_backend_type("local") == "local_filesystem"
        assert self.registry.resolve_backend_type("s3") == "aws_s3"
        assert self.registry.resolve_backend_type("gcs") == "google_cloud_storage"

        # Unknown type
        with pytest.raises(KeyError):
            self.registry.resolve_backend_type("unknown_backend")

    def test_get_backend_class(self):
        """Test getting backend classes."""
        local_class = self.registry.get_backend_class("local")
        assert local_class == LocalStorage

        s3_class = self.registry.get_backend_class("s3")
        assert s3_class == S3Storage

        gcs_class = self.registry.get_backend_class("gcs")
        assert gcs_class == GCSStorage

        db_class = self.registry.get_backend_class("database")
        assert db_class == DatabaseStorage

    def test_create_backend(self):
        """Test creating backend instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create local backend
            local_backend = self.registry.create_backend(
                "local", {"base_path": temp_dir}
            )
            assert isinstance(local_backend, LocalStorage)
            assert local_backend.backend_type == "local_filesystem"

        # Test with invalid configuration
        with pytest.raises(ValueError):
            self.registry.create_backend("local", {})  # Missing base_path

    def test_custom_backend_registration(self):
        """Test registering custom backends."""

        class CustomStorageBackend(StorageBackend):
            @property
            def backend_type(self) -> str:
                return "custom"

            def _validate_config(self) -> None:
                pass

            def save(self, key: str, cloakmap, metadata=None, **kwargs):
                pass

            def load(self, key: str, **kwargs):
                pass

            def exists(self, key: str, **kwargs) -> bool:
                return False

            def delete(self, key: str, **kwargs) -> bool:
                return False

            def list_keys(self, prefix=None, limit=None, **kwargs):
                return []

            def get_metadata(self, key: str, **kwargs):
                pass

        # Register custom backend
        self.registry.register_backend(
            "custom", CustomStorageBackend, ["c", "custom_alias"]
        )

        # Verify registration
        assert "custom" in self.registry.list_backend_types()
        aliases = self.registry.list_aliases()
        assert "c" in aliases
        assert aliases["c"] == "custom"
        assert "custom_alias" in aliases

        # Create instance
        backend = self.registry.create_backend("custom", {})
        assert isinstance(backend, CustomStorageBackend)

        # Test alias usage
        backend_via_alias = self.registry.create_backend("c", {})
        assert isinstance(backend_via_alias, CustomStorageBackend)

    def test_backend_info(self):
        """Test getting backend information."""
        info = self.registry.get_backend_info("local")

        assert info["backend_type"] == "local_filesystem"
        assert info["class_name"] == "LocalStorage"
        assert "local" in info["aliases"]
        assert info["docstring"] is not None

    def test_validate_backend(self):
        """Test backend validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid configuration
            result = self.registry.validate_backend("local", {"base_path": temp_dir})
            assert result["valid"] is True
            assert result["creation_successful"] is True
            assert len(result["errors"]) == 0

        # Invalid configuration
        result = self.registry.validate_backend("local", {})
        assert result["valid"] is False
        assert len(result["errors"]) > 0

        # Unknown backend
        result = self.registry.validate_backend("unknown", {})
        assert result["valid"] is False
        assert len(result["errors"]) > 0


class TestStorageConfig:
    """Test storage configuration system."""

    def test_basic_config(self):
        """Test basic configuration creation."""
        config = StorageConfig(backend_type="local", config={"base_path": "/tmp/test"})

        assert config.backend_type == "local"
        assert config.config["base_path"] == "/tmp/test"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "backend_type": "s3",
            "config": {"bucket_name": "my-bucket", "region_name": "us-west-2"},
            "fallback_backends": [
                {"backend_type": "local", "config": {"base_path": "/tmp"}}
            ],
        }

        config = StorageConfig.from_dict(data)
        assert config.backend_type == "s3"
        assert config.config["bucket_name"] == "my-bucket"
        assert len(config.fallback_backends) == 1

    def test_environment_overrides(self):
        """Test environment variable overrides."""
        with patch.dict(
            "os.environ",
            {
                "CLOAKPIVOT_STORAGE_BACKEND": "s3",
                "CLOAKPIVOT_STORAGE_BUCKET_NAME": "env-bucket",
                "AWS_DEFAULT_REGION": "us-east-1",
            },
        ):
            config = StorageConfig()
            assert config.backend_type == "s3"
            assert config.config["bucket_name"] == "env-bucket"
            assert config.config["region_name"] == "us-east-1"

    def test_yaml_serialization(self):
        """Test YAML serialization."""
        config = StorageConfig(
            backend_type="gcs", config={"bucket_name": "my-gcs-bucket"}
        )

        yaml_str = config.to_yaml()
        assert "backend_type: gcs" in yaml_str
        assert "bucket_name: my-gcs-bucket" in yaml_str

        # Test roundtrip
        data = __import__("yaml").safe_load(yaml_str)
        restored = StorageConfig.from_dict(data)
        assert restored.backend_type == config.backend_type
        assert restored.config == config.config

    def test_create_backend(self):
        """Test backend creation from config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(backend_type="local", config={"base_path": temp_dir})

            backend = config.create_backend()
            assert isinstance(backend, LocalStorage)
            assert backend.backend_type == "local_filesystem"

    def test_validation(self):
        """Test configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid configuration
            config = StorageConfig(backend_type="local", config={"base_path": temp_dir})

            result = config.validate()
            assert result["valid"] is True
            assert len(result["errors"]) == 0

        # Invalid configuration
        config = StorageConfig(
            backend_type="local",
            config={},  # Missing base_path
        )

        result = config.validate()
        assert result["valid"] is False
        assert len(result["errors"]) > 0


@pytest.mark.skipif(
    not __import__("shutil").which("sqlite3"), reason="SQLite not available"
)
class TestDatabaseStorage:
    """Test database storage backend (SQLite only for CI/CD)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.storage = DatabaseStorage(config={"database_url": "sqlite:///:memory:"})

        # Create test CloakMap
        anchors = [
            AnchorEntry.create_from_detection(
                node_id="p1",
                start=0,
                end=8,
                entity_type="PERSON",
                confidence=0.9,
                original_text="Jane Doe",
                masked_value="[PERSON]",
                strategy_used="template",
            )
        ]

        self.test_cloakmap = CloakMap.create(
            doc_id="test_database_document", doc_hash="def456ghi789", anchors=anchors
        )

    def test_backend_type(self):
        """Test backend type identifier."""
        assert self.storage.backend_type == "database"

    def test_save_and_load(self):
        """Test basic save and load operations."""
        key = "test_db_document"

        # Save CloakMap
        metadata = self.storage.save(key, self.test_cloakmap)
        assert isinstance(metadata, StorageMetadata)
        assert metadata.key == key
        assert metadata.doc_id == self.test_cloakmap.doc_id

        # Load CloakMap
        loaded_cloakmap = self.storage.load(key)
        assert isinstance(loaded_cloakmap, CloakMap)
        assert loaded_cloakmap.doc_id == self.test_cloakmap.doc_id
        assert loaded_cloakmap.doc_hash == self.test_cloakmap.doc_hash

    def test_exists_and_delete(self):
        """Test existence checking and deletion."""
        key = "test_db_document"

        # Should not exist initially
        assert not self.storage.exists(key)

        # Save and check existence
        self.storage.save(key, self.test_cloakmap)
        assert self.storage.exists(key)

        # Delete
        deleted = self.storage.delete(key)
        assert deleted is True
        assert not self.storage.exists(key)

    def test_list_keys(self):
        """Test key listing with database backend."""
        keys = ["db_doc1", "db_doc2", "prefix_doc3"]

        # Save multiple CloakMaps
        for key in keys:
            self.storage.save(key, self.test_cloakmap)

        # List all keys
        all_keys = self.storage.list_keys()
        assert len(all_keys) == len(keys)
        for key in keys:
            assert key in all_keys

        # List with prefix
        prefix_keys = self.storage.list_keys(prefix="prefix_")
        assert len(prefix_keys) == 1
        assert "prefix_doc3" in prefix_keys


if __name__ == "__main__":
    unittest.main()
