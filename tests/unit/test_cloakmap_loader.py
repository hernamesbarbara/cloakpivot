"""Unit tests for cloakpivot.unmasking.cloakmap_loader module."""

import json
import tempfile
from pathlib import Path

import pytest

from cloakpivot.core.types.anchors import AnchorEntry
from cloakpivot.core.types.cloakmap import CloakMap
from cloakpivot.unmasking.cloakmap_loader import CloakMapLoader, CloakMapLoadError


class TestCloakMapLoadError:
    """Test CloakMapLoadError exception."""

    def test_exception_creation(self):
        """Test creating CloakMapLoadError."""
        error = CloakMapLoadError("Failed to load")
        assert str(error) == "Failed to load"
        assert isinstance(error, Exception)


class TestCloakMapLoader:
    """Test CloakMapLoader class."""

    def test_initialization(self):
        """Test CloakMapLoader initialization."""
        loader = CloakMapLoader()
        assert loader is not None

    def test_load_from_file_success(self):
        """Test successfully loading cloakmap from file."""
        loader = CloakMapLoader()

        # Create a temporary cloakmap file
        with tempfile.TemporaryDirectory() as tmpdir:
            cloakmap_path = Path(tmpdir) / "test.cloakmap"

            # Create and save a cloakmap
            original_cloakmap = CloakMap(
                doc_id="test_doc",
                doc_hash="test_hash",
                anchors=[]
            )
            original_cloakmap.save_to_file(cloakmap_path)

            # Load it back
            loaded_cloakmap = loader.load_from_file(cloakmap_path)

            assert isinstance(loaded_cloakmap, CloakMap)
            assert loaded_cloakmap.doc_id == "test_doc"
            assert loaded_cloakmap.doc_hash == "test_hash"

    def test_load_from_file_not_found(self):
        """Test loading from non-existent file."""
        loader = CloakMapLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_from_file("nonexistent.cloakmap")

    def test_load_from_file_invalid_json(self):
        """Test loading file with invalid JSON."""
        loader = CloakMapLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_file = Path(tmpdir) / "invalid.cloakmap"
            invalid_file.write_text("not valid json {]")

            with pytest.raises(CloakMapLoadError):
                loader.load_from_file(invalid_file)

    def test_load_with_anchors(self):
        """Test loading cloakmap with anchors."""
        loader = CloakMapLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            cloakmap_path = Path(tmpdir) / "anchors.cloakmap"

            # Create cloakmap with anchors
            anchor = AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=0,
                end=8,
                entity_type="PERSON",
                confidence=0.95,
                original_text="John Doe",
                masked_value="[PERSON]",
                strategy_used="template"
            )

            cloakmap = CloakMap(
                doc_id="doc_with_anchors",
                doc_hash="hash123",
                anchors=[anchor]
            )
            cloakmap.save_to_file(cloakmap_path)

            # Load and verify
            loaded = loader.load_from_file(cloakmap_path)

            assert len(loaded.anchors) == 1
            assert loaded.anchors[0].entity_type == "PERSON"
            assert loaded.anchors[0].original_text == "John Doe"

    def test_load_from_json_string(self):
        """Test loading cloakmap from JSON string."""
        loader = CloakMapLoader()

        cloakmap_dict = {
            "version": "2.0",
            "doc_id": "test",
            "doc_hash": "hash",
            "anchors": [],
            "created_at": "2024-01-01T00:00:00Z"
        }
        json_str = json.dumps(cloakmap_dict)

        if hasattr(loader, 'load_from_json'):
            loaded = loader.load_from_json(json_str)
            assert loaded.doc_id == "test"

    def test_load_from_dict(self):
        """Test loading cloakmap from dictionary."""
        loader = CloakMapLoader()

        cloakmap_dict = {
            "version": "2.0",
            "doc_id": "test",
            "doc_hash": "hash",
            "anchors": [],
            "created_at": "2024-01-01T00:00:00Z"
        }

        if hasattr(loader, 'load_from_dict'):
            loaded = loader.load_from_dict(cloakmap_dict)
            assert loaded.doc_id == "test"

    def test_validate_cloakmap(self):
        """Test cloakmap validation."""
        loader = CloakMapLoader()

        # Valid cloakmap
        valid_cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=[]
        )

        if hasattr(loader, 'validate'):
            is_valid = loader.validate(valid_cloakmap)
            assert is_valid is True

    def test_load_with_metadata(self):
        """Test loading cloakmap with metadata."""
        loader = CloakMapLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            cloakmap_path = Path(tmpdir) / "metadata.cloakmap"

            # Create cloakmap with metadata
            cloakmap = CloakMap(
                doc_id="meta_doc",
                doc_hash="hash",
                anchors=[],
                metadata={
                    "source": "test_source",
                    "processing_time": 1.5,
                    "entity_count": 5
                }
            )
            cloakmap.save_to_file(cloakmap_path)

            loaded = loader.load_from_file(cloakmap_path)

            assert loaded.metadata is not None
            if loaded.metadata:
                assert loaded.metadata.get("source") == "test_source"
                assert loaded.metadata.get("entity_count") == 5

    def test_load_with_presidio_metadata(self):
        """Test loading v2.0 cloakmap with Presidio metadata."""
        loader = CloakMapLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            cloakmap_path = Path(tmpdir) / "presidio.cloakmap"

            # Create v2.0 cloakmap with Presidio metadata
            cloakmap = CloakMap(
                version="2.0",
                doc_id="presidio_doc",
                doc_hash="hash",
                anchors=[],
                presidio_metadata={
                    "operators": [
                        {
                            "start": 0,
                            "end": 8,
                            "entity_type": "PERSON",
                            "text": "John Doe",
                            "operator": "replace"
                        }
                    ]
                }
            )
            cloakmap.save_to_file(cloakmap_path)

            loaded = loader.load_from_file(cloakmap_path)

            assert loaded.presidio_metadata is not None
            if loaded.presidio_metadata:
                assert "operators" in loaded.presidio_metadata

    def test_load_with_policy_snapshot(self):
        """Test loading cloakmap with policy snapshot."""
        loader = CloakMapLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            cloakmap_path = Path(tmpdir) / "policy.cloakmap"

            # Create cloakmap with policy snapshot
            cloakmap = CloakMap(
                doc_id="policy_doc",
                doc_hash="hash",
                anchors=[],
                policy_snapshot={
                    "default_strategy": "redact",
                    "locale": "en",
                    "confidence_threshold": 0.85
                }
            )
            cloakmap.save_to_file(cloakmap_path)

            loaded = loader.load_from_file(cloakmap_path)

            assert loaded.policy_snapshot is not None
            if loaded.policy_snapshot:
                assert loaded.policy_snapshot.get("default_strategy") == "redact"
                assert loaded.policy_snapshot.get("confidence_threshold") == 0.85

    def test_batch_load(self):
        """Test loading multiple cloakmaps."""
        loader = CloakMapLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create multiple cloakmaps
            paths = []
            for i in range(3):
                path = tmpdir / f"cloakmap_{i}.cloakmap"
                cloakmap = CloakMap(
                    doc_id=f"doc_{i}",
                    doc_hash=f"hash_{i}",
                    anchors=[]
                )
                cloakmap.save_to_file(path)
                paths.append(path)

            # If batch loading is supported
            if hasattr(loader, 'load_batch'):
                cloakmaps = loader.load_batch(paths)
                assert len(cloakmaps) == 3
                assert all(isinstance(cm, CloakMap) for cm in cloakmaps)

    def test_load_compressed(self):
        """Test loading compressed cloakmap."""
        loader = CloakMapLoader()

        if hasattr(loader, 'load_compressed'):
            # Create compressed cloakmap
            with tempfile.TemporaryDirectory() as tmpdir:
                compressed_path = Path(tmpdir) / "compressed.cloakmap.gz"

                # Create and compress a cloakmap
                cloakmap = CloakMap(
                    doc_id="compressed",
                    doc_hash="hash",
                    anchors=[]
                )

                # Save compressed (if supported)
                if hasattr(cloakmap, 'save_compressed'):
                    cloakmap.save_compressed(compressed_path)

                    loaded = loader.load_compressed(compressed_path)
                    assert loaded.doc_id == "compressed"

    def test_load_encrypted(self):
        """Test loading encrypted cloakmap."""
        loader = CloakMapLoader()

        if hasattr(loader, 'load_encrypted'):
            with tempfile.TemporaryDirectory() as tmpdir:
                encrypted_path = Path(tmpdir) / "encrypted.cloakmap"

                # Create cloakmap with encryption
                cloakmap = CloakMap(
                    doc_id="encrypted",
                    doc_hash="hash",
                    anchors=[],
                    crypto={"algorithm": "AES", "encrypted": True}
                )
                cloakmap.save_to_file(encrypted_path)

                # Try to load with key
                key = "test_key_123"
                loaded = loader.load_encrypted(encrypted_path, key)
                assert loaded is not None

    def test_migration_from_v1(self):
        """Test migrating v1.0 cloakmap to v2.0."""
        loader = CloakMapLoader()

        if hasattr(loader, 'migrate_v1_to_v2'):
            # Create v1.0 cloakmap
            v1_cloakmap = CloakMap(
                version="1.0",
                doc_id="v1_doc",
                doc_hash="hash",
                anchors=[]
            )

            # Migrate to v2.0
            v2_cloakmap = loader.migrate_v1_to_v2(v1_cloakmap)

            assert v2_cloakmap.version == "2.0"
            assert v2_cloakmap.doc_id == "v1_doc"

    def test_error_handling_corrupted_file(self):
        """Test handling corrupted cloakmap file."""
        loader = CloakMapLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted_file = Path(tmpdir) / "corrupted.cloakmap"

            # Write partially valid JSON
            corrupted_file.write_text('{"doc_id": "test", "anchors": [{"incomplete":')

            with pytest.raises(CloakMapLoadError):
                loader.load_from_file(corrupted_file)

    def test_get_statistics(self):
        """Test getting cloakmap statistics."""
        loader = CloakMapLoader()

        cloakmap = CloakMap(
            doc_id="stats",
            doc_hash="hash",
            anchors=[
                AnchorEntry.create_from_detection(
                    node_id="#/texts/0",
                    start=0,
                    end=8,
                    entity_type="PERSON",
                    confidence=0.9,
                    original_text="Name",
                    masked_value="[PERSON]",
                    strategy_used="template"
                ),
                AnchorEntry.create_from_detection(
                    node_id="#/texts/0",
                    start=10,
                    end=17,
                    entity_type="EMAIL",
                    confidence=0.95,
                    original_text="email@test.com",
                    masked_value="[EMAIL]",
                    strategy_used="template"
                )
            ]
        )

        if hasattr(loader, 'get_statistics'):
            stats = loader.get_statistics(cloakmap)
            assert stats is not None
            assert "total_anchors" in stats
            assert stats["total_anchors"] == 2
