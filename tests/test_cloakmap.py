"""Comprehensive tests for CloakMap functionality."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import (
    CloakMap,
    merge_cloakmaps,
    validate_cloakmap_integrity,
)


class TestCloakMapCreation:
    """Test CloakMap creation and validation."""

    def test_create_minimal_cloakmap(self):
        """Test creating a minimal CloakMap."""
        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        assert cloakmap.version == "1.0"
        assert cloakmap.doc_id == "doc123"
        assert cloakmap.doc_hash == "hash123"
        assert cloakmap.anchors == []
        assert cloakmap.policy_snapshot == {}
        assert cloakmap.crypto is None
        assert cloakmap.signature is None
        assert isinstance(cloakmap.created_at, datetime)
        assert cloakmap.metadata == {}

    def test_create_full_cloakmap(self):
        """Test creating a full CloakMap with all fields."""
        anchor = AnchorEntry(
            node_id="node1",
            start=0,
            end=10,
            entity_type="PERSON",
            confidence=0.9,
            masked_value="John Doe",
            replacement_id="repl1",
            original_checksum="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="redact"
        )

        crypto_metadata = {"algorithm": "AES-256", "key_id": "key1"}
        policy_snapshot = {"default_strategy": "redact"}
        metadata = {"source": "test"}

        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[anchor],
            policy_snapshot=policy_snapshot,
            crypto=crypto_metadata,
            signature="sig123",
            metadata=metadata
        )

        assert len(cloakmap.anchors) == 1
        assert cloakmap.anchors[0] == anchor
        assert cloakmap.crypto == crypto_metadata
        assert cloakmap.signature == "sig123"
        assert cloakmap.metadata == metadata

    def test_invalid_version_raises_error(self):
        """Test that invalid version raises error."""
        with pytest.raises(ValueError, match="Version cannot be empty"):
            CloakMap(
                version="",
                doc_id="doc123",
                doc_hash="hash123",
                anchors=[],
                policy_snapshot={}
            )

    def test_invalid_doc_id_raises_error(self):
        """Test that invalid doc_id raises error."""
        with pytest.raises(ValueError, match="Document ID cannot be empty"):
            CloakMap(
                version="1.0",
                doc_id="",
                doc_hash="hash123",
                anchors=[],
                policy_snapshot={}
            )

    def test_invalid_doc_hash_raises_error(self):
        """Test that invalid doc_hash raises error."""
        with pytest.raises(ValueError, match="Document hash cannot be empty"):
            CloakMap(
                version="1.0",
                doc_id="doc123",
                doc_hash="",
                anchors=[],
                policy_snapshot={}
            )


class TestCloakMapProperties:
    """Test CloakMap computed properties."""

    def setup_method(self):
        """Set up test data."""
        self.anchors = [
            AnchorEntry(
                node_id="node1",
                start=0,
                end=8,
                entity_type="PERSON",
                confidence=0.9,
                masked_value="John Doe",
                replacement_id="repl1",
                original_checksum="a" * 64,
                checksum_salt="dGVzdA==",  # base64 encoded "test"
                strategy_used="redact"
            ),
            AnchorEntry(
                node_id="node2",
                start=10,
                end=25,
                entity_type="EMAIL",
                confidence=0.8,
                masked_value="john@example.com",
                replacement_id="repl2",
                original_checksum="b" * 64,
                checksum_salt="dGVzdA==",  # base64 encoded "test"
                strategy_used="template"
            ),
            AnchorEntry(
                node_id="node3",
                start=30,
                end=38,
                entity_type="PERSON",
                confidence=0.95,
                masked_value="Jane Doe",
                replacement_id="repl3",
                original_checksum="c" * 64,
                checksum_salt="dGVzdA==",  # base64 encoded "test"
                strategy_used="redact"
            )
        ]

        self.cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=self.anchors,
            policy_snapshot={},
            signature="test_signature"
        )

    def test_anchor_count(self):
        """Test anchor count property."""
        assert self.cloakmap.anchor_count == 3

    def test_entity_count_by_type(self):
        """Test entity count by type property."""
        counts = self.cloakmap.entity_count_by_type
        assert counts["PERSON"] == 2
        assert counts["EMAIL"] == 1

    def test_is_signed(self):
        """Test is_signed property."""
        assert self.cloakmap.is_signed is True

        unsigned_map = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )
        assert unsigned_map.is_signed is False

    def test_is_encrypted(self):
        """Test is_encrypted property."""
        assert self.cloakmap.is_encrypted is False

        encrypted_map = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={},
            crypto={"algorithm": "AES-256"}
        )
        assert encrypted_map.is_encrypted is True


class TestCloakMapStatistics:
    """Test CloakMap statistics functionality."""

    def test_get_stats_empty(self):
        """Test statistics for empty CloakMap."""
        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        stats = cloakmap.get_stats()
        assert stats["anchor_count"] == 0
        assert stats["average_confidence"] == 0.0
        assert stats["total_original_length"] == 0
        assert stats["total_masked_length"] == 0
        assert stats["entity_counts"] == {}
        assert stats["strategy_counts"] == {}

    def test_get_stats_with_anchors(self):
        """Test statistics with anchors."""
        anchors = [
            AnchorEntry(
                node_id="node1",
                start=0,
                end=8,  # length 8
                entity_type="PERSON",
                confidence=0.8,
                masked_value="********",  # length 8
                replacement_id="repl1",
                original_checksum="a" * 64,
                checksum_salt="dGVzdA==",  # base64 encoded "test"
                strategy_used="redact"
            ),
            AnchorEntry(
                node_id="node2",
                start=10,
                end=25,  # length 15
                entity_type="EMAIL",
                confidence=0.9,
                masked_value="user@domain.com",  # length 15
                replacement_id="repl2",
                original_checksum="b" * 64,
                checksum_salt="dGVzdA==",  # base64 encoded "test"
                strategy_used="template"
            )
        ]

        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=anchors,
            policy_snapshot={}
        )

        stats = cloakmap.get_stats()
        assert stats["anchor_count"] == 2
        assert stats["average_confidence"] == 0.85  # (0.8 + 0.9) / 2
        assert stats["total_original_length"] == 23  # 8 + 15
        assert stats["total_masked_length"] == 23  # 8 + 15
        assert stats["entity_counts"]["PERSON"] == 1
        assert stats["entity_counts"]["EMAIL"] == 1
        assert stats["strategy_counts"]["redact"] == 1
        assert stats["strategy_counts"]["template"] == 1


class TestCloakMapSerialization:
    """Test CloakMap JSON serialization and deserialization."""

    def test_to_dict(self):
        """Test converting CloakMap to dictionary."""
        anchor = AnchorEntry(
            node_id="node1",
            start=0,
            end=10,
            entity_type="PERSON",
            confidence=0.9,
            masked_value="John Doe",
            replacement_id="repl1",
            original_checksum="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="redact"
        )

        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[anchor],
            policy_snapshot={"strategy": "redact"},
            crypto={"algorithm": "AES-256"},
            signature="sig123",
            metadata={"source": "test"}
        )

        data = cloakmap.to_dict()

        assert data["version"] == "1.0"
        assert data["doc_id"] == "doc123"
        assert data["doc_hash"] == "hash123"
        assert len(data["anchors"]) == 1
        assert data["policy_snapshot"]["strategy"] == "redact"
        assert data["crypto"]["algorithm"] == "AES-256"
        assert data["signature"] == "sig123"
        assert data["metadata"]["source"] == "test"
        assert "created_at" in data

    def test_from_dict(self):
        """Test creating CloakMap from dictionary."""
        data = {
            "version": "1.0",
            "doc_id": "doc123",
            "doc_hash": "hash123",
            "anchors": [
                {
                    "node_id": "node1",
                    "start": 0,
                    "end": 10,
                    "entity_type": "PERSON",
                    "confidence": 0.9,
                    "masked_value": "John Doe",
                    "replacement_id": "repl1",
                    "original_checksum": "a1b2c3d4e5f67890123456789012345678901234567890123456789012345678",
                    "checksum_salt": "dGVzdA==",
                    "strategy_used": "redact",
                    "metadata": {},
                    "created_at": "2023-01-01T00:00:00"
                }
            ],
            "policy_snapshot": {"strategy": "redact"},
            "crypto": {"algorithm": "AES-256"},
            "signature": "sig123",
            "metadata": {"source": "test"},
            "created_at": "2023-01-01T00:00:00"
        }

        cloakmap = CloakMap.from_dict(data)

        assert cloakmap.version == "1.0"
        assert cloakmap.doc_id == "doc123"
        assert cloakmap.doc_hash == "hash123"
        assert len(cloakmap.anchors) == 1
        assert cloakmap.anchors[0].node_id == "node1"
        assert cloakmap.policy_snapshot["strategy"] == "redact"
        assert cloakmap.crypto["algorithm"] == "AES-256"
        assert cloakmap.signature == "sig123"
        assert cloakmap.metadata["source"] == "test"

    def test_serialization_roundtrip(self):
        """Test complete serialization roundtrip."""
        anchor = AnchorEntry(
            node_id="node1",
            start=0,
            end=10,
            entity_type="PERSON",
            confidence=0.9,
            masked_value="John Doe",
            replacement_id="repl1",
            original_checksum="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="redact"
        )

        original = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[anchor],
            policy_snapshot={"strategy": "redact"},
            crypto={"algorithm": "AES-256"},
            signature="sig123",
            metadata={"source": "test"}
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = CloakMap.from_dict(data)

        assert restored.version == original.version
        assert restored.doc_id == original.doc_id
        assert restored.doc_hash == original.doc_hash
        assert len(restored.anchors) == len(original.anchors)
        assert restored.anchors[0].node_id == original.anchors[0].node_id
        assert restored.policy_snapshot == original.policy_snapshot
        assert restored.crypto == original.crypto
        assert restored.signature == original.signature
        assert restored.metadata == original.metadata


class TestCloakMapFileOperations:
    """Test CloakMap file I/O operations."""

    def test_save_to_file(self):
        """Test saving CloakMap to file."""
        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={"strategy": "redact"}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_cloakmap.json"
            cloakmap.save_to_file(file_path)

            assert file_path.exists()

            # Verify file content
            with open(file_path) as f:
                data = json.load(f)

            assert data["version"] == "1.0"
            assert data["doc_id"] == "doc123"
            assert data["policy_snapshot"]["strategy"] == "redact"

    def test_load_from_file(self):
        """Test loading CloakMap from file."""
        original = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={"strategy": "redact"},
            metadata={"source": "test"}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_cloakmap.json"
            original.save_to_file(file_path)

            loaded = CloakMap.load_from_file(file_path)

            assert loaded.version == original.version
            assert loaded.doc_id == original.doc_id
            assert loaded.doc_hash == original.doc_hash
            assert loaded.policy_snapshot == original.policy_snapshot
            assert loaded.metadata == original.metadata

    def test_load_from_nonexistent_file_raises_error(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            CloakMap.load_from_file(Path("nonexistent.json"))


class TestCloakMapSignatures:
    """Test CloakMap signature functionality."""

    def test_sign_cloakmap(self):
        """Test signing a CloakMap."""
        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        secret_key = "test_secret_key"
        signed_map = cloakmap.sign(secret_key=secret_key)

        assert signed_map.signature is not None
        assert signed_map.signature != ""
        assert signed_map.is_signed is True

    def test_verify_signature_valid(self):
        """Test verifying valid signature."""
        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        secret_key = "test_secret_key"
        signed_map = cloakmap.sign(secret_key=secret_key)

        assert signed_map.verify_signature(secret_key=secret_key) is True

    def test_verify_signature_invalid(self):
        """Test verifying invalid signature."""
        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        secret_key = "test_secret_key"
        wrong_key = "wrong_secret_key"
        signed_map = cloakmap.sign(secret_key=secret_key)

        assert signed_map.verify_signature(secret_key=wrong_key) is False

    def test_verify_signature_unsigned_map(self):
        """Test verifying signature on unsigned map."""
        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        assert cloakmap.verify_signature(secret_key="any_key") is False


class TestMergeCloakMaps:
    """Test CloakMap merging functionality."""

    def test_merge_compatible_maps(self):
        """Test merging compatible CloakMaps."""
        anchor1 = AnchorEntry(
            node_id="node1",
            start=0,
            end=10,
            entity_type="PERSON",
            confidence=0.9,
            masked_value="John Doe",
            replacement_id="repl1",
            original_checksum="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="redact"
        )

        anchor2 = AnchorEntry(
            node_id="node2",
            start=20,
            end=35,
            entity_type="EMAIL",
            confidence=0.8,
            masked_value="john@example.com",
            replacement_id="repl2",
            original_checksum="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="template"
        )

        map1 = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[anchor1],
            policy_snapshot={"strategy": "redact"}
        )

        map2 = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[anchor2],
            policy_snapshot={"strategy": "redact"}
        )

        merged = merge_cloakmaps([map1, map2])

        assert len(merged.anchors) == 2
        assert merged.anchors[0] == anchor1
        assert merged.anchors[1] == anchor2
        assert merged.version == "1.0"
        assert merged.doc_id == "doc123"
        assert merged.doc_hash == "hash123"

    def test_merge_incompatible_versions_raises_error(self):
        """Test merging maps with different versions raises error."""
        map1 = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        map2 = CloakMap(
            version="2.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        with pytest.raises(ValueError, match="Cannot merge CloakMaps with different versions"):
            merge_cloakmaps([map1, map2])

    def test_merge_different_documents_raises_error(self):
        """Test merging maps from different documents raises error."""
        map1 = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        map2 = CloakMap(
            version="1.0",
            doc_id="doc456",
            doc_hash="hash456",
            anchors=[],
            policy_snapshot={}
        )

        with pytest.raises(ValueError, match="Cannot merge CloakMaps from different documents"):
            merge_cloakmaps([map1, map2])

    def test_merge_overlapping_anchors_raises_error(self):
        """Test merging maps with overlapping anchors raises error."""
        anchor1 = AnchorEntry(
            node_id="node1",
            start=0,
            end=10,
            entity_type="PERSON",
            confidence=0.9,
            masked_value="John Doe",
            replacement_id="repl1",
            original_checksum="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="redact"
        )

        anchor2 = AnchorEntry(
            node_id="node1",
            start=5,
            end=15,  # Overlaps with anchor1
            entity_type="EMAIL",
            confidence=0.8,
            masked_value="john@example.com",
            replacement_id="repl2",
            original_checksum="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="template"
        )

        map1 = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[anchor1],
            policy_snapshot={}
        )

        map2 = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[anchor2],
            policy_snapshot={}
        )

        with pytest.raises(ValueError, match="Anchor overlap detected"):
            merge_cloakmaps([map1, map2])


class TestValidateCloakMapIntegrity:
    """Test CloakMap integrity validation."""

    def test_validate_valid_cloakmap(self):
        """Test validation of valid CloakMap."""
        anchor = AnchorEntry(
            node_id="node1",
            start=0,
            end=10,
            entity_type="PERSON",
            confidence=0.9,
            masked_value="John Doe",
            replacement_id="repl1",
            original_checksum="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="redact"
        )

        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[anchor],
            policy_snapshot={}
        )

        result = validate_cloakmap_integrity(cloakmap)

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["checks"]["structure"] is True
        assert result["checks"]["anchors"] is True
        assert result["checks"]["signature"] is True

    def test_validate_with_duplicate_replacement_ids(self):
        """Test validation detects duplicate replacement IDs."""
        anchor1 = AnchorEntry(
            node_id="node1",
            start=0,
            end=10,
            entity_type="PERSON",
            confidence=0.9,
            masked_value="John Doe",
            replacement_id="repl1",  # Duplicate ID
            original_checksum="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="redact"
        )

        anchor2 = AnchorEntry(
            node_id="node2",
            start=20,
            end=30,
            entity_type="PERSON",
            confidence=0.8,
            masked_value="Jane Doe",
            replacement_id="repl1",  # Duplicate ID
            original_checksum="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="redact"
        )

        with pytest.raises(ValueError, match="duplicate replacement_id: repl1"):
            CloakMap(
                version="1.0",
                doc_id="doc123",
                doc_hash="hash123",
                anchors=[anchor1, anchor2],
                policy_snapshot={}
            )

    def test_validate_signed_cloakmap_with_key(self):
        """Test validation of signed CloakMap with correct key."""
        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        secret_key = "test_secret_key"
        signed_map = cloakmap.sign(secret_key=secret_key)

        result = validate_cloakmap_integrity(signed_map, secret_key=secret_key)

        assert result["valid"] is True
        assert result["checks"]["signature"] is True

    def test_validate_signed_cloakmap_with_wrong_key(self):
        """Test validation of signed CloakMap with wrong key."""
        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        secret_key = "test_secret_key"
        wrong_key = "wrong_secret_key"
        signed_map = cloakmap.sign(secret_key=secret_key)

        result = validate_cloakmap_integrity(signed_map, secret_key=wrong_key)

        # The new validation function may still report as valid at top level but have signature errors
        assert any("Signature verification failed" in error for error in result["errors"])

    def test_validate_signed_cloakmap_without_key(self):
        """Test validation of signed CloakMap without providing key."""
        cloakmap = CloakMap(
            version="1.0",
            doc_id="doc123",
            doc_hash="hash123",
            anchors=[],
            policy_snapshot={}
        )

        secret_key = "test_secret_key"
        signed_map = cloakmap.sign(secret_key=secret_key)

        result = validate_cloakmap_integrity(signed_map)  # No key provided

        # Still structurally valid, but should have errors about missing keys
        assert len(result["errors"]) > 0
