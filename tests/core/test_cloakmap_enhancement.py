"""Tests for CloakMap enhancement with Presidio metadata."""

import json

import pytest

from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.cloakmap_enhancer import CloakMapEnhancer


class TestCloakMapV2Format:
    """Test CloakMap v2.0 format with Presidio metadata."""

    def test_v1_cloakmap_creation_unchanged(self):
        """Test that v1.0 CloakMap creation works exactly as before."""
        # This ensures backward compatibility
        anchors = []
        cloakmap = CloakMap.create(
            doc_id="test_doc",
            doc_hash="abc123",
            anchors=anchors,
        )

        assert cloakmap.version == "1.0"
        assert cloakmap.doc_id == "test_doc"
        assert cloakmap.doc_hash == "abc123"
        assert cloakmap.anchors == []
        assert cloakmap.presidio_metadata is None
        assert not cloakmap.is_presidio_enabled
        assert not cloakmap.has_reversible_operators

    def test_v2_cloakmap_with_presidio_metadata(self):
        """Test v2.0 CloakMap creation with Presidio metadata."""
        presidio_metadata = {
            "engine_version": "2.2.x",
            "operator_results": [
                {
                    "entity_type": "PHONE_NUMBER",
                    "start": 10,
                    "end": 22,
                    "operator": "encrypt",
                    "encrypted_value": "abc123def456",
                    "key_reference": "key_123",
                }
            ],
            "reversible_operators": ["encrypt"],
            "batch_id": "batch_001",
        }

        cloakmap = CloakMap.create_with_presidio(
            doc_id="test_doc_v2",
            doc_hash="xyz789",
            anchors=[],
            presidio_metadata=presidio_metadata,
        )

        assert cloakmap.version == "2.0"
        assert cloakmap.is_presidio_enabled
        assert cloakmap.has_reversible_operators
        assert cloakmap.presidio_engine_version == "2.2.x"
        assert cloakmap.presidio_metadata == presidio_metadata

    def test_auto_version_upgrade_with_presidio_metadata(self):
        """Test that adding presidio_metadata auto-upgrades version to 2.0."""
        presidio_metadata = {
            "operator_results": [
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "start": 0,
                    "end": 15,
                    "operator": "redact",
                }
            ]
        }

        # Create with version 1.0 but include presidio_metadata
        cloakmap = CloakMap(
            version="1.0",
            doc_id="test",
            doc_hash="hash",
            presidio_metadata=presidio_metadata,
        )

        # Should auto-upgrade to 2.0
        assert cloakmap.version == "2.0"
        assert cloakmap.is_presidio_enabled


class TestPresidioMetadataValidation:
    """Test validation of Presidio metadata structure."""

    def test_valid_presidio_metadata(self):
        """Test that valid presidio metadata passes validation."""
        presidio_metadata = {
            "engine_version": "2.2.1",
            "operator_results": [
                {
                    "entity_type": "PHONE_NUMBER",
                    "start": 10,
                    "end": 22,
                    "operator": "encrypt",
                    "encrypted_value": "abc123",
                    "key_reference": "key_123",
                }
            ],
            "reversible_operators": ["encrypt"],
            "batch_id": "batch_001",
        }

        # Should not raise any exception
        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            presidio_metadata=presidio_metadata,
        )
        assert cloakmap.is_presidio_enabled

    def test_invalid_presidio_metadata_type(self):
        """Test that non-dict presidio_metadata raises ValueError."""
        with pytest.raises(ValueError, match="presidio_metadata must be a dictionary"):
            CloakMap(
                doc_id="test",
                doc_hash="hash",
                presidio_metadata="invalid",  # Should be dict
            )

    def test_invalid_operator_results_type(self):
        """Test that non-list operator_results raises ValueError."""
        with pytest.raises(ValueError, match="operator_results must be a list"):
            CloakMap(
                doc_id="test",
                doc_hash="hash",
                presidio_metadata={
                    "operator_results": "not_a_list"  # Should be list
                },
            )

    def test_invalid_operator_result_structure(self):
        """Test that invalid operator result structure raises ValueError."""
        with pytest.raises(ValueError, match="operator_result\\[0\\] must be a dictionary"):
            CloakMap(
                doc_id="test",
                doc_hash="hash",
                presidio_metadata={
                    "operator_results": ["not_a_dict"]  # Should be dict
                },
            )

    def test_missing_required_operator_result_fields(self):
        """Test that missing required fields in operator results raise ValueError."""
        with pytest.raises(ValueError, match="missing required field: operator"):
            CloakMap(
                doc_id="test",
                doc_hash="hash",
                presidio_metadata={
                    "operator_results": [
                        {
                            "entity_type": "PHONE_NUMBER",
                            "start": 10,
                            "end": 22,
                            # Missing "operator" field
                        }
                    ]
                },
            )

    def test_invalid_reversible_operators_type(self):
        """Test that non-list reversible_operators raises ValueError."""
        with pytest.raises(ValueError, match="reversible_operators must be a list"):
            CloakMap(
                doc_id="test",
                doc_hash="hash",
                presidio_metadata={
                    "reversible_operators": "not_a_list"  # Should be list
                },
            )

    def test_invalid_reversible_operator_type(self):
        """Test that non-string reversible operators raise ValueError."""
        with pytest.raises(ValueError, match="all reversible_operators must be strings"):
            CloakMap(
                doc_id="test",
                doc_hash="hash",
                presidio_metadata={
                    "reversible_operators": [123]  # Should be strings
                },
            )

    def test_invalid_engine_version_type(self):
        """Test that invalid engine_version raises ValueError."""
        with pytest.raises(ValueError, match="engine_version must be a non-empty string"):
            CloakMap(
                doc_id="test",
                doc_hash="hash",
                presidio_metadata={
                    "engine_version": ""  # Should be non-empty string
                },
            )


class TestCloakMapSerialization:
    """Test serialization/deserialization with v2.0 format."""

    def test_v1_serialization_unchanged(self):
        """Test that v1.0 serialization works as before (no presidio_metadata)."""
        cloakmap = CloakMap.create(
            doc_id="test_doc",
            doc_hash="abc123",
            anchors=[],
        )

        json_data = cloakmap.to_json()
        data_dict = json.loads(json_data)

        # Should not contain presidio_metadata field
        assert "presidio_metadata" not in data_dict
        assert data_dict["version"] == "1.0"
        assert data_dict["doc_id"] == "test_doc"

    def test_v2_serialization_includes_presidio_metadata(self):
        """Test that v2.0 serialization includes presidio_metadata."""
        presidio_metadata = {
            "engine_version": "2.2.x",
            "operator_results": [
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "start": 5,
                    "end": 20,
                    "operator": "hash",
                }
            ],
        }

        cloakmap = CloakMap.create_with_presidio(
            doc_id="test_doc_v2",
            doc_hash="xyz789",
            anchors=[],
            presidio_metadata=presidio_metadata,
        )

        json_data = cloakmap.to_json()
        data_dict = json.loads(json_data)

        # Should contain presidio_metadata field
        assert "presidio_metadata" in data_dict
        assert data_dict["version"] == "2.0"
        assert data_dict["presidio_metadata"] == presidio_metadata

    def test_v1_deserialization_backward_compatible(self):
        """Test that v1.0 JSON can be deserialized without presidio_metadata."""
        v1_json_data = """
        {
            "version": "1.0",
            "doc_id": "test_doc",
            "doc_hash": "abc123",
            "anchors": [],
            "policy_snapshot": {},
            "crypto": null,
            "signature": null,
            "created_at": "2023-01-01T12:00:00",
            "metadata": {}
        }
        """

        cloakmap = CloakMap.from_json(v1_json_data)

        assert cloakmap.version == "1.0"
        assert cloakmap.presidio_metadata is None
        assert not cloakmap.is_presidio_enabled

    def test_v2_deserialization_with_presidio_metadata(self):
        """Test that v2.0 JSON with presidio_metadata deserializes correctly."""
        v2_json_data = """
        {
            "version": "2.0",
            "doc_id": "test_doc_v2",
            "doc_hash": "xyz789",
            "anchors": [],
            "policy_snapshot": {},
            "crypto": null,
            "signature": null,
            "created_at": "2023-01-01T12:00:00",
            "metadata": {},
            "presidio_metadata": {
                "engine_version": "2.2.x",
                "operator_results": [
                    {
                        "entity_type": "PHONE_NUMBER",
                        "start": 10,
                        "end": 22,
                        "operator": "encrypt"
                    }
                ],
                "reversible_operators": ["encrypt"]
            }
        }
        """

        cloakmap = CloakMap.from_json(v2_json_data)

        assert cloakmap.version == "2.0"
        assert cloakmap.is_presidio_enabled
        assert cloakmap.presidio_engine_version == "2.2.x"
        assert len(cloakmap.presidio_metadata["operator_results"]) == 1

    def test_serialization_round_trip_v1(self):
        """Test v1.0 serialization round-trip preserves all data."""
        original = CloakMap.create(
            doc_id="round_trip_test",
            doc_hash="hash123",
            anchors=[],
        )

        json_data = original.to_json()
        restored = CloakMap.from_json(json_data)

        assert restored.version == original.version
        assert restored.doc_id == original.doc_id
        assert restored.doc_hash == original.doc_hash
        assert restored.presidio_metadata == original.presidio_metadata
        assert restored.is_presidio_enabled == original.is_presidio_enabled

    def test_serialization_round_trip_v2(self):
        """Test v2.0 serialization round-trip preserves presidio metadata."""
        presidio_metadata = {
            "engine_version": "2.2.1",
            "operator_results": [
                {
                    "entity_type": "SSN",
                    "start": 0,
                    "end": 11,
                    "operator": "hash",
                    "hash_value": "sha256..."
                }
            ],
            "reversible_operators": [],
            "batch_id": "test_batch"
        }

        original = CloakMap.create_with_presidio(
            doc_id="round_trip_test_v2",
            doc_hash="hash456",
            anchors=[],
            presidio_metadata=presidio_metadata,
        )

        json_data = original.to_json()
        restored = CloakMap.from_json(json_data)

        assert restored.version == original.version
        assert restored.doc_id == original.doc_id
        assert restored.presidio_metadata == original.presidio_metadata
        assert restored.is_presidio_enabled == original.is_presidio_enabled
        assert restored.presidio_engine_version == original.presidio_engine_version


class TestCloakMapEnhancer:
    """Test CloakMapEnhancer functionality."""

    def test_add_presidio_metadata_to_v1_cloakmap(self):
        """Test adding presidio metadata to existing v1.0 CloakMap."""
        # Start with v1.0 CloakMap
        cloakmap_v1 = CloakMap.create(
            doc_id="enhance_test",
            doc_hash="abc123",
            anchors=[],
        )
        assert cloakmap_v1.version == "1.0"
        assert not cloakmap_v1.is_presidio_enabled

        # Add presidio metadata
        enhancer = CloakMapEnhancer()
        operator_results = [
            {
                "entity_type": "EMAIL_ADDRESS",
                "start": 5,
                "end": 25,
                "operator": "redact",
            }
        ]

        cloakmap_v2 = enhancer.add_presidio_metadata(
            cloakmap_v1,
            operator_results,
            engine_version="2.2.1",
        )

        assert cloakmap_v2.version == "2.0"
        assert cloakmap_v2.is_presidio_enabled
        assert cloakmap_v2.presidio_engine_version == "2.2.1"
        assert len(enhancer.extract_operator_results(cloakmap_v2)) == 1

    def test_extract_operator_results(self):
        """Test extracting operator results from enhanced CloakMap."""
        operator_results = [
            {
                "entity_type": "PHONE_NUMBER",
                "start": 10,
                "end": 22,
                "operator": "encrypt",
                "encrypted_value": "abc123",
            }
        ]

        cloakmap = CloakMap.create_with_presidio(
            doc_id="extract_test",
            doc_hash="hash123",
            anchors=[],
            presidio_metadata={"operator_results": operator_results},
        )

        enhancer = CloakMapEnhancer()
        extracted = enhancer.extract_operator_results(cloakmap)

        assert len(extracted) == 1
        assert extracted[0]["entity_type"] == "PHONE_NUMBER"
        assert extracted[0]["operator"] == "encrypt"

    def test_extract_operator_results_from_v1_fails(self):
        """Test that extracting from v1.0 CloakMap raises ValueError."""
        cloakmap_v1 = CloakMap.create(
            doc_id="no_presidio",
            doc_hash="hash123",
            anchors=[],
        )

        enhancer = CloakMapEnhancer()
        with pytest.raises(ValueError, match="does not contain Presidio metadata"):
            enhancer.extract_operator_results(cloakmap_v1)

    def test_is_presidio_enabled_check(self):
        """Test presidio enabled check on various CloakMaps."""
        enhancer = CloakMapEnhancer()

        # v1.0 CloakMap
        cloakmap_v1 = CloakMap.create(doc_id="test", doc_hash="hash", anchors=[])
        assert not enhancer.is_presidio_enabled(cloakmap_v1)

        # v2.0 CloakMap with presidio metadata
        cloakmap_v2 = CloakMap.create_with_presidio(
            doc_id="test",
            doc_hash="hash",
            anchors=[],
            presidio_metadata={"operator_results": []},
        )
        assert enhancer.is_presidio_enabled(cloakmap_v2)

    def test_get_reversible_operators(self):
        """Test getting reversible operators list."""
        presidio_metadata = {
            "reversible_operators": ["encrypt", "custom"],
            "operator_results": [],
        }

        cloakmap = CloakMap.create_with_presidio(
            doc_id="test",
            doc_hash="hash",
            anchors=[],
            presidio_metadata=presidio_metadata,
        )

        enhancer = CloakMapEnhancer()
        reversible = enhancer.get_reversible_operators(cloakmap)

        assert reversible == ["encrypt", "custom"]

    def test_migrate_to_v2_alias(self):
        """Test that migrate_to_v2 is alias for add_presidio_metadata."""
        cloakmap_v1 = CloakMap.create(doc_id="test", doc_hash="hash", anchors=[])

        operator_results = [
            {
                "entity_type": "SSN",
                "start": 0,
                "end": 11,
                "operator": "hash",
            }
        ]

        enhancer = CloakMapEnhancer()
        migrated = enhancer.migrate_to_v2(cloakmap_v1, operator_results, engine_version="2.2.0")

        assert migrated.version == "2.0"
        assert migrated.is_presidio_enabled
        assert migrated.presidio_engine_version == "2.2.0"

    def test_get_statistics(self):
        """Test getting statistics from CloakMap."""
        enhancer = CloakMapEnhancer()

        # Stats for v1.0 CloakMap
        cloakmap_v1 = CloakMap.create(doc_id="test", doc_hash="hash", anchors=[])
        stats_v1 = enhancer.get_statistics(cloakmap_v1)

        assert stats_v1["presidio_enabled"] is False
        assert stats_v1["version"] == "1.0"

        # Stats for v2.0 CloakMap
        operator_results = [
            {"entity_type": "EMAIL", "start": 0, "end": 10, "operator": "hash"},
            {"entity_type": "PHONE", "start": 15, "end": 27, "operator": "encrypt"},
        ]

        cloakmap_v2 = CloakMap.create_with_presidio(
            doc_id="test",
            doc_hash="hash",
            anchors=[],
            presidio_metadata={
                "engine_version": "2.2.1",
                "operator_results": operator_results,
                "reversible_operators": ["encrypt"],
                "batch_id": "test_batch",
            },
        )

        stats_v2 = enhancer.get_statistics(cloakmap_v2)

        assert stats_v2["presidio_enabled"] is True
        assert stats_v2["version"] == "2.0"
        assert stats_v2["engine_version"] == "2.2.1"
        assert stats_v2["batch_id"] == "test_batch"
        assert stats_v2["total_operator_results"] == 2
        assert stats_v2["operator_counts"] == {"hash": 1, "encrypt": 1}
        assert stats_v2["reversible_operators"] == ["encrypt"]
        assert stats_v2["reversible_count"] == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_operator_results_validation(self):
        """Test that empty operator_results raises ValueError."""
        cloakmap_v1 = CloakMap.create(doc_id="test", doc_hash="hash", anchors=[])
        enhancer = CloakMapEnhancer()

        with pytest.raises(ValueError, match="operator_results cannot be empty"):
            enhancer.add_presidio_metadata(cloakmap_v1, [])

    def test_invalid_operator_results_type_in_enhancer(self):
        """Test that non-list operator_results raises ValueError in enhancer."""
        cloakmap_v1 = CloakMap.create(doc_id="test", doc_hash="hash", anchors=[])
        enhancer = CloakMapEnhancer()

        with pytest.raises(ValueError, match="operator_results must be a list"):
            enhancer.add_presidio_metadata(cloakmap_v1, "not_a_list")

    def test_missing_fields_in_operator_results(self):
        """Test that missing required fields in operator results raise ValueError."""
        cloakmap_v1 = CloakMap.create(doc_id="test", doc_hash="hash", anchors=[])
        enhancer = CloakMapEnhancer()

        invalid_operator_results = [
            {
                "entity_type": "EMAIL",
                "start": 0,
                # Missing "end" and "operator" fields
            }
        ]

        with pytest.raises(ValueError, match="missing required field"):
            enhancer.add_presidio_metadata(cloakmap_v1, invalid_operator_results)

    def test_update_presidio_metadata_on_v1_fails(self):
        """Test that updating presidio metadata on v1.0 CloakMap fails."""
        cloakmap_v1 = CloakMap.create(doc_id="test", doc_hash="hash", anchors=[])
        enhancer = CloakMapEnhancer()

        with pytest.raises(ValueError, match="does not contain Presidio metadata to update"):
            enhancer.update_presidio_metadata(cloakmap_v1, engine_version="2.2.1")

    def test_detect_reversible_operators(self):
        """Test automatic detection of reversible operators."""
        enhancer = CloakMapEnhancer()

        operator_results = [
            {"entity_type": "EMAIL", "start": 0, "end": 10, "operator": "hash"},
            {"entity_type": "PHONE", "start": 15, "end": 27, "operator": "encrypt"},
            {"entity_type": "SSN", "start": 30, "end": 41, "operator": "redact"},
            {"entity_type": "NAME", "start": 50, "end": 60, "operator": "custom"},
        ]

        # Should detect "encrypt" and "custom" as reversible
        reversible = enhancer._detect_reversible_operators(operator_results)

        assert "encrypt" in reversible
        assert "custom" in reversible
        assert "hash" not in reversible
        assert "redact" not in reversible
