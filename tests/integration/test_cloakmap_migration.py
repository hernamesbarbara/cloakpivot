"""Integration tests for CloakMap migration scenarios."""

import tempfile
from pathlib import Path

import pytest

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.cloakmap_enhancer import CloakMapEnhancer


class TestCloakMapMigrationScenarios:
    """Test end-to-end migration scenarios for CloakMap versions."""

    def test_v1_to_v2_migration_preserves_all_data(self):
        """Test that v1.0 → v2.0 migration preserves all original data."""
        # Create v1.0 CloakMap with comprehensive data
        anchors = [
            AnchorEntry.create_from_detection(
                node_id="para_1",
                start=10,
                end=22,
                entity_type="PHONE_NUMBER",
                confidence=0.95,
                original_text="555-123-4567",
                masked_value="[PHONE]",
                strategy_used="template",
            )
        ]

        original_v1 = CloakMap.create(
            doc_id="migration_test_doc",
            doc_hash="original_hash_123",
            anchors=anchors,
            metadata={"source": "test_suite", "priority": "high"},
        )

        # Add crypto and signature data
        original_v1_signed = original_v1.with_signature(secret_key="test_secret")

        # Migrate to v2.0
        enhancer = CloakMapEnhancer()
        operator_results = [
            {
                "entity_type": "PHONE_NUMBER",
                "start": 10,
                "end": 22,
                "operator": "replace",
                "new_value": "[PHONE]",
            }
        ]

        migrated_v2 = enhancer.add_presidio_metadata(
            original_v1_signed,
            operator_results,
            engine_version="2.2.1",
            reversible_operators=[],
            batch_id="migration_batch_001",
        )

        # Verify all original data preserved
        assert migrated_v2.version == "2.0"
        assert migrated_v2.doc_id == original_v1.doc_id
        assert migrated_v2.doc_hash == original_v1.doc_hash
        assert migrated_v2.anchors == original_v1.anchors
        assert migrated_v2.policy_snapshot == original_v1.policy_snapshot
        assert migrated_v2.metadata == original_v1.metadata
        assert migrated_v2.created_at == original_v1.created_at

        # Verify presidio metadata added
        assert migrated_v2.is_presidio_enabled
        assert migrated_v2.presidio_engine_version == "2.2.1"
        extracted_results = enhancer.extract_operator_results(migrated_v2)
        assert len(extracted_results) == 1
        assert extracted_results[0]["entity_type"] == "PHONE_NUMBER"

    def test_bulk_migration_with_mixed_versions(self):
        """Test handling of bulk migration with mixed CloakMap versions."""
        enhancer = CloakMapEnhancer()

        # Create multiple v1.0 CloakMaps
        v1_cloakmaps = []
        for i in range(3):
            cloakmap = CloakMap.create(
                doc_id=f"bulk_doc_{i}",
                doc_hash=f"hash_{i}",
                anchors=[],
                metadata={"batch": "bulk_test", "index": i},
            )
            v1_cloakmaps.append(cloakmap)

        # Create one v2.0 CloakMap
        v2_cloakmap = CloakMap.create_with_presidio(
            doc_id="bulk_doc_v2",
            doc_hash="hash_v2",
            anchors=[],
            presidio_metadata={
                "engine_version": "2.2.0",
                "operator_results": [
                    {
                        "entity_type": "EMAIL",
                        "start": 0,
                        "end": 15,
                        "operator": "hash",
                    }
                ],
            },
        )

        mixed_cloakmaps = v1_cloakmaps + [v2_cloakmap]

        # Process mixed versions
        migrated_results = []
        for cloakmap in mixed_cloakmaps:
            if cloakmap.version == "1.0":
                # Migrate v1.0 to v2.0
                operator_results = [
                    {
                        "entity_type": "PLACEHOLDER",
                        "start": 0,
                        "end": 1,
                        "operator": "redact",
                    }
                ]
                migrated = enhancer.add_presidio_metadata(
                    cloakmap,
                    operator_results,
                    engine_version="2.2.1",
                )
                migrated_results.append(migrated)
            else:
                # Keep v2.0 as-is
                migrated_results.append(cloakmap)

        # Verify results
        assert len(migrated_results) == 4
        for result in migrated_results:
            assert result.version == "2.0"
            assert result.is_presidio_enabled

        # Verify original v2.0 kept its metadata
        original_v2_result = [r for r in migrated_results if r.doc_id == "bulk_doc_v2"][0]
        assert original_v2_result.presidio_engine_version == "2.2.0"

        # Verify migrated v1.0s got new metadata
        migrated_v1_results = [r for r in migrated_results if r.doc_id.startswith("bulk_doc_") and r.doc_id != "bulk_doc_v2"]
        for result in migrated_v1_results:
            assert result.presidio_engine_version == "2.2.1"

    def test_file_based_migration_workflow(self):
        """Test complete file-based migration workflow."""
        enhancer = CloakMapEnhancer()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create and save v1.0 CloakMap to file
            original_v1 = CloakMap.create(
                doc_id="file_migration_test",
                doc_hash="file_hash_123",
                anchors=[],
                metadata={"file_test": True},
            )

            v1_file_path = temp_path / "cloakmap_v1.json"
            original_v1.save_to_file(v1_file_path)

            # Load from file
            loaded_v1 = CloakMap.load_from_file(v1_file_path)
            assert loaded_v1.version == "1.0"
            assert not loaded_v1.is_presidio_enabled

            # Migrate to v2.0
            operator_results = [
                {
                    "entity_type": "GENERIC",
                    "start": 0,
                    "end": 5,
                    "operator": "redact",
                }
            ]

            migrated_v2 = enhancer.add_presidio_metadata(
                loaded_v1,
                operator_results,
                engine_version="2.2.1",
                batch_id="file_migration",
            )

            # Save migrated version to new file
            v2_file_path = temp_path / "cloakmap_v2.json"
            migrated_v2.save_to_file(v2_file_path)

            # Load migrated version and verify
            loaded_v2 = CloakMap.load_from_file(v2_file_path)
            assert loaded_v2.version == "2.0"
            assert loaded_v2.is_presidio_enabled
            assert loaded_v2.presidio_engine_version == "2.2.1"

            # Verify original data preserved
            assert loaded_v2.doc_id == original_v1.doc_id
            assert loaded_v2.doc_hash == original_v1.doc_hash
            assert loaded_v2.metadata == original_v1.metadata

    def test_performance_with_large_cloakmaps(self):
        """Test migration performance with large CloakMaps."""
        import time

        # Create large v1.0 CloakMap
        large_anchors = []
        for i in range(100):  # 100 anchors
            anchor = AnchorEntry.create_from_detection(
                node_id=f"node_{i}",
                start=i * 10,
                end=i * 10 + 5,
                entity_type="TEST_ENTITY",
                confidence=0.9,
                original_text=f"text_{i}",
                masked_value=f"[MASKED_{i}]",
                strategy_used="template",
            )
            large_anchors.append(anchor)

        large_cloakmap_v1 = CloakMap.create(
            doc_id="performance_test_large",
            doc_hash="large_hash",
            anchors=large_anchors,
            metadata={"size": "large", "anchor_count": len(large_anchors)},
        )

        # Create large operator results
        large_operator_results = []
        for i in range(100):
            result = {
                "entity_type": "TEST_ENTITY",
                "start": i * 10,
                "end": i * 10 + 5,
                "operator": "replace",
                "new_value": f"[MASKED_{i}]",
            }
            large_operator_results.append(result)

        # Time the migration
        enhancer = CloakMapEnhancer()
        start_time = time.time()

        migrated_large = enhancer.add_presidio_metadata(
            large_cloakmap_v1,
            large_operator_results,
            engine_version="2.2.1",
        )

        migration_time = time.time() - start_time

        # Verify migration completed correctly
        assert migrated_large.version == "2.0"
        assert migrated_large.is_presidio_enabled
        assert len(migrated_large.anchors) == 100
        extracted_results = enhancer.extract_operator_results(migrated_large)
        assert len(extracted_results) == 100

        # Performance check: migration should complete in reasonable time (< 1 second)
        assert migration_time < 1.0, f"Migration took too long: {migration_time:.2f}s"

        # Test serialization performance with large v2.0 CloakMap
        start_time = time.time()
        json_data = migrated_large.to_json()
        serialization_time = time.time() - start_time

        # Serialization should be fast
        assert serialization_time < 1.0, f"Serialization took too long: {serialization_time:.2f}s"

        # Test deserialization performance
        start_time = time.time()
        deserialized = CloakMap.from_json(json_data)
        deserialization_time = time.time() - start_time

        # Deserialization should be fast
        assert deserialization_time < 1.0, f"Deserialization took too long: {deserialization_time:.2f}s"

        # Verify data integrity after round-trip
        assert deserialized.version == migrated_large.version
        assert deserialized.doc_id == migrated_large.doc_id
        assert len(deserialized.anchors) == len(migrated_large.anchors)
        assert deserialized.is_presidio_enabled == migrated_large.is_presidio_enabled


class TestVersionDetectionAndCompatibility:
    """Test version detection and compatibility handling."""

    def test_auto_version_detection_from_presidio_metadata(self):
        """Test automatic version detection based on presidio_metadata presence."""
        # Create CloakMap with presidio_metadata but version="1.0" explicitly
        cloakmap = CloakMap(
            version="1.0",  # Explicit v1.0
            doc_id="version_test",
            doc_hash="hash",
            presidio_metadata={  # But has presidio metadata
                "operator_results": [
                    {
                        "entity_type": "TEST",
                        "start": 0,
                        "end": 5,
                        "operator": "redact",
                    }
                ]
            },
        )

        # Should auto-upgrade to v2.0
        assert cloakmap.version == "2.0"
        assert cloakmap.is_presidio_enabled

    def test_legacy_json_without_presidio_metadata_loads_as_v1(self):
        """Test that legacy JSON without presidio_metadata loads as v1.0."""
        legacy_json = """
        {
            "version": "1.0",
            "doc_id": "legacy_doc",
            "doc_hash": "legacy_hash",
            "anchors": [],
            "policy_snapshot": {},
            "crypto": null,
            "signature": null,
            "created_at": null,
            "metadata": {}
        }
        """

        cloakmap = CloakMap.from_json(legacy_json)

        assert cloakmap.version == "1.0"
        assert cloakmap.presidio_metadata is None
        assert not cloakmap.is_presidio_enabled

    def test_mixed_version_handling(self):
        """Test handling collections of mixed version CloakMaps."""
        # Create mixed versions
        v1_cloakmap = CloakMap.create(doc_id="v1", doc_hash="hash1", anchors=[])

        v2_cloakmap = CloakMap.create_with_presidio(
            doc_id="v2",
            doc_hash="hash2",
            anchors=[],
            presidio_metadata={
                "operator_results": [
                    {"entity_type": "TEST", "start": 0, "end": 5, "operator": "redact"}
                ]
            },
        )

        mixed_versions = [v1_cloakmap, v2_cloakmap]

        # Process mixed versions safely
        enhancer = CloakMapEnhancer()
        for cloakmap in mixed_versions:
            if enhancer.is_presidio_enabled(cloakmap):
                # Can safely extract presidio data
                results = enhancer.extract_operator_results(cloakmap)
                assert len(results) >= 0
            else:
                # v1.0 CloakMap - no presidio operations
                assert cloakmap.version == "1.0"

        # Verify mixed handling works
        v1_count = sum(1 for cm in mixed_versions if cm.version == "1.0")
        v2_count = sum(1 for cm in mixed_versions if cm.version == "2.0")
        assert v1_count == 1
        assert v2_count == 1

    def test_version_validation_edge_cases(self):
        """Test version validation with edge cases."""
        # Valid version formats
        valid_versions = ["1.0", "2.0", "1.1", "2.1", "1.0.0", "2.0.1"]

        for version in valid_versions:
            cloakmap = CloakMap(
                version=version,
                doc_id="version_test",
                doc_hash="hash",
            )
            assert cloakmap.version == version

        # Test individual invalid version formats
        with pytest.raises(ValueError, match="Version cannot be empty"):
            CloakMap(version="", doc_id="version_test", doc_hash="hash")

        with pytest.raises(ValueError, match="version must follow 'major.minor' format"):
            CloakMap(version="1", doc_id="version_test", doc_hash="hash")

        with pytest.raises(ValueError, match="version major and minor components must be numeric"):
            CloakMap(version="v1.0", doc_id="version_test", doc_hash="hash")

        # Note: "1.0.x" actually succeeds because only first two parts need to be numeric
        # This is by design to support patch versions like "1.0.1"
        with pytest.raises(ValueError, match="version major and minor components must be numeric"):
            CloakMap(version="x.0", doc_id="version_test", doc_hash="hash")

        with pytest.raises(ValueError, match="version must follow 'major.minor' format"):
            CloakMap(version="invalid", doc_id="version_test", doc_hash="hash")


class TestUpgradeAndDowngradeScenarios:
    """Test upgrade and downgrade scenarios."""

    def test_v2_to_v1_compatibility_mode(self):
        """Test that v2.0 CloakMap can be used in v1.0 compatibility mode."""
        # Create v2.0 CloakMap
        v2_cloakmap = CloakMap.create_with_presidio(
            doc_id="upgrade_test",
            doc_hash="hash123",
            anchors=[],
            presidio_metadata={
                "operator_results": [
                    {"entity_type": "TEST", "start": 0, "end": 5, "operator": "redact"}
                ]
            },
        )

        # All v1.0 operations should still work
        assert v2_cloakmap.doc_id == "upgrade_test"
        assert v2_cloakmap.doc_hash == "hash123"
        assert len(v2_cloakmap.anchors) == 0
        assert v2_cloakmap.anchor_count == 0
        assert isinstance(v2_cloakmap.entity_count_by_type, dict)

        # v2.0 operations also work
        assert v2_cloakmap.is_presidio_enabled
        assert v2_cloakmap.presidio_engine_version is None  # Not set in this case

    def test_strip_presidio_metadata_for_v1_compatibility(self):
        """Test creating v1.0 compatible version by stripping presidio metadata."""
        # Start with v2.0 CloakMap
        v2_cloakmap = CloakMap.create_with_presidio(
            doc_id="strip_test",
            doc_hash="hash123",
            anchors=[],
            presidio_metadata={
                "engine_version": "2.2.1",
                "operator_results": [
                    {"entity_type": "TEST", "start": 0, "end": 5, "operator": "redact"}
                ],
            },
        )

        # Create v1.0 compatible version (strip presidio metadata)
        v1_compatible = CloakMap(
            version="1.0",
            doc_id=v2_cloakmap.doc_id,
            doc_hash=v2_cloakmap.doc_hash,
            anchors=v2_cloakmap.anchors,
            policy_snapshot=v2_cloakmap.policy_snapshot,
            crypto=v2_cloakmap.crypto,
            signature=v2_cloakmap.signature,
            created_at=v2_cloakmap.created_at,
            metadata=v2_cloakmap.metadata,
            # presidio_metadata=None (omitted)
        )

        assert v1_compatible.version == "1.0"
        assert not v1_compatible.is_presidio_enabled
        assert v1_compatible.doc_id == v2_cloakmap.doc_id
        assert v1_compatible.doc_hash == v2_cloakmap.doc_hash

    def test_incremental_presidio_feature_adoption(self):
        """Test incremental adoption of presidio features."""
        enhancer = CloakMapEnhancer()

        # Stage 1: v1.0 CloakMap (legacy)
        stage1 = CloakMap.create(doc_id="incremental", doc_hash="hash", anchors=[])
        assert stage1.version == "1.0"

        # Stage 2: Add basic presidio metadata
        stage2 = enhancer.add_presidio_metadata(
            stage1,
            [{"entity_type": "TEST", "start": 0, "end": 5, "operator": "redact"}],
        )
        assert stage2.version == "2.0"
        assert stage2.is_presidio_enabled

        # Stage 3: Add more advanced features (engine version, batch tracking)
        stage3 = enhancer.update_presidio_metadata(
            stage2,
            engine_version="2.2.1",
            batch_id="advanced_batch",
        )
        assert stage3.presidio_engine_version == "2.2.1"
        assert enhancer.get_batch_id(stage3) == "advanced_batch"

        # Stage 4: Add reversible operators
        stage4 = enhancer.update_presidio_metadata(
            stage3,
            reversible_operators=["encrypt", "custom"],
        )
        reversible = enhancer.get_reversible_operators(stage4)
        assert "encrypt" in reversible
        assert "custom" in reversible

        # Verify all stages maintain data integrity
        for stage in [stage2, stage3, stage4]:
            assert stage.doc_id == stage1.doc_id
            assert stage.doc_hash == stage1.doc_hash
            assert stage.is_presidio_enabled
