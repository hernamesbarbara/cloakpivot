"""Tests for CloakMap migration utilities."""

import tempfile
from pathlib import Path

import pytest
import yaml

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.migration import CloakMapMigrator, StrategyMigrator


class TestCloakMapMigrator:
    """Test CloakMap migration functionality."""
    
    def test_migrate_v1_to_v2_single_file(self):
        """Test migrating a single v1.0 CloakMap to v2.0."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create v1.0 CloakMap
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
            
            v1_cloakmap = CloakMap.create(
                doc_id="test_doc",
                doc_hash="test_hash",
                anchors=anchors,
                metadata={"test": True}
            )
            
            # Save v1.0 CloakMap
            v1_path = temp_path / "test.cloakmap"
            v1_cloakmap.save_to_file(v1_path)
            
            # Migrate
            migrator = CloakMapMigrator()
            output_path = migrator.migrate_cloakmap(v1_path)
            
            # Load migrated CloakMap
            migrated = CloakMap.load_from_file(output_path)
            
            # Verify migration
            assert migrated.version == "2.0"
            assert migrated.doc_id == v1_cloakmap.doc_id
            assert migrated.doc_hash == v1_cloakmap.doc_hash
            assert len(migrated.anchors) == len(v1_cloakmap.anchors)
            assert migrated.is_presidio_enabled
            assert migrated.presidio_metadata is not None
            assert migrated.presidio_metadata["migration_source"] == "legacy_v1.0"
            
    def test_skip_already_v2_cloakmap(self):
        """Test that v2.0 CloakMaps are skipped during migration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create v2.0 CloakMap
            v2_cloakmap = CloakMap.create_with_presidio(
                doc_id="test_doc",
                doc_hash="test_hash",
                anchors=[],
                presidio_metadata={
                    "engine_version": "2.2.0",
                    "operator_results": []
                }
            )
            
            # Save v2.0 CloakMap
            v2_path = temp_path / "test_v2.cloakmap"
            v2_cloakmap.save_to_file(v2_path)
            
            # Attempt migration
            migrator = CloakMapMigrator()
            output_path = migrator.migrate_cloakmap(v2_path)
            
            # Should return same path (no migration needed)
            assert output_path == v2_path
    
    def test_bulk_migrate_mixed_versions(self):
        """Test bulk migration with mixed v1.0 and v2.0 CloakMaps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create v1.0 CloakMaps
            for i in range(2):
                v1_cloakmap = CloakMap.create(
                    doc_id=f"v1_doc_{i}",
                    doc_hash=f"hash_{i}",
                    anchors=[]
                )
                v1_cloakmap.save_to_file(temp_path / f"v1_{i}.cloakmap")
            
            # Create v2.0 CloakMap
            v2_cloakmap = CloakMap.create_with_presidio(
                doc_id="v2_doc",
                doc_hash="v2_hash",
                anchors=[],
                presidio_metadata={"engine_version": "2.2.0", "operator_results": []}
            )
            v2_cloakmap.save_to_file(temp_path / "v2.cloakmap")
            
            # Bulk migrate
            migrator = CloakMapMigrator()
            results = migrator.bulk_migrate(temp_path)
            
            # Verify results
            assert len(results["migrated"]) == 2  # Two v1.0 files migrated
            assert len(results["skipped"]) == 1   # One v2.0 file skipped
            assert len(results["errors"]) == 0
            
            # Verify migrated files exist
            for item in results["migrated"]:
                assert Path(item["target"]).exists()
                migrated = CloakMap.load_from_file(Path(item["target"]))
                assert migrated.version == "2.0"
                assert migrated.is_presidio_enabled
    
    def test_migrate_with_custom_output_path(self):
        """Test migration with custom output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create v1.0 CloakMap
            v1_cloakmap = CloakMap.create(
                doc_id="test_doc",
                doc_hash="test_hash",
                anchors=[]
            )
            
            v1_path = temp_path / "original.cloakmap"
            v1_cloakmap.save_to_file(v1_path)
            
            # Migrate to custom path
            custom_output = temp_path / "migrated" / "custom.cloakmap"
            custom_output.parent.mkdir()
            
            migrator = CloakMapMigrator()
            output_path = migrator.migrate_cloakmap(v1_path, custom_output)
            
            assert output_path == custom_output
            assert custom_output.exists()
            
            # Verify migration
            migrated = CloakMap.load_from_file(custom_output)
            assert migrated.version == "2.0"
    
    def test_error_handling_invalid_file(self):
        """Test error handling for invalid CloakMap files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create invalid file
            invalid_path = temp_path / "invalid.cloakmap"
            invalid_path.write_text("not a valid cloakmap")
            
            migrator = CloakMapMigrator()
            
            with pytest.raises(ValueError, match="Failed to load CloakMap"):
                migrator.migrate_cloakmap(invalid_path)
    
    def test_error_handling_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        migrator = CloakMapMigrator()
        
        with pytest.raises(FileNotFoundError):
            migrator.migrate_cloakmap(Path("/nonexistent/file.cloakmap"))
    
    def test_infer_operator_results(self):
        """Test inference of Presidio operator results from anchors."""
        anchors = [
            AnchorEntry.create_from_detection(
                node_id="para_1",
                start=10,
                end=22,
                entity_type="PHONE_NUMBER",
                confidence=0.95,
                original_text="555-123-4567",
                masked_value="[PHONE]",
                strategy_used="redact",
            ),
            AnchorEntry.create_from_detection(
                node_id="para_2",
                start=30,
                end=50,
                entity_type="EMAIL",
                confidence=0.90,
                original_text="test@example.com",
                masked_value="<EMAIL>",
                strategy_used="template",
            ),
        ]
        
        v1_cloakmap = CloakMap.create(
            doc_id="test_doc",
            doc_hash="test_hash",
            anchors=anchors
        )
        
        migrator = CloakMapMigrator()
        migrated = migrator._enhance_with_presidio_metadata(v1_cloakmap)
        
        # Verify operator results were inferred
        assert migrated.presidio_metadata is not None
        operator_results = migrated.presidio_metadata["operator_results"]
        assert len(operator_results) == 2
        
        # Check first operator result
        assert operator_results[0]["entity_type"] == "PHONE_NUMBER"
        assert operator_results[0]["operator"] == "redact"
        assert operator_results[0]["new_value"] == "[PHONE]"
        
        # Check second operator result
        assert operator_results[1]["entity_type"] == "EMAIL"
        assert operator_results[1]["operator"] == "replace"
        assert operator_results[1]["new_value"] == "<EMAIL>"


class TestStrategyMigrator:
    """Test strategy/policy migration functionality."""
    
    def test_migrate_policy_file(self):
        """Test migrating a policy file to Presidio format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create legacy policy file
            policy_data = {
                "strategies": {
                    "PHONE_NUMBER": {
                        "kind": "redact"
                    },
                    "EMAIL": {
                        "kind": "hash",
                        "algorithm": "md5"
                    },
                    "PERSON": {
                        "kind": "surrogate"
                    }
                }
            }
            
            policy_path = temp_path / "policy.yml"
            with open(policy_path, 'w') as f:
                yaml.dump(policy_data, f)
            
            # Migrate
            migrator = StrategyMigrator()
            output_path = migrator.migrate_policy_file(policy_path)
            
            # Load migrated policy
            with open(output_path) as f:
                migrated_data = yaml.safe_load(f)
            
            # Verify Presidio configuration added
            assert "presidio" in migrated_data
            assert migrated_data["presidio"]["enabled"] is True
            assert migrated_data["presidio"]["fallback_to_legacy"] is True
            
            # Verify strategies migrated
            strategies = migrated_data["strategies"]
            
            # Check redact strategy
            assert strategies["PHONE_NUMBER"]["operator"] == "redact"
            assert strategies["PHONE_NUMBER"]["presidio_optimized"] is True
            
            # Check hash strategy
            assert strategies["EMAIL"]["operator"] == "hash"
            assert strategies["EMAIL"]["algorithm"] == "md5"  # Preserved
            assert strategies["EMAIL"]["presidio_optimized"] is True
            
            # Check surrogate strategy
            assert strategies["PERSON"]["operator"] == "replace"
            assert strategies["PERSON"]["use_presidio_faker"] is True
    
    def test_skip_already_migrated_policy(self):
        """Test that already migrated policies are skipped."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create policy with Presidio config
            policy_data = {
                "presidio": {
                    "enabled": True
                },
                "strategies": {}
            }
            
            policy_path = temp_path / "policy.yml"
            with open(policy_path, 'w') as f:
                yaml.dump(policy_data, f)
            
            # Attempt bulk migration
            migrator = StrategyMigrator()
            results = migrator.bulk_migrate_policies(temp_path)
            
            assert len(results["skipped"]) == 1
            assert results["skipped"][0]["reason"] == "Already has Presidio configuration"
    
    def test_bulk_migrate_policies(self):
        """Test bulk migration of policy files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple policy files
            for i in range(3):
                policy_data = {
                    "strategies": {
                        "TEST_ENTITY": {
                            "kind": "redact"
                        }
                    }
                }
                
                with open(temp_path / f"policy_{i}.yml", 'w') as f:
                    yaml.dump(policy_data, f)
            
            # Create one already migrated
            migrated_data = {
                "presidio": {"enabled": True},
                "strategies": {}
            }
            with open(temp_path / "already_migrated.yml", 'w') as f:
                yaml.dump(migrated_data, f)
            
            # Bulk migrate
            migrator = StrategyMigrator()
            results = migrator.bulk_migrate_policies(temp_path)
            
            assert len(results["migrated"]) == 3
            assert len(results["skipped"]) == 1
            assert len(results["errors"]) == 0
            
            # Verify migrated files
            for item in results["migrated"]:
                assert Path(item["target"]).exists()
                assert item["target"].endswith(".presidio.yml")
    
    def test_migrate_complex_strategies(self):
        """Test migration of complex strategy configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create policy with various strategy types
            policy_data = {
                "strategies": {
                    "SSN": {
                        "kind": "encrypt",
                        "key": "my-key"
                    },
                    "CREDIT_CARD": {
                        "kind": "hash",
                        "algorithm": "sha512"
                    },
                    "DATE": {
                        "kind": "template",
                        "template": "[DATE]"
                    }
                }
            }
            
            policy_path = temp_path / "complex_policy.yml"
            with open(policy_path, 'w') as f:
                yaml.dump(policy_data, f)
            
            # Migrate
            migrator = StrategyMigrator()
            output_path = migrator.migrate_policy_file(policy_path)
            
            # Load and verify
            with open(output_path) as f:
                migrated_data = yaml.safe_load(f)
            
            strategies = migrated_data["strategies"]
            
            # Check encrypt strategy
            assert strategies["SSN"]["operator"] == "encrypt"
            assert strategies["SSN"]["reversible"] is True
            assert strategies["SSN"]["key_id"] == "default"
            
            # Check hash with custom algorithm
            assert strategies["CREDIT_CARD"]["operator"] == "hash"
            assert strategies["CREDIT_CARD"]["algorithm"] == "sha512"
            
            # Check template strategy
            assert strategies["DATE"]["operator"] == "replace"
            assert strategies["DATE"]["presidio_optimized"] is True
    
    def test_error_handling_invalid_policy(self):
        """Test error handling for invalid policy files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create invalid YAML file
            invalid_path = temp_path / "invalid.yml"
            invalid_path.write_text("not: valid: yaml: structure:")
            
            migrator = StrategyMigrator()
            
            with pytest.raises(ValueError, match="Failed to load policy"):
                migrator.migrate_policy_file(invalid_path)
    
    def test_error_handling_nonexistent_policy(self):
        """Test error handling for nonexistent policy files."""
        migrator = StrategyMigrator()
        
        with pytest.raises(FileNotFoundError):
            migrator.migrate_policy_file(Path("/nonexistent/policy.yml"))