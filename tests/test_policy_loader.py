"""Tests for the policy loader system with inheritance and validation."""

import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.policy_loader import (
    PolicyFileSchema,
    PolicyInheritanceError,
    PolicyLoader,
    PolicyValidationError,
)
from cloakpivot.core.strategies import StrategyKind


class TestPolicyFileSchema:
    """Test the Pydantic schema validation."""

    def test_valid_minimal_schema(self):
        """Test minimal valid policy schema."""
        data = {"version": "1.0"}
        schema = PolicyFileSchema(**data)
        assert schema.version == "1.0"
        assert schema.locale == "en"  # default

    def test_valid_complete_schema(self):
        """Test complete policy schema with all fields."""
        data = {
            "version": "1.0",
            "name": "test-policy",
            "description": "Test policy",
            "locale": "en-US",
            "seed": "test-seed",
            "default_strategy": {"kind": "redact", "parameters": {"redact_char": "*"}},
            "per_entity": {
                "PERSON": {
                    "kind": "template",
                    "parameters": {"template": "[PERSON]"},
                    "threshold": 0.8,
                    "enabled": True,
                }
            },
            "thresholds": {"EMAIL_ADDRESS": 0.7},
            "allow_list": ["test@example.com"],
            "deny_list": ["confidential"],
            "min_entity_length": 2,
            "context_rules": {"heading": {"enabled": False}},
            "policy_composition": {
                "merge_strategy": "override",
                "validation_level": "strict",
            },
        }

        schema = PolicyFileSchema(**data)
        assert schema.name == "test-policy"
        assert schema.locale == "en-US"
        assert schema.default_strategy.kind == "redact"
        assert schema.per_entity["PERSON"].threshold == 0.8

    def test_invalid_strategy_kind(self):
        """Test validation fails for invalid strategy kind."""
        data = {"default_strategy": {"kind": "invalid_strategy", "parameters": {}}}

        with pytest.raises(ValueError):  # Pydantic ValidationError
            PolicyFileSchema(**data)

    def test_invalid_locale_format(self):
        """Test validation fails for invalid locale format."""
        data = {"locale": "invalid-locale-format"}

        with pytest.raises(ValueError):  # Pydantic ValidationError
            PolicyFileSchema(**data)

    def test_invalid_threshold_range(self):
        """Test validation fails for threshold outside 0-1 range."""
        data = {
            "per_entity": {
                "PERSON": {
                    "threshold": 1.5  # Invalid: > 1.0
                }
            }
        }

        with pytest.raises(ValueError):  # Pydantic ValidationError
            PolicyFileSchema(**data)


class TestPolicyLoader:
    """Test the PolicyLoader class functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.loader = PolicyLoader(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_simple_policy(self):
        """Test loading a simple policy without inheritance."""
        policy_content = dedent("""
            version: "1.0"
            name: "simple-policy"
            locale: "en"

            default_strategy:
              kind: "redact"
              parameters:
                redact_char: "*"

            per_entity:
              PERSON:
                kind: "template"
                parameters:
                  template: "[PERSON]"
                threshold: 0.8
        """)

        policy_file = self.temp_dir / "simple.yaml"
        policy_file.write_text(policy_content)

        policy = self.loader.load_policy(policy_file)

        assert isinstance(policy, MaskingPolicy)
        assert policy.locale == "en"
        assert policy.default_strategy.kind == StrategyKind.REDACT
        assert "PERSON" in policy.per_entity
        assert policy.per_entity["PERSON"].kind == StrategyKind.TEMPLATE
        assert policy.thresholds["PERSON"] == 0.8

    def test_load_policy_with_inheritance(self):
        """Test loading policy that inherits from base template."""
        # Create base policy
        base_content = dedent("""
            version: "1.0"
            name: "base-policy"

            default_strategy:
              kind: "redact"
              parameters:
                redact_char: "*"

            per_entity:
              PERSON:
                kind: "template"
                parameters:
                  template: "[PERSON]"
                threshold: 0.7
              EMAIL_ADDRESS:
                kind: "partial"
                parameters:
                  visible_chars: 3
                threshold: 0.6
        """)

        base_file = self.temp_dir / "base.yaml"
        base_file.write_text(base_content)

        # Create derived policy
        derived_content = dedent("""
            version: "1.0"
            name: "derived-policy"
            extends: "base.yaml"

            per_entity:
              PERSON:
                threshold: 0.9  # Override threshold
              PHONE_NUMBER:
                kind: "hash"
                parameters:
                  algorithm: "sha256"
                threshold: 0.8
        """)

        derived_file = self.temp_dir / "derived.yaml"
        derived_file.write_text(derived_content)

        policy = self.loader.load_policy(derived_file)

        # Should inherit EMAIL_ADDRESS from base
        assert "EMAIL_ADDRESS" in policy.per_entity
        assert policy.per_entity["EMAIL_ADDRESS"].kind == StrategyKind.PARTIAL

        # Should override PERSON threshold
        assert policy.thresholds["PERSON"] == 0.9

        # Should add new PHONE_NUMBER strategy
        assert "PHONE_NUMBER" in policy.per_entity
        assert policy.per_entity["PHONE_NUMBER"].kind == StrategyKind.HASH

    def test_circular_inheritance_detection(self):
        """Test detection of circular inheritance."""
        # Create policy A that extends B
        policy_a = dedent("""
            version: "1.0"
            extends: "policy_b.yaml"
            name: "policy-a"
        """)

        # Create policy B that extends A (circular)
        policy_b = dedent("""
            version: "1.0"
            extends: "policy_a.yaml"
            name: "policy-b"
        """)

        file_a = self.temp_dir / "policy_a.yaml"
        file_b = self.temp_dir / "policy_b.yaml"

        file_a.write_text(policy_a)
        file_b.write_text(policy_b)

        with pytest.raises(
            PolicyInheritanceError, match="Circular inheritance detected"
        ):
            self.loader.load_policy(file_a)

    def test_missing_base_policy_file(self):
        """Test error when base policy file doesn't exist."""
        derived_content = dedent("""
            version: "1.0"
            extends: "nonexistent.yaml"
            name: "derived-policy"
        """)

        derived_file = self.temp_dir / "derived.yaml"
        derived_file.write_text(derived_content)

        with pytest.raises(FileNotFoundError):
            self.loader.load_policy(derived_file)

    def test_invalid_yaml_syntax(self):
        """Test error handling for invalid YAML syntax."""
        invalid_content = """
        invalid: yaml: syntax: [unclosed
        """

        policy_file = self.temp_dir / "invalid.yaml"
        policy_file.write_text(invalid_content)

        with pytest.raises(PolicyValidationError, match="Invalid YAML"):
            self.loader.load_policy(policy_file)

    def test_policy_validation_errors(self):
        """Test policy validation catches errors."""
        policy_file = self.temp_dir / "test.yaml"

        # Test with invalid strategy kind
        invalid_content = dedent("""
            version: "1.0"
            per_entity:
              PERSON:
                kind: "invalid_strategy"
        """)

        policy_file.write_text(invalid_content)

        errors = self.loader.validate_policy_file(policy_file)
        assert len(errors) > 0
        assert any("strategy" in error.lower() for error in errors)

    def test_complex_inheritance_chain(self):
        """Test multiple levels of inheritance."""
        # Create base policy
        base_content = dedent("""
            version: "1.0"
            name: "base"

            default_strategy:
              kind: "redact"
              parameters:
                redact_char: "*"

            per_entity:
              PERSON:
                kind: "template"
                parameters:
                  template: "[PERSON]"
                threshold: 0.5
        """)

        # Create intermediate policy
        intermediate_content = dedent("""
            version: "1.0"
            name: "intermediate"
            extends: "base.yaml"

            per_entity:
              PERSON:
                threshold: 0.7  # Override threshold
              EMAIL_ADDRESS:
                kind: "partial"
                parameters:
                  visible_chars: 3
                  position: "start"
                threshold: 0.6
        """)

        # Create final policy
        final_content = dedent("""
            version: "1.0"
            name: "final"
            extends: "intermediate.yaml"

            per_entity:
              PERSON:
                threshold: 0.9  # Override again
              PHONE_NUMBER:
                kind: "hash"
                parameters:
                  algorithm: "sha256"
                threshold: 0.8
        """)

        base_file = self.temp_dir / "base.yaml"
        intermediate_file = self.temp_dir / "intermediate.yaml"
        final_file = self.temp_dir / "final.yaml"

        base_file.write_text(base_content)
        intermediate_file.write_text(intermediate_content)
        final_file.write_text(final_content)

        policy = self.loader.load_policy(final_file)

        # Should have all entities from inheritance chain
        assert "PERSON" in policy.per_entity
        assert "EMAIL_ADDRESS" in policy.per_entity
        assert "PHONE_NUMBER" in policy.per_entity

        # Should use final override for PERSON threshold
        assert policy.thresholds["PERSON"] == 0.9

        # Should inherit EMAIL_ADDRESS from intermediate
        assert policy.thresholds["EMAIL_ADDRESS"] == 0.6

        # Should have PHONE_NUMBER from final
        assert policy.thresholds["PHONE_NUMBER"] == 0.8

    def test_policy_caching(self):
        """Test that policies are cached correctly."""
        policy_content = dedent("""
            version: "1.0"
            name: "cached-policy"
        """)

        policy_file = self.temp_dir / "cached.yaml"
        policy_file.write_text(policy_content)

        # Load policy twice
        policy1 = self.loader.load_policy(policy_file)
        policy2 = self.loader.load_policy(policy_file)

        # Should be equivalent (caching doesn't affect functionality)
        assert policy1.locale == policy2.locale

    def test_relative_path_resolution(self):
        """Test relative path resolution for inheritance."""
        # Create subdirectory structure
        subdir = self.temp_dir / "subdir"
        subdir.mkdir()

        base_content = dedent("""
            version: "1.0"
            name: "base-in-subdir"

            default_strategy:
              kind: "redact"
        """)

        derived_content = dedent("""
            version: "1.0"
            name: "derived-in-subdir"
            extends: "base.yaml"
        """)

        base_file = subdir / "base.yaml"
        derived_file = subdir / "derived.yaml"

        base_file.write_text(base_content)
        derived_file.write_text(derived_content)

        policy = self.loader.load_policy(derived_file)
        assert policy.default_strategy.kind == StrategyKind.REDACT


class TestPolicyConversion:
    """Test conversion from schema to MaskingPolicy."""

    def test_schema_to_masking_policy_conversion(self):
        """Test complete conversion from schema to MaskingPolicy."""
        schema_data = {
            "version": "1.0",
            "locale": "en-US",
            "seed": "test-seed",
            "min_entity_length": 3,
            "default_strategy": {"kind": "redact", "parameters": {"redact_char": "X"}},
            "per_entity": {
                "PERSON": {
                    "kind": "hash",
                    "parameters": {"algorithm": "sha256"},
                    "threshold": 0.8,
                    "enabled": True,
                },
                "EMAIL_ADDRESS": {
                    "kind": "partial",
                    "parameters": {"visible_chars": 3},
                    "threshold": 0.7,
                    "enabled": True,
                },
            },
            "allow_list": ["test@example.com", "John Doe"],
            "deny_list": ["confidential"],
            "context_rules": {
                "heading": {"enabled": False},
                "table": {"enabled": True, "threshold_overrides": {"PERSON": 0.9}},
            },
        }

        schema = PolicyFileSchema(**schema_data)
        loader = PolicyLoader()
        policy = loader._schema_to_masking_policy(schema)

        # Test basic conversion
        assert policy.locale == "en-US"
        assert policy.seed == "test-seed"
        assert policy.min_entity_length == 3

        # Test strategy conversion
        assert policy.default_strategy.kind == StrategyKind.REDACT
        assert policy.default_strategy.parameters["redact_char"] == "X"

        # Test per-entity strategies
        assert "PERSON" in policy.per_entity
        assert policy.per_entity["PERSON"].kind == StrategyKind.HASH
        assert policy.thresholds["PERSON"] == 0.8

        assert "EMAIL_ADDRESS" in policy.per_entity
        assert policy.per_entity["EMAIL_ADDRESS"].kind == StrategyKind.PARTIAL
        assert policy.thresholds["EMAIL_ADDRESS"] == 0.7

        # Test allow/deny lists
        assert "test@example.com" in policy.allow_list
        assert "John Doe" in policy.allow_list
        assert "confidential" in policy.deny_list

        # Test context rules
        assert "heading" in policy.context_rules
        assert not policy.context_rules["heading"]["enabled"]
        assert "table" in policy.context_rules
        assert policy.context_rules["table"]["enabled"]
