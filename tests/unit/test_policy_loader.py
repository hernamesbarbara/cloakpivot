"""Unit tests for cloakpivot.core.policy_loader module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.policy_loader import (
    AllowListItem,
    ContextRuleConfig,
    EntityConfig,
    LocaleConfig,
    PolicyCompositionConfig,
    PolicyFileSchema,
    PolicyInheritanceError,
    PolicyLoadContext,
    PolicyLoader,
    PolicyValidationError,
    StrategyConfig,
)
from cloakpivot.core.strategies import StrategyKind


class TestExceptions:
    """Test custom exceptions."""

    def test_policy_validation_error(self):
        """Test PolicyValidationError."""
        error = PolicyValidationError("Invalid policy")
        assert str(error) == "Invalid policy"
        assert isinstance(error, Exception)

    def test_policy_inheritance_error(self):
        """Test PolicyInheritanceError."""
        error = PolicyInheritanceError("Circular inheritance")
        assert str(error) == "Circular inheritance"
        assert isinstance(error, Exception)


class TestPolicyLoadContext:
    """Test PolicyLoadContext dataclass."""

    def test_initialization(self):
        """Test PolicyLoadContext initialization."""
        context = PolicyLoadContext(
            current_file=Path("/policies/test.yaml"),
            base_path=Path("/policies"),
            inheritance_chain=[Path("/policies/base.yaml")],
        )
        assert context.current_file == Path("/policies/test.yaml")
        assert context.base_path == Path("/policies")
        assert context.inheritance_chain == [Path("/policies/base.yaml")]

    def test_derive_path_absolute(self):
        """Test derive_path with absolute path."""
        context = PolicyLoadContext(
            current_file=Path("/policies/test.yaml"),
            base_path=Path("/policies"),
            inheritance_chain=[],
        )
        result = context.derive_path("/other/policy.yaml")
        assert result == Path("/other/policy.yaml")

    def test_derive_path_relative(self):
        """Test derive_path with relative path."""
        context = PolicyLoadContext(
            current_file=Path("/policies/test.yaml"),
            base_path=Path("/policies"),
            inheritance_chain=[],
        )
        result = context.derive_path("base.yaml")
        assert result == Path("/policies/base.yaml").resolve()

    def test_derive_path_parent_directory(self):
        """Test derive_path with parent directory reference."""
        context = PolicyLoadContext(
            current_file=Path("/policies/subdir/test.yaml"),
            base_path=Path("/policies"),
            inheritance_chain=[],
        )
        result = context.derive_path("../base.yaml")
        assert result == Path("/policies/base.yaml").resolve()


class TestStrategyConfig:
    """Test StrategyConfig Pydantic model."""

    def test_valid_strategy_config(self):
        """Test valid StrategyConfig."""
        config = StrategyConfig(kind="redact", parameters={"char": "#"})
        assert config.kind == "redact"
        assert config.parameters == {"char": "#"}

    def test_valid_strategy_kinds(self):
        """Test all valid strategy kinds."""
        for kind in ["redact", "hash", "template", "partial", "surrogate"]:
            config = StrategyConfig(kind=kind)
            assert config.kind == kind

    def test_invalid_strategy_kind(self):
        """Test invalid strategy kind."""
        with pytest.raises(ValueError, match="Invalid strategy kind"):
            StrategyConfig(kind="invalid_kind")

    def test_empty_parameters(self):
        """Test strategy config with no parameters."""
        config = StrategyConfig(kind="redact")
        assert config.parameters == {}


class TestEntityConfig:
    """Test EntityConfig Pydantic model."""

    def test_valid_entity_config(self):
        """Test valid EntityConfig."""
        config = EntityConfig(kind="hash", parameters={"salt": "test"}, threshold=0.8, enabled=True)
        assert config.kind == "hash"
        assert config.parameters == {"salt": "test"}
        assert config.threshold == 0.8
        assert config.enabled is True

    def test_all_none_values(self):
        """Test EntityConfig with all None values."""
        config = EntityConfig()
        assert config.kind is None
        assert config.parameters is None
        assert config.threshold is None
        assert config.enabled is None

    def test_invalid_kind(self):
        """Test EntityConfig with invalid kind."""
        with pytest.raises(ValueError, match="Invalid strategy kind"):
            EntityConfig(kind="invalid")

    def test_threshold_validation(self):
        """Test threshold validation."""
        # Valid thresholds
        EntityConfig(threshold=0.0)
        EntityConfig(threshold=0.5)
        EntityConfig(threshold=1.0)

        # Invalid thresholds
        with pytest.raises(ValueError):
            EntityConfig(threshold=-0.1)
        with pytest.raises(ValueError):
            EntityConfig(threshold=1.1)


class TestLocaleConfig:
    """Test LocaleConfig Pydantic model."""

    def test_valid_locale_config(self):
        """Test valid LocaleConfig."""
        entity_overrides = {
            "PERSON": EntityConfig(kind="redact"),
            "EMAIL": EntityConfig(threshold=0.9),
        }
        config = LocaleConfig(recognizers=["custom_recognizer"], entity_overrides=entity_overrides)
        assert config.recognizers == ["custom_recognizer"]
        assert "PERSON" in config.entity_overrides
        assert config.entity_overrides["PERSON"].kind == "redact"

    def test_empty_locale_config(self):
        """Test empty LocaleConfig."""
        config = LocaleConfig()
        assert config.recognizers is None
        assert config.entity_overrides is None


class TestContextRuleConfig:
    """Test ContextRuleConfig Pydantic model."""

    def test_valid_context_rule(self):
        """Test valid ContextRuleConfig."""
        config = ContextRuleConfig(
            enabled=True,
            strategy_overrides={"PERSON": EntityConfig(kind="hash")},
            threshold_overrides={"EMAIL": 0.95},
        )
        assert config.enabled is True
        assert "PERSON" in config.strategy_overrides
        assert config.threshold_overrides["EMAIL"] == 0.95

    def test_disabled_context(self):
        """Test disabled context rule."""
        config = ContextRuleConfig(enabled=False)
        assert config.enabled is False
        assert config.strategy_overrides is None
        assert config.threshold_overrides is None


class TestAllowListItem:
    """Test AllowListItem Pydantic model."""

    def test_pattern_based_item(self):
        """Test AllowListItem with pattern."""
        config = AllowListItem(pattern=r"^test.*")
        assert config.pattern == r"^test.*"
        assert config.value is None

    def test_value_based_item(self):
        """Test AllowListItem with value."""
        config = AllowListItem(value="test@example.com")
        assert config.value == "test@example.com"
        assert config.pattern is None

    def test_invalid_regex_pattern(self):
        """Test AllowListItem with invalid regex."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            AllowListItem(pattern="[invalid")

    def test_both_pattern_and_value(self):
        """Test AllowListItem with both pattern and value."""
        config = AllowListItem(pattern=r"^test", value="test")
        assert config.pattern == r"^test"
        assert config.value == "test"


class TestPolicyCompositionConfig:
    """Test PolicyCompositionConfig Pydantic model."""

    def test_default_values(self):
        """Test default values."""
        config = PolicyCompositionConfig()
        assert config.merge_strategy == "override"
        assert config.validation_level == "strict"

    def test_valid_merge_strategies(self):
        """Test valid merge strategies."""
        for strategy in ["override", "merge", "strict"]:
            config = PolicyCompositionConfig(merge_strategy=strategy)
            assert config.merge_strategy == strategy

    def test_valid_validation_levels(self):
        """Test valid validation levels."""
        for level in ["strict", "warn", "permissive"]:
            config = PolicyCompositionConfig(validation_level=level)
            assert config.validation_level == level

    def test_invalid_merge_strategy(self):
        """Test invalid merge strategy."""
        with pytest.raises(ValueError, match="Invalid merge strategy"):
            PolicyCompositionConfig(merge_strategy="invalid")

    def test_invalid_validation_level(self):
        """Test invalid validation level."""
        with pytest.raises(ValueError, match="Invalid validation level"):
            PolicyCompositionConfig(validation_level="invalid")


class TestPolicyFileSchema:
    """Test PolicyFileSchema Pydantic model."""

    def test_minimal_schema(self):
        """Test minimal valid schema."""
        schema = PolicyFileSchema()
        assert schema.version == "1.0"
        assert schema.locale == "en"
        assert schema.min_entity_length == 1

    def test_full_schema(self):
        """Test full schema with all fields."""
        schema = PolicyFileSchema(
            version="2.0",
            name="Test Policy",
            description="Test description",
            extends="base.yaml",
            locale="fr-CA",
            seed="test-seed",
            default_strategy=StrategyConfig(kind="redact"),
            per_entity={"PERSON": EntityConfig(kind="hash")},
            thresholds={"EMAIL": 0.9},
            allow_list=["allowed"],
            deny_list=["denied"],
            min_entity_length=3,
            locales={"en": LocaleConfig()},
            context_rules={"medical": ContextRuleConfig(enabled=True)},
            policy_composition=PolicyCompositionConfig(),
        )
        assert schema.name == "Test Policy"
        assert schema.locale == "fr-CA"
        assert schema.min_entity_length == 3

    def test_locale_validation(self):
        """Test locale format validation."""
        # Valid locales
        PolicyFileSchema(locale="en")
        PolicyFileSchema(locale="fr-CA")

        # Invalid locales
        with pytest.raises(ValueError, match="Locale must follow format"):
            PolicyFileSchema(locale="ENG")
        with pytest.raises(ValueError, match="Locale must follow format"):
            PolicyFileSchema(locale="en_US")  # Wrong separator

    def test_version_validation(self):
        """Test version format validation."""
        # Valid versions
        PolicyFileSchema(version="1.0")
        PolicyFileSchema(version="2.1.3")

        # Invalid versions
        with pytest.raises(ValueError, match="Version must follow format"):
            PolicyFileSchema(version="v1.0")
        with pytest.raises(ValueError, match="Version must follow format"):
            PolicyFileSchema(version="1")

    def test_extends_list(self):
        """Test extends field with list."""
        schema = PolicyFileSchema(extends=["base1.yaml", "base2.yaml"])
        assert schema.extends == ["base1.yaml", "base2.yaml"]


class TestPolicyLoader:
    """Test PolicyLoader class."""

    def test_initialization_default(self):
        """Test PolicyLoader initialization with defaults."""
        loader = PolicyLoader()
        assert loader.base_path == Path.cwd()
        assert loader._policy_cache == {}

    def test_initialization_with_base_path(self):
        """Test PolicyLoader initialization with base path."""
        base = Path("/custom/path")
        loader = PolicyLoader(base_path=base)
        assert loader.base_path == base

    @patch("cloakpivot.core.policy_loader.Path.exists")
    @patch("cloakpivot.core.policy_loader.Path.open")
    def test_load_policy_simple(self, mock_open, mock_exists):
        """Test loading a simple policy."""
        mock_exists.return_value = True
        policy_yaml = """
        version: "1.0"
        name: Test Policy
        locale: en
        default_strategy:
          kind: redact
        """
        mock_open.return_value.__enter__.return_value.read.return_value = policy_yaml
        mock_file = mock_open.return_value.__enter__.return_value
        type(mock_file).read = lambda self: policy_yaml

        with patch("yaml.safe_load", return_value=yaml.safe_load(policy_yaml)):
            loader = PolicyLoader()
            policy = loader.load_policy("test.yaml")

            assert isinstance(policy, MaskingPolicy)
            assert policy.locale == "en"
            assert policy.default_strategy.kind == StrategyKind.REDACT

    @patch("cloakpivot.core.policy_loader.Path.exists")
    def test_load_policy_file_not_found(self, mock_exists):
        """Test loading non-existent policy file."""
        mock_exists.return_value = False
        loader = PolicyLoader()

        with pytest.raises(FileNotFoundError, match="Policy file not found"):
            loader.load_policy("nonexistent.yaml")

    @patch("cloakpivot.core.policy_loader.Path.exists")
    @patch("cloakpivot.core.policy_loader.Path.open")
    def test_load_policy_invalid_yaml(self, mock_open, mock_exists):
        """Test loading policy with invalid YAML."""
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = "invalid: yaml: content:"

        with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
            loader = PolicyLoader()
            with pytest.raises(PolicyValidationError, match="Invalid YAML"):
                loader.load_policy("invalid.yaml")

    @patch("cloakpivot.core.policy_loader.Path.exists")
    @patch("cloakpivot.core.policy_loader.Path.open")
    def test_circular_inheritance_detection(self, mock_open, mock_exists):
        """Test circular inheritance detection."""
        mock_exists.return_value = True

        # Create mock policy files that reference each other
        policy_a = {"extends": "b.yaml"}
        policy_b = {"extends": "a.yaml"}

        def mock_yaml_load(content):
            if "a.yaml" in str(content):
                return policy_a
            return policy_b

        with patch("yaml.safe_load", side_effect=[policy_a, policy_b]):
            loader = PolicyLoader()
            with pytest.raises(PolicyInheritanceError, match="Circular inheritance"):
                loader.load_policy("a.yaml")

    def test_validate_policy_file(self):
        """Test validate_policy_file method."""
        loader = PolicyLoader()

        with patch.object(loader, "load_policy") as mock_load:
            # Valid policy
            mock_load.return_value = Mock(spec=MaskingPolicy)
            errors = loader.validate_policy_file("valid.yaml")
            assert errors == []

            # Invalid policy
            mock_load.side_effect = PolicyValidationError("Invalid")
            errors = loader.validate_policy_file("invalid.yaml")
            assert len(errors) == 1
            assert "Invalid" in errors[0]

    def test_merge_policies_empty_list(self):
        """Test _merge_policies with empty list."""
        loader = PolicyLoader()
        with pytest.raises(ValueError, match="Cannot merge empty policy list"):
            loader._merge_policies([])

    def test_merge_policies_single_policy(self):
        """Test _merge_policies with single policy."""
        loader = PolicyLoader()
        policy = PolicyFileSchema(name="Test")
        result = loader._merge_policies([policy])
        assert result == policy

    def test_merge_two_policies_override_strategy(self):
        """Test _merge_two_policies with override strategy."""
        loader = PolicyLoader()
        base = PolicyFileSchema(name="Base", locale="en", thresholds={"PERSON": 0.8})
        override = PolicyFileSchema(
            name="Override",
            locale="fr",
            thresholds={"EMAIL": 0.9},
            policy_composition=PolicyCompositionConfig(merge_strategy="override"),
        )

        result = loader._merge_two_policies(base, override)
        assert result.name == "Override"
        assert result.locale == "fr"
        assert result.thresholds == {"PERSON": 0.8, "EMAIL": 0.9}

    def test_schema_to_masking_policy_conversion(self):
        """Test _schema_to_masking_policy conversion."""
        loader = PolicyLoader()
        schema = PolicyFileSchema(
            locale="en-US",
            seed="test-seed",
            default_strategy=StrategyConfig(kind="hash"),
            per_entity={
                "PERSON": EntityConfig(kind="hash", threshold=0.9),
                "EMAIL": EntityConfig(threshold=0.85),
            },
            thresholds={"PHONE": 0.7},
            allow_list=["allowed"],
            deny_list=["denied"],
            min_entity_length=2,
        )

        policy = loader._schema_to_masking_policy(schema)

        assert isinstance(policy, MaskingPolicy)
        assert policy.locale == "en-US"
        assert policy.seed == "test-seed"
        assert policy.default_strategy.kind == StrategyKind.HASH
        assert "PERSON" in policy.per_entity
        assert policy.thresholds.get("PERSON") == 0.9
        assert policy.thresholds.get("EMAIL") == 0.85
        assert policy.thresholds.get("PHONE") == 0.7
        assert "allowed" in policy.allow_list
        assert "denied" in policy.deny_list
        assert policy.min_entity_length == 2
