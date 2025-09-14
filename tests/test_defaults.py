"""Test default configurations and policies."""

import pytest

from cloakpivot.defaults import (
    DEFAULT_ENTITIES,
    HIGH_RISK_ENTITIES,
    KEEP_BY_DEFAULT,
    get_default_policy,
    get_conservative_policy,
    get_permissive_policy,
    get_default_analyzer_config,
    get_multilingual_analyzer_config,
    get_high_precision_analyzer_config,
    get_high_recall_analyzer_config,
    get_policy_preset,
    get_analyzer_preset,
    POLICY_PRESETS,
    ANALYZER_PRESETS
)
from cloakpivot.core.strategies import StrategyKind
from cloakpivot.core.policies import MaskingPolicy


class TestDefaultEntities:
    """Test default entity configurations."""

    def test_default_entities_list(self):
        """Test DEFAULT_ENTITIES contains common PII types."""
        assert isinstance(DEFAULT_ENTITIES, list)
        assert len(DEFAULT_ENTITIES) > 0

        # Should include common entities
        assert "EMAIL_ADDRESS" in DEFAULT_ENTITIES
        assert "PERSON" in DEFAULT_ENTITIES
        assert "PHONE_NUMBER" in DEFAULT_ENTITIES
        assert "CREDIT_CARD" in DEFAULT_ENTITIES
        assert "US_SSN" in DEFAULT_ENTITIES

    def test_high_risk_entities(self):
        """Test HIGH_RISK_ENTITIES contains sensitive types."""
        assert isinstance(HIGH_RISK_ENTITIES, list)
        assert len(HIGH_RISK_ENTITIES) > 0

        # Should include highly sensitive entities
        assert "CREDIT_CARD" in HIGH_RISK_ENTITIES
        assert "US_SSN" in HIGH_RISK_ENTITIES
        assert "MEDICAL_LICENSE" in HIGH_RISK_ENTITIES

    def test_keep_by_default_entities(self):
        """Test KEEP_BY_DEFAULT contains contextual entities."""
        assert isinstance(KEEP_BY_DEFAULT, list)

        # These are often needed for context
        assert "DATE_TIME" in KEEP_BY_DEFAULT
        assert "URL" in KEEP_BY_DEFAULT


class TestDefaultPolicies:
    """Test default masking policies."""

    def test_get_default_policy(self):
        """Test default policy structure and content."""
        policy = get_default_policy()

        assert isinstance(policy, MaskingPolicy)
        assert policy.per_entity is not None
        assert policy.default_strategy is not None

        # Check specific entity strategies
        assert "EMAIL_ADDRESS" in policy.per_entity
        email_strategy = policy.per_entity["EMAIL_ADDRESS"]
        assert email_strategy.kind == StrategyKind.TEMPLATE
        assert email_strategy.parameters["template"] == "[EMAIL]"

        assert "PHONE_NUMBER" in policy.per_entity
        phone_strategy = policy.per_entity["PHONE_NUMBER"]
        assert phone_strategy.kind == StrategyKind.TEMPLATE
        assert phone_strategy.parameters["template"] == "[PHONE]"

        # Check SSN uses partial masking
        assert "US_SSN" in policy.per_entity
        ssn_strategy = policy.per_entity["US_SSN"]
        assert ssn_strategy.kind == StrategyKind.PARTIAL

    def test_get_conservative_policy(self):
        """Test conservative policy uses aggressive masking."""
        policy = get_conservative_policy()

        assert isinstance(policy, MaskingPolicy)
        # Conservative policy should have minimal per-entity rules
        assert len(policy.per_entity) == 0
        # Should use REDACT for everything
        assert policy.default_strategy.kind == StrategyKind.REDACT
        assert policy.default_strategy.parameters["replacement"] == "[REMOVED]"

    def test_get_permissive_policy(self):
        """Test permissive policy with minimal masking."""
        policy = get_permissive_policy()

        assert isinstance(policy, MaskingPolicy)
        # Should only mask high-risk entities
        for entity in HIGH_RISK_ENTITIES:
            assert entity in policy.per_entity

        # Default should be less aggressive
        assert policy.default_strategy.kind == StrategyKind.TEMPLATE
        assert policy.default_strategy.parameters["template"] == "[PII]"

    def test_policy_strategies_are_valid(self):
        """Test all policy strategies have valid configurations."""
        policies = [
            get_default_policy(),
            get_conservative_policy(),
            get_permissive_policy()
        ]

        for policy in policies:
            # Default strategy should be valid
            assert policy.default_strategy is not None
            assert policy.default_strategy.kind in StrategyKind

            # Per-entity strategies should be valid
            for entity_type, strategy in policy.per_entity.items():
                assert isinstance(entity_type, str)
                assert strategy is not None
                assert strategy.kind in StrategyKind
                assert strategy.parameters is not None


class TestAnalyzerConfigs:
    """Test analyzer configurations."""

    def test_get_default_analyzer_config(self):
        """Test default analyzer configuration."""
        config = get_default_analyzer_config()

        assert isinstance(config, dict)
        assert "languages" in config
        assert config["languages"] == ["en"]
        assert "confidence_threshold" in config
        assert config["confidence_threshold"] == 0.7
        assert config["return_decision_process"] is False

    def test_get_multilingual_analyzer_config(self):
        """Test multilingual analyzer configuration."""
        languages = ["en", "es", "fr"]
        config = get_multilingual_analyzer_config(languages)

        assert isinstance(config, dict)
        assert config["languages"] == languages
        # Lower threshold for multilingual
        assert config["confidence_threshold"] == 0.65

    def test_get_high_precision_analyzer_config(self):
        """Test high precision analyzer configuration."""
        config = get_high_precision_analyzer_config()

        assert isinstance(config, dict)
        # Higher threshold for precision
        assert config["confidence_threshold"] == 0.85
        # Include decision process for debugging
        assert config["return_decision_process"] is True

    def test_get_high_recall_analyzer_config(self):
        """Test high recall analyzer configuration."""
        config = get_high_recall_analyzer_config()

        assert isinstance(config, dict)
        # Lower threshold for recall
        assert config["confidence_threshold"] == 0.5
        assert config["return_decision_process"] is False


class TestPresets:
    """Test preset management."""

    def test_policy_presets_available(self):
        """Test policy presets are defined."""
        assert isinstance(POLICY_PRESETS, dict)
        assert "default" in POLICY_PRESETS
        assert "conservative" in POLICY_PRESETS
        assert "permissive" in POLICY_PRESETS

    def test_get_policy_preset(self):
        """Test retrieving policy presets by name."""
        # Valid preset
        policy = get_policy_preset("default")
        assert isinstance(policy, MaskingPolicy)

        policy = get_policy_preset("conservative")
        assert isinstance(policy, MaskingPolicy)
        assert policy.default_strategy.kind == StrategyKind.REDACT

        policy = get_policy_preset("permissive")
        assert isinstance(policy, MaskingPolicy)

    def test_get_policy_preset_invalid(self):
        """Test invalid policy preset raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_policy_preset("invalid_preset")

        assert "Unknown policy preset" in str(exc_info.value)

    def test_analyzer_presets_available(self):
        """Test analyzer presets are defined."""
        assert isinstance(ANALYZER_PRESETS, dict)
        assert "default" in ANALYZER_PRESETS
        assert "high_precision" in ANALYZER_PRESETS
        assert "high_recall" in ANALYZER_PRESETS

    def test_get_analyzer_preset(self):
        """Test retrieving analyzer presets by name."""
        # Valid presets
        config = get_analyzer_preset("default")
        assert isinstance(config, dict)
        assert config["confidence_threshold"] == 0.7

        config = get_analyzer_preset("high_precision")
        assert isinstance(config, dict)
        assert config["confidence_threshold"] == 0.85

        config = get_analyzer_preset("high_recall")
        assert isinstance(config, dict)
        assert config["confidence_threshold"] == 0.5

    def test_get_analyzer_preset_invalid(self):
        """Test invalid analyzer preset raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_analyzer_preset("invalid_preset")

        assert "Unknown analyzer preset" in str(exc_info.value)


class TestDefaultsIntegration:
    """Test defaults work with CloakEngine."""

    def test_default_policy_with_engine(self):
        """Test default policy works with CloakEngine."""
        from cloakpivot.engine import CloakEngine
        from docling_core.types import DoclingDocument
        from docling_core.types.doc.document import TextItem

        policy = get_default_policy()
        engine = CloakEngine(default_policy=policy)

        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="Email: test@example.com",
            self_ref="#/texts/0",
            label="text",
            orig="Email: test@example.com"
        )]

        result = engine.mask_document(doc)
        assert result is not None
        assert "test@example.com" not in result.document.texts[0].text

    def test_conservative_policy_with_engine(self):
        """Test conservative policy works with CloakEngine."""
        from cloakpivot.engine import CloakEngine
        from docling_core.types import DoclingDocument
        from docling_core.types.doc.document import TextItem

        policy = get_conservative_policy()
        engine = CloakEngine(default_policy=policy)

        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="Call 555-123-4567 or email test@example.com",
            self_ref="#/texts/0",
            label="text",
            orig="Call 555-123-4567 or email test@example.com"
        )]

        result = engine.mask_document(doc)
        assert result is not None
        masked_text = result.document.texts[0].text
        # Should mask aggressively
        assert "test@example.com" not in masked_text
        # Conservative policy should mask detected entities
        if result.entities_found > 0:
            # If phone was detected, it should be masked
            assert "555-123-4567" not in masked_text

    def test_analyzer_config_with_engine(self):
        """Test analyzer configs work with CloakEngine."""
        from cloakpivot.engine import CloakEngine

        config = get_high_precision_analyzer_config()
        engine = CloakEngine(analyzer_config=config)

        assert engine is not None
        assert engine.analyzer_config.min_confidence == 0.85