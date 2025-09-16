"""Unit tests for CloakEngineBuilder."""

import pytest

from cloakpivot import CloakEngineBuilder
from cloakpivot.core import StrategyKind


class TestCloakEngineBuilder:
    """Test CloakEngineBuilder functionality."""

    def test_builder_creation(self):
        """Test creating a builder."""
        builder = CloakEngineBuilder()
        assert builder is not None
        assert isinstance(builder, CloakEngineBuilder)

    def test_builder_with_confidence_threshold(self):
        """Test setting confidence threshold."""
        engine = CloakEngineBuilder().with_confidence_threshold(0.75).build()
        assert engine._analyzer_config.min_confidence == 0.75

    def test_builder_with_languages(self):
        """Test setting custom languages."""
        languages = ["en", "es", "fr"]
        engine = CloakEngineBuilder().with_languages(languages).build()
        assert engine is not None

    def test_builder_with_custom_policy(self):
        """Test using custom policy."""
        from cloakpivot.core import MaskingPolicy, Strategy

        policy = MaskingPolicy(default_strategy=Strategy(StrategyKind.REDACT))
        engine = CloakEngineBuilder().with_custom_policy(policy).build()
        assert engine is not None

    def test_builder_with_decision_process(self):
        """Test enabling decision process."""
        engine = CloakEngineBuilder().with_decision_process(True).build()
        assert engine is not None

    def test_builder_with_analyzer_config(self):
        """Test using analyzer config dict."""
        config = {
            "confidence_threshold": 0.8,
            "languages": ["en", "es"],
        }
        engine = CloakEngineBuilder().with_analyzer_config(config).build()
        assert engine is not None

    def test_builder_with_additional_recognizers(self):
        """Test adding additional recognizers."""
        recognizers = ["CUSTOM_ENTITY_1", "CUSTOM_ENTITY_2"]
        engine = CloakEngineBuilder().with_additional_recognizers(recognizers).build()
        assert engine is not None

    def test_builder_with_presidio_disabled(self):
        """Test disabling Presidio engine."""
        engine = CloakEngineBuilder().with_presidio_engine(False).build()
        assert engine is not None
        # Should still work but without Presidio detection

    def test_builder_with_conflict_resolution(self):
        """Test setting conflict resolution configuration."""
        from cloakpivot.core.normalization import ConflictResolutionConfig

        config = ConflictResolutionConfig(merge_threshold_chars=10)
        engine = CloakEngineBuilder().with_conflict_resolution(config).build()
        assert engine is not None

    def test_builder_chaining(self):
        """Test chaining multiple builder methods."""
        from cloakpivot.core import MaskingPolicy, Strategy

        policy = MaskingPolicy(default_strategy=Strategy(StrategyKind.REDACT))
        engine = (
            CloakEngineBuilder()
            .with_confidence_threshold(0.8)
            .with_languages(["en", "es"])
            .with_custom_policy(policy)
            .with_presidio_engine(True)
            .build()
        )

        assert engine is not None

    def test_builder_reset(self):
        """Test that each builder call starts fresh."""
        builder1 = CloakEngineBuilder().with_confidence_threshold(0.9)
        builder2 = CloakEngineBuilder().with_confidence_threshold(0.5)

        engine1 = builder1.build()
        engine2 = builder2.build()

        # Each builder should create independent engines
        assert engine1 is not engine2

    def test_builder_invalid_confidence(self):
        """Test that invalid confidence threshold raises error."""
        builder = CloakEngineBuilder()

        with pytest.raises(ValueError):
            builder.with_confidence_threshold(-0.5)

        with pytest.raises(ValueError):
            builder.with_confidence_threshold(2.0)

    def test_builder_invalid_languages(self):
        """Test that empty languages list fails."""
        builder = CloakEngineBuilder()
        # Empty list should cause an error when building
        with pytest.raises(IndexError):
            builder.with_languages([]).build()

    def test_builder_multiple_calls(self):
        """Test that multiple calls work correctly."""
        builder = CloakEngineBuilder()

        # Successive calls should override
        builder.with_confidence_threshold(0.5)
        builder.with_confidence_threshold(0.9)

        engine = builder.build()
        assert engine is not None
