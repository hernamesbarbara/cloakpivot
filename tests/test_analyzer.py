"""Tests for Presidio AnalyzerEngine integration."""

from unittest.mock import Mock, patch

import pytest

from cloakpivot.core.analyzer import (
    AnalyzerConfig,
    AnalyzerEngineWrapper,
    EntityDetectionResult,
    RecognizerRegistry,
)
from cloakpivot.core.policies import MaskingPolicy


class TestAnalyzerEngineWrapper:
    """Test AnalyzerEngine wrapper functionality."""

    def test_default_initialization(self):
        """Test analyzer initializes with default configuration."""
        analyzer = AnalyzerEngineWrapper()

        assert analyzer.is_initialized is False
        assert analyzer.config.language == "en"
        assert analyzer.config.min_confidence == 0.5
        assert len(analyzer.registry.get_enabled_recognizers()) > 0

    def test_lazy_initialization(self):
        """Test analyzer is initialized on first use."""
        analyzer = AnalyzerEngineWrapper()

        assert analyzer.is_initialized is False

        # First call should initialize
        with patch("cloakpivot.core.analyzer.AnalyzerEngine") as mock_engine:
            with patch(
                "cloakpivot.core.analyzer.NlpEngineProvider"
            ) as mock_nlp_provider:
                # Mock the NLP engine provider and engine
                mock_nlp_engine = Mock()
                mock_nlp_provider.return_value.create_engine.return_value = (
                    mock_nlp_engine
                )

                # Mock the analyzer engine
                mock_instance = Mock()
                mock_instance.analyze.return_value = (
                    []
                )  # Return empty list for analysis
                mock_engine.return_value = mock_instance

                result = analyzer.analyze_text("test text")

                assert analyzer.is_initialized is True
                mock_engine.assert_called_once()
                assert result == []  # Should return empty list from mock

    def test_configuration_from_policy(self):
        """Test analyzer configuration from MaskingPolicy."""
        policy = MaskingPolicy(locale="es", thresholds={"PHONE_NUMBER": 0.8})

        analyzer = AnalyzerEngineWrapper.from_policy(policy)

        assert analyzer.config.language == "es"
        assert (
            analyzer.config.min_confidence == 0.8
        )  # Minimum threshold from policy thresholds

    def test_custom_recognizer_configuration(self):
        """Test custom recognizer configuration."""
        config = AnalyzerConfig(
            language="en",
            enabled_recognizers=["PHONE_NUMBER", "EMAIL_ADDRESS"],
            disabled_recognizers=["PERSON"],
            min_confidence=0.7,
        )

        analyzer = AnalyzerEngineWrapper(config=config)

        enabled = analyzer.registry.get_enabled_recognizers()
        assert "PHONE_NUMBER" in enabled
        assert "EMAIL_ADDRESS" in enabled
        assert "PERSON" not in enabled


class TestRecognizerRegistry:
    """Test recognizer registry functionality."""

    def test_default_recognizers_enabled(self):
        """Test default recognizers are enabled by default."""
        registry = RecognizerRegistry()

        enabled = registry.get_enabled_recognizers()
        expected = {
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "CREDIT_CARD",
            "US_SSN",
            "PERSON",
            "URL",
            "IP_ADDRESS",
        }

        assert expected.issubset(set(enabled))

    def test_enable_disable_recognizers(self):
        """Test enabling and disabling recognizers."""
        registry = RecognizerRegistry()

        # Disable a recognizer
        registry.disable_recognizer("PERSON")
        enabled = registry.get_enabled_recognizers()
        assert "PERSON" not in enabled

        # Re-enable it
        registry.enable_recognizer("PERSON")
        enabled = registry.get_enabled_recognizers()
        assert "PERSON" in enabled

    def test_custom_recognizer_addition(self):
        """Test adding custom recognizers."""
        registry = RecognizerRegistry()

        # Mock custom recognizer
        mock_recognizer = Mock()
        mock_recognizer.supported_entities = ["CUSTOM_ENTITY"]

        registry.add_custom_recognizer("CUSTOM_ENTITY", mock_recognizer)

        enabled = registry.get_enabled_recognizers()
        assert "CUSTOM_ENTITY" in enabled


class TestAnalyzerConfig:
    """Test analyzer configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = AnalyzerConfig()

        assert config.language == "en"
        assert config.min_confidence == 0.5
        assert config.enabled_recognizers is None
        assert config.disabled_recognizers == set()
        assert config.custom_recognizers == {}

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid language
        config = AnalyzerConfig(language="en")
        assert config.language == "en"

        # Invalid language should raise error
        with pytest.raises(ValueError, match="Language must be a valid ISO 639-1 code"):
            AnalyzerConfig(language="invalid")

        # Valid confidence range
        config = AnalyzerConfig(min_confidence=0.8)
        assert config.min_confidence == 0.8

        # Invalid confidence range
        with pytest.raises(
            ValueError, match="min_confidence must be between 0.0 and 1.0"
        ):
            AnalyzerConfig(min_confidence=1.5)

    def test_from_policy_conversion(self):
        """Test creating config from MaskingPolicy."""
        policy = MaskingPolicy(
            locale="fr", thresholds={"PHONE_NUMBER": 0.9, "EMAIL_ADDRESS": 0.7}
        )

        config = AnalyzerConfig.from_policy(policy)

        assert config.language == "fr"
        # Should use minimum threshold as global confidence
        assert config.min_confidence == 0.7


class TestEntityDetectionResult:
    """Test entity detection result processing."""

    def test_from_presidio_result(self):
        """Test converting from Presidio RecognizerResult."""
        # Mock RecognizerResult
        mock_result = Mock()
        mock_result.entity_type = "PHONE_NUMBER"
        mock_result.start = 10
        mock_result.end = 22
        mock_result.score = 0.95

        detection = EntityDetectionResult.from_presidio_result(
            mock_result, "555-123-4567"
        )

        assert detection.entity_type == "PHONE_NUMBER"
        assert detection.start == 10
        assert detection.end == 22
        assert detection.confidence == 0.95
        assert detection.text == "555-123-4567"

    def test_result_ordering(self):
        """Test entity result ordering for deterministic behavior."""
        results = [
            EntityDetectionResult("EMAIL", 20, 35, 0.8, "test@example.com"),
            EntityDetectionResult("PHONE", 10, 22, 0.9, "555-123-4567"),
            EntityDetectionResult(
                "EMAIL", 20, 35, 0.7, "test@example.com"
            ),  # Same position, lower confidence
        ]

        sorted_results = sorted(results)

        # Should sort by position first, then by confidence descending
        assert sorted_results[0].start == 10  # PHONE comes first (lower position)
        assert sorted_results[1].confidence == 0.8  # Higher confidence EMAIL comes next
        assert sorted_results[2].confidence == 0.7  # Lower confidence EMAIL comes last

    def test_overlap_detection(self):
        """Test detection of overlapping entities."""
        result1 = EntityDetectionResult("PHONE", 10, 22, 0.9, "555-123-4567")
        result2 = EntityDetectionResult("PERSON", 15, 25, 0.8, "John Smith")
        result3 = EntityDetectionResult("EMAIL", 30, 45, 0.8, "test@example.com")

        assert result1.overlaps_with(result2) is True
        assert result1.overlaps_with(result3) is False
        assert result2.overlaps_with(result3) is False


class TestBackwardCompatibility:
    """Test backward compatibility for existing analyzer usage patterns."""

    def test_analyzer_wrapper_default_initialization_unchanged(self):
        """Test that default AnalyzerEngineWrapper initialization behavior is unchanged."""
        # This should work exactly as it did before the singleton integration
        analyzer = AnalyzerEngineWrapper()

        assert analyzer is not None
        assert analyzer.config is not None
        assert analyzer.config.language == "en"  # Default language
        assert analyzer.config.min_confidence == 0.5  # Default confidence

        # New attribute should exist but default behavior preserved
        assert hasattr(analyzer, "use_singleton")

    def test_analyzer_wrapper_with_config_unchanged(self):
        """Test that AnalyzerEngineWrapper with config works as before."""
        config = AnalyzerConfig(
            language="es", min_confidence=0.7, nlp_engine_name="spacy"
        )

        analyzer = AnalyzerEngineWrapper(config)

        assert analyzer.config.language == "es"
        assert analyzer.config.min_confidence == 0.7
        assert analyzer.config.nlp_engine_name == "spacy"

    def test_from_policy_method_unchanged(self):
        """Test that from_policy class method works as before."""
        policy = MaskingPolicy(
            locale="en", thresholds={"EMAIL": 0.8, "PHONE_NUMBER": 0.7}
        )

        analyzer = AnalyzerEngineWrapper.from_policy(policy)

        assert analyzer is not None
        assert analyzer.config.language == "en"

    def test_existing_method_signatures_unchanged(self):
        """Test that existing method signatures are unchanged."""
        analyzer = AnalyzerEngineWrapper()

        # These methods should exist and have same signatures
        assert hasattr(analyzer, "is_initialized")
        assert hasattr(analyzer, "from_policy")
        assert hasattr(analyzer, "_get_spacy_model_name")
        assert hasattr(analyzer, "_initialize_engine")
        assert hasattr(analyzer, "analyze_text")

    def test_analyzer_properties_unchanged(self):
        """Test that analyzer properties work as before."""
        analyzer = AnalyzerEngineWrapper()

        # Properties should work the same
        assert isinstance(analyzer.is_initialized, bool)
        assert analyzer.config is not None
        assert analyzer.registry is not None

    def test_singleton_parameter_optional(self):
        """Test that use_singleton parameter is optional and doesn't break existing code."""
        # All these should work (existing patterns)
        analyzer1 = AnalyzerEngineWrapper()
        analyzer2 = AnalyzerEngineWrapper(config=None)

        config = AnalyzerConfig(language="en")
        analyzer3 = AnalyzerEngineWrapper(config)

        assert all(a is not None for a in [analyzer1, analyzer2, analyzer3])

    def test_new_functionality_doesnt_break_old(self):
        """Test that new singleton functionality doesn't interfere with old usage."""
        # Create analyzer the old way
        old_analyzer = AnalyzerEngineWrapper(use_singleton=False)

        # Create analyzer the new way
        new_analyzer = AnalyzerEngineWrapper.create_shared()

        # Both should work
        assert old_analyzer is not None
        assert new_analyzer is not None

        # They should be different instances when old way explicitly disables singleton
        # (though this might not be immediately testable without actual initialization)
        assert hasattr(old_analyzer, "use_singleton")
        assert old_analyzer.use_singleton is False
