"""Tests for Presidio AnalyzerEngine integration."""

import pytest
from typing import List, Optional
from unittest.mock import Mock, patch

from cloakpivot.core.analyzer import (
    AnalyzerEngineWrapper,
    RecognizerRegistry,
    AnalyzerConfig,
    EntityDetectionResult,
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
        with patch('presidio_analyzer.AnalyzerEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.analyze.return_value = []  # Return empty list for analysis
            mock_engine.return_value = mock_instance
            
            result = analyzer.analyze_text("test text")
            
            assert analyzer.is_initialized is True
            mock_engine.assert_called_once()
            assert result == []  # Should return empty list from mock
    
    def test_configuration_from_policy(self):
        """Test analyzer configuration from MaskingPolicy."""
        policy = MaskingPolicy(
            locale="es",
            thresholds={"PHONE_NUMBER": 0.8}
        )
        
        analyzer = AnalyzerEngineWrapper.from_policy(policy)
        
        assert analyzer.config.language == "es"
        assert analyzer.config.min_confidence == 0.8  # Minimum threshold from policy thresholds
    
    def test_custom_recognizer_configuration(self):
        """Test custom recognizer configuration."""
        config = AnalyzerConfig(
            language="en",
            enabled_recognizers=["PHONE_NUMBER", "EMAIL_ADDRESS"],
            disabled_recognizers=["PERSON"],
            min_confidence=0.7
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
            "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", 
            "US_SSN", "PERSON", "URL", "IP_ADDRESS"
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
        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            AnalyzerConfig(min_confidence=1.5)
    
    def test_from_policy_conversion(self):
        """Test creating config from MaskingPolicy."""
        policy = MaskingPolicy(
            locale="fr",
            thresholds={"PHONE_NUMBER": 0.9, "EMAIL_ADDRESS": 0.7}
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
        
        detection = EntityDetectionResult.from_presidio_result(mock_result, "555-123-4567")
        
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
            EntityDetectionResult("EMAIL", 20, 35, 0.7, "test@example.com"),  # Same position, lower confidence
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