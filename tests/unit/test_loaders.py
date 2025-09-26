"""Comprehensive unit tests for cloakpivot.loaders module.

This test module provides coverage of the loaders system validation functions and exceptions.
"""

from unittest.mock import Mock, patch

import pytest

from cloakpivot.loaders import (
    ConfigurationError,
    InitializationError,
    LoaderError,
    _validate_confidence,
    _validate_language,
    _validate_nlp_engine,
    clear_all_caches,
    get_detection_pipeline,
    get_document_processor,
    get_presidio_analyzer,
)


class TestExceptions:
    """Test the exception classes."""

    def test_loader_error(self):
        """Test LoaderError exception."""
        error = LoaderError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        error = ConfigurationError("Invalid config")
        assert str(error) == "Invalid config"
        assert isinstance(error, LoaderError)

    def test_initialization_error(self):
        """Test InitializationError exception."""
        error = InitializationError("Failed to initialize")
        assert str(error) == "Failed to initialize"
        assert isinstance(error, LoaderError)


class TestValidationFunctions:
    """Test validation functions."""

    def test_validate_language_valid(self):
        """Test _validate_language with valid inputs."""
        _validate_language("en")
        _validate_language("es")
        _validate_language("fr")
        _validate_language("pt-BR")
        _validate_language("zh-CN")
        assert True

    def test_validate_language_invalid_type(self):
        """Test _validate_language with invalid type."""
        with pytest.raises(ConfigurationError) as exc_info:
            _validate_language(123)
        assert "Language must be a string" in str(exc_info.value)

    def test_validate_language_empty(self):
        """Test _validate_language with empty string."""
        with pytest.raises(ConfigurationError) as exc_info:
            _validate_language("")
        assert "Language cannot be empty" in str(exc_info.value)

    def test_validate_language_invalid_length(self):
        """Test _validate_language with invalid length."""
        with pytest.raises(ConfigurationError) as exc_info:
            _validate_language("e")
        assert "should be 2-5 characters" in str(exc_info.value)

        with pytest.raises(ConfigurationError) as exc_info:
            _validate_language("en-US-extra")
        assert "should be 2-5 characters" in str(exc_info.value)

    def test_validate_confidence_valid(self):
        """Test _validate_confidence with valid inputs."""
        _validate_confidence(0.0)
        _validate_confidence(0.5)
        _validate_confidence(0.85)
        _validate_confidence(1.0)
        assert True

    def test_validate_confidence_invalid_type(self):
        """Test _validate_confidence with invalid type."""
        with pytest.raises(ConfigurationError) as exc_info:
            _validate_confidence("0.5")
        assert "Confidence must be a number" in str(exc_info.value)

    def test_validate_confidence_out_of_range(self):
        """Test _validate_confidence with out of range values."""
        with pytest.raises(ConfigurationError) as exc_info:
            _validate_confidence(-0.1)
        assert "Confidence must be between 0.0 and 1.0" in str(exc_info.value)

        with pytest.raises(ConfigurationError) as exc_info:
            _validate_confidence(1.1)
        assert "Confidence must be between 0.0 and 1.0" in str(exc_info.value)

    def test_validate_nlp_engine_valid(self):
        """Test _validate_nlp_engine with valid inputs."""
        _validate_nlp_engine("spacy")
        _validate_nlp_engine("transformers")
        assert True

    def test_validate_nlp_engine_invalid_type(self):
        """Test _validate_nlp_engine with invalid type."""
        with pytest.raises(ConfigurationError) as exc_info:
            _validate_nlp_engine(123)
        assert "NLP engine must be a string" in str(exc_info.value)

    def test_validate_nlp_engine_invalid_name(self):
        """Test _validate_nlp_engine with invalid engine name."""
        with pytest.raises(ConfigurationError) as exc_info:
            _validate_nlp_engine("invalid")
        assert "Invalid NLP engine" in str(exc_info.value)


class TestLoaderFunctions:
    """Test loader functions."""

    @patch("cloakpivot.loaders.AnalyzerEngineWrapper")
    def test_get_presidio_analyzer_default(self, mock_wrapper):
        """Test get_presidio_analyzer with default parameters."""
        mock_instance = Mock()
        mock_wrapper.return_value = mock_instance

        result = get_presidio_analyzer()
        assert result is not None

    @patch("cloakpivot.loaders.AnalyzerEngineWrapper")
    def test_get_presidio_analyzer_with_params(self, mock_wrapper):
        """Test get_presidio_analyzer with custom parameters."""
        mock_instance = Mock()
        mock_wrapper.return_value = mock_instance

        result = get_presidio_analyzer(language="es", min_confidence=0.9, nlp_engine_name="spacy")
        assert result is not None

    @patch("cloakpivot.loaders.DocumentProcessor")
    def test_get_document_processor(self, mock_processor):
        """Test get_document_processor."""
        mock_instance = Mock()
        mock_processor.return_value = mock_instance

        result = get_document_processor()
        assert result is not None

    @patch("cloakpivot.loaders.EntityDetectionPipeline")
    def test_get_detection_pipeline(self, mock_pipeline):
        """Test get_detection_pipeline."""
        mock_instance = Mock()
        mock_pipeline.return_value = mock_instance

        result = get_detection_pipeline()
        assert result is not None

    def test_clear_all_caches(self):
        """Test clear_all_caches function."""
        # Just ensure it doesn't raise
        clear_all_caches()
        assert True
