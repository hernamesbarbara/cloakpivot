"""Tests for environment variable configuration system."""

import os
from unittest.mock import patch

from cloakpivot.core.analyzer import AnalyzerEngineWrapper
from cloakpivot.core.config import PerformanceConfig, reset_performance_config
from cloakpivot.core.model_info import (
    MODEL_CHARACTERISTICS,
    get_model_recommendations,
    validate_model_availability,
)
from cloakpivot.loaders import get_presidio_analyzer


class TestPerformanceConfig:
    """Test PerformanceConfig dataclass and environment variable loading."""

    def test_default_configuration(self):
        """Test default configuration values without environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            config = PerformanceConfig.from_environment()

            assert config.model_size == "small"
            assert config.use_singleton_analyzers is True
            assert config.analyzer_cache_size == 8
            assert config.enable_parallel_processing is True
            assert config.max_worker_threads is None
            assert config.enable_memory_optimization is True
            assert config.gc_frequency == 100

    def test_model_size_environment_variable(self):
        """Test MODEL_SIZE affects model selection."""
        test_cases = [
            ("small", "small"),
            ("medium", "medium"),
            ("large", "large"),
            ("LARGE", "large"),  # Test case insensitive
            ("invalid", "small"),  # Test fallback to default
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"MODEL_SIZE": env_value}):
                config = PerformanceConfig.from_environment()
                assert config.model_size == expected

    def test_singleton_disable(self):
        """Test CLOAKPIVOT_USE_SINGLETON=false behavior."""
        test_cases = [
            ("true", True),
            ("false", False),
            ("TRUE", True),
            ("FALSE", False),
            ("1", True),  # Invalid values use default (True)
            ("invalid", True),  # Invalid values use default (True)
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"CLOAKPIVOT_USE_SINGLETON": env_value}):
                config = PerformanceConfig.from_environment()
                assert config.use_singleton_analyzers is expected

    def test_cache_size_configuration(self):
        """Test ANALYZER_CACHE_SIZE affects cache behavior."""
        test_cases = [
            ("16", 16),
            ("32", 32),
            ("1", 1),
            ("invalid", 8),  # Should fallback to default
            ("-1", 8),  # Should fallback to default
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"ANALYZER_CACHE_SIZE": env_value}):
                config = PerformanceConfig.from_environment()
                assert config.analyzer_cache_size == expected

    def test_parallel_processing_configuration(self):
        """Test ENABLE_PARALLEL configuration."""
        test_cases = [
            ("true", True),
            ("false", False),
            ("TRUE", True),
            ("FALSE", False),
            ("invalid", True),  # Should default to True
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"ENABLE_PARALLEL": env_value}):
                reset_performance_config()  # Force reload of config
                config = PerformanceConfig.from_environment()
                assert config.enable_parallel_processing is expected

    def test_max_workers_configuration(self):
        """Test MAX_WORKERS configuration."""
        test_cases = [
            ("4", 4),
            ("8", 8),
            ("1", 1),
            (None, None),  # Not set
            ("invalid", None),  # Should fallback to None
            ("0", None),  # Should fallback to None
            ("-1", None),  # Should fallback to None
        ]

        for env_value, expected in test_cases:
            env_dict = {"MAX_WORKERS": env_value} if env_value is not None else {}
            with patch.dict(os.environ, env_dict, clear=True):
                config = PerformanceConfig.from_environment()
                assert config.max_worker_threads == expected

    def test_memory_optimization_configuration(self):
        """Test MEMORY_OPTIMIZATION configuration."""
        test_cases = [
            ("true", True),
            ("false", False),
            ("TRUE", True),
            ("FALSE", False),
            ("invalid", True),  # Should default to True
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"MEMORY_OPTIMIZATION": env_value}):
                reset_performance_config()  # Force reload of config
                config = PerformanceConfig.from_environment()
                assert config.enable_memory_optimization is expected

    def test_gc_frequency_configuration(self):
        """Test GC_FREQUENCY configuration."""
        test_cases = [
            ("50", 50),
            ("200", 200),
            ("1", 1),
            ("invalid", 100),  # Should fallback to default
            ("0", 100),  # Should fallback to default
            ("-1", 100),  # Should fallback to default
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"GC_FREQUENCY": env_value}):
                config = PerformanceConfig.from_environment()
                assert config.gc_frequency == expected


class TestModelInfo:
    """Test model characteristics and validation functions."""

    def test_model_characteristics_structure(self):
        """Test MODEL_CHARACTERISTICS has expected structure."""
        expected_sizes = {"small", "medium", "large"}
        expected_fields = {"memory_mb", "load_time_ms", "accuracy_score", "description"}

        assert set(MODEL_CHARACTERISTICS.keys()) == expected_sizes

        for _size, characteristics in MODEL_CHARACTERISTICS.items():
            assert set(characteristics.keys()) == expected_fields
            assert isinstance(characteristics["memory_mb"], int)
            assert isinstance(characteristics["load_time_ms"], int)
            assert isinstance(characteristics["accuracy_score"], float)
            assert isinstance(characteristics["description"], str)

            # Validate ranges
            assert 0 < characteristics["memory_mb"] <= 500
            assert 0 < characteristics["load_time_ms"] <= 10000
            assert 0.0 <= characteristics["accuracy_score"] <= 1.0
            assert len(characteristics["description"]) > 10

    def test_model_characteristics_ordering(self):
        """Test that model sizes have expected performance characteristics."""
        small = MODEL_CHARACTERISTICS["small"]
        medium = MODEL_CHARACTERISTICS["medium"]
        large = MODEL_CHARACTERISTICS["large"]

        # Memory usage should increase
        assert small["memory_mb"] < medium["memory_mb"] < large["memory_mb"]

        # Load time should increase
        assert small["load_time_ms"] < medium["load_time_ms"] < large["load_time_ms"]

        # Accuracy should increase
        assert (
            small["accuracy_score"] < medium["accuracy_score"] < large["accuracy_score"]
        )

    def test_validate_model_availability(self):
        """Test model availability validation."""
        # Test valid combinations
        assert validate_model_availability("en", "small") is True
        assert validate_model_availability("en", "medium") is True
        assert validate_model_availability("en", "large") is True
        assert validate_model_availability("es", "small") is True

        # Test invalid combinations
        assert validate_model_availability("invalid_lang", "small") is False
        assert validate_model_availability("en", "invalid_size") is False
        assert validate_model_availability("", "small") is False
        assert validate_model_availability("en", "") is False

    def test_get_model_recommendations_memory_limit(self):
        """Test model recommendations based on memory constraints."""
        # Low memory should recommend small
        recommendations = get_model_recommendations(memory_limit_mb=20)
        assert "small" in recommendations["recommended_size"]

        # Medium memory should allow medium
        recommendations = get_model_recommendations(memory_limit_mb=75)
        assert "medium" in recommendations["recommended_size"]

        # High memory should allow large
        recommendations = get_model_recommendations(memory_limit_mb=200)
        assert "large" in recommendations["recommended_size"]

    def test_get_model_recommendations_speed_priority(self):
        """Test model recommendations with speed priority."""
        recommendations = get_model_recommendations(speed_priority=True)
        assert "small" in recommendations["recommended_size"]
        assert "fast" in recommendations["reason"].lower()

        recommendations = get_model_recommendations(speed_priority=False)
        # Should not prioritize small when speed is not priority
        assert len(recommendations["alternatives"]) > 0


class TestAnalyzerEnhancement:
    """Test enhancements to AnalyzerEngineWrapper for environment variable support."""

    def test_spacy_model_name_with_model_size_small(self):
        """Test _get_spacy_model_name with MODEL_SIZE=small."""
        with patch.dict(os.environ, {"MODEL_SIZE": "small"}):
            reset_performance_config()  # Force reload of config
            wrapper = AnalyzerEngineWrapper()

            # Test English
            assert wrapper._get_spacy_model_name("en") == "en_core_web_sm"

            # Test other languages
            assert wrapper._get_spacy_model_name("es") == "es_core_news_sm"
            assert wrapper._get_spacy_model_name("fr") == "fr_core_news_sm"
            assert wrapper._get_spacy_model_name("de") == "de_core_news_sm"

    def test_spacy_model_name_with_model_size_medium(self):
        """Test _get_spacy_model_name with MODEL_SIZE=medium."""
        with patch.dict(os.environ, {"MODEL_SIZE": "medium"}):
            reset_performance_config()  # Force reload of config
            wrapper = AnalyzerEngineWrapper()

            # Test English
            assert wrapper._get_spacy_model_name("en") == "en_core_web_md"

            # Test other languages
            assert wrapper._get_spacy_model_name("es") == "es_core_news_md"
            assert wrapper._get_spacy_model_name("fr") == "fr_core_news_md"
            assert wrapper._get_spacy_model_name("de") == "de_core_news_md"

    def test_spacy_model_name_with_model_size_large(self):
        """Test _get_spacy_model_name with MODEL_SIZE=large."""
        with patch.dict(os.environ, {"MODEL_SIZE": "large"}):
            reset_performance_config()  # Force reload of config
            wrapper = AnalyzerEngineWrapper()

            # Test English
            assert wrapper._get_spacy_model_name("en") == "en_core_web_lg"

            # Test other languages
            assert wrapper._get_spacy_model_name("es") == "es_core_news_lg"
            assert wrapper._get_spacy_model_name("fr") == "fr_core_news_lg"
            assert wrapper._get_spacy_model_name("de") == "de_core_news_lg"

    def test_spacy_model_name_fallback_invalid_size(self):
        """Test _get_spacy_model_name falls back to small for invalid MODEL_SIZE."""
        with patch.dict(os.environ, {"MODEL_SIZE": "invalid"}):
            reset_performance_config()  # Force reload of config
            wrapper = AnalyzerEngineWrapper()

            # Should fallback to small
            assert wrapper._get_spacy_model_name("en") == "en_core_web_sm"
            assert wrapper._get_spacy_model_name("es") == "es_core_news_sm"

    def test_spacy_model_name_unknown_language(self):
        """Test _get_spacy_model_name with unknown language."""
        with patch.dict(os.environ, {"MODEL_SIZE": "medium"}):
            reset_performance_config()  # Force reload of config
            wrapper = AnalyzerEngineWrapper()

            # Unknown language should use generic pattern
            assert wrapper._get_spacy_model_name("xx") == "xx_core_web_md"


class TestLoaderIntegration:
    """Test integration with loaders and singleton behavior."""

    def test_singleton_behavior_enabled(self):
        """Test singleton behavior when CLOAKPIVOT_USE_SINGLETON=true."""
        with patch.dict(os.environ, {"CLOAKPIVOT_USE_SINGLETON": "true"}):
            # Multiple calls should return the same cached instance
            analyzer1 = get_presidio_analyzer(language="en")
            analyzer2 = get_presidio_analyzer(language="en")

            # Should be the same object due to caching
            assert analyzer1 is analyzer2

    def test_singleton_behavior_disabled_via_config(self):
        """Test that PerformanceConfig can control singleton behavior."""
        with patch.dict(os.environ, {"CLOAKPIVOT_USE_SINGLETON": "false"}):
            config = PerformanceConfig.from_environment()

            # Create wrapper directly (bypassing singleton loader)
            wrapper = AnalyzerEngineWrapper(
                use_singleton=config.use_singleton_analyzers
            )

            assert wrapper.use_singleton is False

    def test_cache_size_affects_behavior(self):
        """Test that ANALYZER_CACHE_SIZE affects loader caching."""
        # This test verifies the cache size configuration is used

        with patch.dict(os.environ, {"ANALYZER_CACHE_SIZE": "16"}):
            reset_performance_config()  # Force reload of config
            config = PerformanceConfig.from_environment()
            assert config.analyzer_cache_size == 16

        with patch.dict(os.environ, {"ANALYZER_CACHE_SIZE": "4"}):
            reset_performance_config()  # Force reload of config
            config = PerformanceConfig.from_environment()
            assert config.analyzer_cache_size == 4


class TestConfigurationIntegration:
    """Test end-to-end configuration integration."""

    def test_development_profile(self):
        """Test development environment variable profile."""
        dev_env = {
            "MODEL_SIZE": "small",
            "ANALYZER_CACHE_SIZE": "4",
            "CLOAKPIVOT_USE_SINGLETON": "true",
        }

        with patch.dict(os.environ, dev_env):
            config = PerformanceConfig.from_environment()

            assert config.model_size == "small"
            assert config.analyzer_cache_size == 4
            assert config.use_singleton_analyzers is True

    def test_production_profile(self):
        """Test production environment variable profile."""
        prod_env = {
            "MODEL_SIZE": "medium",
            "ANALYZER_CACHE_SIZE": "16",
            "CLOAKPIVOT_USE_SINGLETON": "true",
            "MAX_WORKERS": "4",
        }

        with patch.dict(os.environ, prod_env):
            config = PerformanceConfig.from_environment()

            assert config.model_size == "medium"
            assert config.analyzer_cache_size == 16
            assert config.use_singleton_analyzers is True
            assert config.max_worker_threads == 4

    def test_high_accuracy_profile(self):
        """Test high accuracy environment variable profile."""
        high_acc_env = {
            "MODEL_SIZE": "large",
            "ANALYZER_CACHE_SIZE": "32",
            "MAX_WORKERS": "4",
            "MEMORY_OPTIMIZATION": "true",
        }

        with patch.dict(os.environ, high_acc_env):
            config = PerformanceConfig.from_environment()

            assert config.model_size == "large"
            assert config.analyzer_cache_size == 32
            assert config.max_worker_threads == 4
            assert config.enable_memory_optimization is True

    def test_config_validation_prevents_issues(self):
        """Test that configuration validation prevents common issues."""
        # Test that invalid values fallback to safe defaults
        invalid_env = {
            "MODEL_SIZE": "huge",  # Invalid, should fallback to small
            "ANALYZER_CACHE_SIZE": "-1",  # Invalid, should fallback to 8
            "MAX_WORKERS": "abc",  # Invalid, should fallback to None
            "GC_FREQUENCY": "0",  # Invalid, should fallback to 100
        }

        with patch.dict(os.environ, invalid_env):
            config = PerformanceConfig.from_environment()

            # Should use safe defaults for invalid values
            assert config.model_size == "small"
            assert config.analyzer_cache_size == 8
            assert config.max_worker_threads is None
            assert config.gc_frequency == 100
