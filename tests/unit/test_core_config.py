"""Comprehensive unit tests for cloakpivot.core.utilities.config module.

This test module provides full coverage of the performance configuration
system, including environment variable loading, validation, and global
configuration management.
"""

import os
from unittest.mock import patch

import pytest

from cloakpivot.core.utilities.config import (
    PerformanceConfig,
    get_performance_config,
    performance_config,
    reset_performance_config,
)


class TestPerformanceConfig:
    """Test the PerformanceConfig dataclass."""

    def test_default_initialization(self):
        """Test PerformanceConfig with default values."""
        config = PerformanceConfig()
        assert config.model_size == "small"
        assert config.use_singleton_analyzers is True
        assert config.analyzer_cache_size == 8
        assert config.enable_parallel_processing is True
        assert config.max_worker_threads is None
        assert config.enable_memory_optimization is True
        assert config.gc_frequency == 100

    def test_custom_initialization(self):
        """Test PerformanceConfig with custom values."""
        config = PerformanceConfig(
            model_size="large",
            use_singleton_analyzers=False,
            analyzer_cache_size=16,
            enable_parallel_processing=False,
            max_worker_threads=4,
            enable_memory_optimization=False,
            gc_frequency=50,
        )
        assert config.model_size == "large"
        assert config.use_singleton_analyzers is False
        assert config.analyzer_cache_size == 16
        assert config.enable_parallel_processing is False
        assert config.max_worker_threads == 4
        assert config.enable_memory_optimization is False
        assert config.gc_frequency == 50

    def test_validate_model_size_valid(self):
        """Test model size validation with valid sizes."""
        for size in ["small", "medium", "large", "SMALL", "Medium", "LARGE"]:
            config = PerformanceConfig(model_size=size)
            assert config.model_size in {"small", "medium", "large"}

    def test_validate_model_size_invalid(self):
        """Test model size validation with invalid size."""
        with patch("cloakpivot.core.utilities.config.logger") as mock_logger:
            config = PerformanceConfig(model_size="extra-large")
            assert config.model_size == "small"
            mock_logger.warning.assert_called_once()
            assert "Invalid model_size" in str(mock_logger.warning.call_args)

    def test_validate_cache_size_valid(self):
        """Test cache size validation with valid values."""
        config = PerformanceConfig(analyzer_cache_size=10)
        assert config.analyzer_cache_size == 10

    def test_validate_cache_size_invalid_zero(self):
        """Test cache size validation with zero."""
        with patch("cloakpivot.core.utilities.config.logger") as mock_logger:
            config = PerformanceConfig(analyzer_cache_size=0)
            assert config.analyzer_cache_size == 8
            mock_logger.warning.assert_called_once()

    def test_validate_cache_size_invalid_negative(self):
        """Test cache size validation with negative value."""
        with patch("cloakpivot.core.utilities.config.logger") as mock_logger:
            config = PerformanceConfig(analyzer_cache_size=-5)
            assert config.analyzer_cache_size == 8
            mock_logger.warning.assert_called_once()

    def test_validate_worker_threads_valid(self):
        """Test worker threads validation with valid values."""
        # Test with None
        config1 = PerformanceConfig(max_worker_threads=None)
        assert config1.max_worker_threads is None

        # Test with positive integer
        config2 = PerformanceConfig(max_worker_threads=8)
        assert config2.max_worker_threads == 8

    def test_validate_worker_threads_invalid(self):
        """Test worker threads validation with invalid values."""
        with patch("cloakpivot.core.utilities.config.logger") as mock_logger:
            config = PerformanceConfig(max_worker_threads=0)
            assert config.max_worker_threads is None
            mock_logger.warning.assert_called_once()
            assert "max_worker_threads must be positive" in str(mock_logger.warning.call_args)

        with patch("cloakpivot.core.utilities.config.logger") as mock_logger:
            config = PerformanceConfig(max_worker_threads=-2)
            assert config.max_worker_threads is None
            mock_logger.warning.assert_called_once()

    def test_validate_gc_frequency_valid(self):
        """Test GC frequency validation with valid values."""
        config = PerformanceConfig(gc_frequency=50)
        assert config.gc_frequency == 50

    def test_validate_gc_frequency_invalid(self):
        """Test GC frequency validation with invalid values."""
        with patch("cloakpivot.core.utilities.config.logger") as mock_logger:
            config = PerformanceConfig(gc_frequency=0)
            assert config.gc_frequency == 100
            mock_logger.warning.assert_called_once()

        with patch("cloakpivot.core.utilities.config.logger") as mock_logger:
            config = PerformanceConfig(gc_frequency=-10)
            assert config.gc_frequency == 100
            mock_logger.warning.assert_called_once()

    def test_post_init_logging(self):
        """Test that post_init logs debug information."""
        with patch("cloakpivot.core.utilities.config.logger") as mock_logger:
            PerformanceConfig()
            mock_logger.debug.assert_called_once()
            debug_msg = str(mock_logger.debug.call_args)
            assert "PerformanceConfig initialized" in debug_msg

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = PerformanceConfig(
            model_size="medium",
            analyzer_cache_size=12,
            max_worker_threads=6,
        )
        result = config.to_dict()

        assert result == {
            "model_size": "medium",
            "use_singleton_analyzers": True,
            "analyzer_cache_size": 12,
            "enable_parallel_processing": True,
            "max_worker_threads": 6,
            "enable_memory_optimization": True,
            "gc_frequency": 100,
        }

    def test_repr(self):
        """Test string representation."""
        config = PerformanceConfig(model_size="large", analyzer_cache_size=16)
        repr_str = repr(config)
        assert "PerformanceConfig" in repr_str
        assert "model_size='large'" in repr_str
        assert "use_singleton=True" in repr_str
        assert "cache_size=16" in repr_str

    def test_get_model_characteristics(self):
        """Test getting model characteristics."""
        with patch("cloakpivot.core.types.model_info.MODEL_CHARACTERISTICS") as mock_chars:
            mock_chars.__getitem__.return_value = {"memory": "100MB", "accuracy": "high"}
            mock_chars.get.return_value = {"memory": "100MB", "accuracy": "high"}

            config = PerformanceConfig(model_size="medium")
            chars = config.get_model_characteristics()

            mock_chars.get.assert_called_once_with("medium", mock_chars["small"])
            assert chars == {"memory": "100MB", "accuracy": "high"}


class TestEnvironmentLoading:
    """Test loading configuration from environment variables."""

    def test_from_environment_all_defaults(self):
        """Test from_environment with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = PerformanceConfig.from_environment()
            assert config.model_size == "small"
            assert config.use_singleton_analyzers is True
            assert config.analyzer_cache_size == 8
            assert config.enable_parallel_processing is True
            assert config.max_worker_threads is None
            assert config.enable_memory_optimization is True
            assert config.gc_frequency == 100

    def test_from_environment_all_custom(self):
        """Test from_environment with all environment variables set."""
        env_vars = {
            "MODEL_SIZE": "large",
            "CLOAKPIVOT_USE_SINGLETON": "false",
            "ANALYZER_CACHE_SIZE": "16",
            "ENABLE_PARALLEL": "false",
            "MAX_WORKERS": "4",
            "MEMORY_OPTIMIZATION": "false",
            "GC_FREQUENCY": "50",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = PerformanceConfig.from_environment()
            assert config.model_size == "large"
            assert config.use_singleton_analyzers is False
            assert config.analyzer_cache_size == 16
            assert config.enable_parallel_processing is False
            assert config.max_worker_threads == 4
            assert config.enable_memory_optimization is False
            assert config.gc_frequency == 50

    def test_from_environment_mixed_case(self):
        """Test from_environment with mixed case values."""
        env_vars = {
            "MODEL_SIZE": "MEDIUM",
            "CLOAKPIVOT_USE_SINGLETON": "True",
            "ENABLE_PARALLEL": "FALSE",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = PerformanceConfig.from_environment()
            assert config.model_size == "medium"
            assert config.use_singleton_analyzers is True
            assert config.enable_parallel_processing is False

    def test_from_environment_invalid_values(self):
        """Test from_environment with invalid values falls back to defaults."""
        env_vars = {
            "ANALYZER_CACHE_SIZE": "not_a_number",
            "MAX_WORKERS": "-5",
            "GC_FREQUENCY": "0",
        }
        with (
            patch.dict(os.environ, env_vars, clear=True),
            patch("cloakpivot.core.utilities.config.logger") as mock_logger,
        ):
            config = PerformanceConfig.from_environment()
            assert config.analyzer_cache_size == 8
            assert config.max_worker_threads is None
            assert config.gc_frequency == 100
            # Check that warnings were logged
            assert mock_logger.warning.call_count >= 2

    def test_from_environment_exception_handling(self):
        """Test from_environment handles exceptions gracefully."""
        with patch(
            "cloakpivot.core.utilities.config.PerformanceConfig._get_env_string"
        ) as mock_get:
            mock_get.side_effect = Exception("Test error")
            with patch("cloakpivot.core.utilities.config.logger") as mock_logger:
                config = PerformanceConfig.from_environment()
                # Should return default config
                assert config.model_size == "small"
                mock_logger.error.assert_called_once()
                assert "Error loading configuration" in str(mock_logger.error.call_args)

    def test_get_env_string(self):
        """Test _get_env_string static method."""
        # Test with value present
        with patch.dict(os.environ, {"TEST_KEY": "  test_value  "}, clear=True):
            result = PerformanceConfig._get_env_string("TEST_KEY", "default")
            assert result == "test_value"

        # Test with value absent
        with patch.dict(os.environ, {}, clear=True):
            result = PerformanceConfig._get_env_string("TEST_KEY", "default")
            assert result == "default"

    def test_get_env_bool(self):
        """Test _get_env_bool static method."""
        # Test "true" values
        for value in ["true", "True", "TRUE", "  true  "]:
            with patch.dict(os.environ, {"TEST_KEY": value}, clear=True):
                result = PerformanceConfig._get_env_bool("TEST_KEY", False)
                assert result is True

        # Test "false" values
        for value in ["false", "False", "FALSE", "  false  "]:
            with patch.dict(os.environ, {"TEST_KEY": value}, clear=True):
                result = PerformanceConfig._get_env_bool("TEST_KEY", True)
                assert result is False

        # Test invalid values return default
        for value in ["yes", "1", "on", "invalid"]:
            with patch.dict(os.environ, {"TEST_KEY": value}, clear=True):
                result = PerformanceConfig._get_env_bool("TEST_KEY", True)
                assert result is True  # Returns default

        # Test absent value
        with patch.dict(os.environ, {}, clear=True):
            result = PerformanceConfig._get_env_bool("TEST_KEY", False)
            assert result is False

    def test_get_env_int(self):
        """Test _get_env_int static method."""
        # Test valid positive integer
        with patch.dict(os.environ, {"TEST_KEY": "42"}, clear=True):
            result = PerformanceConfig._get_env_int("TEST_KEY", 10)
            assert result == 42

        # Test valid integer with whitespace
        with patch.dict(os.environ, {"TEST_KEY": "  25  "}, clear=True):
            result = PerformanceConfig._get_env_int("TEST_KEY", 10)
            assert result == 25

        # Test zero (not allowed by default)
        with (
            patch.dict(os.environ, {"TEST_KEY": "0"}, clear=True),
            patch("cloakpivot.core.utilities.config.logger") as mock_logger,
        ):
            result = PerformanceConfig._get_env_int("TEST_KEY", 10)
            assert result == 10  # Returns default
            mock_logger.warning.assert_called_once()

        # Test negative (not allowed by default)
        with (
            patch.dict(os.environ, {"TEST_KEY": "-5"}, clear=True),
            patch("cloakpivot.core.utilities.config.logger") as mock_logger,
        ):
            result = PerformanceConfig._get_env_int("TEST_KEY", 10)
            assert result == 10  # Returns default
            mock_logger.warning.assert_called_once()

        # Test allow_none parameter
        with patch.dict(os.environ, {"TEST_KEY": "-5"}, clear=True):
            result = PerformanceConfig._get_env_int("TEST_KEY", None, allow_none=True)
            assert result == -5  # Negative allowed with allow_none

        # Test invalid integer
        with (
            patch.dict(os.environ, {"TEST_KEY": "not_a_number"}, clear=True),
            patch("cloakpivot.core.utilities.config.logger") as mock_logger,
        ):
            result = PerformanceConfig._get_env_int("TEST_KEY", 15)
            assert result == 15  # Returns default
            mock_logger.warning.assert_called_once()

        # Test absent value
        with patch.dict(os.environ, {}, clear=True):
            result = PerformanceConfig._get_env_int("TEST_KEY", None, allow_none=True)
            assert result is None


class TestGlobalConfiguration:
    """Test global configuration management functions."""

    def setup_method(self):
        """Reset global configuration before each test."""
        reset_performance_config()

    def test_get_performance_config_creates_singleton(self):
        """Test that get_performance_config creates and caches config."""
        with patch.dict(os.environ, {"MODEL_SIZE": "medium"}, clear=True):
            config1 = get_performance_config()
            config2 = get_performance_config()

            # Should be the same instance
            assert config1 is config2
            assert config1.model_size == "medium"

    def test_reset_performance_config(self):
        """Test that reset_performance_config clears the cache."""
        with patch.dict(os.environ, {"MODEL_SIZE": "large"}, clear=True):
            config1 = get_performance_config()
            assert config1.model_size == "large"

            # Reset and change environment
            reset_performance_config()

            with patch.dict(os.environ, {"MODEL_SIZE": "small"}, clear=True):
                config2 = get_performance_config()
                assert config2.model_size == "small"
                assert config1 is not config2

    def test_config_proxy_attribute_access(self):
        """Test _ConfigProxy provides attribute access to config."""
        with patch.dict(os.environ, {"MODEL_SIZE": "medium"}, clear=True):
            # Access through proxy
            assert performance_config.model_size == "medium"
            assert performance_config.analyzer_cache_size == 8
            assert performance_config.use_singleton_analyzers is True

    def test_config_proxy_repr(self):
        """Test _ConfigProxy repr method."""
        with patch.dict(os.environ, {"MODEL_SIZE": "large"}, clear=True):
            repr_str = repr(performance_config)
            assert "PerformanceConfig" in repr_str
            assert "model_size='large'" in repr_str

    def test_config_proxy_method_access(self):
        """Test _ConfigProxy provides method access to config."""
        with patch.dict(os.environ, {"MODEL_SIZE": "small"}, clear=True):
            # Access method through proxy
            config_dict = performance_config.to_dict()
            assert config_dict["model_size"] == "small"

    def test_config_proxy_invalid_attribute(self):
        """Test _ConfigProxy raises AttributeError for invalid attributes."""
        with pytest.raises(AttributeError):
            _ = performance_config.non_existent_attribute


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Reset global configuration before each test."""
        reset_performance_config()

    def test_empty_string_environment_values(self):
        """Test handling of empty string environment values."""
        env_vars = {
            "MODEL_SIZE": "",
            "ANALYZER_CACHE_SIZE": "",
            "CLOAKPIVOT_USE_SINGLETON": "",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = PerformanceConfig.from_environment()
            # Empty strings should use defaults
            assert config.model_size == "small"  # Empty string becomes "small" after validation
            assert config.analyzer_cache_size == 8
            assert config.use_singleton_analyzers is True

    def test_whitespace_only_environment_values(self):
        """Test handling of whitespace-only environment values."""
        env_vars = {
            "MODEL_SIZE": "   ",
            "ANALYZER_CACHE_SIZE": "  \t  ",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = PerformanceConfig.from_environment()
            assert config.model_size == "small"  # Whitespace becomes empty after strip
            assert config.analyzer_cache_size == 8

    def test_extremely_large_cache_size(self):
        """Test handling of extremely large cache size."""
        config = PerformanceConfig(analyzer_cache_size=999999999)
        assert config.analyzer_cache_size == 999999999  # Should accept large values

    def test_extremely_large_gc_frequency(self):
        """Test handling of extremely large GC frequency."""
        config = PerformanceConfig(gc_frequency=1000000)
        assert config.gc_frequency == 1000000  # Should accept large values

    def test_model_size_with_special_characters(self):
        """Test model size validation with special characters."""
        with patch("cloakpivot.core.utilities.config.logger") as mock_logger:
            config = PerformanceConfig(model_size="small-v2")
            assert config.model_size == "small"
            mock_logger.warning.assert_called_once()

    def test_concurrent_config_access(self):
        """Test that concurrent access to global config works correctly."""
        with patch.dict(os.environ, {"MODEL_SIZE": "medium"}, clear=True):
            # Simulate concurrent access
            configs = []
            for _ in range(5):
                configs.append(get_performance_config())

            # All should be the same instance
            assert all(c is configs[0] for c in configs)
            assert all(c.model_size == "medium" for c in configs)
