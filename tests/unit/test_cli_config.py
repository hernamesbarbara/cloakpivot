"""Comprehensive unit tests for cloakpivot.cli.config module.

This test module provides full coverage of the CLI configuration
management including PresidioConfig class and helper functions.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from cloakpivot.cli.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_BATCH_SIZE,
    MAX_BATCH_SIZE,
    MAX_CONFIDENCE_THRESHOLD,
    MIN_BATCH_SIZE,
    MIN_CONFIDENCE_THRESHOLD,
    PresidioConfig,
    create_masking_engine,
    get_config_from_env,
    load_presidio_config,
    merge_configs,
)


class TestPresidioConfig:
    """Test the PresidioConfig dataclass."""

    def test_default_initialization(self):
        """Test PresidioConfig with default values."""
        config = PresidioConfig()
        assert config.engine == "auto"
        assert config.fallback_on_error is True
        assert config.batch_processing is True
        assert config.connection_pooling is True
        assert config.max_batch_size == DEFAULT_MAX_BATCH_SIZE
        assert config.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
        assert config.operator_chaining is False
        assert config.operators == {}
        assert config.custom_recognizers == []

    def test_custom_initialization(self):
        """Test PresidioConfig with custom values."""
        operators = {"redact": {"char": "*"}}
        recognizers = ["custom1", "custom2"]

        config = PresidioConfig(
            engine="presidio",
            fallback_on_error=False,
            batch_processing=False,
            connection_pooling=False,
            max_batch_size=50,
            confidence_threshold=0.9,
            operator_chaining=True,
            operators=operators,
            custom_recognizers=recognizers,
        )

        assert config.engine == "presidio"
        assert config.fallback_on_error is False
        assert config.batch_processing is False
        assert config.connection_pooling is False
        assert config.max_batch_size == 50
        assert config.confidence_threshold == 0.9
        assert config.operator_chaining is True
        assert config.operators == operators
        assert config.custom_recognizers == recognizers

    def test_create_default(self):
        """Test create_default class method."""
        config = PresidioConfig.create_default()

        assert config.engine == "auto"
        assert config.fallback_on_error is True
        assert config.batch_processing is True
        assert config.connection_pooling is True
        assert config.max_batch_size == DEFAULT_MAX_BATCH_SIZE
        assert config.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
        assert config.operator_chaining is False

        # Check default operators
        assert "default_redact" in config.operators
        assert "encryption" in config.operators
        assert "hash" in config.operators
        assert config.operators["default_redact"]["redact_char"] == "*"
        assert config.operators["encryption"]["key_provider"] == "environment"
        assert config.operators["hash"]["algorithm"] == "sha256"

    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = {
                "engine": "presidio",
                "fallback_on_error": False,
                "batch_processing": True,
                "max_batch_size": 200,
                "confidence_threshold": 0.85,
                "operator_chaining": True,
                "operators": {"test": {"param": "value"}},
                "custom_recognizers": ["recognizer1"],
            }
            yaml.dump(yaml_content, f)
            config_path = Path(f.name)

        try:
            config = PresidioConfig.load_from_file(config_path)

            assert config.engine == "presidio"
            assert config.fallback_on_error is False
            assert config.batch_processing is True
            assert config.max_batch_size == 200
            assert config.confidence_threshold == 0.85
            assert config.operator_chaining is True
            assert config.operators == {"test": {"param": "value"}}
            assert config.custom_recognizers == ["recognizer1"]
        finally:
            config_path.unlink()

    def test_load_from_yaml_file_with_presidio_section(self):
        """Test loading configuration from YAML file with presidio section."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = {
                "presidio": {
                    "engine": "legacy",
                    "max_batch_size": 50,
                }
            }
            yaml.dump(yaml_content, f)
            config_path = Path(f.name)

        try:
            config = PresidioConfig.load_from_file(config_path)
            assert config.engine == "legacy"
            assert config.max_batch_size == 50
        finally:
            config_path.unlink()

    def test_load_from_json_file(self):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_content = {
                "engine": "auto",
                "fallback_on_error": True,
                "confidence_threshold": 0.7,
            }
            json.dump(json_content, f)
            config_path = Path(f.name)

        try:
            config = PresidioConfig.load_from_file(config_path)
            assert config.engine == "auto"
            assert config.fallback_on_error is True
            assert config.confidence_threshold == 0.7
        finally:
            config_path.unlink()

    def test_load_from_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            PresidioConfig.load_from_file(Path("nonexistent.yaml"))
        assert "Configuration file not found" in str(exc_info.value)

    def test_load_from_file_unsupported_format(self):
        """Test loading from unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                PresidioConfig.load_from_file(config_path)
            assert "Unsupported config format" in str(exc_info.value)
        finally:
            config_path.unlink()

    def test_load_from_file_invalid_engine(self):
        """Test loading configuration with invalid engine type."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"engine": "invalid"}, f)
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                PresidioConfig.load_from_file(config_path)
            assert "Invalid engine type" in str(exc_info.value)
        finally:
            config_path.unlink()

    def test_load_from_file_invalid_boolean_fields(self):
        """Test loading configuration with invalid boolean values."""
        test_cases = [
            ("fallback_on_error", "not_bool", "fallback_on_error must be a boolean"),
            ("batch_processing", 123, "batch_processing must be a boolean"),
            ("connection_pooling", "yes", "connection_pooling must be a boolean"),
            ("operator_chaining", None, "operator_chaining must be a boolean"),
        ]

        for field, value, expected_error in test_cases:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump({field: value}, f)
                config_path = Path(f.name)

            try:
                with pytest.raises(ValueError) as exc_info:
                    PresidioConfig.load_from_file(config_path)
                assert expected_error in str(exc_info.value)
            finally:
                config_path.unlink()

    def test_load_from_file_invalid_max_batch_size(self):
        """Test loading configuration with invalid max_batch_size."""
        test_cases = [
            ("not_int", "Invalid max_batch_size"),
            (0, "Invalid max_batch_size"),  # Below MIN_BATCH_SIZE
            (MAX_BATCH_SIZE + 1, "Invalid max_batch_size"),  # Above MAX_BATCH_SIZE
            (-10, "Invalid max_batch_size"),  # Negative
        ]

        for value, expected_error in test_cases:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump({"max_batch_size": value}, f)
                config_path = Path(f.name)

            try:
                with pytest.raises(ValueError) as exc_info:
                    PresidioConfig.load_from_file(config_path)
                assert expected_error in str(exc_info.value)
            finally:
                config_path.unlink()

    def test_load_from_file_invalid_confidence_threshold(self):
        """Test loading configuration with invalid confidence_threshold."""
        test_cases = [
            ("not_number", "Invalid confidence_threshold"),
            (-0.1, "Invalid confidence_threshold"),  # Below MIN
            (1.1, "Invalid confidence_threshold"),  # Above MAX
        ]

        for value, expected_error in test_cases:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump({"confidence_threshold": value}, f)
                config_path = Path(f.name)

            try:
                with pytest.raises(ValueError) as exc_info:
                    PresidioConfig.load_from_file(config_path)
                assert expected_error in str(exc_info.value)
            finally:
                config_path.unlink()

    def test_load_from_file_invalid_operators(self):
        """Test loading configuration with invalid operators."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"operators": "not_dict"}, f)
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                PresidioConfig.load_from_file(config_path)
            assert "operators must be a dictionary" in str(exc_info.value)
        finally:
            config_path.unlink()

    def test_load_from_file_invalid_custom_recognizers(self):
        """Test loading configuration with invalid custom_recognizers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"custom_recognizers": "not_list"}, f)
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                PresidioConfig.load_from_file(config_path)
            assert "custom_recognizers must be a list" in str(exc_info.value)
        finally:
            config_path.unlink()

    def test_to_engine_params_presidio(self):
        """Test converting config to engine params for Presidio."""
        config = PresidioConfig(engine="presidio")
        params = config.to_engine_params()

        assert params["use_presidio_engine"] is True
        assert params["resolve_conflicts"] is True

    def test_to_engine_params_legacy(self):
        """Test converting config to engine params for legacy engine."""
        config = PresidioConfig(engine="legacy")
        params = config.to_engine_params()

        assert params["use_presidio_engine"] is False
        assert params["resolve_conflicts"] is True

    def test_to_engine_params_auto(self):
        """Test converting config to engine params for auto mode."""
        config = PresidioConfig(engine="auto")
        params = config.to_engine_params()

        assert "use_presidio_engine" not in params
        assert params["resolve_conflicts"] is True

    def test_to_dict(self):
        """Test converting config to dictionary."""
        operators = {"test": {"param": "value"}}
        recognizers = ["rec1", "rec2"]

        config = PresidioConfig(
            engine="presidio",
            fallback_on_error=False,
            batch_processing=True,
            connection_pooling=False,
            max_batch_size=150,
            confidence_threshold=0.75,
            operator_chaining=True,
            operators=operators,
            custom_recognizers=recognizers,
        )

        result = config.to_dict()

        assert result["engine"] == "presidio"
        assert result["fallback_on_error"] is False
        assert result["batch_processing"] is True
        assert result["connection_pooling"] is False
        assert result["max_batch_size"] == 150
        assert result["confidence_threshold"] == 0.75
        assert result["operator_chaining"] is True
        assert result["operators"] == operators
        assert result["custom_recognizers"] == recognizers


class TestHelperFunctions:
    """Test helper functions in cli.config module."""

    def test_load_presidio_config(self):
        """Test load_presidio_config function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"engine": "presidio", "max_batch_size": 75}, f)
            config_path = Path(f.name)

        try:
            config_dict = load_presidio_config(config_path)
            assert config_dict["engine"] == "presidio"
            assert config_dict["max_batch_size"] == 75
        finally:
            config_path.unlink()

    @patch("cloakpivot.cli.config.MaskingEngine")
    def test_create_masking_engine(self, mock_engine_class):
        """Test create_masking_engine function."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        presidio_settings = {
            "max_batch_size": 200,
            "confidence_threshold": 0.9,
        }

        engine = create_masking_engine(
            engine_type="presidio",
            presidio_settings=presidio_settings,
            fallback_enabled=True,
        )

        assert engine == mock_engine
        mock_engine_class.assert_called_once()

        # Check that proper params were passed
        call_kwargs = mock_engine_class.call_args[1]
        assert call_kwargs["use_presidio_engine"] is True
        assert call_kwargs["resolve_conflicts"] is True

    @patch("cloakpivot.cli.config.MaskingEngine")
    def test_create_masking_engine_legacy(self, mock_engine_class):
        """Test create_masking_engine with legacy type."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        engine = create_masking_engine(
            engine_type="legacy",
            presidio_settings=None,
            fallback_enabled=False,
        )

        assert engine == mock_engine
        call_kwargs = mock_engine_class.call_args[1]
        assert call_kwargs["use_presidio_engine"] is False

    def test_get_config_from_env_empty(self):
        """Test get_config_from_env with no environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_config_from_env()
            assert config == {}

    def test_get_config_from_env_all_vars(self):
        """Test get_config_from_env with all environment variables."""
        env_vars = {
            "CLOAKPIVOT_ENGINE": "presidio",
            "CLOAKPIVOT_PRESIDIO_FALLBACK": "true",
            "CLOAKPIVOT_MAX_BATCH_SIZE": "150",
            "CLOAKPIVOT_CONFIDENCE_THRESHOLD": "0.85",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = get_config_from_env()

            assert config["engine"] == "presidio"
            assert config["fallback_on_error"] is True
            assert config["max_batch_size"] == 150
            assert config["confidence_threshold"] == 0.85

    def test_get_config_from_env_fallback_variations(self):
        """Test get_config_from_env with different fallback values."""
        for value, expected in [
            ("true", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("YES", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("invalid", False),
        ]:
            with patch.dict(os.environ, {"CLOAKPIVOT_PRESIDIO_FALLBACK": value}, clear=True):
                config = get_config_from_env()
                assert config.get("fallback_on_error", False) == expected

    def test_get_config_from_env_invalid_batch_size(self):
        """Test get_config_from_env with invalid batch size."""
        test_cases = [
            ("not_a_number", "Invalid CLOAKPIVOT_MAX_BATCH_SIZE"),
            ("0", "Invalid CLOAKPIVOT_MAX_BATCH_SIZE"),
            (str(MAX_BATCH_SIZE + 1), "Invalid CLOAKPIVOT_MAX_BATCH_SIZE"),
        ]

        for value, expected_log in test_cases:
            with patch.dict(os.environ, {"CLOAKPIVOT_MAX_BATCH_SIZE": value}, clear=True):
                with patch("cloakpivot.cli.config.logger") as mock_logger:
                    config = get_config_from_env()
                    assert "max_batch_size" not in config
                    mock_logger.warning.assert_called_once()
                    assert expected_log in str(mock_logger.warning.call_args)

    def test_get_config_from_env_invalid_confidence(self):
        """Test get_config_from_env with invalid confidence threshold."""
        test_cases = [
            ("not_a_number", "Invalid CLOAKPIVOT_CONFIDENCE_THRESHOLD"),
            ("-0.1", "Invalid CLOAKPIVOT_CONFIDENCE_THRESHOLD"),
            ("1.5", "Invalid CLOAKPIVOT_CONFIDENCE_THRESHOLD"),
        ]

        for value, expected_log in test_cases:
            with patch.dict(os.environ, {"CLOAKPIVOT_CONFIDENCE_THRESHOLD": value}, clear=True):
                with patch("cloakpivot.cli.config.logger") as mock_logger:
                    config = get_config_from_env()
                    assert "confidence_threshold" not in config
                    mock_logger.warning.assert_called_once()
                    assert expected_log in str(mock_logger.warning.call_args)

    def test_merge_configs_empty(self):
        """Test merge_configs with empty configs."""
        result = merge_configs()
        assert result == {}

        result = merge_configs({}, {})
        assert result == {}

    def test_merge_configs_single(self):
        """Test merge_configs with single config."""
        config = {"engine": "presidio", "max_batch_size": 100}
        result = merge_configs(config)
        assert result == config

    def test_merge_configs_multiple(self):
        """Test merge_configs with multiple configs."""
        config1 = {"engine": "auto", "max_batch_size": 50}
        config2 = {"engine": "presidio", "confidence_threshold": 0.8}
        config3 = {"max_batch_size": 100}

        result = merge_configs(config1, config2, config3)

        # Later configs override earlier ones
        assert result["engine"] == "presidio"
        assert result["max_batch_size"] == 100
        assert result["confidence_threshold"] == 0.8

    def test_merge_configs_with_none(self):
        """Test merge_configs with None values."""
        config1 = {"engine": "auto"}
        config2 = None
        config3 = {"max_batch_size": 75}

        result = merge_configs(config1, config2, config3)

        assert result["engine"] == "auto"
        assert result["max_batch_size"] == 75


class TestConstants:
    """Test module constants are properly defined."""

    def test_batch_size_constants(self):
        """Test batch size constants."""
        assert DEFAULT_MAX_BATCH_SIZE == 100
        assert MIN_BATCH_SIZE == 1
        assert MAX_BATCH_SIZE == 10000
        assert MIN_BATCH_SIZE < DEFAULT_MAX_BATCH_SIZE < MAX_BATCH_SIZE

    def test_confidence_constants(self):
        """Test confidence threshold constants."""
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.8
        assert MIN_CONFIDENCE_THRESHOLD == 0.0
        assert MAX_CONFIDENCE_THRESHOLD == 1.0
        assert MIN_CONFIDENCE_THRESHOLD <= DEFAULT_CONFIDENCE_THRESHOLD <= MAX_CONFIDENCE_THRESHOLD
