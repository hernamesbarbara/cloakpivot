"""Tests for Presidio configuration management."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from cloakpivot.cli.config import (
    PresidioConfig,
    create_masking_engine,
    get_config_from_env,
    load_presidio_config,
    merge_configs,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def yaml_config_file(temp_dir):
    """Create a YAML configuration file."""
    config_data = {
        "presidio": {
            "engine": "presidio",
            "fallback_on_error": False,
            "batch_processing": True,
            "max_batch_size": 50,
            "confidence_threshold": 0.75,
            "operators": {
                "default_redact": {
                    "redact_char": "#",
                    "preserve_length": False,
                },
                "hash": {
                    "algorithm": "sha512",
                    "truncate": 16,
                },
            },
            "custom_recognizers": ["custom_ssn", "custom_id"],
        }
    }

    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path


@pytest.fixture
def json_config_file(temp_dir):
    """Create a JSON configuration file."""
    config_data = {
        "engine": "legacy",
        "fallback_on_error": True,
        "batch_processing": False,
        "max_batch_size": 25,
        "confidence_threshold": 0.9,
        "operators": {
            "encryption": {
                "key_provider": "file",
                "algorithm": "AES-128",
            }
        },
    }

    config_path = temp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    return config_path


class TestPresidioConfig:
    """Test PresidioConfig class."""

    def test_default_config(self):
        """Test creating default configuration."""
        config = PresidioConfig()

        assert config.engine == "auto"
        assert config.fallback_on_error is True
        assert config.batch_processing is True
        assert config.max_batch_size == 100
        assert config.confidence_threshold == 0.8
        assert config.operator_chaining is False
        assert config.operators == {}
        assert config.custom_recognizers == []

    def test_create_default(self):
        """Test create_default class method."""
        config = PresidioConfig.create_default()

        assert config.engine == "auto"
        assert "default_redact" in config.operators
        assert "encryption" in config.operators
        assert "hash" in config.operators
        assert config.operators["default_redact"]["redact_char"] == "*"

    def test_load_from_yaml_file(self, yaml_config_file):
        """Test loading configuration from YAML file."""
        config = PresidioConfig.load_from_file(yaml_config_file)

        assert config.engine == "presidio"
        assert config.fallback_on_error is False
        assert config.batch_processing is True
        assert config.max_batch_size == 50
        assert config.confidence_threshold == 0.75
        assert len(config.operators) == 2
        assert config.operators["default_redact"]["redact_char"] == "#"
        assert config.operators["hash"]["algorithm"] == "sha512"
        assert config.custom_recognizers == ["custom_ssn", "custom_id"]

    def test_load_from_json_file(self, json_config_file):
        """Test loading configuration from JSON file."""
        config = PresidioConfig.load_from_file(json_config_file)

        assert config.engine == "legacy"
        assert config.fallback_on_error is True
        assert config.batch_processing is False
        assert config.max_batch_size == 25
        assert config.confidence_threshold == 0.9
        assert "encryption" in config.operators
        assert config.operators["encryption"]["key_provider"] == "file"

    def test_load_from_file_not_found(self, temp_dir):
        """Test loading from non-existent file."""
        config_path = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            PresidioConfig.load_from_file(config_path)

    def test_load_from_unsupported_format(self, temp_dir):
        """Test loading from unsupported file format."""
        config_path = temp_dir / "config.txt"
        config_path.write_text("unsupported format")

        with pytest.raises(ValueError, match="Unsupported config format"):
            PresidioConfig.load_from_file(config_path)

    def test_to_engine_params_presidio(self):
        """Test converting to engine parameters for Presidio."""
        config = PresidioConfig(engine="presidio")
        params = config.to_engine_params()

        assert params["use_presidio_engine"] is True
        assert params["resolve_conflicts"] is True

    def test_to_engine_params_legacy(self):
        """Test converting to engine parameters for legacy."""
        config = PresidioConfig(engine="legacy")
        params = config.to_engine_params()

        assert params["use_presidio_engine"] is False
        assert params["resolve_conflicts"] is True

    def test_to_engine_params_auto(self):
        """Test converting to engine parameters for auto."""
        config = PresidioConfig(engine="auto")
        params = config.to_engine_params()

        assert "use_presidio_engine" not in params
        assert params["resolve_conflicts"] is True

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = PresidioConfig(
            engine="presidio",
            max_batch_size=75,
            operators={"test": {"key": "value"}},
        )

        config_dict = config.to_dict()

        assert config_dict["engine"] == "presidio"
        assert config_dict["max_batch_size"] == 75
        assert config_dict["operators"] == {"test": {"key": "value"}}
        assert "fallback_on_error" in config_dict
        assert "batch_processing" in config_dict


class TestConfigLoading:
    """Test configuration loading functions."""

    def test_load_presidio_config(self, yaml_config_file):
        """Test load_presidio_config function."""
        config_dict = load_presidio_config(yaml_config_file)

        assert isinstance(config_dict, dict)
        assert config_dict["engine"] == "presidio"
        assert config_dict["max_batch_size"] == 50
        assert "operators" in config_dict

    def test_load_presidio_config_not_found(self, temp_dir):
        """Test loading non-existent config."""
        config_path = temp_dir / "missing.yaml"

        with pytest.raises(FileNotFoundError):
            load_presidio_config(config_path)


class TestMaskingEngineFactory:
    """Test MaskingEngine factory function."""

    @patch("cloakpivot.masking.engine.MaskingEngine")
    def test_create_masking_engine_presidio(self, mock_engine_class):
        """Test creating MaskingEngine with Presidio settings."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        engine = create_masking_engine(
            engine_type="presidio",
            presidio_settings={"confidence_threshold": 0.85},
            fallback_enabled=True,
        )

        assert engine == mock_engine
        mock_engine_class.assert_called_once()
        kwargs = mock_engine_class.call_args.kwargs
        assert kwargs["use_presidio_engine"] is True
        assert kwargs["resolve_conflicts"] is True

    @patch("cloakpivot.masking.engine.MaskingEngine")
    def test_create_masking_engine_legacy(self, mock_engine_class):
        """Test creating MaskingEngine with legacy settings."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        engine = create_masking_engine(
            engine_type="legacy",
            presidio_settings={},
            fallback_enabled=False,
        )

        assert engine == mock_engine
        mock_engine_class.assert_called_once()
        kwargs = mock_engine_class.call_args.kwargs
        assert kwargs["use_presidio_engine"] is False
        assert kwargs["resolve_conflicts"] is True

    @patch("cloakpivot.masking.engine.MaskingEngine")
    def test_create_masking_engine_auto(self, mock_engine_class):
        """Test creating MaskingEngine with auto settings."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        engine = create_masking_engine(
            engine_type="auto",
            presidio_settings={},
            fallback_enabled=True,
        )

        assert engine == mock_engine
        mock_engine_class.assert_called_once()
        kwargs = mock_engine_class.call_args.kwargs
        assert "use_presidio_engine" not in kwargs
        assert kwargs["resolve_conflicts"] is True


class TestEnvironmentConfig:
    """Test environment variable configuration."""

    def test_get_config_from_env_engine(self, monkeypatch):
        """Test getting engine selection from environment."""
        monkeypatch.setenv("CLOAKPIVOT_ENGINE", "presidio")

        config = get_config_from_env()

        assert config["engine"] == "presidio"

    def test_get_config_from_env_fallback_true(self, monkeypatch):
        """Test getting fallback setting from environment (true)."""
        monkeypatch.setenv("CLOAKPIVOT_PRESIDIO_FALLBACK", "true")

        config = get_config_from_env()

        assert config["fallback_on_error"] is True

    def test_get_config_from_env_fallback_false(self, monkeypatch):
        """Test getting fallback setting from environment (false)."""
        monkeypatch.setenv("CLOAKPIVOT_PRESIDIO_FALLBACK", "false")

        config = get_config_from_env()

        assert config["fallback_on_error"] is False

    def test_get_config_from_env_batch_size(self, monkeypatch):
        """Test getting batch size from environment."""
        monkeypatch.setenv("CLOAKPIVOT_MAX_BATCH_SIZE", "200")

        config = get_config_from_env()

        assert config["max_batch_size"] == 200

    def test_get_config_from_env_batch_size_invalid(self, monkeypatch):
        """Test invalid batch size in environment."""
        monkeypatch.setenv("CLOAKPIVOT_MAX_BATCH_SIZE", "not_a_number")

        config = get_config_from_env()

        assert "max_batch_size" not in config

    def test_get_config_from_env_confidence_threshold(self, monkeypatch):
        """Test getting confidence threshold from environment."""
        monkeypatch.setenv("CLOAKPIVOT_CONFIDENCE_THRESHOLD", "0.95")

        config = get_config_from_env()

        assert config["confidence_threshold"] == 0.95

    def test_get_config_from_env_confidence_invalid(self, monkeypatch):
        """Test invalid confidence threshold in environment."""
        monkeypatch.setenv("CLOAKPIVOT_CONFIDENCE_THRESHOLD", "invalid")

        config = get_config_from_env()

        assert "confidence_threshold" not in config

    def test_get_config_from_env_empty(self):
        """Test getting config from empty environment."""
        # Clear relevant environment variables
        for key in list(os.environ.keys()):
            if key.startswith("CLOAKPIVOT_"):
                del os.environ[key]

        config = get_config_from_env()

        assert config == {}


class TestConfigMerging:
    """Test configuration merging."""

    def test_merge_configs_empty(self):
        """Test merging empty configurations."""
        result = merge_configs({}, {})

        assert result == {}

    def test_merge_configs_single(self):
        """Test merging single configuration."""
        config = {"engine": "presidio", "max_batch_size": 50}

        result = merge_configs(config)

        assert result == config

    def test_merge_configs_override(self):
        """Test merging with override."""
        config1 = {"engine": "legacy", "max_batch_size": 50}
        config2 = {"engine": "presidio", "confidence_threshold": 0.9}

        result = merge_configs(config1, config2)

        assert result["engine"] == "presidio"
        assert result["max_batch_size"] == 50
        assert result["confidence_threshold"] == 0.9

    def test_merge_configs_multiple(self):
        """Test merging multiple configurations."""
        config1 = {"engine": "auto"}
        config2 = {"max_batch_size": 75}
        config3 = {"engine": "presidio", "confidence_threshold": 0.85}

        result = merge_configs(config1, config2, config3)

        assert result["engine"] == "presidio"
        assert result["max_batch_size"] == 75
        assert result["confidence_threshold"] == 0.85

    def test_merge_configs_with_none(self):
        """Test merging with None values."""
        config1 = {"engine": "legacy"}
        config2 = None
        config3 = {"max_batch_size": 100}

        result = merge_configs(config1, config2, config3)

        assert result["engine"] == "legacy"
        assert result["max_batch_size"] == 100


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_yaml_config(self, yaml_config_file):
        """Test that valid YAML config loads without errors."""
        config = PresidioConfig.load_from_file(yaml_config_file)

        # Should not raise any exceptions
        assert config is not None
        assert isinstance(config, PresidioConfig)

    def test_partial_config(self, temp_dir):
        """Test loading partial configuration."""
        config_data = {
            "presidio": {
                "engine": "presidio",
                # Only some fields specified
            }
        }

        config_path = temp_dir / "partial.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = PresidioConfig.load_from_file(config_path)

        # Should use defaults for missing fields
        assert config.engine == "presidio"
        assert config.fallback_on_error is True  # default
        assert config.max_batch_size == 100  # default

    def test_config_without_presidio_section(self, temp_dir):
        """Test loading config without presidio section."""
        config_data = {
            "engine": "legacy",
            "max_batch_size": 75,
        }

        config_path = temp_dir / "no_section.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = PresidioConfig.load_from_file(config_path)

        # Should handle configs without presidio section
        assert config.engine == "legacy"
        assert config.max_batch_size == 75
