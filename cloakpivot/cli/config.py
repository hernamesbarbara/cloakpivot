"""Configuration management for Presidio integration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_MAX_BATCH_SIZE = 100
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 10000
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
MIN_CONFIDENCE_THRESHOLD = 0.0
MAX_CONFIDENCE_THRESHOLD = 1.0
DEFAULT_TIMEOUT_SECONDS = 5


@dataclass
class PresidioConfig:
    """Presidio-specific configuration management.

    This class manages configuration settings for the Presidio integration,
    including engine selection, processing options, and operator configurations.

    Attributes:
        engine: Engine selection mode ('auto', 'presidio', or 'legacy')
        fallback_on_error: Whether to fallback to legacy engine on Presidio errors
        batch_processing: Enable batch processing for improved performance
        connection_pooling: Enable connection pooling for Presidio services
        max_batch_size: Maximum number of entities to process in a batch
        confidence_threshold: Minimum confidence score for entity detection
        operator_chaining: Enable operator chaining for complex transformations
        operators: Custom operator configurations
        custom_recognizers: List of custom recognizer patterns

    Examples:
        >>> # Load from file
        >>> config = PresidioConfig.load_from_file(Path("config.yml"))

        >>> # Create with defaults
        >>> config = PresidioConfig.create_default()

        >>> # Convert to engine parameters
        >>> params = config.to_engine_params()
    """

    engine: str = "auto"  # auto, presidio, legacy
    fallback_on_error: bool = True
    batch_processing: bool = True
    connection_pooling: bool = True
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    operator_chaining: bool = False
    operators: dict[str, dict[str, Any]] = field(default_factory=dict)
    custom_recognizers: list[str] = field(default_factory=list)

    @classmethod
    def load_from_file(cls, config_path: Path) -> PresidioConfig:
        """Load Presidio configuration from YAML/JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            PresidioConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        # Extract presidio section if present
        if "presidio" in data:
            data = data["presidio"]

        # Create config instance
        config = cls()

        # Update with loaded values with validation
        if "engine" in data:
            engine = data["engine"]
            if engine not in ["auto", "presidio", "legacy"]:
                raise ValueError(
                    f"Invalid engine type: {engine}. "
                    "Must be 'auto', 'presidio', or 'legacy'"
                )
            config.engine = engine

        if "fallback_on_error" in data:
            if not isinstance(data["fallback_on_error"], bool):
                raise ValueError("fallback_on_error must be a boolean")
            config.fallback_on_error = data["fallback_on_error"]

        if "batch_processing" in data:
            if not isinstance(data["batch_processing"], bool):
                raise ValueError("batch_processing must be a boolean")
            config.batch_processing = data["batch_processing"]

        if "connection_pooling" in data:
            if not isinstance(data["connection_pooling"], bool):
                raise ValueError("connection_pooling must be a boolean")
            config.connection_pooling = data["connection_pooling"]

        if "max_batch_size" in data:
            batch_size = data["max_batch_size"]
            if not isinstance(batch_size, int) or batch_size < MIN_BATCH_SIZE or batch_size > MAX_BATCH_SIZE:
                raise ValueError(
                    f"Invalid max_batch_size: {batch_size}. "
                    f"Must be an integer between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}"
                )
            config.max_batch_size = batch_size

        if "confidence_threshold" in data:
            threshold = data["confidence_threshold"]
            if (
                not isinstance(threshold, (int, float)) or
                threshold < MIN_CONFIDENCE_THRESHOLD or threshold > MAX_CONFIDENCE_THRESHOLD
            ):
                raise ValueError(
                    f"Invalid confidence_threshold: {threshold}. "
                    f"Must be a number between {MIN_CONFIDENCE_THRESHOLD} and {MAX_CONFIDENCE_THRESHOLD}"
                )
            config.confidence_threshold = float(threshold)

        if "operator_chaining" in data:
            if not isinstance(data["operator_chaining"], bool):
                raise ValueError("operator_chaining must be a boolean")
            config.operator_chaining = data["operator_chaining"]

        if "operators" in data:
            if not isinstance(data["operators"], dict):
                raise ValueError("operators must be a dictionary")
            config.operators = data["operators"]

        if "custom_recognizers" in data:
            if not isinstance(data["custom_recognizers"], list):
                raise ValueError("custom_recognizers must be a list")
            config.custom_recognizers = data["custom_recognizers"]

        return config

    @classmethod
    def create_default(cls) -> PresidioConfig:
        """Create default Presidio configuration.

        Returns:
            Default PresidioConfig instance
        """
        return cls(
            operators={
                "default_redact": {
                    "redact_char": "*",
                    "preserve_length": True,
                },
                "encryption": {
                    "key_provider": "environment",
                    "key_rotation": True,
                },
                "hash": {
                    "algorithm": "sha256",
                    "salt_source": "random",
                    "truncate": 8,
                },
            }
        )

    def to_engine_params(self) -> dict[str, Any]:
        """Convert to MaskingEngine parameters.

        Returns:
            Dictionary of parameters for MaskingEngine initialization
        """
        params = {}

        # Determine use_presidio_engine parameter
        if self.engine == "presidio":
            params["use_presidio_engine"] = True
        elif self.engine == "legacy":
            params["use_presidio_engine"] = False
        # For "auto", don't set it (let engine decide)

        # Add other relevant parameters
        params["resolve_conflicts"] = True  # Always enable conflict resolution

        return params

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "engine": self.engine,
            "fallback_on_error": self.fallback_on_error,
            "batch_processing": self.batch_processing,
            "connection_pooling": self.connection_pooling,
            "max_batch_size": self.max_batch_size,
            "confidence_threshold": self.confidence_threshold,
            "operator_chaining": self.operator_chaining,
            "operators": self.operators,
            "custom_recognizers": self.custom_recognizers,
        }


def load_presidio_config(config_path: Path) -> dict[str, Any]:
    """Load and validate Presidio configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config = PresidioConfig.load_from_file(config_path)
    return config.to_dict()


def create_masking_engine(
    engine_type: str,
    presidio_settings: dict[str, Any],
    fallback_enabled: bool,
) -> Any:
    """Factory function for creating MaskingEngine with proper configuration.

    Args:
        engine_type: Type of engine (auto, presidio, legacy)
        presidio_settings: Presidio-specific settings
        fallback_enabled: Whether to enable fallback to legacy engine

    Returns:
        Configured MaskingEngine instance
    """
    from cloakpivot.masking.engine import MaskingEngine

    # Create config from settings
    config = PresidioConfig()
    config.engine = engine_type
    config.fallback_on_error = fallback_enabled

    # Update with presidio_settings if provided
    if presidio_settings:
        for key, value in presidio_settings.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Get engine parameters
    engine_params = config.to_engine_params()

    # Create and return engine
    return MaskingEngine(**engine_params)


def get_config_from_env() -> dict[str, Any]:
    """Get configuration from environment variables.

    Returns:
        Configuration dictionary from environment
    """
    config: dict[str, Any] = {}

    # Check for engine selection
    if "CLOAKPIVOT_ENGINE" in os.environ:
        config["engine"] = os.environ["CLOAKPIVOT_ENGINE"]

    # Check for fallback setting
    if "CLOAKPIVOT_PRESIDIO_FALLBACK" in os.environ:
        config["fallback_on_error"] = (
            os.environ["CLOAKPIVOT_PRESIDIO_FALLBACK"].lower() in ["true", "1", "yes"]
        )

    # Check for batch size
    if "CLOAKPIVOT_MAX_BATCH_SIZE" in os.environ:
        try:
            batch_size = int(os.environ["CLOAKPIVOT_MAX_BATCH_SIZE"])
            if batch_size < MIN_BATCH_SIZE or batch_size > MAX_BATCH_SIZE:
                logger.warning(
                    f"Invalid CLOAKPIVOT_MAX_BATCH_SIZE: {batch_size}. "
                    f"Must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}. Using defaults."
                )
            else:
                config["max_batch_size"] = batch_size
        except ValueError:
            logger.warning(
                f"Invalid CLOAKPIVOT_MAX_BATCH_SIZE: '{os.environ['CLOAKPIVOT_MAX_BATCH_SIZE']}'. "
                "Must be an integer. Using defaults."
            )

    # Check for confidence threshold
    if "CLOAKPIVOT_CONFIDENCE_THRESHOLD" in os.environ:
        try:
            threshold = float(os.environ["CLOAKPIVOT_CONFIDENCE_THRESHOLD"])
            if threshold < MIN_CONFIDENCE_THRESHOLD or threshold > MAX_CONFIDENCE_THRESHOLD:
                logger.warning(
                    f"Invalid CLOAKPIVOT_CONFIDENCE_THRESHOLD: {threshold}. "
                    f"Must be between {MIN_CONFIDENCE_THRESHOLD} and {MAX_CONFIDENCE_THRESHOLD}. Using defaults."
                )
            else:
                config["confidence_threshold"] = threshold
        except ValueError:
            logger.warning(
                f"Invalid CLOAKPIVOT_CONFIDENCE_THRESHOLD: '{os.environ['CLOAKPIVOT_CONFIDENCE_THRESHOLD']}'. "
                "Must be a number. Using defaults."
            )

    return config


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configuration dictionaries.

    Later configs override earlier ones.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration dictionary
    """
    result = {}
    for config in configs:
        if config:
            result.update(config)
    return result
