"""Base classes and interfaces for the plugin system."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """Status of a plugin in the registry."""

    UNLOADED = "unloaded"
    LOADED = "loaded"
    VALIDATED = "validated"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginInfo:
    """Information about a plugin."""

    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    entry_point: Optional[str] = None
    status: PluginStatus = field(default=PluginStatus.UNLOADED)
    error_message: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    config_schema: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate plugin info after initialization."""
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        if not self.version:
            raise ValueError("Plugin version cannot be empty")
        if not self.plugin_type:
            raise ValueError("Plugin type cannot be empty")


class BasePlugin(ABC):
    """
    Base class for all CloakPivot plugins.

    All custom plugins must inherit from this class and implement the required methods.
    This provides a standard interface for plugin management, validation, and execution.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """
        Initialize the plugin.

        Args:
            config: Optional configuration dictionary for the plugin
        """
        self.config = config or {}
        self.plugin_id = str(uuid4())
        self.is_initialized = False
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Get plugin information."""
        pass

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        Validate plugin configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid

        Raises:
            PluginValidationError: If configuration is invalid
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the plugin.

        This method is called after the plugin is loaded and configured.
        Use this to set up any resources needed for plugin operation.

        Raises:
            PluginExecutionError: If initialization fails
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up plugin resources.

        This method is called when the plugin is being unloaded.
        Use this to clean up any resources allocated during initialization.
        """
        pass

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with optional default."""
        return self.config.get(key, default)

    def update_config(self, new_config: dict[str, Any]) -> None:
        """Update plugin configuration."""
        if self.validate_config(new_config):
            self.config.update(new_config)
            self.logger.info(f"Plugin {self.info.name} configuration updated")

    def get_health_status(self) -> dict[str, Any]:
        """Get plugin health status for monitoring."""
        return {
            "plugin_id": self.plugin_id,
            "name": self.info.name,
            "version": self.info.version,
            "status": "healthy" if self.is_initialized else "not_initialized",
            "config_keys": list(self.config.keys()),
            "last_check": None,  # Could be implemented for periodic health checks
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.info.name}, version={self.info.version})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.info.name}', "
            f"version='{self.info.version}', "
            f"status={self.is_initialized}"
            f")"
        )
