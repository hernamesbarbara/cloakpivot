"""Central plugin registry for managing strategy and recognizer plugins."""

import importlib
import logging
import sys
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Optional

from .base import BasePlugin, PluginInfo, PluginStatus
from .exceptions import (
    PluginDependencyError,
    PluginError,
    PluginExecutionError,
    PluginRegistrationError,
    PluginValidationError,
)
from .recognizers.base import BaseRecognizerPlugin
from .strategies.base import BaseStrategyPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Central registry for managing CloakPivot plugins.

    This registry provides plugin discovery via Python entry points,
    plugin lifecycle management, and error isolation between plugins.

    Example:
        >>> registry = get_plugin_registry()
        >>> registry.discover_plugins()
        >>> strategy_plugins = registry.get_strategy_plugins()
        >>> recognizer_plugins = registry.get_recognizer_plugins()
    """

    _instance: Optional["PluginRegistry"] = None

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        if PluginRegistry._instance is not None:
            raise RuntimeError(
                "PluginRegistry is a singleton. Use get_plugin_registry() instead."
            )

        self._plugins: dict[str, BasePlugin] = {}
        self._plugin_infos: dict[str, PluginInfo] = {}
        self._strategy_plugins: dict[str, BaseStrategyPlugin] = {}
        self._recognizer_plugins: dict[str, BaseRecognizerPlugin] = {}
        self._discovery_completed = False

        logger.info("Plugin registry initialized")

    @classmethod
    def get_instance(cls) -> "PluginRegistry":
        """Get the singleton plugin registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def discover_plugins(self, force_rediscovery: bool = False) -> None:
        """
        Discover plugins using Python entry points.

        Args:
            force_rediscovery: Force rediscovery even if already completed
        """
        if self._discovery_completed and not force_rediscovery:
            logger.debug("Plugin discovery already completed, skipping")
            return

        logger.info("Starting plugin discovery via entry points")

        try:
            # Discover strategy plugins
            self._discover_entry_point_plugins(
                "cloakpivot.plugins.strategies",
                BaseStrategyPlugin,  # type: ignore[type-abstract]
                "strategy",
            )

            # Discover recognizer plugins
            self._discover_entry_point_plugins(
                "cloakpivot.plugins.recognizers",
                BaseRecognizerPlugin,  # type: ignore[type-abstract]
                "recognizer",
            )

            self._discovery_completed = True
            logger.info(
                f"Plugin discovery completed. Found {len(self._plugins)} plugins: "
                f"{len(self._strategy_plugins)} strategy, {len(self._recognizer_plugins)} recognizer"
            )

        except Exception as e:
            logger.error(f"Plugin discovery failed: {e}")
            raise PluginRegistrationError(f"Plugin discovery failed: {e}") from e

    def _discover_entry_point_plugins(
        self, entry_point_group: str, base_class: type[BasePlugin], plugin_type: str
    ) -> None:
        """
        Discover plugins from a specific entry point group.

        Args:
            entry_point_group: Entry point group name
            base_class: Expected base class for plugins
            plugin_type: Type of plugin (strategy/recognizer)
        """
        try:
            # Use importlib.metadata.entry_points for Python 3.10+
            if sys.version_info >= (3, 10):
                eps = entry_points(group=entry_point_group)
            else:
                # Fallback for older Python versions
                eps = entry_points().get(entry_point_group, [])

            logger.debug(f"Found {len(eps)} entry points for {entry_point_group}")

            for entry_point in eps:
                try:
                    self._load_entry_point_plugin(entry_point, base_class, plugin_type)
                except Exception as e:
                    logger.warning(
                        f"Failed to load plugin from entry point {entry_point.name}: {e}"
                    )
                    # Continue with other plugins
                    continue

        except Exception as e:
            logger.error(
                f"Failed to discover entry points for {entry_point_group}: {e}"
            )
            # Don't raise - continue with other discovery methods

    def _load_entry_point_plugin(
        self, entry_point: EntryPoint, base_class: type[BasePlugin], plugin_type: str
    ) -> None:
        """
        Load a plugin from an entry point.

        Args:
            entry_point: Entry point to load
            base_class: Expected base class
            plugin_type: Type of plugin
        """
        plugin_name = entry_point.name

        try:
            # Load the plugin class
            plugin_class = entry_point.load()

            # Validate plugin class
            if not issubclass(plugin_class, base_class):
                raise PluginValidationError(
                    f"Plugin {plugin_name} does not inherit from {base_class.__name__}",
                    plugin_name=plugin_name,
                )

            # Create plugin instance with empty config initially
            plugin_instance = plugin_class({})

            # Validate plugin info
            plugin_info = plugin_instance.info
            if plugin_info.plugin_type != plugin_type:
                raise PluginValidationError(
                    f"Plugin {plugin_name} has incorrect type: {plugin_info.plugin_type}, expected {plugin_type}",
                    plugin_name=plugin_name,
                )

            # Register the plugin
            self.register_plugin(plugin_instance)

            logger.info(
                f"Loaded {plugin_type} plugin: {plugin_name} v{plugin_info.version}"
            )

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            raise PluginRegistrationError(
                f"Failed to load plugin {plugin_name}: {e}", plugin_name=plugin_name
            ) from e

    def register_plugin(self, plugin: BasePlugin) -> None:
        """
        Register a plugin instance.

        Args:
            plugin: Plugin instance to register

        Raises:
            PluginRegistrationError: If registration fails
        """
        plugin_info = plugin.info
        plugin_name = plugin_info.name

        if plugin_name in self._plugins:
            raise PluginRegistrationError(
                f"Plugin {plugin_name} is already registered", plugin_name=plugin_name
            )

        try:
            # Validate plugin
            self._validate_plugin(plugin)

            # Store in appropriate registry
            self._plugins[plugin_name] = plugin
            self._plugin_infos[plugin_name] = plugin_info

            if isinstance(plugin, BaseStrategyPlugin):
                self._strategy_plugins[plugin_name] = plugin
            elif isinstance(plugin, BaseRecognizerPlugin):
                self._recognizer_plugins[plugin_name] = plugin

            # Update plugin status
            plugin_info.status = PluginStatus.LOADED

            logger.info(f"Registered plugin: {plugin_name} ({plugin_info.plugin_type})")

        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_name}: {e}")
            raise PluginRegistrationError(
                f"Failed to register plugin {plugin_name}: {e}", plugin_name=plugin_name
            ) from e

    def _validate_plugin(self, plugin: BasePlugin) -> None:
        """
        Validate a plugin instance.

        Args:
            plugin: Plugin to validate

        Raises:
            PluginValidationError: If validation fails
        """
        plugin_info = plugin.info
        plugin_name = plugin_info.name

        # Basic validation
        if not plugin_name:
            raise PluginValidationError("Plugin name cannot be empty")

        if not plugin_info.version:
            raise PluginValidationError(
                "Plugin version cannot be empty", plugin_name=plugin_name
            )

        if not plugin_info.plugin_type:
            raise PluginValidationError(
                "Plugin type cannot be empty", plugin_name=plugin_name
            )

        # Validate dependencies
        self._validate_plugin_dependencies(plugin)

        # Validate configuration schema if provided
        if plugin_info.config_schema:
            self._validate_config_schema(plugin_info.config_schema, plugin_name)

    def _validate_plugin_dependencies(self, plugin: BasePlugin) -> None:
        """
        Validate plugin dependencies.

        Args:
            plugin: Plugin to validate

        Raises:
            PluginDependencyError: If dependencies are not satisfied
        """
        plugin_info = plugin.info

        for dependency in plugin_info.dependencies:
            if not self._is_dependency_available(dependency):
                raise PluginDependencyError(
                    f"Plugin dependency not available: {dependency}",
                    plugin_name=plugin_info.name,
                )

    def _is_dependency_available(self, dependency: str) -> bool:
        """
        Check if a dependency is available.

        Args:
            dependency: Dependency specification

        Returns:
            True if dependency is available
        """
        # Simple check for now - can be enhanced with version checking
        try:
            importlib.import_module(dependency)
            return True
        except ImportError:
            return False

    def _validate_config_schema(self, schema: dict[str, Any], plugin_name: str) -> None:
        """
        Validate plugin configuration schema.

        Args:
            schema: JSON schema dictionary
            plugin_name: Name of the plugin

        Raises:
            PluginValidationError: If schema is invalid
        """
        # Basic schema validation - could be enhanced with jsonschema library
        if not isinstance(schema, dict):
            raise PluginValidationError(
                "Config schema must be a dictionary", plugin_name=plugin_name
            )

        if "type" not in schema:
            raise PluginValidationError(
                "Config schema must have a 'type' field", plugin_name=plugin_name
            )

    def initialize_plugin(
        self, plugin_name: str, config: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize a plugin with optional configuration.

        Args:
            plugin_name: Name of plugin to initialize
            config: Optional configuration dictionary

        Raises:
            PluginError: If plugin not found or initialization fails
        """
        if plugin_name not in self._plugins:
            raise PluginError(
                f"Plugin {plugin_name} not found", plugin_name=plugin_name
            )

        plugin = self._plugins[plugin_name]
        plugin_info = self._plugin_infos[plugin_name]

        try:
            # Update configuration if provided
            if config:
                plugin.update_config(config)

            # Initialize the plugin
            plugin.initialize()

            # Update status
            plugin_info.status = PluginStatus.ACTIVE
            plugin_info.error_message = None

            logger.info(f"Plugin {plugin_name} initialized successfully")

        except Exception as e:
            plugin_info.status = PluginStatus.ERROR
            plugin_info.error_message = str(e)

            logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
            raise PluginExecutionError(
                f"Failed to initialize plugin {plugin_name}: {e}",
                plugin_name=plugin_name,
                original_exception=e,
            ) from e

    def cleanup_plugin(self, plugin_name: str) -> None:
        """
        Clean up a plugin.

        Args:
            plugin_name: Name of plugin to clean up
        """
        if plugin_name not in self._plugins:
            logger.warning(f"Plugin {plugin_name} not found for cleanup")
            return

        plugin = self._plugins[plugin_name]
        plugin_info = self._plugin_infos[plugin_name]

        try:
            plugin.cleanup()
            plugin_info.status = PluginStatus.LOADED
            logger.info(f"Plugin {plugin_name} cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up plugin {plugin_name}: {e}")

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self._plugins.get(plugin_name)

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information by name."""
        return self._plugin_infos.get(plugin_name)

    def get_strategy_plugins(self) -> dict[str, BaseStrategyPlugin]:
        """Get all registered strategy plugins."""
        return self._strategy_plugins.copy()

    def get_recognizer_plugins(self) -> dict[str, BaseRecognizerPlugin]:
        """Get all registered recognizer plugins."""
        return self._recognizer_plugins.copy()

    def get_active_plugins(self) -> dict[str, BasePlugin]:
        """Get all active (initialized) plugins."""
        return {
            name: plugin
            for name, plugin in self._plugins.items()
            if self._plugin_infos[name].status == PluginStatus.ACTIVE
        }

    def list_plugins(self) -> list[PluginInfo]:
        """Get list of all plugin information."""
        return list(self._plugin_infos.values())

    def get_plugins_by_type(self, plugin_type: str) -> dict[str, BasePlugin]:
        """
        Get plugins by type.

        Args:
            plugin_type: Type of plugins to retrieve

        Returns:
            Dictionary of plugins of the specified type
        """
        return {
            name: plugin
            for name, plugin in self._plugins.items()
            if self._plugin_infos[name].plugin_type == plugin_type
        }

    def cleanup_all_plugins(self) -> None:
        """Clean up all plugins."""
        for plugin_name in list(self._plugins.keys()):
            self.cleanup_plugin(plugin_name)

    def get_registry_status(self) -> dict[str, Any]:
        """
        Get overall registry status.

        Returns:
            Dictionary with registry statistics and status
        """
        status_counts: dict[str, int] = {}
        for info in self._plugin_infos.values():
            status = info.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_plugins": len(self._plugins),
            "strategy_plugins": len(self._strategy_plugins),
            "recognizer_plugins": len(self._recognizer_plugins),
            "status_counts": status_counts,
            "discovery_completed": self._discovery_completed,
        }


# Global plugin registry instance
_global_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.

    Returns:
        The singleton plugin registry
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = PluginRegistry.get_instance()

    return _global_registry


def reset_plugin_registry() -> None:
    """
    Reset the global plugin registry.

    This is primarily for testing purposes.
    """
    global _global_registry

    if _global_registry is not None:
        _global_registry.cleanup_all_plugins()
        PluginRegistry._instance = None
        _global_registry = None
