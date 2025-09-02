"""Registry for managing custom strategy plugins."""

import logging
from typing import Any, Optional

from ..exceptions import PluginError, PluginExecutionError
from .base import BaseStrategyPlugin, StrategyPluginResult

logger = logging.getLogger(__name__)


class StrategyPluginRegistry:
    """
    Registry for managing custom masking strategy plugins.

    This registry provides integration between strategy plugins and
    the existing StrategyApplicator system.
    """

    def __init__(self, main_registry: Optional[Any] = None) -> None:
        """
        Initialize the strategy plugin registry.

        Args:
            main_registry: Reference to main plugin registry
        """
        self._main_registry = main_registry
        self._active_strategies: dict[str, BaseStrategyPlugin] = {}

    def register_strategy_plugin(
        self, plugin: BaseStrategyPlugin, config: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Register and initialize a strategy plugin.

        Args:
            plugin: Strategy plugin to register
            config: Optional configuration for the plugin

        Raises:
            PluginError: If registration fails
        """
        plugin_name = plugin.info.name

        try:
            # Initialize plugin with configuration
            if config:
                plugin.update_config(config)

            plugin.initialize()

            self._active_strategies[plugin_name] = plugin

            logger.info(f"Strategy plugin {plugin_name} registered and initialized")

        except Exception as e:
            logger.error(f"Failed to register strategy plugin {plugin_name}: {e}")
            raise PluginError(
                f"Failed to register strategy plugin {plugin_name}: {e}",
                plugin_name=plugin_name,
            ) from e

    def apply_strategy(
        self,
        plugin_name: str,
        original_text: str,
        entity_type: str,
        confidence: float,
        context: Optional[dict[str, Any]] = None,
    ) -> StrategyPluginResult:
        """
        Apply a custom strategy plugin.

        Args:
            plugin_name: Name of strategy plugin to use
            original_text: Original PII text to mask
            entity_type: Type of entity
            confidence: Detection confidence
            context: Optional context information

        Returns:
            StrategyPluginResult with masked text and metadata

        Raises:
            PluginError: If plugin not found or execution fails
        """
        if plugin_name not in self._active_strategies:
            raise PluginError(
                f"Strategy plugin {plugin_name} not found or not active",
                plugin_name=plugin_name,
            )

        plugin = self._active_strategies[plugin_name]

        try:
            result = plugin.apply_strategy_safe(
                original_text, entity_type, confidence, context
            )

            if not result.success:
                logger.warning(
                    f"Strategy plugin {plugin_name} returned failure: {result.error_message}"
                )

            return result

        except Exception as e:
            logger.error(f"Strategy plugin {plugin_name} execution failed: {e}")
            raise PluginExecutionError(
                f"Strategy plugin {plugin_name} execution failed: {e}",
                plugin_name=plugin_name,
                original_exception=e,
            ) from e

    def get_active_strategy_plugins(self) -> dict[str, BaseStrategyPlugin]:
        """Get all active strategy plugins."""
        return self._active_strategies.copy()

    def get_strategy_plugin(self, plugin_name: str) -> Optional[BaseStrategyPlugin]:
        """Get a specific strategy plugin by name."""
        return self._active_strategies.get(plugin_name)

    def list_strategy_plugins(self) -> list[str]:
        """Get list of active strategy plugin names."""
        return list(self._active_strategies.keys())

    def supports_entity_type(self, plugin_name: str, entity_type: str) -> bool:
        """
        Check if a strategy plugin supports a given entity type.

        Args:
            plugin_name: Name of strategy plugin
            entity_type: Entity type to check

        Returns:
            True if plugin supports the entity type
        """
        if plugin_name not in self._active_strategies:
            return False

        plugin = self._active_strategies[plugin_name]
        supported_types = plugin.get_supported_entity_types()

        # None means supports all types
        if supported_types is None:
            return True

        return entity_type in supported_types

    def get_plugin_parameters_schema(
        self, plugin_name: str
    ) -> Optional[dict[str, Any]]:
        """
        Get parameter schema for a strategy plugin.

        Args:
            plugin_name: Name of strategy plugin

        Returns:
            JSON schema for plugin parameters
        """
        if plugin_name not in self._active_strategies:
            return None

        plugin = self._active_strategies[plugin_name]
        return plugin.get_strategy_parameters_schema()

    def cleanup_strategy_plugin(self, plugin_name: str) -> None:
        """
        Clean up a strategy plugin.

        Args:
            plugin_name: Name of plugin to clean up
        """
        if plugin_name in self._active_strategies:
            plugin = self._active_strategies[plugin_name]
            try:
                plugin.cleanup()
                del self._active_strategies[plugin_name]
                logger.info(f"Strategy plugin {plugin_name} cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up strategy plugin {plugin_name}: {e}")

    def cleanup_all_strategies(self) -> None:
        """Clean up all strategy plugins."""
        for plugin_name in list(self._active_strategies.keys()):
            self.cleanup_strategy_plugin(plugin_name)

    def get_strategy_registry_status(self) -> dict[str, Any]:
        """Get status of strategy plugin registry."""
        plugin_status = {}

        for name, plugin in self._active_strategies.items():
            plugin_status[name] = {
                "name": plugin.info.name,
                "version": plugin.info.version,
                "initialized": plugin.is_initialized,
                "supported_entities": plugin.get_supported_entity_types(),
            }

        return {
            "active_strategies": len(self._active_strategies),
            "plugins": plugin_status,
        }
