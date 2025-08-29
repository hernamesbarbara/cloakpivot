"""Plugin-aware StrategyApplicator extension."""

import logging
from typing import Any, Optional

from ..core.strategies import Strategy, StrategyKind
from ..masking.applicator import StrategyApplicator
from .exceptions import PluginExecutionError
from .strategies.registry import StrategyPluginRegistry

logger = logging.getLogger(__name__)


class PluginAwareStrategyApplicator(StrategyApplicator):
    """
    Extended StrategyApplicator that supports custom strategy plugins.

    This class extends the base StrategyApplicator to handle custom
    plugin strategies while maintaining full backward compatibility.
    """

    def __init__(
        self,
        seed: Optional[str] = None,
        strategy_registry: Optional[StrategyPluginRegistry] = None
    ) -> None:
        """
        Initialize the plugin-aware strategy applicator.

        Args:
            seed: Optional seed for deterministic random generation
            strategy_registry: Strategy plugin registry to use
        """
        super().__init__(seed)
        self.strategy_registry = strategy_registry or StrategyPluginRegistry()

        logger.debug("PluginAwareStrategyApplicator initialized")

    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float,
    ) -> str:
        """
        Apply strategy with plugin support.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity
            strategy: The masking strategy to apply
            confidence: Detection confidence score

        Returns:
            The masked replacement text
        """
        # Check if this is a plugin strategy
        if strategy.kind == StrategyKind.CUSTOM:
            plugin_name = strategy.get_parameter("plugin_name")

            if plugin_name:
                return self._apply_plugin_strategy(
                    original_text, entity_type, strategy, confidence
                )

        # Fall back to base implementation for non-plugin strategies
        return super().apply_strategy(original_text, entity_type, strategy, confidence)

    def _apply_plugin_strategy(
        self,
        original_text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float,
    ) -> str:
        """
        Apply a custom plugin strategy.

        Args:
            original_text: Original PII text to mask
            entity_type: Type of entity
            strategy: Plugin strategy configuration
            confidence: Detection confidence

        Returns:
            Masked text from plugin
        """
        plugin_name = strategy.get_parameter("plugin_name")
        plugin_config = strategy.get_parameter("plugin_config", {})
        strategy.get_parameter("fallback_strategy")

        if not plugin_name:
            error_msg = "Plugin strategy missing plugin_name parameter"
            logger.error(error_msg)
            return self._apply_fallback_strategy(
                original_text, entity_type, strategy, confidence, error_msg
            )

        try:
            # Apply the plugin strategy
            result = self.strategy_registry.apply_strategy(
                plugin_name=plugin_name,
                original_text=original_text,
                entity_type=entity_type,
                confidence=confidence,
                context={"plugin_config": plugin_config}
            )

            if result.success:
                logger.debug(
                    f"Plugin strategy {plugin_name} applied successfully "
                    f"in {result.execution_time_ms:.1f}ms"
                )
                return result.masked_text
            else:
                error_msg = f"Plugin strategy {plugin_name} failed: {result.error_message}"
                logger.warning(error_msg)
                return self._apply_fallback_strategy(
                    original_text, entity_type, strategy, confidence, error_msg
                )

        except PluginExecutionError as e:
            error_msg = f"Plugin strategy {plugin_name} execution error: {e}"
            logger.error(error_msg)
            return self._apply_fallback_strategy(
                original_text, entity_type, strategy, confidence, error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error in plugin strategy {plugin_name}: {e}"
            logger.error(error_msg)
            return self._apply_fallback_strategy(
                original_text, entity_type, strategy, confidence, error_msg
            )

    def _apply_fallback_strategy(
        self,
        original_text: str,
        entity_type: str,
        primary_strategy: Strategy,
        confidence: float,
        error_msg: str,
    ) -> str:
        """
        Apply fallback strategy when plugin fails.

        Args:
            original_text: Original text
            entity_type: Entity type
            primary_strategy: Primary strategy to apply
            confidence: Confidence score
            error_msg: Error message from failed plugin

        Returns:
            Fallback masked text
        """
        try:
            logger.info(f"Applying fallback strategy for {entity_type}: {error_msg}")
            return super().apply_strategy(
                original_text, entity_type, primary_strategy, confidence
            )
        except Exception as e:
                logger.error(f"Fallback strategy also failed: {e}")

        # Ultimate fallback - simple redaction
        logger.warning(f"Using ultimate fallback (redaction) for {entity_type}")
        return "*" * len(original_text)

    def register_strategy_plugin_with_config(
        self,
        plugin: Any,
        config: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Register a strategy plugin with configuration.

        Args:
            plugin: Strategy plugin instance
            config: Optional plugin configuration
        """
        self.strategy_registry.register_strategy_plugin(plugin, config)
        logger.info(f"Registered strategy plugin: {plugin.info.name}")

    def list_available_strategy_plugins(self) -> list[str]:
        """Get list of available strategy plugin names."""
        return self.strategy_registry.list_strategy_plugins()

    def supports_entity_type_with_plugin(
        self,
        plugin_name: str,
        entity_type: str
    ) -> bool:
        """
        Check if a plugin supports a given entity type.

        Args:
            plugin_name: Name of strategy plugin
            entity_type: Entity type to check

        Returns:
            True if plugin supports the entity type
        """
        return self.strategy_registry.supports_entity_type(plugin_name, entity_type)

    def get_plugin_parameters_schema(self, plugin_name: str) -> Optional[dict[str, Any]]:
        """
        Get parameter schema for a strategy plugin.

        Args:
            plugin_name: Name of strategy plugin

        Returns:
            JSON schema for plugin parameters
        """
        return self.strategy_registry.get_plugin_parameters_schema(plugin_name)

    def get_strategy_registry_status(self) -> dict[str, Any]:
        """Get status of the strategy plugin registry."""
        return self.strategy_registry.get_strategy_registry_status()

    def cleanup_strategy_plugins(self) -> None:
        """Clean up all strategy plugins."""
        self.strategy_registry.cleanup_all_strategies()
        logger.info("All strategy plugins cleaned up")
