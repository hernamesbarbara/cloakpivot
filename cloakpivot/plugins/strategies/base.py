"""Base class for custom masking strategy plugins."""

import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from ..base import BasePlugin, PluginInfo
from ..exceptions import PluginExecutionError, PluginValidationError


@dataclass
class StrategyPluginResult:
    """Result from a strategy plugin execution."""

    masked_text: str
    execution_time_ms: float
    metadata: dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class BaseStrategyPlugin(BasePlugin):
    """
    Base class for custom masking strategy plugins.

    Strategy plugins implement custom masking logic that can be used
    in place of or alongside the built-in masking strategies.

    Example:
        >>> class ROT13StrategyPlugin(BaseStrategyPlugin):
        ...     @property
        ...     def info(self) -> PluginInfo:
        ...         return PluginInfo(
        ...             name="rot13_strategy",
        ...             version="1.0.0",
        ...             description="ROT13 text transformation strategy",
        ...             author="CloakPivot Team",
        ...             plugin_type="strategy"
        ...         )
        ...
        ...     def apply_strategy(self, original_text, entity_type, confidence):
        ...         # Implement ROT13 transformation
        ...         result = original_text.translate(str.maketrans(
        ...             "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        ...             "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijkl"
        ...         ))
        ...         return StrategyPluginResult(
        ...             masked_text=result,
        ...             execution_time_ms=0.1,
        ...             metadata={"algorithm": "rot13"}
        ...         )
    """

    @property
    def info(self) -> PluginInfo:
        """Get plugin information - must be implemented by subclasses."""
        return PluginInfo(
            name=self.__class__.__name__.lower(),
            version="1.0.0",
            description="Custom strategy plugin",
            author="Unknown",
            plugin_type="strategy",
        )

    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        Validate strategy plugin configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if valid

        Raises:
            PluginValidationError: If configuration is invalid
        """
        # Base validation - check for common configuration errors
        if not isinstance(config, dict):
            raise PluginValidationError(
                "Configuration must be a dictionary", plugin_name=self.info.name
            )

        # Validate any strategy-specific parameters
        return self._validate_strategy_config(config)

    def _validate_strategy_config(self, config: dict[str, Any]) -> bool:
        """
        Override this method to add strategy-specific validation.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid

        Raises:
            PluginValidationError: If configuration is invalid
        """
        return True

    def initialize(self) -> None:
        """Initialize the strategy plugin."""
        try:
            self._initialize_strategy()
            self.is_initialized = True
            self.logger.info(
                f"Strategy plugin {self.info.name} initialized successfully"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to initialize strategy plugin {self.info.name}: {e}"
            )
            raise PluginExecutionError(
                f"Strategy plugin initialization failed: {e}",
                plugin_name=self.info.name,
                original_exception=e,
            ) from e

    def _initialize_strategy(self) -> None:
        """
        Override this method to add strategy-specific initialization.

        This is called during initialize() and should set up any
        resources needed for the strategy to function.
        """
        pass

    def cleanup(self) -> None:
        """Clean up strategy plugin resources."""
        try:
            self._cleanup_strategy()
            self.is_initialized = False
            self.logger.info(
                f"Strategy plugin {self.info.name} cleaned up successfully"
            )
        except Exception as e:
            self.logger.warning(f"Error during strategy plugin cleanup: {e}")

    def _cleanup_strategy(self) -> None:
        """
        Override this method to add strategy-specific cleanup.

        This is called during cleanup() and should release any
        resources allocated during initialization.
        """
        pass

    @abstractmethod
    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        confidence: float,
        context: Optional[dict[str, Any]] = None,
    ) -> StrategyPluginResult:
        """
        Apply the custom masking strategy to the input text.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity (e.g., 'PHONE_NUMBER', 'EMAIL_ADDRESS')
            confidence: Detection confidence score (0.0-1.0)
            context: Optional contextual information (document structure, etc.)

        Returns:
            StrategyPluginResult containing the masked text and metadata

        Raises:
            PluginExecutionError: If strategy application fails
        """
        pass

    def apply_strategy_safe(
        self,
        original_text: str,
        entity_type: str,
        confidence: float,
        context: Optional[dict[str, Any]] = None,
    ) -> StrategyPluginResult:
        """
        Apply strategy with error handling and timing.

        This is the main entry point used by the plugin registry.
        It wraps apply_strategy() with error handling and performance monitoring.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity
            confidence: Detection confidence score
            context: Optional contextual information

        Returns:
            StrategyPluginResult with success/failure information
        """
        if not self.is_initialized:
            return StrategyPluginResult(
                masked_text=original_text,
                execution_time_ms=0.0,
                metadata={},
                success=False,
                error_message="Plugin not initialized",
            )

        start_time = time.perf_counter()

        try:
            result = self.apply_strategy(
                original_text, entity_type, confidence, context
            )

            # Ensure result is valid
            if not isinstance(result, StrategyPluginResult):
                raise PluginExecutionError(
                    "Strategy plugin must return StrategyPluginResult",
                    plugin_name=self.info.name,
                )

            # Update timing
            end_time = time.perf_counter()
            result.execution_time_ms = (end_time - start_time) * 1000

            return result

        except Exception as e:
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000

            self.logger.error(
                f"Strategy plugin {self.info.name} failed: {e}", exc_info=True
            )

            return StrategyPluginResult(
                masked_text=original_text,  # Fallback to original text
                execution_time_ms=execution_time,
                metadata={"error_type": type(e).__name__},
                success=False,
                error_message=str(e),
            )

    def get_supported_entity_types(self) -> Optional[list[str]]:
        """
        Get list of entity types this strategy supports.

        Returns:
            List of supported entity types, or None if supports all types
        """
        return None  # Default: supports all entity types

    def get_strategy_parameters_schema(self) -> dict[str, Any]:
        """
        Get JSON schema for strategy parameters.

        Returns:
            JSON schema dictionary describing accepted parameters
        """
        return {"type": "object", "properties": {}, "additionalProperties": True}
