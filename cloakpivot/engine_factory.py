"""Factory for creating masking and unmasking engines.

This module provides a factory pattern for creating engine instances,
enabling dependency injection and better testability.
"""

import logging
from typing import Any

from .masking.engine import MaskingEngine
from .unmasking.engine import UnmaskingEngine

logger = logging.getLogger(__name__)


class EngineFactory:
    """Factory for creating masking and unmasking engines.

    This factory provides centralized engine creation with support for
    dependency injection and configuration management.

    Examples:
        >>> factory = EngineFactory()
        >>> masking_engine = factory.create_masking_engine()
        >>> unmasking_engine = factory.create_unmasking_engine()
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the engine factory.

        Args:
            config: Optional configuration dictionary for engine creation
        """
        self.config = config or {}
        logger.debug(f"EngineFactory initialized with config: {self.config}")

    def create_masking_engine(self, **kwargs: Any) -> MaskingEngine:
        """Create a new MaskingEngine instance.

        Args:
            **kwargs: Additional arguments to pass to MaskingEngine constructor

        Returns:
            Configured MaskingEngine instance
        """
        # Merge factory config with provided kwargs
        engine_config = {**self.config, **kwargs}

        logger.info("Creating MaskingEngine instance")
        engine = MaskingEngine(**engine_config)

        # Apply any additional factory-level configuration
        self._configure_masking_engine(engine)

        return engine

    def create_unmasking_engine(self, **kwargs: Any) -> UnmaskingEngine:
        """Create a new UnmaskingEngine instance.

        Args:
            **kwargs: Additional arguments to pass to UnmaskingEngine constructor

        Returns:
            Configured UnmaskingEngine instance
        """
        # Merge factory config with provided kwargs
        engine_config = {**self.config, **kwargs}

        logger.info("Creating UnmaskingEngine instance")
        engine = UnmaskingEngine(**engine_config)

        # Apply any additional factory-level configuration
        self._configure_unmasking_engine(engine)

        return engine

    def create_engine_pair(self, **kwargs: Any) -> tuple[MaskingEngine, UnmaskingEngine]:
        """Create a matched pair of masking and unmasking engines.

        This ensures both engines are configured consistently for round-trip
        masking/unmasking operations.

        Args:
            **kwargs: Additional arguments for engine configuration

        Returns:
            Tuple of (MaskingEngine, UnmaskingEngine)
        """
        logger.info("Creating matched engine pair")

        # Ensure consistent configuration for both engines
        masking_engine = self.create_masking_engine(**kwargs)
        unmasking_engine = self.create_unmasking_engine(**kwargs)

        return masking_engine, unmasking_engine

    def _configure_masking_engine(self, engine: MaskingEngine) -> None:
        """Apply factory-specific configuration to a masking engine.

        Args:
            engine: MaskingEngine to configure
        """
        # Add any factory-level configuration here
        # For example, setting default adapters, policies, etc.
        if "default_policy" in self.config:
            logger.debug(f"Setting default policy: {self.config['default_policy']}")
            # engine.set_default_policy(self.config['default_policy'])

    def _configure_unmasking_engine(self, engine: UnmaskingEngine) -> None:
        """Apply factory-specific configuration to an unmasking engine.

        Args:
            engine: UnmaskingEngine to configure
        """
        # Add any factory-level configuration here
        # For example, setting default options, verification settings, etc.
        if "verify_integrity" in self.config:
            logger.debug(f"Setting integrity verification: {self.config['verify_integrity']}")
            # engine.set_verify_integrity(self.config['verify_integrity'])


# Singleton instance for convenience
_default_factory = None


def get_default_factory() -> EngineFactory:
    """Get the default engine factory instance.

    Returns:
        Default EngineFactory instance (singleton)
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = EngineFactory()
    return _default_factory


def create_masking_engine(**kwargs: Any) -> MaskingEngine:
    """Convenience function to create a masking engine using the default factory.

    Args:
        **kwargs: Arguments to pass to engine creation

    Returns:
        MaskingEngine instance
    """
    return get_default_factory().create_masking_engine(**kwargs)


def create_unmasking_engine(**kwargs: Any) -> UnmaskingEngine:
    """Convenience function to create an unmasking engine using the default factory.

    Args:
        **kwargs: Arguments to pass to engine creation

    Returns:
        UnmaskingEngine instance
    """
    return get_default_factory().create_unmasking_engine(**kwargs)