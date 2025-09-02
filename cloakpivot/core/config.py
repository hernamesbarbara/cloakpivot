"""Runtime performance configuration from environment variables.

This module provides centralized configuration management for CloakPivot performance
settings through environment variables. It supports model size selection, singleton
behavior control, caching configuration, and performance tuning options.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Runtime performance configuration from environment variables.

    This dataclass encapsulates all performance-related configuration options
    that can be controlled through environment variables, providing a single
    source of truth for performance settings across the application.

    Attributes:
        model_size: Size of spaCy models to use (small|medium|large)
        use_singleton_analyzers: Whether to use singleton analyzer pattern
        analyzer_cache_size: Size of LRU cache for analyzers
        enable_parallel_processing: Whether to enable parallel processing
        max_worker_threads: Maximum number of worker threads (None for auto)
        enable_memory_optimization: Whether to enable memory optimization
        gc_frequency: Number of operations between garbage collection runs
    """

    # Model configuration
    model_size: str = "small"  # small|medium|large

    # Singleton behavior
    use_singleton_analyzers: bool = True
    analyzer_cache_size: int = 8

    # Performance tuning
    enable_parallel_processing: bool = True
    max_worker_threads: Optional[int] = None

    # Memory optimization
    enable_memory_optimization: bool = True
    gc_frequency: int = 100  # operations between garbage collection

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_model_size()
        self._validate_cache_size()
        self._validate_worker_threads()
        self._validate_gc_frequency()

        logger.debug(
            f"PerformanceConfig initialized: model_size={self.model_size}, "
            f"cache_size={self.analyzer_cache_size}, use_singleton={self.use_singleton_analyzers}"
        )

    def _validate_model_size(self) -> None:
        """Validate and normalize model_size."""

        self.model_size = self.model_size.lower()
        valid_sizes = {"small", "medium", "large"}

        if self.model_size not in valid_sizes:
            logger.warning(
                f"Invalid model_size '{self.model_size}', using 'small'. Valid: {valid_sizes}"
            )
            self.model_size = "small"

    def _validate_cache_size(self) -> None:
        """Validate and normalize analyzer_cache_size."""
        if (
            not isinstance(self.analyzer_cache_size, int)
            or self.analyzer_cache_size <= 0
        ):
            logger.warning(
                f"analyzer_cache_size must be positive integer, got "
                f"{self.analyzer_cache_size}, using 8"
            )
            self.analyzer_cache_size = 8

    def _validate_worker_threads(self) -> None:
        """Validate max_worker_threads."""
        if self.max_worker_threads is not None:
            if (
                not isinstance(self.max_worker_threads, int)
                or self.max_worker_threads <= 0
            ):
                logger.warning(
                    f"max_worker_threads must be positive integer or None, got "
                    f"{self.max_worker_threads}, using None"
                )
                self.max_worker_threads = None

    def _validate_gc_frequency(self) -> None:
        """Validate gc_frequency."""
        if not isinstance(self.gc_frequency, int) or self.gc_frequency <= 0:
            logger.warning(
                f"gc_frequency must be positive integer, got {self.gc_frequency}, using 100"
            )
            self.gc_frequency = 100

    @classmethod
    def from_environment(cls) -> "PerformanceConfig":
        """Load configuration from environment variables.

        Loads configuration values from environment variables with fallback
        to safe defaults. All environment variable parsing is tolerant of
        invalid values and will log warnings while falling back to defaults.

        Environment Variables:
            MODEL_SIZE: Model size selection (small|medium|large)
            CLOAKPIVOT_USE_SINGLETON: Enable singleton analyzers (true|false)
            ANALYZER_CACHE_SIZE: LRU cache size for analyzers (positive integer)
            ENABLE_PARALLEL: Enable parallel processing (true|false)
            MAX_WORKERS: Maximum worker threads (positive integer)
            MEMORY_OPTIMIZATION: Enable memory optimization (true|false)
            GC_FREQUENCY: Garbage collection frequency (positive integer)

        Returns:
            PerformanceConfig instance with values from environment or defaults
        """
        try:
            # Model configuration
            model_size = cls._get_env_string("MODEL_SIZE", "small").lower()

            # Singleton behavior
            use_singleton = cls._get_env_bool("CLOAKPIVOT_USE_SINGLETON", True)
            cache_size = cls._get_env_int("ANALYZER_CACHE_SIZE", 8) or 8

            # Performance tuning
            enable_parallel = cls._get_env_bool("ENABLE_PARALLEL", True)
            max_workers = cls._get_env_int("MAX_WORKERS", None, allow_none=True)

            # Memory optimization
            enable_memory_opt = cls._get_env_bool("MEMORY_OPTIMIZATION", True)
            gc_frequency = cls._get_env_int("GC_FREQUENCY", 100) or 100

            config = cls(
                model_size=model_size,
                use_singleton_analyzers=use_singleton,
                analyzer_cache_size=cache_size,
                enable_parallel_processing=enable_parallel,
                max_worker_threads=max_workers,
                enable_memory_optimization=enable_memory_opt,
                gc_frequency=gc_frequency,
            )

            logger.info(f"Loaded configuration from environment: {config}")
            return config

        except Exception as e:
            logger.error(
                f"Error loading configuration from environment: {e}, using defaults"
            )
            return cls()

    @staticmethod
    def _get_env_string(key: str, default: str) -> str:
        """Get string value from environment with default fallback."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.strip()

    @staticmethod
    def _get_env_bool(key: str, default: bool) -> bool:
        """Get boolean value from environment with default fallback.

        Only 'true' (case insensitive) evaluates to True, everything else
        defaults to the provided default.
        """
        value = os.getenv(key)
        if value is None:
            return default

        cleaned_value = value.strip().lower()
        if cleaned_value == "true":
            return True
        elif cleaned_value == "false":
            return False
        else:
            # Invalid value, return default
            return default

    @staticmethod
    def _get_env_int(
        key: str, default: Optional[int], allow_none: bool = False
    ) -> Optional[int]:
        """Get integer value from environment with default fallback."""
        value = os.getenv(key)
        if value is None:
            return default

        try:
            parsed = int(value.strip())
            if parsed <= 0 and not allow_none:
                logger.warning(
                    f"Environment variable {key}={value} must be positive, using default {default}"
                )
                return default
            return parsed
        except ValueError:
            logger.warning(
                f"Environment variable {key}={value} is not a valid integer, "
                f"using default {default}"
            )
            return default

    def get_model_characteristics(self) -> dict[str, Any]:
        """Get performance characteristics for current model size.

        Returns:
            Dictionary with memory, performance, and accuracy information
        """
        from .model_info import MODEL_CHARACTERISTICS

        return MODEL_CHARACTERISTICS.get(
            self.model_size, MODEL_CHARACTERISTICS["small"]
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "model_size": self.model_size,
            "use_singleton_analyzers": self.use_singleton_analyzers,
            "analyzer_cache_size": self.analyzer_cache_size,
            "enable_parallel_processing": self.enable_parallel_processing,
            "max_worker_threads": self.max_worker_threads,
            "enable_memory_optimization": self.enable_memory_optimization,
            "gc_frequency": self.gc_frequency,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PerformanceConfig(model_size={self.model_size!r}, "
            f"use_singleton={self.use_singleton_analyzers}, "
            f"cache_size={self.analyzer_cache_size})"
        )


# Global configuration instance - loaded lazily to support testing
# This provides easy access throughout the application
_performance_config = None


def get_performance_config() -> PerformanceConfig:
    """Get the global performance configuration, creating it if needed."""
    global _performance_config
    if _performance_config is None:
        _performance_config = PerformanceConfig.from_environment()
    return _performance_config


def reset_performance_config() -> None:
    """Reset the global configuration for testing purposes."""
    global _performance_config
    _performance_config = None


# For backwards compatibility, provide the performance_config as a property-like access
class _ConfigProxy:
    """Proxy object that provides lazy access to performance config."""

    def __getattr__(self, name: str) -> Any:
        config = get_performance_config()
        return getattr(config, name)

    def __repr__(self) -> str:
        return repr(get_performance_config())


performance_config = _ConfigProxy()
