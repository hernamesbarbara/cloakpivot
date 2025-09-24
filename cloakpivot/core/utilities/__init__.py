"""Utility functions and helpers for CloakPivot core."""

# Import specific utility functionality for backward compatibility
from .cloakmap_serializer import CloakMapSerializer
from .cloakmap_validator import CloakMapValidator, merge_cloakmaps, validate_cloakmap_integrity
from .config import PerformanceConfig, get_performance_config, reset_performance_config
from .error_handling import (
    CircuitBreaker,
    ErrorCollector,
    ErrorRecord,
    FailureToleranceConfig,
    FailureToleranceLevel,
    PartialFailureManager,
    RetryManager,
    create_partial_failure_manager,
    with_circuit_breaker,
    with_error_isolation,
    with_retry,
)
from .validation import (
    DocumentValidator,
    InputValidator,
    PolicyValidator,
    SystemValidator,
    validate_for_masking,
    validate_for_unmasking,
)

__all__ = [
    # From config.py
    "PerformanceConfig",
    "get_performance_config",
    "reset_performance_config",

    # From error_handling.py
    "FailureToleranceLevel",
    "FailureToleranceConfig",
    "ErrorRecord",
    "ErrorCollector",
    "PartialFailureManager",
    "CircuitBreaker",
    "RetryManager",
    "create_partial_failure_manager",
    "with_error_isolation",
    "with_retry",
    "with_circuit_breaker",

    # From cloakmap_serializer.py
    "CloakMapSerializer",

    # From cloakmap_validator.py
    "CloakMapValidator",
    "validate_cloakmap_integrity",
    "merge_cloakmaps",

    # From validation.py
    "SystemValidator",
    "DocumentValidator",
    "PolicyValidator",
    "InputValidator",
    "validate_for_masking",
    "validate_for_unmasking",
]
