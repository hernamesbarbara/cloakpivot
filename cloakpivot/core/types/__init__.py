"""Type definitions and data structures for CloakPivot core."""

# Import core type definitions
from .anchors import AnchorEntry, AnchorIndex
from .cloakmap import CloakMap
from .exceptions import (
    CloakPivotError,
    ConfigurationError,
    DependencyError,
    DetectionError,
    IntegrityError,
    MaskingError,
    PartialProcessingError,
    PolicyError,
    ProcessingError,
    UnmaskingError,
    ValidationError,
)
from .model_info import get_model_name, get_supported_languages, validate_model_availability
from .results import (
    BatchResult,
    DiagnosticInfo,
    MaskResult,
    OperationStatus,
    PerformanceMetrics,
    ProcessingStats,
    UnmaskResult,
)
from .strategies import (
    DEFAULT_REDACT,
    EMAIL_TEMPLATE,
    PHONE_TEMPLATE,
    SSN_PARTIAL,
    Strategy,
    StrategyKind,
)
from .types import DoclingDocument, UnmaskingResult

__all__ = [
    # From types.py
    "DoclingDocument",
    "UnmaskingResult",

    # From anchors.py
    "AnchorEntry",
    "AnchorIndex",

    # From strategies.py
    "Strategy",
    "StrategyKind",
    "DEFAULT_REDACT",
    "PHONE_TEMPLATE",
    "EMAIL_TEMPLATE",
    "SSN_PARTIAL",

    # From results.py
    "OperationStatus",
    "ProcessingStats",
    "PerformanceMetrics",
    "DiagnosticInfo",
    "MaskResult",
    "UnmaskResult",
    "BatchResult",

    # From exceptions.py
    "CloakPivotError",
    "ValidationError",
    "ProcessingError",
    "DetectionError",
    "MaskingError",
    "UnmaskingError",
    "PolicyError",
    "IntegrityError",
    "PartialProcessingError",
    "ConfigurationError",
    "DependencyError",

    # From cloakmap.py
    "CloakMap",

    # From model_info.py
    "get_model_name",
    "validate_model_availability",
    "get_supported_languages",
]
