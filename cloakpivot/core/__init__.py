"""Core functionality for CloakPivot masking and unmasking operations."""

# Strategy system
# Presidio integration system
from .analyzer import (
    AnalyzerConfig,
    AnalyzerEngineWrapper,
    EntityDetectionResult,
    RecognizerRegistry,
)

# Anchor system
from .anchors import AnchorEntry, AnchorIndex

# CloakMap system
from .cloakmap import CloakMap, merge_cloakmaps, validate_cloakmap_integrity

# Entity detection pipeline
from .detection import (
    DocumentAnalysisResult,
    EntityDetectionPipeline,
    SegmentAnalysisResult,
)

# Entity normalization
from .normalization import (
    ConflictResolutionConfig,
    ConflictResolutionStrategy,
    EntityNormalizer,
    NormalizationResult,
)

# Error handling system
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
    create_dependency_error,
    create_processing_error,
    create_validation_error,
)
from .error_handling import (
    CircuitBreaker,
    ErrorCollector,
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
    CloakMapValidator,
    DocumentValidator,
    InputValidator,
    PolicyValidator,
    SystemValidator,
    validate_for_masking,
    validate_for_unmasking,
)

# Policy system
from .policies import (
    CONSERVATIVE_POLICY,
    PARTIAL_POLICY,
    TEMPLATE_POLICY,
    MaskingPolicy,
)

# Result system
from .results import (
    BatchResult,
    DiagnosticInfo,
    MaskResult,
    OperationStatus,
    PerformanceMetrics,
    ProcessingStats,
    UnmaskResult,
    create_diagnostics,
    create_performance_metrics,
    create_processing_stats,
)
from .strategies import (
    DEFAULT_REDACT,
    EMAIL_TEMPLATE,
    HASH_SHA256,
    PHONE_TEMPLATE,
    SSN_PARTIAL,
    Strategy,
    StrategyKind,
)

__all__ = [
    # Strategy exports
    "StrategyKind",
    "Strategy",
    "DEFAULT_REDACT",
    "PHONE_TEMPLATE",
    "EMAIL_TEMPLATE",
    "SSN_PARTIAL",
    "HASH_SHA256",
    # Policy exports
    "MaskingPolicy",
    "CONSERVATIVE_POLICY",
    "TEMPLATE_POLICY",
    "PARTIAL_POLICY",
    # Anchor exports
    "AnchorEntry",
    "AnchorIndex",
    # CloakMap exports
    "CloakMap",
    "merge_cloakmaps",
    "validate_cloakmap_integrity",
    # Result exports
    "OperationStatus",
    "ProcessingStats",
    "PerformanceMetrics",
    "DiagnosticInfo",
    "MaskResult",
    "UnmaskResult",
    "BatchResult",
    "create_performance_metrics",
    "create_processing_stats",
    "create_diagnostics",
    # Presidio integration exports
    "AnalyzerEngineWrapper",
    "AnalyzerConfig",
    "RecognizerRegistry",
    "EntityDetectionResult",
    # Detection pipeline exports
    "EntityDetectionPipeline",
    "DocumentAnalysisResult",
    "SegmentAnalysisResult",
    # Normalization exports
    "EntityNormalizer",
    "ConflictResolutionStrategy",
    "ConflictResolutionConfig",
    "NormalizationResult",
    # Error handling exports
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
    "create_validation_error",
    "create_processing_error",
    "create_dependency_error",
    "PartialFailureManager",
    "ErrorCollector",
    "FailureToleranceConfig",
    "FailureToleranceLevel",
    "CircuitBreaker",
    "RetryManager",
    "create_partial_failure_manager",
    "with_error_isolation",
    "with_circuit_breaker",
    "with_retry",
    "InputValidator",
    "SystemValidator",
    "DocumentValidator",
    "PolicyValidator",
    "CloakMapValidator",
    "validate_for_masking",
    "validate_for_unmasking",
]
