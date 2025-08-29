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
]
