"""Core functionality for CloakPivot masking and unmasking operations."""

# Strategy system
from .strategies import (
    StrategyKind,
    Strategy,
    DEFAULT_REDACT,
    PHONE_TEMPLATE,
    EMAIL_TEMPLATE,
    SSN_PARTIAL,
    HASH_SHA256
)

# Policy system
from .policies import (
    MaskingPolicy,
    CONSERVATIVE_POLICY,
    TEMPLATE_POLICY,
    PARTIAL_POLICY
)

# Anchor system
from .anchors import (
    AnchorEntry,
    AnchorIndex
)

# CloakMap system
from .cloakmap import (
    CloakMap,
    merge_cloakmaps,
    validate_cloakmap_integrity
)

# Result system
from .results import (
    OperationStatus,
    ProcessingStats,
    PerformanceMetrics,
    DiagnosticInfo,
    MaskResult,
    UnmaskResult,
    BatchResult,
    create_performance_metrics,
    create_processing_stats,
    create_diagnostics
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
]
