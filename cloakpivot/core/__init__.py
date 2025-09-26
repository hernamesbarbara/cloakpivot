"""Core functionality for CloakPivot masking and unmasking operations."""

from typing import Any

# Import modules as submodules for test compatibility
from . import policies
from .policies import policy_loader

# Essential imports for backward compatibility - only import what we know exists and is needed
# Core data structures
# Essential policies
from .policies.policies import CONSERVATIVE_POLICY, PARTIAL_POLICY, TEMPLATE_POLICY, MaskingPolicy
from .processing import (
    analyzer,
    cloakmap_enhancer,
    detection,
    normalization,
    presidio_mapper,
    surrogate,
)

# Essential processing
from .processing.surrogate import SurrogateGenerator
from .types import (
    anchors,
    cloakmap,
    model_info,  # Need to expose this directly
    results,
    strategies,
)
from .types.anchors import AnchorEntry, AnchorIndex
from .types.cloakmap import CloakMap

# Results and other types needed for imports
from .types.results import (
    BatchResult,
    DiagnosticInfo,
    MaskResult,
    OperationStatus,
    PerformanceMetrics,
    ProcessingStats,
    UnmaskResult,
)
from .types.strategies import (
    DEFAULT_REDACT,
    EMAIL_TEMPLATE,
    HASH_SHA256,
    PHONE_TEMPLATE,
    SSN_PARTIAL,
    Strategy,
    StrategyKind,
)
from .utilities import validation

# Essential utilities
from .utilities.cloakmap_serializer import CloakMapSerializer
from .utilities.cloakmap_validator import CloakMapValidator


# Validation functions
def merge_cloakmaps(*cloakmaps: Any) -> None:
    """Placeholder for merge_cloakmaps function."""
    pass


def validate_cloakmap_integrity(cloakmap: Any) -> bool:
    """Placeholder for validate_cloakmap_integrity function."""
    return True


# Export the essentials for backward compatibility
__all__ = [
    # Modules
    "analyzer",
    "anchors",
    "cloakmap",
    "cloakmap_enhancer",
    "detection",
    "model_info",
    "normalization",
    "policies",
    "policy_loader",
    "presidio_mapper",
    "results",
    "strategies",
    "surrogate",
    "validation",
    # Classes and functions
    "Strategy",
    "StrategyKind",
    "DEFAULT_REDACT",
    "PHONE_TEMPLATE",
    "EMAIL_TEMPLATE",
    "SSN_PARTIAL",
    "HASH_SHA256",
    "AnchorEntry",
    "AnchorIndex",
    "CloakMap",
    "MaskingPolicy",
    "CONSERVATIVE_POLICY",
    "TEMPLATE_POLICY",
    "PARTIAL_POLICY",
    "SurrogateGenerator",
    "CloakMapSerializer",
    "CloakMapValidator",
    "BatchResult",
    "MaskResult",
    "UnmaskResult",
    "OperationStatus",
    "PerformanceMetrics",
    "ProcessingStats",
    "DiagnosticInfo",
    "merge_cloakmaps",
    "validate_cloakmap_integrity",
]
