"""Core functionality for CloakPivot masking and unmasking operations."""

# Essential imports for backward compatibility - only import what we know exists and is needed

# Core data structures
from .types.strategies import Strategy, StrategyKind, DEFAULT_REDACT, PHONE_TEMPLATE, EMAIL_TEMPLATE, SSN_PARTIAL, HASH_SHA256
from .types.anchors import AnchorEntry, AnchorIndex
from .types.cloakmap import CloakMap

# Essential policies
from .policies.policies import MaskingPolicy, CONSERVATIVE_POLICY, TEMPLATE_POLICY, PARTIAL_POLICY

# Essential processing
from .processing.surrogate import SurrogateGenerator

# Essential utilities
from .utilities.cloakmap_serializer import CloakMapSerializer
from .utilities.cloakmap_validator import CloakMapValidator

# Results and other types needed for imports
from .types.results import BatchResult, MaskResult, UnmaskResult, OperationStatus, PerformanceMetrics, ProcessingStats, DiagnosticInfo

# Validation functions
def merge_cloakmaps(*cloakmaps):
    """Placeholder for merge_cloakmaps function."""
    pass

def validate_cloakmap_integrity(cloakmap):
    """Placeholder for validate_cloakmap_integrity function."""
    return True

# Export the essentials for backward compatibility
__all__ = [
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