"""CloakPivot: PII masking/unmasking on top of DocPivot and Presidio.

CloakPivot enables reversible document masking while preserving structure
and formatting using DocPivot for document processing and Presidio for
PII detection and anonymization.
"""

__version__ = "0.1.0"
__author__ = "CloakPivot Team"
__email__ = "contact@example.com"

# Core API exports
from .core import (
    CONSERVATIVE_POLICY,
    DEFAULT_REDACT,
    EMAIL_TEMPLATE,
    HASH_SHA256,
    PARTIAL_POLICY,
    PHONE_TEMPLATE,
    SSN_PARTIAL,
    TEMPLATE_POLICY,
    # Anchor system
    AnchorEntry,
    AnchorIndex,
    BatchResult,
    # CloakMap system
    CloakMap,
    DiagnosticInfo,
    # Policy system
    MaskingPolicy,
    MaskResult,
    # Result system
    OperationStatus,
    PerformanceMetrics,
    ProcessingStats,
    Strategy,
    # Strategy system
    StrategyKind,
    UnmaskResult,
    merge_cloakmaps,
    validate_cloakmap_integrity,
)

# Document processing system
from .document import (
    AnchorMapper,
    DocumentProcessor,
    NodeReference,
    TextExtractor,
    TextSegment,
)

# Masking system
from .masking import DocumentMasker, MaskingEngine, MaskingResult, StrategyApplicator

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Strategy system
    "StrategyKind",
    "Strategy",
    "DEFAULT_REDACT",
    "PHONE_TEMPLATE",
    "EMAIL_TEMPLATE",
    "SSN_PARTIAL",
    "HASH_SHA256",
    # Policy system
    "MaskingPolicy",
    "CONSERVATIVE_POLICY",
    "TEMPLATE_POLICY",
    "PARTIAL_POLICY",
    # Anchor system
    "AnchorEntry",
    "AnchorIndex",
    # CloakMap system
    "CloakMap",
    "merge_cloakmaps",
    "validate_cloakmap_integrity",
    # Result system
    "OperationStatus",
    "ProcessingStats",
    "PerformanceMetrics",
    "DiagnosticInfo",
    "MaskResult",
    "UnmaskResult",
    "BatchResult",
    # Document processing
    "DocumentProcessor",
    "TextExtractor",
    "TextSegment",
    "AnchorMapper",
    "NodeReference",
    # Masking system
    "MaskingEngine",
    "MaskingResult",
    "StrategyApplicator",
    "DocumentMasker",
]
