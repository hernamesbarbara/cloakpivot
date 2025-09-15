"""CloakPivot: PII masking/unmasking on top of DocPivot and Presidio.

CloakPivot enables reversible document masking while preserving structure
and formatting using DocPivot for document processing and Presidio for
PII detection and anonymization.

Supports DoclingDocument versions 1.2.0 through 1.7.0+, including proper
handling of v1.7.0's segment-local charspan changes.
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
from .defaults import (
    DEFAULT_ENTITIES,
    get_analyzer_preset,
    get_conservative_policy,
    get_default_analyzer_config,
    get_default_policy,
    get_permissive_policy,
    get_policy_preset,
)

# Document processing system
from .document import (
    AnchorMapper,
    DocumentProcessor,
    NodeReference,
    TextExtractor,
    TextSegment,
)

# New simplified API
from .engine import CloakEngine, MaskResult
from .engine_builder import CloakEngineBuilder

# Masking system
from .masking import MaskingEngine, MaskingResult, StrategyApplicator
from .registration import is_registered, register_cloak_methods, unregister_cloak_methods

# Unmasking system
from .unmasking import UnmaskingEngine
from .wrappers import CloakedDocument

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # New simplified API
    "CloakEngine",
    "CloakEngineBuilder",
    "register_cloak_methods",
    "unregister_cloak_methods",
    "is_registered",
    "CloakedDocument",
    "MaskResult",
    "DEFAULT_ENTITIES",
    "get_default_policy",
    "get_conservative_policy",
    "get_permissive_policy",
    "get_default_analyzer_config",
    "get_policy_preset",
    "get_analyzer_preset",
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
    # Unmasking system
    "UnmaskingEngine",
]
