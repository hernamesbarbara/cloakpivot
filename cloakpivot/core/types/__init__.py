"""Type definitions and data structures for CloakPivot core."""

# Import core type definitions
from .types import DoclingDocument, UnmaskingResult
from .anchors import AnchorEntry, AnchorIndex
from .strategies import Strategy, StrategyKind, DEFAULT_REDACT, PHONE_TEMPLATE, EMAIL_TEMPLATE, SSN_PARTIAL
from .results import *
from .exceptions import *
from .cloakmap import CloakMap
from .model_info import get_model_name, validate_model_availability, get_supported_languages

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

    # From results.py (imported via *)
    # From exceptions.py (imported via *)

    # From cloakmap.py
    "CloakMap",

    # From model_info.py
    "get_model_name",
    "validate_model_availability",
    "get_supported_languages",
]