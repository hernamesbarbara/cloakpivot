"""Format support and serialization for CloakPivot.

This module provides enhanced format support leveraging docpivot's SerializerProvider
with CloakPivot-specific features like format detection, validation, and
masking-aware serialization.
"""

from .registry import FormatRegistry, SupportedFormat
from .serialization import (
    CloakPivotSerializer,
    SerializationError,
    SerializationResult,
)

__all__ = [
    "FormatRegistry",
    "CloakPivotSerializer",
    "SerializationError",
    "SerializationResult",
    "SupportedFormat",
]
