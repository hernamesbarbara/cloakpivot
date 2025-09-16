"""Centralized type imports for consistent usage across the codebase.

This module provides a single place to import commonly used types,
following the same import patterns as the upstream libraries themselves.
"""

from typing import TYPE_CHECKING, Any

# Use the most direct import path to avoid mypy issues
try:
    from docling_core.types.doc.document import DoclingDocument
except ImportError:
    # Fallback for when docling_core is not installed
    if not TYPE_CHECKING:
        DoclingDocument = Any  # type: ignore[misc]
    else:
        # For type checking, we need to provide something
        DoclingDocument = Any  # type: ignore[misc,assignment]

# Export the properly typed version
__all__ = ["DoclingDocument"]
