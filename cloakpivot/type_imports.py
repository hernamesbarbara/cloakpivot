"""Centralized type imports for consistent usage across the codebase.

This module provides a single place to import commonly used types,
following the same import patterns as the upstream libraries themselves.
"""

from typing import TYPE_CHECKING, Any

# Use TYPE_CHECKING to handle optional imports cleanly
if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument
else:
    try:
        from docling_core.types.doc.document import DoclingDocument
    except ImportError:
        # Fallback for when docling_core is not installed
        DoclingDocument = Any

# Export the properly typed version
__all__ = ["DoclingDocument"]
