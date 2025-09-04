"""Centralized type definitions to avoid import issues."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docling_core.types import DoclingDocument
else:
    try:
        from docling_core.types import DoclingDocument
    except ImportError:
        DoclingDocument = Any  # type: ignore[misc,assignment]

__all__ = ["DoclingDocument"]
