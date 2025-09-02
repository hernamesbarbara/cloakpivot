"""Centralized type definitions to avoid import issues."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docling_core.types import DoclingDocument  # type: ignore[attr-defined]
else:
    try:
        from docling_core.types import DoclingDocument  # type: ignore[attr-defined]
    except ImportError:
        DoclingDocument = Any  # type: ignore[misc,assignment]

__all__ = ["DoclingDocument"]
