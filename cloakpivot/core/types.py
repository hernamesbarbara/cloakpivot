"""Centralized type definitions to avoid import issues."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from docling_core.types import DoclingDocument
    from .cloakmap import CloakMap
else:
    try:
        from docling_core.types import DoclingDocument
    except ImportError:
        DoclingDocument = Any  # type: ignore[misc,assignment]
    try:
        from .cloakmap import CloakMap
    except ImportError:
        CloakMap = Any  # type: ignore[misc,assignment]


@dataclass
class UnmaskingResult:
    """
    Result of an unmasking operation containing the restored document.

    Attributes:
        restored_document: The DoclingDocument with original content restored
        cloakmap: The CloakMap that was used for restoration
        stats: Statistics about the unmasking operation
        integrity_report: Report on restoration integrity and any issues
    """

    restored_document: DoclingDocument
    cloakmap: CloakMap
    stats: Optional[dict[str, Any]] = None
    integrity_report: Optional[dict[str, Any]] = None

    @property
    def unmasked_document(self) -> DoclingDocument:
        """Alias for restored_document for backward compatibility."""
        return self.restored_document


__all__ = ["DoclingDocument", "UnmaskingResult"]
