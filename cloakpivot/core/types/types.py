"""Centralized type definitions to avoid import issues."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cloakpivot.type_imports import DoclingDocument

if TYPE_CHECKING:
    from .cloakmap import CloakMap
else:
    try:
        from .cloakmap import CloakMap
    except ImportError:
        CloakMap = Any


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
    stats: dict[str, Any] | None = None
    integrity_report: dict[str, Any] | None = None

    @property
    def unmasked_document(self) -> DoclingDocument:
        """Alias for restored_document for backward compatibility."""
        return self.restored_document


__all__ = ["DoclingDocument", "UnmaskingResult"]
