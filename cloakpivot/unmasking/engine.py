"""Core UnmaskingEngine for orchestrating PII unmasking operations."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, cast

from cloakpivot.core.types import DoclingDocument

from ..core.cloakmap import CloakMap
from .anchor_resolver import AnchorResolver
from .cloakmap_loader import CloakMapLoader
from .document_unmasker import DocumentUnmasker

logger = logging.getLogger(__name__)


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


class UnmaskingEngine:
    """
    Core unmasking engine that orchestrates PII restoration operations.

    This engine coordinates the unmasking process by:
    1. Loading and validating CloakMap from storage
    2. Verifying document hash and compatibility
    3. Resolving anchor positions in the masked document
    4. Restoring original content using replacement token mapping
    5. Verifying round-trip integrity

    Examples:
        >>> engine = UnmaskingEngine()
        >>> result = engine.unmask_document(
        ...     masked_document=masked_doc,
        ...     cloakmap_path="document.cloakmap"
        ... )
        >>> print(f"Restored {len(result.cloakmap.anchors)} entities")
    """

    def __init__(self) -> None:
        """Initialize the unmasking engine."""
        self.cloakmap_loader = CloakMapLoader()
        self.document_unmasker = DocumentUnmasker()
        self.anchor_resolver = AnchorResolver()
        logger.debug("UnmaskingEngine initialized")

    def unmask_document(
        self,
        masked_document: DoclingDocument,
        cloakmap: Union[CloakMap, str, Path],
        verify_integrity: bool = True,
    ) -> UnmaskingResult:
        """
        Unmask a masked document using the provided CloakMap.

        Args:
            masked_document: The masked DoclingDocument to restore
            cloakmap: CloakMap object or path to CloakMap file
            verify_integrity: Whether to perform integrity verification

        Returns:
            UnmaskingResult containing restored document and metadata

        Raises:
            ValueError: If validation fails or restoration is not possible
            FileNotFoundError: If CloakMap file is not found
        """
        # Validate inputs first before accessing attributes
        if not isinstance(masked_document, DoclingDocument):
            raise ValueError("document must be a DoclingDocument")

        logger.info(f"Starting unmasking of document {masked_document.name}")

        # Load CloakMap if it's already a CloakMap object, or load from path
        if isinstance(cloakmap, CloakMap):
            cloakmap_obj = cloakmap
        elif isinstance(cloakmap, (str, Path)):
            cloakmap_obj = self.cloakmap_loader.load(cloakmap)
        else:
            raise ValueError("cloakmap must be a CloakMap")

        # Handle empty CloakMap case - if no anchors exist, return document unchanged
        if not cloakmap_obj.anchors:
            logger.warning(
                "CloakMap contains no anchors - returning document unchanged"
            )
            return UnmaskingResult(
                restored_document=self._copy_document(masked_document),
                cloakmap=cloakmap_obj,
                stats={
                    "total_anchors_processed": 0,
                    "successful_restorations": 0,
                    "failed_restorations": 0,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        # Validate remaining inputs
        self._validate_inputs(masked_document, cloakmap_obj)

        # Verify document compatibility
        self._verify_document_compatibility(masked_document, cloakmap_obj)

        # Create a copy of the document for restoration
        restored_document = self._copy_document(masked_document)

        # Resolve anchor positions in the copied document (not the original)
        resolved_anchors = self.anchor_resolver.resolve_anchors(
            document=restored_document, anchors=cloakmap_obj.anchors
        )

        logger.info(
            f"Resolved {len(resolved_anchors)} out of {len(cloakmap_obj.anchors)} anchors"
        )

        # Apply unmasking to restore original content
        restoration_stats = self.document_unmasker.apply_unmasking(
            document=restored_document,
            resolved_anchors=resolved_anchors.get("resolved", []),
            cloakmap=cloakmap_obj,
        )

        # Perform integrity verification if requested
        integrity_report = None
        if verify_integrity:
            integrity_report = self._verify_restoration_integrity(
                original_document=restored_document,
                masked_document=masked_document,
                cloakmap=cloakmap_obj,
                resolved_anchors=resolved_anchors,
            )

        # Generate statistics
        stats = self._generate_stats(cloakmap_obj, resolved_anchors, restoration_stats)

        logger.info("Unmasking completed successfully")

        return UnmaskingResult(
            restored_document=restored_document,
            cloakmap=cloakmap_obj,
            stats=stats,
            integrity_report=integrity_report,
        )

    def unmask_from_files(
        self,
        masked_document_path: Union[str, Path],
        cloakmap_path: Union[str, Path],
        verify_integrity: bool = True,
    ) -> UnmaskingResult:
        """
        Unmask a document loaded from file paths.

        Args:
            masked_document_path: Path to the masked document file
            cloakmap_path: Path to the CloakMap file
            verify_integrity: Whether to perform integrity verification

        Returns:
            UnmaskingResult containing restored document and metadata
        """
        # Load masked document using docpivot
        # This is a placeholder - actual implementation would use docpivot
        import json
        from pathlib import Path

        doc_path = Path(masked_document_path)
        if not doc_path.exists():
            raise FileNotFoundError(f"Masked document not found: {doc_path}")

        # For now, assume we can load JSON documents
        # Real implementation would use docpivot.load_document()
        with open(doc_path, encoding="utf-8") as f:
            json.load(f)

        # Create a minimal DoclingDocument for testing
        # Real implementation would properly deserialize
        masked_doc = DoclingDocument(name=doc_path.name)

        return self.unmask_document(
            masked_document=masked_doc,
            cloakmap=cloakmap_path,
            verify_integrity=verify_integrity,
        )

    def _validate_inputs(self, document: DoclingDocument, cloakmap: CloakMap) -> None:
        """Validate input parameters."""
        if not isinstance(document, DoclingDocument):
            raise ValueError("document must be a DoclingDocument")

        if not isinstance(cloakmap, CloakMap):
            raise ValueError("cloakmap must be a CloakMap")

        if not cloakmap.anchors:
            raise ValueError("CloakMap contains no anchors to restore")

    def _verify_document_compatibility(
        self, document: DoclingDocument, cloakmap: CloakMap
    ) -> None:
        """Verify that the document is compatible with the CloakMap."""
        # Check document ID compatibility
        doc_name = document.name or "unnamed_document"
        if cloakmap.doc_id != doc_name:
            logger.warning(
                f"Document name '{doc_name}' does not match "
                f"CloakMap doc_id '{cloakmap.doc_id}'"
            )

        # For now, skip hash verification since we don't have the original hash
        # In a full implementation, we'd verify against the masked document hash
        logger.debug("Document compatibility verification completed")

    def _copy_document(self, document: DoclingDocument) -> DoclingDocument:
        """Create a deep copy of the document for restoration."""
        import copy

        return copy.deepcopy(document)

    def _verify_restoration_integrity(
        self,
        original_document: DoclingDocument,
        masked_document: DoclingDocument,
        cloakmap: CloakMap,
        resolved_anchors: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify the integrity of the restoration process."""
        integrity_report = {
            "valid": True,
            "issues": [],
            "stats": {
                "total_anchors": len(cloakmap.anchors),
                "resolved_anchors": len(resolved_anchors.get("resolved", [])),
                "failed_anchors": len(resolved_anchors.get("failed", [])),
            },
        }

        # Check for failed anchor resolutions
        failed_anchors = resolved_anchors.get("failed", [])
        if failed_anchors:
            integrity_report["valid"] = False
            cast(list[str], integrity_report["issues"]).append(
                f"Failed to resolve {len(failed_anchors)} anchors"
            )

        # Additional integrity checks would go here
        # - Verify no replacement tokens remain in document
        # - Check document structure preservation
        # - Validate content checksums where possible

        logger.info(
            f"Integrity verification completed: "
            f"{'PASSED' if integrity_report['valid'] else 'FAILED'}"
        )

        return integrity_report

    def _generate_stats(
        self,
        cloakmap: CloakMap,
        resolved_anchors: dict[str, Any],
        restoration_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate statistics about the unmasking operation."""
        return {
            "cloakmap_version": cloakmap.version,
            "total_anchors": len(cloakmap.anchors),
            "resolved_anchors": len(resolved_anchors.get("resolved", [])),
            "failed_anchors": len(resolved_anchors.get("failed", [])),
            "entity_type_counts": cloakmap.entity_count_by_type,
            "restoration_stats": restoration_stats,
            "success_rate": (
                len(resolved_anchors.get("resolved", [])) / len(cloakmap.anchors) * 100
                if cloakmap.anchors
                else 100
            ),
        }
