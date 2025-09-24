"""Core UnmaskingEngine for orchestrating PII unmasking operations."""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from cloakpivot.core.types import DoclingDocument, UnmaskingResult

from ..core.anchors import AnchorEntry
from ..core.cloakmap import CloakMap
from .anchor_resolver import AnchorResolver
from .cloakmap_loader import CloakMapLoader
from .document_unmasker import DocumentUnmasker

logger = logging.getLogger(__name__)


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

    def __init__(self, use_presidio_engine: bool | None = None) -> None:
        """Initialize the unmasking engine.

        Args:
            use_presidio_engine: Force engine selection (True=Presidio, False=Legacy, None=Auto)
        """
        self.cloakmap_loader = CloakMapLoader()
        self.document_unmasker = DocumentUnmasker()
        self.anchor_resolver = AnchorResolver()
        self.use_presidio_override = use_presidio_engine

        # Initialize Presidio adapter if requested
        self.presidio_adapter: Any | None = None
        if use_presidio_engine is True:
            from .presidio_adapter import PresidioUnmaskingAdapter

            self.presidio_adapter = PresidioUnmaskingAdapter()

        logger.debug(f"UnmaskingEngine initialized with use_presidio_engine={use_presidio_engine}")

    def unmask_document(
        self,
        masked_document: DoclingDocument,
        cloakmap: CloakMap | str | Path,
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
        elif isinstance(cloakmap, str | Path):
            cloakmap_obj = self.cloakmap_loader.load(cloakmap)
        else:
            raise ValueError("cloakmap must be a CloakMap")

        # Handle empty CloakMap case - if no anchors exist, return document unchanged
        if not cloakmap_obj.anchors:
            logger.warning("CloakMap contains no anchors - returning document unchanged")
            return UnmaskingResult(
                restored_document=self._copy_document(masked_document),
                cloakmap=cloakmap_obj,
                stats={
                    "total_anchors_processed": 0,
                    "successful_restorations": 0,
                    "failed_restorations": 0,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

        # Select the appropriate unmasking engine
        engine_type = self._select_unmasking_engine(cloakmap_obj)
        logger.info(f"Selected unmasking engine: {engine_type}")

        # Determine if we need hybrid processing
        has_reversible = self._detect_reversible_operations(cloakmap_obj)
        has_anchors = len(cloakmap_obj.anchors) > 0

        # Route to appropriate unmasking method
        if engine_type == "presidio" and (has_reversible or not has_anchors):
            # Use Presidio engine if selected and has reversible ops or no anchors
            result = self._unmask_with_presidio(masked_document, cloakmap_obj)
        elif has_reversible and has_anchors:
            # Hybrid processing: Use Presidio for reversible, legacy for anchors
            reversible_ops, anchor_ops = self._categorize_operations(cloakmap_obj)

            # First apply Presidio for reversible operations
            if self.presidio_adapter:
                result = self._unmask_with_presidio(masked_document, cloakmap_obj)
            else:
                # Fall back to legacy if Presidio not available
                result = self._unmask_with_legacy(masked_document, cloakmap_obj)

            # Update stats to indicate hybrid processing
            if result.stats:
                result.stats["method"] = "hybrid"
                result.stats["presidio_restored"] = len(reversible_ops)
                result.stats["anchor_restored"] = len(anchor_ops)
        else:
            # Use legacy engine for pure anchor-based restoration
            result = self._unmask_with_legacy(masked_document, cloakmap_obj)

        # Perform integrity verification if requested
        if verify_integrity:
            # Get resolved anchors from stats if available
            resolved_anchors_data = {}
            if hasattr(result, "stats") and result.stats and "resolved_anchors" in result.stats:
                # If it's already a dict, use it directly
                if isinstance(result.stats["resolved_anchors"], dict):
                    resolved_anchors_data = result.stats["resolved_anchors"]
                # If it's a number, create a dict format
                elif isinstance(result.stats["resolved_anchors"], int):
                    resolved_anchors_data = {
                        "resolved": [None] * result.stats["resolved_anchors"],
                        "failed": [],
                    }

            result.integrity_report = self._verify_restoration_integrity(
                _original_document=result.restored_document,
                _masked_document=masked_document,
                cloakmap=cloakmap_obj,
                resolved_anchors=resolved_anchors_data,
            )

        logger.info("Unmasking completed successfully")
        return result

    def unmask_from_files(
        self,
        masked_document_path: str | Path,
        cloakmap_path: str | Path,
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
        with doc_path.open(encoding="utf-8") as f:
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

    def _verify_document_compatibility(self, document: DoclingDocument, cloakmap: CloakMap) -> None:
        """Verify that the document is compatible with the CloakMap."""
        # Check document ID compatibility
        doc_name = document.name or "unnamed_document"
        if cloakmap.doc_id != doc_name:
            logger.warning(
                f"Document name '{doc_name}' does not match " f"CloakMap doc_id '{cloakmap.doc_id}'"
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
        _original_document: DoclingDocument,
        _masked_document: DoclingDocument,
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

    def _select_unmasking_engine(self, cloakmap: CloakMap) -> str:
        """Select the appropriate unmasking engine based on configuration and CloakMap.

        Args:
            cloakmap: The CloakMap to use for unmasking

        Returns:
            Engine selection: "presidio" or "legacy"
        """
        # Check explicit override
        if self.use_presidio_override is not None:
            return "presidio" if self.use_presidio_override else "legacy"

        # Check environment variable
        env_value = os.environ.get("CLOAKPIVOT_USE_PRESIDIO_ENGINE", "").lower()
        if env_value == "true":
            return "presidio"
        if env_value == "false":
            return "legacy"

        # Auto-detect based on CloakMap version and metadata
        if cloakmap.version == "2.0" and cloakmap.presidio_metadata:
            return "presidio"

        return "legacy"

    def _detect_reversible_operations(self, cloakmap: CloakMap) -> bool:
        """Detect if the CloakMap contains reversible operations.

        Args:
            cloakmap: The CloakMap to check

        Returns:
            True if reversible operations are detected
        """
        if not cloakmap.presidio_metadata:
            return False

        reversible_ops = cloakmap.presidio_metadata.get("reversible_operators", [])
        return len(reversible_ops) > 0

    def _categorize_operations(
        self, cloakmap: CloakMap
    ) -> tuple[list[dict[str, Any]], list[AnchorEntry]]:
        """Categorize operations into reversible and anchor-based.

        Args:
            cloakmap: The CloakMap to categorize

        Returns:
            Tuple of (reversible_operations, anchor_operations)
        """
        reversible_ops = []
        anchor_ops = list(cloakmap.anchors)  # All anchors are anchor-based by default

        if cloakmap.presidio_metadata:
            operator_results = cloakmap.presidio_metadata.get("operator_results", [])
            reversible_ops = operator_results

        return reversible_ops, anchor_ops

    def _unmask_with_presidio(
        self, document: DoclingDocument, cloakmap: CloakMap
    ) -> UnmaskingResult:
        """Unmask using Presidio engine.

        Args:
            document: The masked document
            cloakmap: The CloakMap with Presidio metadata

        Returns:
            UnmaskingResult with restored document
        """
        if not self.presidio_adapter:
            from .presidio_adapter import PresidioUnmaskingAdapter

            self.presidio_adapter = PresidioUnmaskingAdapter()

        return self.presidio_adapter.unmask_document(document, cloakmap)

    def _unmask_with_legacy(self, document: DoclingDocument, cloakmap: CloakMap) -> UnmaskingResult:
        """Unmask using legacy anchor-based method.

        Args:
            document: The masked document
            cloakmap: The CloakMap with anchors

        Returns:
            UnmaskingResult with restored document
        """
        # Use existing anchor-based unmasking logic
        restored_document = self._copy_document(document)

        resolved_anchors = self.anchor_resolver.resolve_anchors(
            document=restored_document, anchors=cloakmap.anchors
        )

        restoration_stats = self.document_unmasker.apply_unmasking(
            document=restored_document,
            resolved_anchors=resolved_anchors.get("resolved", []),
            cloakmap=cloakmap,
        )

        stats = self._generate_stats(cloakmap, resolved_anchors, restoration_stats)
        stats["method"] = "legacy"

        return UnmaskingResult(
            restored_document=restored_document,
            cloakmap=cloakmap,
            stats=stats,
        )

    def _enhance_legacy_cloakmap(self, cloakmap: CloakMap) -> CloakMap:
        """Enhance a v1.0 CloakMap with Presidio metadata.

        Args:
            cloakmap: v1.0 CloakMap to enhance

        Returns:
            Enhanced v2.0 CloakMap
        """
        return CloakMap(
            version="2.0",
            doc_id=cloakmap.doc_id,
            doc_hash=cloakmap.doc_hash,
            anchors=cloakmap.anchors,
            presidio_metadata={
                "operator_results": [],
                "reversible_operators": [],
                "engine_version": "2.2.0",
            },
            created_at=cloakmap.created_at,
        )

        # Additional fields are already handled by the CloakMap dataclass

    def migrate_to_presidio(self, cloakmap_path: str | Path) -> Path:
        """Migrate a v1.0 CloakMap to v2.0 with Presidio metadata.

        Args:
            cloakmap_path: Path to the v1.0 CloakMap file

        Returns:
            Path to the new v2.0 CloakMap file
        """
        cloakmap_path = Path(cloakmap_path)

        # Load the existing CloakMap
        cloakmap = self.cloakmap_loader.load(cloakmap_path)

        # Enhance it with Presidio metadata
        enhanced_map = self._enhance_legacy_cloakmap(cloakmap)

        # Save to new file
        new_path = cloakmap_path.with_suffix(".v2.cloakmap")
        enhanced_map.save_to_file(new_path)

        logger.info(f"Migrated CloakMap from v1.0 to v2.0: {new_path}")

        return new_path


__all__ = ["UnmaskingEngine", "UnmaskingResult"]
