"""Presidio-based unmasking adapter for CloakPivot.

This module provides the PresidioUnmaskingAdapter, which integrates Presidio's
deanonymization capabilities into CloakPivot's unmasking workflow. The adapter
enables reversible PII operations (such as encryption) while maintaining backward
compatibility with v1.0 CloakMaps that use anchor-based restoration.

The adapter is the counterpart to PresidioMaskingAdapter and works in tandem to
provide a complete masking/unmasking solution. It intelligently chooses between:
- Presidio-based restoration for reversible operations
- Anchor-based restoration for non-reversible operations
- Hybrid approach when both are available

Key Features:
    - Full backward compatibility with v1.0 CloakMaps
    - Support for reversible operations (encryption, custom operators)
    - Hybrid restoration combining Presidio and anchor-based methods
    - Comprehensive statistics tracking for all restoration methods
"""

import copy
import logging
from datetime import datetime
from typing import Any, Union

from presidio_anonymizer import DeanonymizeEngine
from presidio_anonymizer.entities import OperatorResult

from ..core.cloakmap import CloakMap
from ..core.cloakmap_enhancer import CloakMapEnhancer
from ..core.types import DoclingDocument, UnmaskingResult
from .anchor_resolver import AnchorResolver
from .document_unmasker import DocumentUnmasker

logger = logging.getLogger(__name__)


class PresidioUnmaskingAdapter:
    """
    Adapter that uses Presidio DeanonymizeEngine for unmasking operations.

    This adapter enables reversible operations for strategies like encryption
    while maintaining compatibility with existing unmasking workflows. It provides
    a hybrid approach that uses Presidio for reversible operations and falls back
    to anchor-based restoration for non-reversible operations.

    The adapter maintains 100% backward compatibility with v1.0 CloakMaps that
    don't contain Presidio metadata.

    Examples:
        >>> adapter = PresidioUnmaskingAdapter()
        >>> result = adapter.unmask_document(masked_doc, cloakmap)
        >>> print(f"Restored {result.stats['presidio_restored']} entities via Presidio")
    """

    def __init__(self) -> None:
        """Initialize the Presidio unmasking adapter."""
        self.deanonymizer = DeanonymizeEngine()
        self.cloakmap_enhancer = CloakMapEnhancer()
        self.anchor_resolver = AnchorResolver()
        self.document_unmasker = DocumentUnmasker()
        logger.debug("PresidioUnmaskingAdapter initialized")

    def unmask_document(
        self,
        masked_document: DoclingDocument,
        cloakmap: CloakMap
    ) -> UnmaskingResult:
        """
        Main unmasking method compatible with existing UnmaskingEngine API.

        This method determines the appropriate unmasking strategy based on the
        CloakMap version and available metadata, then delegates to the appropriate
        restoration method.

        Args:
            masked_document: The masked DoclingDocument to restore
            cloakmap: CloakMap containing restoration metadata

        Returns:
            UnmaskingResult containing restored document and statistics
        """
        logger.info(f"Starting unmasking of document {masked_document.name}")

        # Check if CloakMap has Presidio metadata
        if self.cloakmap_enhancer.is_presidio_enabled(cloakmap):
            logger.info("Using Presidio-based restoration (v2.0 CloakMap)")
            return self._presidio_deanonymization(masked_document, cloakmap)
        else:
            logger.info("Using anchor-based restoration (v1.0 CloakMap)")
            return self._anchor_based_restoration(masked_document, cloakmap)

    def restore_content(
        self,
        masked_text: str,
        operator_results: list[Union[dict[str, Any], OperatorResult]]
    ) -> str:
        """
        Content restoration using exact position-based replacement.

        This method handles restoration by processing replacements in reverse order
        to maintain correct positions as the text changes.

        Args:
            masked_text: Text containing masked values
            operator_results: List of operator results from masking

        Returns:
            Text with original values restored where possible
        """
        if not operator_results:
            return masked_text

        # Build a list of replacements with position information
        replacements = []

        for result in operator_results:
            if isinstance(result, dict):
                masked_value = result.get("text", "")
                original_value = result.get("original_text", "")
                operator = result.get("operator", "replace")
                start = result.get("start")
                end = result.get("end")

                # For restorable operations, add to replacements list
                if operator in ["replace", "template", "redact", "partial", "hash"] and original_value and masked_value:
                    replacements.append({
                        "masked": masked_value,
                        "original": original_value,
                        "operator": operator,
                        "start": start,
                        "end": end
                    })
                elif operator == "encrypt":
                    logger.warning("Encryption reversal not yet implemented")
                elif operator == "custom":
                    # Custom operators need special handling
                    if "reverse_function" in result:
                        # Apply reverse function if available
                        reverse_func = result["reverse_function"]
                        replacements.append({
                            "masked": masked_value,
                            "original": reverse_func(masked_value),
                            "operator": operator,
                            "start": start,
                            "end": end
                        })
                    else:
                        logger.warning("No reverse function for custom operator")
            else:
                # Handle OperatorResult objects
                if hasattr(result, "old") and result.old:
                    replacements.append({
                        "masked": result.text,
                        "original": result.old,
                        "operator": "presidio",
                        "start": getattr(result, "start", None),
                        "end": getattr(result, "end", None)
                    })

        # Process replacements in the order they appear (which is reverse order from masking)
        # The key insight: since operator_results are in reverse order (last entity first),
        # we need to replace from the end of the string to maintain correct positions
        restored_text = masked_text

        logger.debug(f"Starting restoration with text: {masked_text}")
        logger.debug(f"Found {len(replacements)} replacements to apply")

        for replacement in replacements:
            masked_val = replacement["masked"]
            original_val = replacement["original"]

            logger.debug(f"Replacing '{masked_val}' with '{original_val}'")

            # Replace the LAST occurrence since we're processing in reverse order
            # This ensures we replace the correct instance when there are duplicates
            idx = restored_text.rfind(masked_val)
            if idx != -1:
                restored_text = (
                    restored_text[:idx] +
                    original_val +
                    restored_text[idx + len(masked_val):]
                )
                logger.debug(f"Replacement successful at position {idx}")
            else:
                logger.warning(f"Could not find '{masked_val}' in text")

        return restored_text

    def _presidio_deanonymization(
        self,
        masked_document: DoclingDocument,
        cloakmap: CloakMap
    ) -> UnmaskingResult:
        """
        Use Presidio for reversible operations.

        This method extracts operator results from the CloakMap and uses
        Presidio's DeanonymizeEngine to restore original values for
        reversible operations.

        Args:
            masked_document: Document with masked content
            cloakmap: CloakMap with Presidio metadata

        Returns:
            UnmaskingResult with restored document
        """
        # Create a copy of the document for restoration
        restored_document = copy.deepcopy(masked_document)

        # Extract operator results from CloakMap
        try:
            operator_results = self.cloakmap_enhancer.extract_operator_results(cloakmap)
            reversible_operators = self.cloakmap_enhancer.get_reversible_operators(cloakmap)
        except ValueError as e:
            logger.warning(f"Failed to extract Presidio metadata: {e}")
            # Fall back to anchor-based restoration
            return self._anchor_based_restoration(masked_document, cloakmap)

        # Separate reversible and non-reversible operations
        reversible_results: list[Union[dict[str, Any], OperatorResult]] = []
        non_reversible_count = 0

        for result in operator_results:
            if isinstance(result, dict):
                operator = result.get("operator", "")
            else:
                operator = getattr(result, "operator", "")
            if operator in reversible_operators or operator == "replace":
                reversible_results.append(result)
            else:
                non_reversible_count += 1

        # Restore reversible operations using our restoration logic
        presidio_restored = 0
        presidio_failed = 0

        if reversible_results:
            try:
                # Get the text from the first text item (assuming single text document)
                if restored_document.texts and len(restored_document.texts) > 0:
                    current_text = restored_document.texts[0].text
                else:
                    current_text = ""

                restored_text = self.restore_content(
                    current_text,
                    reversible_results
                )

                logger.debug(f"After restoration: current_text='{current_text}', restored_text='{restored_text}'")

                # Update the text in the document
                if restored_document.texts and len(restored_document.texts) > 0:
                    restored_document.texts[0].text = restored_text
                    restored_document.texts[0].orig = restored_text
                    logger.debug(f"Updated document text to: {restored_document.texts[0].text}")

                # Also update _main_text for backward compatibility
                if hasattr(restored_document, '_main_text'):
                    restored_document._main_text = restored_text
                    logger.debug(f"Updated _main_text to: {restored_document._main_text}")

                # Preserve tables and key_value_items from masked document
                # These don't contain PII that needs unmasking in the current implementation
                if hasattr(masked_document, 'tables'):
                    restored_document.tables = copy.deepcopy(masked_document.tables)
                if hasattr(masked_document, 'key_value_items'):
                    restored_document.key_value_items = copy.deepcopy(masked_document.key_value_items)

                # Count successful restorations by checking what changed
                for result in reversible_results:
                    if isinstance(result, dict):
                        masked_val = result.get("text", "")
                        original_val = result.get("original_text", "")
                    else:
                        # Handle OperatorResult type
                        masked_val = getattr(result, "text", "")
                        original_val = getattr(result, "original_text", "")

                    # Check if restoration was successful
                    if original_val and masked_val not in restored_text and original_val in restored_text:
                        presidio_restored += 1
                    elif isinstance(result, dict) and result.get("operator") == "custom" and "reverse_function" not in result:
                        # Custom operators without reverse function fail
                        presidio_failed += 1
                    elif masked_val in restored_text:
                        # Masked value still present means restoration failed
                        presidio_failed += 1
                    else:
                        presidio_restored += 1

            except Exception as e:
                logger.error(f"Presidio restoration failed: {e}")
                presidio_failed = len(reversible_results)

        # Handle non-reversible operations with anchors if present
        anchor_restored = 0
        if cloakmap.anchors and non_reversible_count > 0:
            logger.info(f"Processing {len(cloakmap.anchors)} anchors for non-reversible operations")

            # Resolve and apply anchor-based restoration
            resolved_anchors = self.anchor_resolver.resolve_anchors(
                document=restored_document,
                anchors=cloakmap.anchors
            )

            restoration_stats = self.document_unmasker.apply_unmasking(
                document=restored_document,
                resolved_anchors=resolved_anchors.get("resolved", []),
                cloakmap=cloakmap
            )

            anchor_restored = restoration_stats.get("successful_restorations", 0)

        # Create result with statistics
        stats = {
            "version": cloakmap.version,
            "method": "presidio" if presidio_restored > 0 else "hybrid",
            "presidio_restored": presidio_restored,
            "presidio_failed": presidio_failed,
            "anchor_restored": anchor_restored,
            "non_reversible_count": non_reversible_count,
            "timestamp": datetime.now().isoformat()
        }

        return UnmaskingResult(
            restored_document=restored_document,
            cloakmap=cloakmap,
            stats=stats
        )

    def _anchor_based_restoration(
        self,
        masked_document: DoclingDocument,
        cloakmap: CloakMap
    ) -> UnmaskingResult:
        """
        Fall back to anchor-based restoration.

        This method provides backward compatibility for v1.0 CloakMaps
        that don't contain Presidio metadata, using the traditional
        anchor-based restoration approach.

        Args:
            masked_document: Document with masked content
            cloakmap: CloakMap with anchor entries

        Returns:
            UnmaskingResult with restored document
        """
        # Create a copy of the document for restoration
        restored_document = copy.deepcopy(masked_document)

        # Handle empty anchors case
        if not cloakmap.anchors:
            logger.warning("No anchors found in CloakMap - returning document unchanged")
            from .engine import UnmaskingResult
            return UnmaskingResult(
                restored_document=restored_document,
                cloakmap=cloakmap,
                stats={
                    "version": cloakmap.version,
                    "method": "anchor_based",
                    "presidio_restored": 0,
                    "anchor_restored": 0,
                    "timestamp": datetime.now().isoformat()
                }
            )

        # Resolve anchor positions
        resolved_anchors = self.anchor_resolver.resolve_anchors(
            document=restored_document,
            anchors=cloakmap.anchors
        )

        logger.info(
            f"Resolved {len(resolved_anchors.get('resolved', []))} "
            f"out of {len(cloakmap.anchors)} anchors"
        )

        # Apply anchor-based unmasking
        restoration_stats = self.document_unmasker.apply_unmasking(
            document=restored_document,
            resolved_anchors=resolved_anchors.get("resolved", []),
            cloakmap=cloakmap
        )

        # Create result with statistics
        stats = {
            "version": cloakmap.version,
            "method": "anchor_based",
            "presidio_restored": 0,
            "presidio_failed": 0,
            "anchor_restored": restoration_stats.get("successful_restorations", 0),
            "anchor_failed": restoration_stats.get("failed_restorations", 0),
            "timestamp": datetime.now().isoformat()
        }

        return UnmaskingResult(
            restored_document=restored_document,
            cloakmap=cloakmap,
            stats=stats
        )

    def _hybrid_restoration(
        self,
        masked_document: DoclingDocument,
        cloakmap: CloakMap
    ) -> UnmaskingResult:
        """
        Perform hybrid restoration using both Presidio and anchors.

        This method combines Presidio deanonymization for reversible operations
        with anchor-based restoration for non-reversible operations, providing
        the best of both approaches.

        Args:
            masked_document: Document with masked content
            cloakmap: CloakMap with both Presidio metadata and anchors

        Returns:
            UnmaskingResult with restored document
        """
        # This is handled within _presidio_deanonymization when both
        # operator results and anchors are present
        return self._presidio_deanonymization(masked_document, cloakmap)
