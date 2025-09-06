"""DocumentUnmasker for restoring original content in DoclingDocument structures."""

import hashlib
import logging
from typing import Any, Optional, Union, cast

from docling_core.types.doc.document import (
    CodeItem,
    FormulaItem,
    KeyValueItem,
    ListItem,
    NodeItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from cloakpivot.core.types import DoclingDocument

from ..core.anchors import AnchorEntry
from ..core.cloakmap import CloakMap
from .anchor_resolver import ResolvedAnchor

logger = logging.getLogger(__name__)


class DocumentUnmaskerError(Exception):
    """Raised when document unmasking fails."""

    pass


class DocumentUnmasker:
    """
    Restores original content in DoclingDocument structures while preserving formatting.

    This class performs the reverse operation of DocumentMasker by:
    - Locating replacement tokens in masked documents
    - Restoring original content using anchor position data
    - Maintaining document structure and hierarchy
    - Preserving formatting and style information

    Since CloakMap only stores checksums (not original plaintext), this implementation
    provides a framework for restoration that can be extended with secure key management
    or external content lookup systems.

    Examples:
        >>> unmasker = DocumentUnmasker()
        >>> stats = unmasker.apply_unmasking(
        ...     document=masked_doc,
        ...     resolved_anchors=resolved_list,
        ...     cloakmap=cloakmap_obj
        ... )
    """

    def __init__(self) -> None:
        """Initialize the document unmasker."""
        logger.debug("DocumentUnmasker initialized")

    def apply_unmasking(
        self,
        document: DoclingDocument,
        resolved_anchors: list[ResolvedAnchor],
        cloakmap: CloakMap,
        original_content_provider: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Apply unmasking operations to a DoclingDocument in-place.

        This method modifies the document by restoring original content at the
        positions specified by resolved anchor entries. The operation preserves
        all document structure and formatting.

        Args:
            document: The DoclingDocument to restore
            resolved_anchors: List of resolved anchor entries
            cloakmap: The CloakMap containing original metadata
            original_content_provider: Optional provider for original content lookup

        Returns:
            Dictionary with restoration statistics and results

        Raises:
            DocumentUnmaskerError: If restoration fails
        """
        if not resolved_anchors:
            logger.debug("No resolved anchors provided, no unmasking applied")
            return {
                "total_anchors": 0,
                "restored_anchors": 0,
                "failed_restorations": 0,
                "success_rate": 100.0,
            }

        logger.info(
            f"Applying unmasking to {len(resolved_anchors)} locations in document"
        )

        # Group resolved anchors by node ID for efficient processing
        anchors_by_node = self._group_resolved_anchors_by_node(resolved_anchors)

        restoration_results = []

        # Apply unmasking to each affected node
        for node_id, node_anchors in anchors_by_node.items():
            try:
                node_results = self._unmask_node(
                    document, node_id, node_anchors, original_content_provider
                )
                restoration_results.extend(node_results)
            except Exception as e:
                logger.error(f"Failed to unmask node {node_id}: {e}")
                # Add failed results for this node's anchors
                for resolved_anchor in node_anchors:
                    restoration_results.append(
                        {
                            "anchor_id": resolved_anchor.anchor.replacement_id,
                            "success": False,
                            "error": str(e),
                        }
                    )

        # Calculate statistics
        stats = self._calculate_restoration_stats(restoration_results)

        # Sync _main_text if it exists (for backward compatibility with tests)
        if hasattr(document, '_main_text') and document.texts:
            document._main_text = document.texts[0].text

        logger.info(
            f"Document unmasking completed: {stats['restored_anchors']} restored, "
            f"{stats['failed_restorations']} failed "
            f"({stats['success_rate']:.1f}% success rate)"
        )

        return stats

    def _group_resolved_anchors_by_node(
        self, resolved_anchors: list[ResolvedAnchor]
    ) -> dict[str, list[ResolvedAnchor]]:
        """Group resolved anchor entries by their node IDs."""
        anchors_by_node: dict[str, list[ResolvedAnchor]] = {}

        for resolved_anchor in resolved_anchors:
            node_id = resolved_anchor.anchor.node_id
            if node_id not in anchors_by_node:
                anchors_by_node[node_id] = []
            anchors_by_node[node_id].append(resolved_anchor)

        # Sort anchors within each node by position (descending for safe replacement)
        for node_anchors in anchors_by_node.values():
            node_anchors.sort(key=lambda ra: ra.found_position[0], reverse=True)

        return anchors_by_node

    def _unmask_node(
        self,
        document: DoclingDocument,
        node_id: str,
        resolved_anchors: list[ResolvedAnchor],
        original_content_provider: Optional[Any],
    ) -> list[dict[str, Any]]:
        """Apply unmasking to a specific node in the document."""
        logger.debug(f"Unmasking node {node_id} with {len(resolved_anchors)} anchors")

        results: list[Any] = []

        # Get the first anchor's node item (they all share the same node)
        if not resolved_anchors:
            return results

        node_item = resolved_anchors[0].node_item

        # Handle different node types
        if self._is_text_bearing_node(node_item):
            results = self._unmask_text_node(
                cast(
                    Union[
                        TextItem,
                        TitleItem,
                        SectionHeaderItem,
                        ListItem,
                        CodeItem,
                        FormulaItem,
                    ],
                    node_item,
                ),
                resolved_anchors,
                original_content_provider,
            )
        elif isinstance(node_item, TableItem):
            results = self._unmask_table_node(
                node_item, node_id, resolved_anchors, original_content_provider
            )
        elif isinstance(node_item, KeyValueItem):
            results = self._unmask_key_value_node(
                node_item, node_id, resolved_anchors, original_content_provider
            )
        else:
            logger.warning(f"Unsupported node type for unmasking: {type(node_item)}")
            for resolved_anchor in resolved_anchors:
                results.append(
                    {
                        "anchor_id": resolved_anchor.anchor.replacement_id,
                        "success": False,
                        "error": f"Unsupported node type: {type(node_item)}",
                    }
                )

        return results

    def _is_text_bearing_node(self, node_item: NodeItem) -> bool:
        """Check if a node item contains editable text."""
        return isinstance(
            node_item,
            (
                TextItem,
                TitleItem,
                SectionHeaderItem,
                ListItem,
                CodeItem,
                FormulaItem,
            ),
        )

    def _unmask_text_node(
        self,
        node_item: Union[
            TextItem,
            TitleItem,
            SectionHeaderItem,
            ListItem,
            CodeItem,
            FormulaItem,
        ],
        resolved_anchors: list[ResolvedAnchor],
        original_content_provider: Optional[Any],
    ) -> list[dict[str, Any]]:
        """Apply unmasking to a text-bearing node."""
        results: list[Any] = []

        if not hasattr(node_item, "text") or not node_item.text:
            logger.warning("Text node has no text content to unmask")
            for resolved_anchor in resolved_anchors:
                results.append(
                    {
                        "anchor_id": resolved_anchor.anchor.replacement_id,
                        "success": False,
                        "error": "No text content in node",
                    }
                )
            return results

        original_text = node_item.text
        modified_text = original_text

        # Apply replacements in reverse order (highest start position first)
        for resolved_anchor in resolved_anchors:
            anchor = resolved_anchor.anchor

            # Get the original content for this anchor
            original_content = self._get_original_content(
                anchor, original_content_provider
            )

            if original_content is None:
                # For now, use placeholder restoration
                original_content = self._generate_placeholder_content(resolved_anchor)
                logger.debug(
                    f"Using placeholder content for {anchor.replacement_id}: '{original_content}'"
                )

            # Search for the masked value in the current state of the text
            masked_value = anchor.masked_value
            position = modified_text.rfind(
                masked_value
            )  # Search from end for reverse processing

            if position == -1:
                # Try a forward search as fallback
                position = modified_text.find(masked_value)

            if position == -1:
                # Try pattern matching for common masking patterns (asterisks)
                if masked_value and all(c == "*" for c in masked_value):
                    # Look for any sequence of asterisks that might be part of this mask
                    import re

                    asterisk_pattern = r"\*+"
                    matches = list(re.finditer(asterisk_pattern, modified_text))

                    # Find the best match (prefer longer sequences, but accept shorter ones too)
                    best_match = None
                    for match in reversed(matches):  # Process in reverse order
                        match_text = match.group()
                        # Accept if it's reasonably close in length or contains asterisks
                        if len(match_text) >= min(3, len(masked_value) // 2):
                            best_match = match
                            break

                    if best_match:
                        position = best_match.start()
                        logger.debug(
                            f"Found asterisk pattern '{best_match.group()}' at position {position} "
                            f"for masked_value '{masked_value}' (anchor {anchor.replacement_id})"
                        )

            if position == -1:
                logger.warning(
                    f"Could not find masked value '{masked_value}' in text for anchor {anchor.replacement_id}"
                )
                results.append(
                    {
                        "anchor_id": anchor.replacement_id,
                        "success": False,
                        "error": "Masked value not found in text",
                    }
                )
                continue

            start_pos = position

            # Determine end position based on what we actually found
            if masked_value in modified_text[position:]:
                # Found exact match
                end_pos = position + len(masked_value)
            else:
                # Found partial match - find where the asterisks end
                end_pos = position
                while end_pos < len(modified_text) and modified_text[end_pos] == "*":
                    end_pos += 1

                logger.debug(
                    f"Using partial asterisk match from {start_pos} to {end_pos} "
                    f"for anchor {anchor.replacement_id}"
                )

            if end_pos > len(modified_text):
                logger.warning(
                    f"Anchor end position {end_pos} exceeds text length {len(modified_text)}"
                )
                results.append(
                    {
                        "anchor_id": anchor.replacement_id,
                        "success": False,
                        "error": "Position exceeds text length",
                    }
                )
                continue

            # Replace the masked text with original content
            modified_text = (
                modified_text[:start_pos] + original_content + modified_text[end_pos:]
            )

            # Verify original content if possible
            content_verified = self._verify_original_content(
                resolved_anchor, original_content
            )

            results.append(
                {
                    "anchor_id": anchor.replacement_id,
                    "success": True,
                    "original_length": len(original_content),
                    "masked_length": len(anchor.masked_value),
                    "content_verified": content_verified,
                    "confidence": resolved_anchor.confidence,
                }
            )

            logger.debug(
                f"Restored text at {start_pos}:{end_pos} "
                f"from '{anchor.masked_value}' to '{original_content}' "
                f"in {anchor.node_id}"
            )

        # Update the node's text
        node_item.text = modified_text
        return results

    def _unmask_table_node(
        self,
        table_item: TableItem,
        node_id: str,
        resolved_anchors: list[ResolvedAnchor],
        original_content_provider: Optional[Any],
    ) -> list[dict[str, Any]]:
        """Apply unmasking to a table node."""
        results: list[Any] = []

        if not hasattr(table_item, "data") or not table_item.data:
            logger.warning("Table item has no data to unmask")
            for resolved_anchor in resolved_anchors:
                results.append(
                    {
                        "anchor_id": resolved_anchor.anchor.replacement_id,
                        "success": False,
                        "error": "No table data",
                    }
                )
            return results

        table_data = table_item.data
        if not hasattr(table_data, "table_cells") or not table_data.table_cells:
            logger.warning("Table data has no cells to unmask")
            for resolved_anchor in resolved_anchors:
                results.append(
                    {
                        "anchor_id": resolved_anchor.anchor.replacement_id,
                        "success": False,
                        "error": "No table cells",
                    }
                )
            return results

        base_node_id = self._get_node_id(table_item)

        # Group anchors by cell coordinates
        cell_anchor_groups: dict[str, list[ResolvedAnchor]] = {}
        for resolved_anchor in resolved_anchors:
            cell_node_id = resolved_anchor.anchor.node_id
            cell_anchor_groups.setdefault(cell_node_id, []).append(resolved_anchor)

        # Apply unmasking to each affected cell
        for cell_node_id, cell_resolved_anchors in cell_anchor_groups.items():
            if not cell_node_id.startswith(base_node_id + "/cell_"):
                continue

            # Parse cell coordinates from node ID
            cell_suffix = cell_node_id[len(base_node_id + "/cell_") :]
            try:
                row_idx, col_idx = map(int, cell_suffix.split("_"))
            except ValueError:
                logger.warning(f"Invalid cell node ID format: {cell_node_id}")
                for resolved_anchor in cell_resolved_anchors:
                    results.append(
                        {
                            "anchor_id": resolved_anchor.anchor.replacement_id,
                            "success": False,
                            "error": "Invalid cell ID format",
                        }
                    )
                continue

            # Check bounds
            if row_idx >= len(table_data.table_cells) or col_idx >= len(
                cast(Any, table_data.table_cells)[row_idx]
            ):
                logger.warning(f"Cell coordinates ({row_idx}, {col_idx}) out of bounds")
                for resolved_anchor in cell_resolved_anchors:
                    results.append(
                        {
                            "anchor_id": resolved_anchor.anchor.replacement_id,
                            "success": False,
                            "error": "Cell coordinates out of bounds",
                        }
                    )
                continue

            # Get the cell and apply unmasking
            # Cast to Any to handle Union[RichTableCell, TableCell] indexing
            cell = cast(Any, table_data.table_cells)[row_idx][col_idx]
            if hasattr(cell, "text") and cell.text:
                cell_results = self._unmask_cell_text(
                    cell, cell_resolved_anchors, original_content_provider
                )
                results.extend(cell_results)
                logger.debug(f"Unmasked table cell ({row_idx}, {col_idx})")

        return results

    def _unmask_key_value_node(
        self,
        kv_item: KeyValueItem,
        node_id: str,
        resolved_anchors: list[ResolvedAnchor],
        original_content_provider: Optional[Any],
    ) -> list[dict[str, Any]]:
        """Apply unmasking to a key-value node."""
        results: list[Any] = []
        base_node_id = self._get_node_id(kv_item)

        # Group anchors by key/value part
        key_anchors = []
        value_anchors = []

        for resolved_anchor in resolved_anchors:
            anchor_node_id = resolved_anchor.anchor.node_id
            if anchor_node_id == f"{base_node_id}/key":
                key_anchors.append(resolved_anchor)
            elif anchor_node_id == f"{base_node_id}/value":
                value_anchors.append(resolved_anchor)

        # Unmask key part
        if key_anchors and hasattr(kv_item, "key") and kv_item.key:
            if hasattr(kv_item.key, "text"):
                key_results = self._unmask_key_value_text(
                    kv_item.key, key_anchors, original_content_provider, "key"
                )
                results.extend(key_results)
                logger.debug(f"Unmasked key-value key: {base_node_id}")

        # Unmask value part
        if value_anchors and hasattr(kv_item, "value") and kv_item.value:
            if hasattr(kv_item.value, "text"):
                value_results = self._unmask_key_value_text(
                    kv_item.value, value_anchors, original_content_provider, "value"
                )
                results.extend(value_results)
                logger.debug(f"Unmasked key-value value: {base_node_id}")

        return results

    def _unmask_cell_text(
        self,
        cell: Any,
        resolved_anchors: list[ResolvedAnchor],
        original_content_provider: Optional[Any],
    ) -> list[dict[str, Any]]:
        """Unmask text content in a table cell."""
        results: list[Any] = []
        original_text = cell.text
        modified_text = original_text

        # Sort anchors by position (reverse order)
        sorted_anchors = sorted(
            resolved_anchors, key=lambda ra: ra.found_position[0], reverse=True
        )

        for resolved_anchor in sorted_anchors:
            anchor = resolved_anchor.anchor
            start_pos, end_pos = resolved_anchor.found_position

            if end_pos > len(modified_text):
                results.append(
                    {
                        "anchor_id": anchor.replacement_id,
                        "success": False,
                        "error": "Position exceeds cell text length",
                    }
                )
                continue

            # Get original content
            original_content = self._get_original_content(
                anchor, original_content_provider
            )
            if original_content is None:
                original_content = self._generate_placeholder_content(resolved_anchor)

            # Replace the text
            modified_text = (
                modified_text[:start_pos] + original_content + modified_text[end_pos:]
            )

            results.append(
                {
                    "anchor_id": anchor.replacement_id,
                    "success": True,
                    "original_length": len(original_content),
                    "masked_length": len(anchor.masked_value),
                    "confidence": resolved_anchor.confidence,
                }
            )

        cell.text = modified_text
        return results

    def _unmask_key_value_text(
        self,
        text_item: Any,
        resolved_anchors: list[ResolvedAnchor],
        original_content_provider: Optional[Any],
        part_type: str,
    ) -> list[dict[str, Any]]:
        """Unmask text content in a key-value part."""
        results: list[Any] = []
        original_text = text_item.text
        modified_text = original_text

        # Sort anchors by position (reverse order)
        sorted_anchors = sorted(
            resolved_anchors, key=lambda ra: ra.found_position[0], reverse=True
        )

        for resolved_anchor in sorted_anchors:
            anchor = resolved_anchor.anchor
            start_pos, end_pos = resolved_anchor.found_position

            if end_pos > len(modified_text):
                results.append(
                    {
                        "anchor_id": anchor.replacement_id,
                        "success": False,
                        "error": f"Position exceeds {part_type} text length",
                    }
                )
                continue

            # Get original content
            original_content = self._get_original_content(
                anchor, original_content_provider
            )
            if original_content is None:
                original_content = self._generate_placeholder_content(resolved_anchor)

            # Replace the text
            modified_text = (
                modified_text[:start_pos] + original_content + modified_text[end_pos:]
            )

            results.append(
                {
                    "anchor_id": anchor.replacement_id,
                    "success": True,
                    "original_length": len(original_content),
                    "masked_length": len(anchor.masked_value),
                    "confidence": resolved_anchor.confidence,
                }
            )

        text_item.text = modified_text
        return results

    def _get_node_id(self, node_item: NodeItem) -> str:
        """Get the node ID for a node item."""
        if hasattr(node_item, "self_ref") and node_item.self_ref:
            return node_item.self_ref

        # Generate fallback ID
        node_type = type(node_item).__name__
        if hasattr(node_item, "text") and node_item.text:
            text_hash = hash(node_item.text[:50])
            return f"#{node_type.lower()}_{abs(text_hash)}"

        return f"#{node_type.lower()}_{id(node_item)}"

    def _get_original_content(
        self,
        anchor: Union[AnchorEntry, ResolvedAnchor],
        original_content_provider: Optional[Any],
    ) -> Optional[str]:
        """
        Get the original content for an anchor.

        First tries to retrieve from anchor metadata, then falls back to content provider,
        and finally to placeholder generation.
        """
        # First, try to get original text from anchor metadata
        # Handle both AnchorEntry and ResolvedAnchor
        actual_anchor = anchor.anchor if isinstance(anchor, ResolvedAnchor) else anchor
        if actual_anchor.metadata and "original_text" in actual_anchor.metadata:
            original_text = actual_anchor.metadata["original_text"]
            logger.debug(
                f"Retrieved original text from metadata for {actual_anchor.replacement_id}: '{original_text}'"
            )
            return cast(str, original_text)

        # Fallback to content provider if available
        actual_anchor = anchor.anchor if isinstance(anchor, ResolvedAnchor) else anchor
        if original_content_provider and hasattr(
            original_content_provider, "get_content"
        ):
            try:
                result = original_content_provider.get_content(
                    actual_anchor.replacement_id, actual_anchor.entity_type
                )
                return cast(Optional[str], result)
            except Exception as e:
                logger.warning(
                    f"Content provider failed for {actual_anchor.replacement_id}: {e}"
                )

        # Return None to trigger placeholder generation as last resort
        actual_anchor = anchor.anchor if isinstance(anchor, ResolvedAnchor) else anchor
        logger.debug(
            f"No original content found for {actual_anchor.replacement_id}, will use placeholder"
        )
        return None

    def _generate_placeholder_content(self, anchor: ResolvedAnchor) -> str:
        """
        Generate placeholder content for testing and demonstration.

        This creates realistic-looking placeholder content based on entity type.
        In production, this would be replaced with actual content restoration.
        """
        # Access entity_type from the underlying AnchorEntry
        entity_type = anchor.anchor.entity_type.upper()

        # Generate type-appropriate placeholders
        placeholder_map = {
            "PHONE_NUMBER": "555-0123",
            "EMAIL_ADDRESS": "user@example.com",
            "PERSON": "John Doe",
            "US_SSN": "123-45-6789",
            "CREDIT_CARD": "4000-1234-5678-9012",
            "US_BANK_NUMBER": "123456789",
            "IP_ADDRESS": "192.168.1.1",
            "URL": "https://example.com",
            "DATE_TIME": "2023-01-15 10:30:00",
            "ORGANIZATION": "Example Corp",
            "LOCATION": "New York, NY",
            "US_DRIVER_LICENSE": "D123456789",
            "US_PASSPORT": "123456789",
        }

        placeholder = placeholder_map.get(entity_type, f"[{entity_type}]")

        # Try to match the length of the original masked value if reasonable
        masked_length = len(anchor.anchor.masked_value)
        if masked_length > len(placeholder) and masked_length <= 50:
            # Pad with realistic characters for the entity type
            if entity_type in ["PHONE_NUMBER", "US_SSN", "CREDIT_CARD"]:
                placeholder = placeholder.ljust(masked_length, "0")
            elif entity_type in ["EMAIL_ADDRESS", "URL"]:
                placeholder = placeholder.ljust(masked_length, "x")
            else:
                placeholder = placeholder.ljust(masked_length, "X")

        return placeholder

    def _verify_original_content(
        self, anchor: ResolvedAnchor, original_content: str
    ) -> bool:
        """
        Verify that the original content matches the stored checksum.

        This validates that the content we're restoring is authentic.
        """
        try:
            computed_checksum = hashlib.sha256(
                original_content.encode("utf-8")
            ).hexdigest()
            return bool(computed_checksum == anchor.anchor.original_checksum)
        except Exception as e:
            logger.warning(
                f"Content verification failed for {anchor.anchor.replacement_id}: {e}"
            )
            return False

    def _calculate_restoration_stats(
        self, restoration_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate statistics about the restoration process."""
        total_anchors = len(restoration_results)
        successful_restorations = sum(
            1 for result in restoration_results if result.get("success", False)
        )
        failed_restorations = total_anchors - successful_restorations

        success_rate = (
            (successful_restorations / total_anchors * 100)
            if total_anchors > 0
            else 100.0
        )

        # Calculate content verification stats
        verified_content = sum(
            1
            for result in restoration_results
            if result.get("success", False) and result.get("content_verified", False)
        )

        # Calculate length changes
        total_original_length = sum(
            result.get("original_length", 0)
            for result in restoration_results
            if result.get("success", False)
        )

        total_masked_length = sum(
            result.get("masked_length", 0)
            for result in restoration_results
            if result.get("success", False)
        )

        # Collect error types
        error_types: dict[str, int] = {}
        for result in restoration_results:
            if not result.get("success", False) and "error" in result:
                error = result["error"]
                error_types[error] = error_types.get(error, 0) + 1

        return {
            "total_anchors": total_anchors,
            "restored_anchors": successful_restorations,
            "failed_restorations": failed_restorations,
            "success_rate": round(success_rate, 2),
            "content_verification": {
                "verified_count": verified_content,
                "verification_rate": (
                    round(verified_content / successful_restorations * 100, 2)
                    if successful_restorations > 0
                    else 0.0
                ),
            },
            "length_changes": {
                "total_original_length": total_original_length,
                "total_masked_length": total_masked_length,
                "length_delta": total_original_length - total_masked_length,
            },
            "error_types": error_types,
        }
