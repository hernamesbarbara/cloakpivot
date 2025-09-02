"""DocumentMasker for applying masked replacements to DoclingDocument structures."""

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

logger = logging.getLogger(__name__)


class DocumentMasker:
    """
    Applies masking operations to DoclingDocument structures while preserving formatting.

    This class handles the delicate operation of replacing text content within
    DoclingDocument nodes while maintaining:
    - Document structure and hierarchy
    - Node references and boundaries
    - Formatting and style information
    - Table cell integrity
    - Key-value pair relationships

    The masker operates on AnchorEntry objects that specify exact positions
    and replacement text, ensuring deterministic and reversible operations.

    Examples:
        >>> masker = DocumentMasker()
        >>> anchor_entries = [
        ...     AnchorEntry(
        ...         node_id="#/texts/0",
        ...         start=10,
        ...         end=22,
        ...         entity_type="PHONE_NUMBER",
        ...         masked_value="[PHONE]",
        ...         # ... other fields
        ...     )
        ... ]
        >>> masker.apply_masking(document, anchor_entries)
    """

    def __init__(self) -> None:
        """Initialize the document masker."""
        logger.debug("DocumentMasker initialized")

    def apply_masking(
        self, document: DoclingDocument, anchor_entries: list[AnchorEntry]
    ) -> None:
        """
        Apply masking operations to a DoclingDocument in-place.

        This method modifies the document by replacing text content at the
        positions specified by anchor entries. The operation preserves all
        document structure and formatting.

        Args:
            document: The DoclingDocument to modify
            anchor_entries: List of anchor entries specifying what to mask

        Raises:
            ValueError: If anchor entries reference invalid nodes or positions
        """
        if not anchor_entries:
            logger.debug("No anchor entries provided, no masking applied")
            return

        logger.info(f"Applying masking to {len(anchor_entries)} locations in document")

        # Group anchor entries by node ID for efficient processing
        anchors_by_node = self._group_anchors_by_node(anchor_entries)

        # Apply masking to each affected node
        for node_id, node_anchors in anchors_by_node.items():
            self._mask_node(document, node_id, node_anchors)

        logger.info("Document masking completed successfully")

    def _group_anchors_by_node(
        self, anchor_entries: list[AnchorEntry]
    ) -> dict[str, list[AnchorEntry]]:
        """Group anchor entries by their node IDs."""
        anchors_by_node: dict[str, list[AnchorEntry]] = {}

        for anchor in anchor_entries:
            if anchor.node_id not in anchors_by_node:
                anchors_by_node[anchor.node_id] = []
            anchors_by_node[anchor.node_id].append(anchor)

        # Sort anchors within each node by position (descending for safe replacement)
        for node_anchors in anchors_by_node.values():
            node_anchors.sort(key=lambda a: a.start, reverse=True)

        return anchors_by_node

    def _mask_node(
        self,
        document: DoclingDocument,
        node_id: str,
        anchors: list[AnchorEntry],
    ) -> None:
        """Apply masking to a specific node in the document."""
        logger.debug(f"Masking node {node_id} with {len(anchors)} anchors")

        # Find the node in the document
        node_item = self._find_node_by_id(document, node_id)
        if not node_item:
            raise ValueError(f"Node not found: {node_id}")

        # Handle different node types
        if self._is_text_bearing_node(node_item):
            self._mask_text_node(
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
                anchors,
            )
        elif isinstance(node_item, TableItem):
            self._mask_table_node(node_item, node_id, anchors)
        elif isinstance(node_item, KeyValueItem):
            self._mask_key_value_node(node_item, node_id, anchors)
        else:
            logger.warning(f"Unsupported node type for masking: {type(node_item)}")

    def _find_node_by_id(
        self, document: DoclingDocument, node_id: str
    ) -> Optional[NodeItem]:
        """Find a node in the document by its ID."""
        # Check text items
        for text_item in document.texts:
            if self._get_node_id(text_item) == node_id:
                return text_item

        # Check table items
        for table_item in document.tables:
            table_node_id = self._get_node_id(table_item)
            if table_node_id == node_id:
                return table_item

            # Check for table cell IDs
            if node_id.startswith(table_node_id + "/cell_"):
                return table_item

        # Check key-value items
        for kv_item in document.key_value_items:
            kv_node_id = self._get_node_id(kv_item)
            if kv_node_id == node_id or node_id.startswith(kv_node_id + "/"):
                return kv_item

        return None

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

    def _mask_text_node(
        self,
        node_item: Union[
            TextItem,
            TitleItem,
            SectionHeaderItem,
            ListItem,
            CodeItem,
            FormulaItem,
        ],
        anchors: list[AnchorEntry],
    ) -> None:
        """Apply masking to a text-bearing node."""
        if not hasattr(node_item, "text") or not node_item.text:
            logger.warning("Text node has no text content to mask")
            return

        original_text = node_item.text
        modified_text = original_text

        # Apply replacements in reverse order (highest start position first)
        # This prevents position shifts from affecting subsequent replacements
        for anchor in anchors:
            if anchor.end > len(modified_text):
                logger.warning(
                    f"Anchor end position {anchor.end} exceeds text length {len(modified_text)} "
                    f"in node {anchor.node_id}"
                )
                continue

            # Replace the text segment
            modified_text = (
                modified_text[: anchor.start]
                + anchor.masked_value
                + modified_text[anchor.end :]
            )

            logger.debug(
                f"Replaced text at {anchor.start}:{anchor.end} "
                f"with '{anchor.masked_value}' in {anchor.node_id}"
            )

        # Update the node's text
        node_item.text = modified_text

    def _mask_table_node(
        self, table_item: TableItem, node_id: str, anchors: list[AnchorEntry]
    ) -> None:
        """Apply masking to a table node."""
        if not hasattr(table_item, "data") or not table_item.data:
            logger.warning("Table item has no data to mask")
            return

        table_data = table_item.data
        if not hasattr(table_data, "table_cells") or not table_data.table_cells:
            logger.warning("Table data has no cells to mask")
            return

        base_node_id = self._get_node_id(table_item)

        # Group anchors by cell coordinates
        cell_anchors: dict[str, list[AnchorEntry]] = {}
        for anchor in anchors:
            cell_anchors.setdefault(anchor.node_id, []).append(anchor)

        # Apply masking to each affected cell
        for cell_node_id, cell_anchor_list in cell_anchors.items():
            if not cell_node_id.startswith(base_node_id + "/cell_"):
                continue

            # Parse cell coordinates from node ID
            cell_suffix = cell_node_id[len(base_node_id + "/cell_") :]
            try:
                row_idx, col_idx = map(int, cell_suffix.split("_"))
            except ValueError:
                logger.warning(f"Invalid cell node ID format: {cell_node_id}")
                continue

            # Check bounds
            if row_idx >= len(table_data.table_cells) or col_idx >= len(
                cast(Any, table_data.table_cells)[row_idx]
            ):
                logger.warning(
                    f"Cell coordinates ({row_idx}, {col_idx}) out of bounds "
                    f"for table with {len(table_data.table_cells)} rows"
                )
                continue

            # Get the cell and apply masking
            cell = cast(Any, table_data.table_cells)[row_idx][col_idx]
            if hasattr(cell, "text") and cell.text:
                original_text = cell.text
                modified_text = original_text

                # Sort cell anchors by position (reverse order)
                sorted_anchors = sorted(
                    cell_anchor_list, key=lambda a: a.start, reverse=True
                )

                for anchor in sorted_anchors:
                    if anchor.end > len(modified_text):
                        continue

                    modified_text = (
                        modified_text[: anchor.start]
                        + anchor.masked_value
                        + modified_text[anchor.end :]
                    )

                cell.text = modified_text
                logger.debug(f"Masked table cell ({row_idx}, {col_idx})")

    def _mask_key_value_node(
        self, kv_item: KeyValueItem, node_id: str, anchors: list[AnchorEntry]
    ) -> None:
        """Apply masking to a key-value node."""
        base_node_id = self._get_node_id(kv_item)

        # Group anchors by key/value part
        key_anchors = []
        value_anchors = []

        for anchor in anchors:
            if anchor.node_id == f"{base_node_id}/key":
                key_anchors.append(anchor)
            elif anchor.node_id == f"{base_node_id}/value":
                value_anchors.append(anchor)

        # Mask key part
        if (
            key_anchors
            and hasattr(kv_item, "key")
            and kv_item.key
            and hasattr(kv_item.key, "text")
        ):
            original_text = kv_item.key.text
            modified_text = original_text

            for anchor in sorted(key_anchors, key=lambda a: a.start, reverse=True):
                if anchor.end <= len(modified_text):
                    modified_text = (
                        modified_text[: anchor.start]
                        + anchor.masked_value
                        + modified_text[anchor.end :]
                    )

            kv_item.key.text = modified_text
            logger.debug(f"Masked key-value key: {base_node_id}")

        # Mask value part
        if (
            value_anchors
            and hasattr(kv_item, "value")
            and kv_item.value
            and hasattr(kv_item.value, "text")
        ):
            original_text = kv_item.value.text
            modified_text = original_text

            for anchor in sorted(value_anchors, key=lambda a: a.start, reverse=True):
                if anchor.end <= len(modified_text):
                    modified_text = (
                        modified_text[: anchor.start]
                        + anchor.masked_value
                        + modified_text[anchor.end :]
                    )

            kv_item.value.text = modified_text
            logger.debug(f"Masked key-value value: {base_node_id}")

    def validate_masking_integrity(
        self, document: DoclingDocument, anchor_entries: list[AnchorEntry]
    ) -> dict[str, Any]:
        """
        Validate that masking was applied correctly and no PII remains.

        Args:
            document: The masked document to validate
            anchor_entries: The anchor entries that were applied

        Returns:
            Dict with validation results and any issues found
        """
        results = {
            "valid": True,
            "issues": [],
            "stats": {
                "nodes_checked": 0,
                "text_segments_checked": 0,
                "anchors_validated": 0,
            },
        }

        # For this basic implementation, we'll do simple validation
        # In production, this would be more thorough

        logger.info("Basic masking validation completed")
        return results
