"""Document reconstruction utilities for the PresidioMaskingAdapter.

This module contains logic for reconstructing masked documents, including
updating text segments and table cells with masked values.
"""

import copy
import json
import logging
from typing import Any

from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import DoclingDocument, TextItem

from cloakpivot.core.types.anchors import AnchorEntry
from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class DocumentReconstructor:
    """Reconstruct documents with masked values."""

    def __init__(self):
        """Initialize the document reconstructor."""
        self.text_processor = TextProcessor()

    def create_masked_document(
        self,
        document: DoclingDocument,
        text_segments: list[TextSegment],
        anchor_entries: list[AnchorEntry],
        masked_text: str,
    ) -> DoclingDocument:
        """Create the masked document preserving structure.

        Args:
            document: Original document
            text_segments: Text segments from document
            anchor_entries: Anchor entries with masked values
            masked_text: Fully masked document text

        Returns:
            Masked DoclingDocument
        """
        # Serialize the document to preserve all structure
        doc_dict = json.loads(document.model_dump_json())
        masked_document = DoclingDocument.model_validate(doc_dict)

        # Apply masking to each segment individually
        if hasattr(document, "texts") and document.texts:
            masked_segments = []

            for i, original_item in enumerate(document.texts):
                if i < len(text_segments):
                    segment = text_segments[i]
                    segment_text = original_item.text

                    # Find entities that affect this segment
                    segment_entities = self.text_processor.extract_segment_entities(
                        segment, anchor_entries, segment_text
                    )

                    # Apply masks to this segment using efficient O(n) approach
                    if segment_entities:
                        # Build spans for this segment
                        segment_spans: list[tuple[int, int, str]] = [
                            (
                                entity_info["local_start"],
                                entity_info["local_end"],
                                entity_info["anchor"].masked_value,
                            )
                            for entity_info in segment_entities
                        ]
                        masked_segment_text = self.text_processor.apply_spans(
                            segment_text, segment_spans
                        )
                    else:
                        masked_segment_text = segment_text

                    # Create new text item preserving original structure
                    masked_text_item = self._create_masked_text_item(
                        original_item, masked_segment_text, i
                    )
                    masked_segments.append(masked_text_item)
                else:
                    masked_segments.append(copy.deepcopy(original_item))

            # Update texts in place
            for i, masked_item in enumerate(masked_segments):
                if i < len(masked_document.texts):
                    masked_document.texts[i].text = masked_item.text
                    if hasattr(masked_item, "orig"):
                        masked_document.texts[i].orig = masked_item.orig

        # Preserve _main_text for backward compatibility
        if hasattr(document, "_main_text"):
            setattr(masked_document, "_main_text", masked_text)  # noqa: B010

        # Update table cells
        self.update_table_cells(masked_document, text_segments, anchor_entries)

        return masked_document

    def _create_masked_text_item(
        self, original_item: Any, masked_text: str, index: int
    ) -> TextItem:
        """Create a masked text item preserving original structure.

        Args:
            original_item: Original text item
            masked_text: Masked text content
            index: Item index

        Returns:
            New TextItem with masked content
        """
        valid_text_labels = {
            DocItemLabel.CAPTION,
            DocItemLabel.CHECKBOX_SELECTED,
            DocItemLabel.CHECKBOX_UNSELECTED,
            DocItemLabel.FOOTNOTE,
            DocItemLabel.PAGE_FOOTER,
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PARAGRAPH,
            DocItemLabel.REFERENCE,
            DocItemLabel.TEXT,
            DocItemLabel.EMPTY_VALUE,
        }

        item_label = DocItemLabel.TEXT
        if hasattr(original_item, "label"):
            item_label = (
                original_item.label
                if original_item.label in valid_text_labels
                else DocItemLabel.TEXT
            )

        masked_text_item = TextItem(
            text=masked_text,
            self_ref=(
                original_item.self_ref if hasattr(original_item, "self_ref") else f"#/texts/{index}"
            ),
            label=item_label,
            orig=masked_text,
        )

        if hasattr(original_item, "prov"):
            masked_text_item.prov = original_item.prov

        return masked_text_item

    def update_table_cells(
        self,
        document: DoclingDocument,
        text_segments: list[TextSegment],
        anchor_entries: list[AnchorEntry],
    ) -> None:
        """Update table cells with masked values.

        Args:
            document: Document to update (modified in place)
            text_segments: Text segments from original extraction
            anchor_entries: Anchor entries with masked values
        """
        if not hasattr(document, "tables") or not document.tables:
            return

        # Build a mapping of node_id to masked content
        masked_content_map = {}
        for anchor in anchor_entries:
            node_id = anchor.node_id
            if node_id not in masked_content_map:
                masked_content_map[node_id] = []
            masked_content_map[node_id].append(anchor)

        for table_idx, table in enumerate(document.tables):
            if not hasattr(table, "data") or not table.data:
                continue

            table_data = table.data

            # Process grid structure (modern format)
            if hasattr(table_data, "grid") and table_data.grid:
                for row_idx, row in enumerate(table_data.grid):
                    for col_idx, cell in enumerate(row):
                        if hasattr(cell, "text") and cell.text:
                            # Generate the expected node_id for this cell
                            cell_node_id = self._get_cell_node_id(
                                table, table_idx, row_idx, col_idx
                            )

                            # Check if we have masked content for this cell
                            if cell_node_id in masked_content_map:
                                # Apply masking to this cell
                                masked_cell_text = self._apply_masks_to_cell(
                                    cell.text, masked_content_map[cell_node_id]
                                )
                                cell.text = masked_cell_text

            # Process legacy table_cells structure (backward compatibility)
            elif hasattr(table_data, "table_cells") and table_data.table_cells:
                flat_cells = table_data.table_cells
                if (
                    flat_cells
                    and hasattr(table_data, "num_rows")
                    and hasattr(table_data, "num_cols")
                ):
                    num_cols = table_data.num_cols
                    for i in range(table_data.num_rows):
                        for j in range(num_cols):
                            idx = i * num_cols + j
                            if idx < len(flat_cells):
                                cell = flat_cells[idx]
                                if hasattr(cell, "text") and cell.text:
                                    cell_node_id = self._get_cell_node_id(table, table_idx, i, j)
                                    if cell_node_id in masked_content_map:
                                        masked_cell_text = self._apply_masks_to_cell(
                                            cell.text, masked_content_map[cell_node_id]
                                        )
                                        cell.text = masked_cell_text

    def _get_cell_node_id(self, table: Any, table_idx: int, row_idx: int, col_idx: int) -> str:
        """Generate the node ID for a table cell.

        Args:
            table: Table object
            table_idx: Table index in document
            row_idx: Row index
            col_idx: Column index

        Returns:
            Node ID string
        """
        table_node_id = self._get_table_node_id(table, table_idx)
        return f"{table_node_id}/cell_{row_idx}_{col_idx}"

    def _get_table_node_id(self, table_item: Any, table_idx: int | None = None) -> str:
        """Get the node ID for a table.

        Args:
            table_item: Table item from document
            table_idx: Optional table index

        Returns:
            Table node ID
        """
        if hasattr(table_item, "self_ref") and table_item.self_ref:
            return str(table_item.self_ref)
        if table_idx is not None:
            return f"#/tables/{table_idx}"
        return "#/tables/0"

    def _apply_masks_to_cell(self, cell_text: str, anchors: list[AnchorEntry]) -> str:
        """Apply masks to a table cell.

        Args:
            cell_text: Original cell text
            anchors: Anchors that apply to this cell

        Returns:
            Masked cell text
        """
        # Sort anchors by position
        sorted_anchors = sorted(anchors, key=lambda a: a.start)

        # Build replacement spans
        spans = []
        for anchor in sorted_anchors:
            # The anchor positions are relative to the cell text
            if anchor.metadata and "original_text" in anchor.metadata:
                # Find the position of original text in the cell
                original = anchor.metadata["original_text"]
                pos = cell_text.find(original)
                if pos >= 0:
                    spans.append((pos, pos + len(original), anchor.masked_value))

        # Apply replacements
        if spans:
            return self.text_processor.apply_spans(cell_text, spans)
        return cell_text
