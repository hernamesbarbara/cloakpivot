"""AnchorResolver for resolving anchor positions in masked documents."""

import logging
from dataclasses import dataclass
from typing import Any, cast

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


@dataclass
class ResolvedAnchor:
    """
    Represents an anchor that has been successfully resolved in a document.

    Attributes:
        anchor: The original anchor entry
        node_item: The document node containing the anchor
        found_position: The actual position where the masked text was found
        found_text: The actual text found at the position
        position_delta: Difference between expected and actual position
        confidence: Confidence score for the resolution (0.0-1.0)
    """

    anchor: AnchorEntry
    node_item: NodeItem
    found_position: tuple[int, int]  # (start, end)
    found_text: str
    position_delta: int
    confidence: float


@dataclass
class FailedAnchor:
    """
    Represents an anchor that could not be resolved in a document.

    Attributes:
        anchor: The original anchor entry
        failure_reason: Description of why resolution failed
        node_found: Whether the target node was found
        attempted_positions: List of positions that were tried
    """

    anchor: AnchorEntry
    failure_reason: str
    node_found: bool
    attempted_positions: list[tuple[int, int]]


class AnchorResolver:
    """
    Resolves anchor positions in masked documents to locate replacement tokens.

    This class handles the complex task of finding masked replacement tokens
    in documents using anchor position data, accounting for:
    - Anchor drift from serialization/deserialization
    - Position misalignments due to formatting changes
    - Missing or modified nodes
    - Text content changes

    Examples:
        >>> resolver = AnchorResolver()
        >>> results = resolver.resolve_anchors(document, anchor_list)
        >>> print(f"Resolved {len(results['resolved'])} anchors")
    """

    # Configuration for anchor resolution
    MAX_POSITION_DRIFT = 100  # Maximum characters to search around expected position (increased)
    MIN_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to accept resolution (lowered)
    FUZZY_SEARCH_WINDOW = 20  # Characters to search in each direction (increased)

    def __init__(self) -> None:
        """Initialize the anchor resolver."""
        logger.debug("AnchorResolver initialized")

    def resolve_anchors(
        self, document: DoclingDocument, anchors: list[AnchorEntry]
    ) -> dict[str, Any]:
        """
        Resolve a list of anchors in a masked document.

        Args:
            document: The masked DoclingDocument to search
            anchors: List of anchor entries to resolve

        Returns:
            Dictionary containing resolved and failed anchors with statistics
        """
        logger.info(f"Resolving {len(anchors)} anchors in document")

        resolved_anchors: list[ResolvedAnchor] = []
        failed_anchors: list[FailedAnchor] = []

        # Group anchors by node ID for better position tracking
        anchors_by_node: dict[str, list[AnchorEntry]] = {}
        for anchor in anchors:
            if anchor.node_id not in anchors_by_node:
                anchors_by_node[anchor.node_id] = []
            anchors_by_node[anchor.node_id].append(anchor)

        # Process each node's anchors
        for _node_id, node_anchors in anchors_by_node.items():
            # Sort anchors by start position for cumulative shift tracking
            sorted_anchors = sorted(node_anchors, key=lambda a: a.start)

            # Track cumulative position shift for this node
            cumulative_shift = 0

            for anchor in sorted_anchors:
                try:
                    # Calculate adjusted position based on cumulative shift
                    adjusted_anchor = AnchorEntry(
                        node_id=anchor.node_id,
                        start=anchor.start + cumulative_shift,
                        end=anchor.start
                        + cumulative_shift
                        + len(anchor.masked_value),  # Use masked value length
                        entity_type=anchor.entity_type,
                        confidence=anchor.confidence,
                        masked_value=anchor.masked_value,
                        replacement_id=anchor.replacement_id,
                        original_checksum=anchor.original_checksum,
                        checksum_salt=anchor.checksum_salt,
                        strategy_used=anchor.strategy_used,
                        timestamp=anchor.timestamp,
                        metadata=anchor.metadata,
                    )

                    resolved = self._resolve_single_anchor_with_adjusted(
                        document, anchor, adjusted_anchor
                    )
                    if resolved:
                        resolved_anchors.append(resolved)
                        logger.debug(
                            f"Resolved anchor {anchor.replacement_id} "
                            f"with confidence {resolved.confidence:.2f}"
                        )

                        # Update cumulative shift based on length difference
                        original_length = anchor.end - anchor.start
                        masked_length = len(anchor.masked_value)
                        shift_delta = masked_length - original_length
                        cumulative_shift += shift_delta
                    else:
                        failed_anchors.append(
                            FailedAnchor(
                                anchor=anchor,
                                failure_reason="Could not locate replacement token",
                                node_found=self._find_node_by_id(document, anchor.node_id)
                                is not None,
                                attempted_positions=[],
                            )
                        )
                        logger.warning(f"Failed to resolve anchor {anchor.replacement_id}")

                except Exception as e:
                    logger.error(f"Error resolving anchor {anchor.replacement_id}: {e}")
                    failed_anchors.append(
                        FailedAnchor(
                            anchor=anchor,
                            failure_reason=f"Resolution error: {e}",
                            node_found=False,
                            attempted_positions=[],
                        )
                    )

        success_rate = len(resolved_anchors) / len(anchors) * 100 if anchors else 100

        logger.info(
            f"Anchor resolution completed: {len(resolved_anchors)} resolved, "
            f"{len(failed_anchors)} failed ({success_rate:.1f}% success rate)"
        )

        return {
            "resolved": resolved_anchors,
            "failed": failed_anchors,
            "stats": {
                "total": len(anchors),
                "resolved": len(resolved_anchors),
                "failed": len(failed_anchors),
                "success_rate": success_rate,
            },
        }

    def _resolve_single_anchor_with_adjusted(
        self, document: DoclingDocument, original_anchor: AnchorEntry, adjusted_anchor: AnchorEntry
    ) -> ResolvedAnchor | None:
        """Resolve a single anchor using adjusted positions."""
        # Find the target node
        node_item = self._find_node_by_id(document, original_anchor.node_id)
        if not node_item:
            logger.debug(f"Node not found: {original_anchor.node_id}")
            return None

        # Extract text content from the node
        node_text = self._extract_node_text(node_item, original_anchor.node_id)
        if not node_text:
            logger.debug(f"No text content in node: {original_anchor.node_id}")
            return None

        # Try exact position match first with adjusted positions
        exact_match = self._try_exact_position_match(adjusted_anchor, node_text)
        if exact_match:
            return ResolvedAnchor(
                anchor=original_anchor,  # Use original anchor for metadata
                node_item=node_item,
                found_position=exact_match["position"],
                found_text=exact_match["text"],
                position_delta=0,
                confidence=1.0,
            )

        # Try fuzzy position matching with adjusted positions
        fuzzy_match = self._try_fuzzy_position_match(adjusted_anchor, node_text)
        if fuzzy_match and fuzzy_match["confidence"] >= self.MIN_CONFIDENCE_THRESHOLD:
            return ResolvedAnchor(
                anchor=original_anchor,  # Use original anchor for metadata
                node_item=node_item,
                found_position=fuzzy_match["position"],
                found_text=fuzzy_match["text"],
                position_delta=fuzzy_match["delta"],
                confidence=fuzzy_match["confidence"],
            )

        # Try content-based search as last resort with adjusted positions
        content_match = self._try_content_based_search(adjusted_anchor, node_text)
        if content_match and content_match["confidence"] >= self.MIN_CONFIDENCE_THRESHOLD:
            return ResolvedAnchor(
                anchor=original_anchor,  # Use original anchor for metadata
                node_item=node_item,
                found_position=content_match["position"],
                found_text=content_match["text"],
                position_delta=content_match["delta"],
                confidence=content_match["confidence"],
            )

        return None

    def _resolve_single_anchor(
        self, document: DoclingDocument, anchor: AnchorEntry
    ) -> ResolvedAnchor | None:
        """Resolve a single anchor in the document (backward compatibility)."""
        # Find the target node
        node_item = self._find_node_by_id(document, anchor.node_id)
        if not node_item:
            logger.debug(f"Node not found: {anchor.node_id}")
            return None

        # Extract text content from the node
        node_text = self._extract_node_text(node_item, anchor.node_id)
        if not node_text:
            logger.debug(f"No text content in node: {anchor.node_id}")
            return None

        # Try exact position match first
        exact_match = self._try_exact_position_match(anchor, node_text)
        if exact_match:
            return ResolvedAnchor(
                anchor=anchor,
                node_item=node_item,
                found_position=exact_match["position"],
                found_text=exact_match["text"],
                position_delta=0,
                confidence=1.0,
            )

        # Try fuzzy position matching
        fuzzy_match = self._try_fuzzy_position_match(anchor, node_text)
        if fuzzy_match and fuzzy_match["confidence"] >= self.MIN_CONFIDENCE_THRESHOLD:
            return ResolvedAnchor(
                anchor=anchor,
                node_item=node_item,
                found_position=fuzzy_match["position"],
                found_text=fuzzy_match["text"],
                position_delta=fuzzy_match["delta"],
                confidence=fuzzy_match["confidence"],
            )

        # Try content-based search as last resort
        content_match = self._try_content_based_search(anchor, node_text)
        if content_match and content_match["confidence"] >= self.MIN_CONFIDENCE_THRESHOLD:
            return ResolvedAnchor(
                anchor=anchor,
                node_item=node_item,
                found_position=content_match["position"],
                found_text=content_match["text"],
                position_delta=content_match["delta"],
                confidence=content_match["confidence"],
            )

        return None

    def _try_exact_position_match(
        self, anchor: AnchorEntry, node_text: str
    ) -> dict[str, Any] | None:
        """Try to match the anchor at its exact expected position."""
        if anchor.end > len(node_text):
            return None

        found_text = node_text[anchor.start : anchor.end]
        if found_text == anchor.masked_value:
            return {
                "position": (anchor.start, anchor.end),
                "text": found_text,
                "confidence": 1.0,
                "delta": 0,
            }

        return None

    def _try_fuzzy_position_match(
        self, anchor: AnchorEntry, node_text: str
    ) -> dict[str, Any] | None:
        """Try to match the anchor within a window around the expected position."""
        expected_length = len(anchor.masked_value)
        search_start = max(0, anchor.start - self.FUZZY_SEARCH_WINDOW)
        search_end = min(len(node_text), anchor.end + self.FUZZY_SEARCH_WINDOW)

        best_match = None
        best_confidence = 0.0

        # Search for the masked value in the fuzzy window
        for start_pos in range(search_start, search_end - expected_length + 1):
            end_pos = start_pos + expected_length
            candidate_text = node_text[start_pos:end_pos]

            if candidate_text == anchor.masked_value:
                delta = abs(start_pos - anchor.start)
                confidence = max(0.0, 1.0 - (delta / self.MAX_POSITION_DRIFT))

                if confidence > best_confidence:
                    best_match = {
                        "position": (start_pos, end_pos),
                        "text": candidate_text,
                        "confidence": confidence,
                        "delta": delta,
                    }
                    best_confidence = confidence

        return best_match

    def _try_content_based_search(
        self, anchor: AnchorEntry, node_text: str
    ) -> dict[str, Any] | None:
        """Search for the masked value anywhere in the node text."""
        masked_value = anchor.masked_value

        # Special handling for asterisk patterns to avoid matching substrings
        if masked_value and all(c == "*" for c in masked_value):
            # For asterisk patterns, look for isolated sequences of the exact length
            import re

            pattern = r"(?<!\*)\*{" + str(len(masked_value)) + r"}(?!\*)"
            occurrences = []
            for match in re.finditer(pattern, node_text):
                occurrences.append((match.start(), match.end()))
        else:
            # For non-asterisk patterns, use the original logic
            occurrences = []
            start_idx = 0
            while True:
                idx = node_text.find(masked_value, start_idx)
                if idx == -1:
                    break
                occurrences.append((idx, idx + len(masked_value)))
                start_idx = idx + 1

        if not occurrences:
            return None

        # Choose the occurrence closest to the expected position
        best_match = None
        best_distance = float("inf")

        for start_pos, end_pos in occurrences:
            distance = abs(start_pos - anchor.start)
            if distance < best_distance and distance <= self.MAX_POSITION_DRIFT:
                confidence = max(0.0, 1.0 - (distance / self.MAX_POSITION_DRIFT))
                best_match = {
                    "position": (start_pos, end_pos),
                    "text": masked_value,
                    "confidence": confidence * 0.8,  # Reduce confidence for content search
                    "delta": distance,
                }
                best_distance = distance

        return best_match

    def _find_node_by_id(self, document: DoclingDocument, node_id: str) -> NodeItem | None:
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

    def _extract_node_text(self, node_item: NodeItem, node_id: str) -> str | None:
        """Extract text content from a node item."""
        # Handle text-bearing nodes
        if isinstance(
            node_item,
            (
                TextItem,
                TitleItem,
                SectionHeaderItem,
                ListItem,
                CodeItem,
                FormulaItem,
            ),
        ):
            return getattr(node_item, "text", None)

        # Handle table nodes
        if isinstance(node_item, TableItem):
            return self._extract_table_text(node_item, node_id)

        # Handle key-value nodes
        if isinstance(node_item, KeyValueItem):
            return self._extract_key_value_text(node_item, node_id)

        return None

    def _extract_table_text(self, table_item: TableItem, node_id: str) -> str | None:
        """Extract text from a table node or specific cell."""
        if not hasattr(table_item, "data") or not table_item.data:
            return None

        table_data = table_item.data
        if not hasattr(table_data, "table_cells") or not table_data.table_cells:
            return None

        base_node_id = self._get_node_id(table_item)

        # Check if this is a specific cell
        if node_id.startswith(base_node_id + "/cell_"):
            cell_suffix = node_id[len(base_node_id + "/cell_") :]
            try:
                row_idx, col_idx = map(int, cell_suffix.split("_"))
                if row_idx < len(table_data.table_cells) and col_idx < len(
                    cast(Any, table_data.table_cells)[row_idx]
                ):
                    cell = cast(Any, table_data.table_cells)[row_idx][col_idx]
                    return getattr(cell, "text", None)
            except ValueError:
                pass

        # Return concatenated text of all cells if node_id matches table
        if node_id == base_node_id:
            all_text = []
            for row in table_data.table_cells:
                for cell in row:
                    if hasattr(cell, "text") and cell.text:
                        all_text.append(cell.text)
            return " ".join(all_text) if all_text else None

        return None

    def _extract_key_value_text(self, kv_item: KeyValueItem, node_id: str) -> str | None:
        """Extract text from a key-value node."""
        base_node_id = self._get_node_id(kv_item)

        # Check for key-specific node ID
        if node_id == f"{base_node_id}/key":
            if hasattr(kv_item, "key") and kv_item.key and hasattr(kv_item.key, "text"):
                return kv_item.key.text  # type: ignore[no-any-return]

        # Check for value-specific node ID
        elif node_id == f"{base_node_id}/value":
            if hasattr(kv_item, "value") and kv_item.value and hasattr(kv_item.value, "text"):
                return kv_item.value.text  # type: ignore[no-any-return]

        # Return concatenated key-value text if node_id matches the item
        elif node_id == base_node_id:
            texts = []
            if hasattr(kv_item, "key") and kv_item.key and hasattr(kv_item.key, "text"):
                texts.append(kv_item.key.text)
            if hasattr(kv_item, "value") and kv_item.value and hasattr(kv_item.value, "text"):
                texts.append(kv_item.value.text)
            return " ".join(texts) if texts else None

        return None

    def get_resolution_stats(self, results: dict[str, Any]) -> dict[str, Any]:
        """Get detailed statistics about anchor resolution results."""
        resolved = results.get("resolved", [])
        failed = results.get("failed", [])

        # Confidence distribution
        confidence_buckets = {"high": 0, "medium": 0, "low": 0}
        position_deltas = []

        for resolved_anchor in resolved:
            if resolved_anchor.confidence >= 0.9:
                confidence_buckets["high"] += 1
            elif resolved_anchor.confidence >= 0.7:
                confidence_buckets["medium"] += 1
            else:
                confidence_buckets["low"] += 1

            position_deltas.append(resolved_anchor.position_delta)

        # Failure analysis
        failure_reasons: dict[str, int] = {}
        for failed_anchor in failed:
            reason = failed_anchor.failure_reason
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        avg_confidence = sum(r.confidence for r in resolved) / len(resolved) if resolved else 0.0

        avg_position_delta = sum(position_deltas) / len(position_deltas) if position_deltas else 0

        return {
            "total_anchors": len(resolved) + len(failed),
            "resolved_count": len(resolved),
            "failed_count": len(failed),
            "success_rate": results.get("stats", {}).get("success_rate", 0),
            "average_confidence": round(avg_confidence, 3),
            "confidence_distribution": confidence_buckets,
            "average_position_delta": round(avg_position_delta, 2),
            "max_position_delta": max(position_deltas) if position_deltas else 0,
            "failure_reasons": failure_reasons,
        }
