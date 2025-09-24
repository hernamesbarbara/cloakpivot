"""Presidio-based masking adapter for CloakPivot."""

import base64
import copy
import hashlib
import logging
import threading
from typing import Any

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult

try:
    import presidio_anonymizer

    PRESIDIO_VERSION = getattr(presidio_anonymizer, "__version__", "2.x.x")
except (ImportError, AttributeError):
    PRESIDIO_VERSION = "2.x.x"

from ..core.types.cloakmap import AnchorEntry, CloakMap
from ..core.processing.cloakmap_enhancer import CloakMapEnhancer
from ..core.policies.policies import MaskingPolicy
from ..core.processing.presidio_mapper import StrategyToOperatorMapper
from ..core.processing.surrogate import SurrogateGenerator
from ..core.types import DoclingDocument
from ..core.types.strategies import Strategy, StrategyKind
from ..document.extractor import TextSegment
from .document_reconstructor import DocumentReconstructor
from .engine import MaskingResult
from .entity_processor import EntityProcessor
from .metadata_manager import MetadataManager
from .protocols import OperatorResultLike, SegmentBoundary, SyntheticOperatorResult
from .strategy_processors import StrategyProcessor
from .text_processor import TextProcessor

logger = logging.getLogger(__name__)

# Constants for consistent usage
UNKNOWN_ENTITY = "UNKNOWN"
SEGMENT_SEPARATOR = "\n\n"
SEGMENT_SEPARATOR_LEN = len(SEGMENT_SEPARATOR)


class PresidioMaskingAdapter:
    """
    Adapter that uses Presidio AnonymizerEngine for masking operations.

    This adapter translates CloakPivot masking concepts to Presidio operations
    while maintaining API compatibility with the existing StrategyApplicator.
    It captures Presidio operator results for perfect reversibility and provides
    robust error handling with fallback mechanisms.

    Examples:
        >>> adapter = PresidioMaskingAdapter()
        >>> result = adapter.apply_strategy(
        ...     "555-123-4567",
        ...     "PHONE_NUMBER",
        ...     Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
        ...     0.95
        ... )
        >>> assert result == "[PHONE]"
    """

    def __init__(self, engine_config: dict[str, Any] | None = None) -> None:
        """
        Initialize the Presidio masking adapter.

        Args:
            engine_config: Optional configuration for Presidio engine
        """
        self._anonymizer_instance: AnonymizerEngine | None = None  # Lazy loading
        self._anonymizer_lock = threading.Lock()  # Thread-safe initialization
        self.operator_mapper = StrategyToOperatorMapper()
        self.cloakmap_enhancer = CloakMapEnhancer()
        self._fallback_char = "*"
        self.engine_config = engine_config or {}
        self._surrogate_generator = SurrogateGenerator()
        self._segment_starts: list[int] | None = None  # Cache for binary search

        # Initialize processors (will be properly initialized when anonymizer is created)
        self.entity_processor: EntityProcessor | None = None
        self.strategy_processor: StrategyProcessor | None = None
        self.text_processor: TextProcessor | None = None
        self.document_reconstructor: DocumentReconstructor | None = None
        self.metadata_manager: MetadataManager | None = None

        logger.debug(f"PresidioMaskingAdapter initialized with config: {self.engine_config}")

    @property
    def anonymizer(self) -> AnonymizerEngine:
        """Lazy-load the AnonymizerEngine on first access (thread-safe)."""
        if self._anonymizer_instance is None:
            with self._anonymizer_lock:
                # Double-check locking pattern
                if self._anonymizer_instance is None:
                    self._anonymizer_instance = AnonymizerEngine()
                    logger.debug("AnonymizerEngine initialized")
                    # Initialize processors now that anonymizer exists
                    self.entity_processor = EntityProcessor(self._anonymizer_instance, self.operator_mapper)
                    self.strategy_processor = StrategyProcessor(self._anonymizer_instance, self.operator_mapper)
                    self.text_processor = TextProcessor()
                    self.document_reconstructor = DocumentReconstructor()
                    self.metadata_manager = MetadataManager()
        return self._anonymizer_instance

    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float,
    ) -> str:
        """
        Apply a masking strategy using Presidio with fallback support.

        This method maintains API compatibility with StrategyApplicator while
        delegating to Presidio's AnonymizerEngine for actual masking operations.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity (e.g., 'PHONE_NUMBER', 'EMAIL_ADDRESS')
            strategy: The masking strategy to apply
            confidence: Detection confidence score

        Returns:
            The masked replacement text
        """
        logger.debug(f"Applying {strategy.kind.value} strategy to {entity_type} via Presidio")

        try:
            # Handle CUSTOM strategy specially
            if strategy.kind == StrategyKind.CUSTOM:
                return self._apply_custom_strategy(original_text, strategy)

            # Handle SURROGATE strategy specially for better quality
            if strategy.kind == StrategyKind.SURROGATE:
                return self._apply_surrogate_strategy(original_text, entity_type, strategy)

            # Special handling for REDACT strategy since Presidio's redact operator has issues
            if strategy.kind == StrategyKind.REDACT:
                params = strategy.parameters or {}
                char = str(params.get("char", params.get("redact_char", "*")))
                return char * len(original_text)

            # Special handling for HASH strategy to support prefix
            if strategy.kind == StrategyKind.HASH:
                return self._apply_hash_strategy(original_text, entity_type, strategy, confidence)

            # Special handling for PARTIAL strategy
            if strategy.kind == StrategyKind.PARTIAL:
                return self._apply_partial_strategy(
                    original_text, entity_type, strategy, confidence
                )

            # Map CloakPivot strategy to Presidio operator
            operator_config = self.operator_mapper.strategy_to_operator(strategy)

            # Create a single entity for this text
            entity = RecognizerResult(
                entity_type=entity_type, start=0, end=len(original_text), score=confidence
            )

            # Use Presidio to anonymize
            result = self.anonymizer.anonymize(
                text=original_text,
                analyzer_results=[entity],
                operators={entity_type: operator_config},
            )

            return str(result.text)

        except Exception as e:
            logger.warning(
                f"Presidio anonymization failed for {entity_type}: {e}. "
                f"Falling back to simple redaction."
            )
            return self._fallback_redaction(original_text)

    def _build_full_text_and_boundaries(
        self, text_segments: list[TextSegment]
    ) -> tuple[str, list[SegmentBoundary]]:
        """Build the full document text and segment boundaries.

        Args:
            text_segments: List of text segments from document

        Returns:
            Tuple of (full_text, segment_boundaries)
        """
        document_text = ""
        segment_boundaries: list[SegmentBoundary] = []

        # Cache segment starts for later binary search
        self._segment_starts = [s.start_offset for s in text_segments]

        for i, segment in enumerate(text_segments):
            start = len(document_text)
            end = start + len(segment.text)
            segment_boundaries.append(
                SegmentBoundary(
                    segment_index=i,
                    start=start,
                    end=end,
                    node_id=segment.node_id,
                )
            )
            document_text += segment.text
            if i < len(text_segments) - 1:
                document_text += SEGMENT_SEPARATOR

        return document_text, segment_boundaries

    # Entity validation delegated to EntityProcessor
    def _validate_entities(
        self, entities: list[RecognizerResult], document_length: int
    ) -> list[RecognizerResult]:
        """Validate entity positions against document length.

        Args:
            entities: List of entities to validate
            document_length: Total length of document text

        Returns:
            List of valid entities within document bounds
        """
        # Delegate to EntityProcessor
        if self.entity_processor is None:
            # Ensure processor is initialized
            _ = self.anonymizer
        return self.entity_processor.validate_entities(entities, document_length)

    # Entity boundary validation delegated to EntityProcessor
    def _validate_entities_against_boundaries(
        self,
        entities: list[RecognizerResult],
        document_text: str,
        segment_boundaries: list[SegmentBoundary],
    ) -> list[RecognizerResult]:
        """Validate and adjust entities to not span across segment boundaries."""
        # Delegate to EntityProcessor
        if self.entity_processor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.entity_processor.validate_entities_against_boundaries(
            entities, document_text, segment_boundaries
        )

    def _prepare_strategies(
        self, entities: list[RecognizerResult], policy: MaskingPolicy
    ) -> dict[str, Strategy]:
        """Prepare masking strategies for each entity type."""
        # Delegate to MetadataManager
        if self.metadata_manager is None:
            _ = self.anonymizer  # Ensure processors are initialized
        return self.metadata_manager.prepare_strategies(entities, policy)

    def _compute_replacements(
        self,
        document_text: str,
        entities: list[RecognizerResult],
        strategies: dict[str, Strategy],
    ) -> tuple[list[OperatorResultLike], dict[tuple[int, int], OperatorResultLike]]:
        """Compute replacement operations for all entities.

        Args:
            document_text: Full document text
            entities: List of entities to mask
            strategies: Masking strategies per entity type

        Returns:
            Tuple of (operator_results, results_by_position)
        """
        # Process all entities in batch
        operator_results = self._batch_process_entities(document_text, entities, strategies)

        # Map operator results by position for correct matching
        op_results_by_pos: dict[tuple[int, int], OperatorResultLike] = {}
        for op_result in operator_results:
            key = (op_result.start, op_result.end)
            op_results_by_pos[key] = op_result

        return operator_results, op_results_by_pos

    def _create_anchor_entries(
        self,
        document_text: str,
        entities: list[RecognizerResult],
        strategies: dict[str, Strategy],
        op_results_by_pos: dict[tuple[int, int], OperatorResultLike],
        text_segments: list[TextSegment],
    ) -> list[AnchorEntry]:
        """Create anchor entries for the CloakMap.

        Args:
            document_text: Full document text
            entities: List of entities being masked
            strategies: Masking strategies per entity type
            op_results_by_pos: Operator results mapped by position
            text_segments: Original text segments

        Returns:
            List of anchor entries for CloakMap
        """
        anchor_entries = []
        import secrets
        import uuid

        # Process entities in reverse order to maintain positions
        sorted_entities = sorted(entities, key=lambda x: x.start, reverse=True)

        for entity in sorted_entities:
            # Find matching operator result by position
            key = (entity.start, entity.end)
            matched_result = op_results_by_pos.get(key)

            # If no exact match, create a synthetic result
            if matched_result is None:
                matched_result = self._create_synthetic_result(
                    entity,
                    strategies.get(
                        entity.entity_type, Strategy(StrategyKind.REDACT, {"char": "*"})
                    ),
                    document_text,
                )

            # Extract original text
            original = document_text[entity.start : entity.end]

            # Get masked value from operator result
            masked_value = (
                matched_result.text if matched_result else self._fallback_redaction(original)
            )

            # Generate secure salt (must be valid base64)
            salt_bytes = secrets.token_bytes(8)
            salt = base64.b64encode(salt_bytes).decode()
            checksum_hash = hashlib.sha256(f"{salt}{original}".encode()).hexdigest()

            # Ensure masked value is not empty
            if not masked_value:
                masked_value = "*"

            # Find the segment containing this entity
            node_id = (
                self._find_segment_for_position(entity.start, text_segments)
                if text_segments
                else "#/texts/0"
            ) or "#/texts/0"

            anchor = AnchorEntry(
                node_id=node_id,
                start=entity.start,
                end=entity.end,  # Use original end position
                entity_type=entity.entity_type,
                masked_value=masked_value,
                confidence=entity.score,
                replacement_id=f"repl_{uuid.uuid4().hex[:12]}",
                original_checksum=checksum_hash,
                checksum_salt=salt,
                strategy_used=strategies[entity.entity_type].kind.value,
                timestamp=None,
                metadata={
                    "original_text": original,
                    "presidio_operator": matched_result.operator if matched_result else "fallback",
                },
            )
            anchor_entries.append(anchor)

        return anchor_entries

    def _apply_spans(self, text: str, spans: list[tuple[int, int, str]]) -> str:
        """Apply non-overlapping replacement spans to text in O(n + k) time.

        This is much more efficient than repeated string slicing which is O(n²).

        Args:
            text: Original text
            spans: List of (start, end, replacement) tuples. Must be non-overlapping.

        Returns:
            Text with all spans applied
        """
        if not spans:
            return text

        # Sort spans by start position for sequential processing
        spans = sorted(spans, key=lambda s: s[0])

        # Build result in O(n + k) time using a list
        result = []
        cursor = 0

        for start, end, replacement in spans:
            # Add text between last replacement and this one
            if cursor < start:
                result.append(text[cursor:start])
            # Add the replacement
            result.append(replacement)
            cursor = end

        # Add any remaining text after last replacement
        if cursor < len(text):
            result.append(text[cursor:])

        return "".join(result)

    def _apply_masks_to_text(
        self,
        document_text: str,
        entities: list[RecognizerResult],
        op_results_by_pos: dict[tuple[int, int], OperatorResultLike],
    ) -> str:
        """Apply all masks to the document text using efficient O(n) algorithm.

        Args:
            document_text: Original document text
            entities: List of entities to mask
            op_results_by_pos: Operator results by position

        Returns:
            Masked text
        """
        # Build spans for efficient application
        spans = []

        for entity in entities:
            key = (entity.start, entity.end)
            matched_result = op_results_by_pos.get(key)

            if matched_result:
                masked_value = matched_result.text
            else:
                original = document_text[entity.start : entity.end]
                masked_value = self._fallback_redaction(original)

            spans.append((entity.start, entity.end, masked_value))

        # Apply all replacements in O(n) time
        return self._apply_spans(document_text, spans)

    def _create_masked_document(
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
        import json

        from docling_core.types.doc import DocItemLabel
        from docling_core.types.doc.document import TextItem

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
                    segment_entities: list[dict[str, Any]] = []
                    for anchor in anchor_entries:
                        # Check if anchor overlaps with this segment
                        if (
                            anchor.metadata
                            and "original_text" in anchor.metadata
                            and segment.start_offset <= anchor.start < segment.end_offset
                        ):
                            local_start = anchor.start - segment.start_offset
                            local_end = min(
                                local_start + len(anchor.metadata["original_text"]),
                                len(segment_text),
                            )
                            if local_start < local_end:
                                segment_entities.append(
                                    {
                                        "anchor": anchor,
                                        "local_start": local_start,
                                        "local_end": local_end,
                                    }
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
                        masked_segment_text = self._apply_spans(segment_text, segment_spans)
                    else:
                        masked_segment_text = segment_text

                    # Create new text item preserving original structure
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
                        text=masked_segment_text,
                        self_ref=(
                            original_item.self_ref
                            if hasattr(original_item, "self_ref")
                            else f"#/texts/{i}"
                        ),
                        label=item_label,
                        orig=masked_segment_text,
                    )

                    if hasattr(original_item, "prov"):
                        masked_text_item.prov = original_item.prov

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
        self._update_table_cells(masked_document, text_segments, anchor_entries)

        return masked_document

    def mask_document(
        self,
        document: DoclingDocument,
        entities: list[RecognizerResult],
        policy: MaskingPolicy,
        text_segments: list[TextSegment],
        original_format: str | None = None,
    ) -> MaskingResult:
        """
        Mask PII entities in a document using Presidio.

        This method orchestrates the masking process by delegating to
        specialized helper methods, making the flow clear and testable.

        Args:
            document: The DoclingDocument to mask
            entities: List of detected PII entities
            policy: Masking policy defining strategies per entity type
            text_segments: Text segments extracted from the document
            original_format: Original document format

        Returns:
            MaskingResult containing masked document and enhanced CloakMap
        """
        logger.info(
            f"Masking document {document.name} with {len(entities)} entities using Presidio"
        )

        # Step 1: Filter overlapping entities
        filtered_entities = self._filter_overlapping_entities(entities)

        # Step 2: Prepare strategies
        strategies = self._prepare_strategies(filtered_entities, policy)

        # Step 3: Build full text and boundaries
        document_text, segment_boundaries = self._build_full_text_and_boundaries(text_segments)

        # Step 4: Validate entities
        valid_entities = self._validate_entities(filtered_entities, len(document_text))

        # Step 4b: Validate entities don't span segment boundaries
        valid_entities = self._validate_entities_against_boundaries(
            valid_entities, document_text, segment_boundaries
        )

        # Step 5: Compute replacements
        operator_results, op_results_by_pos = self._compute_replacements(
            document_text, valid_entities, strategies
        )

        # Step 6: Create anchor entries
        anchor_entries = self._create_anchor_entries(
            document_text, valid_entities, strategies, op_results_by_pos, text_segments
        )

        # Step 7: Apply masks to get masked text
        masked_text = self._apply_masks_to_text(document_text, valid_entities, op_results_by_pos)

        # Step 8: Create masked document
        masked_document = self._create_masked_document(
            document, text_segments, anchor_entries, masked_text
        )

        # Step 9: Build CloakMap
        base_cloakmap = CloakMap.create(
            doc_id=document.name,
            doc_hash=hashlib.sha256(document_text.encode()).hexdigest(),
            anchors=anchor_entries,
            policy=policy,
            metadata={"original_format": original_format} if original_format else {},
        )

        # Step 10: Enhance with metadata
        enhanced_cloakmap = self._enhance_cloakmap_with_metadata(
            base_cloakmap,
            valid_entities,
            op_results_by_pos,
            strategies,
            document_text,
        )

        # Step 11: Calculate statistics
        stats = {
            "entities_masked": len(entities),
            "unique_entity_types": len(strategies),
            "presidio_engine_used": True,
            "fallback_used": any(
                a.metadata and a.metadata.get("presidio_operator") == "fallback"
                for a in anchor_entries
            ),
        }

        return MaskingResult(
            masked_document=masked_document, cloakmap=enhanced_cloakmap, stats=stats
        )

    def _enhance_cloakmap_with_metadata(
        self,
        base_cloakmap: CloakMap,
        entities: list[RecognizerResult],
        op_results_by_pos: dict[tuple[int, int], OperatorResultLike],
        strategies: dict[str, Strategy],
        document_text: str,
    ) -> CloakMap:
        """Enhance CloakMap with Presidio metadata.

        Args:
            base_cloakmap: Base CloakMap to enhance
            entities: List of entities that were masked
            op_results_by_pos: Operator results by position
            strategies: Strategies used for masking
            document_text: Original document text

        Returns:
            Enhanced CloakMap
        """
        # Delegate to MetadataManager
        if self.metadata_manager is None:
            _ = self.anonymizer  # Ensure processor is initialized

        # Convert op_results_by_pos to list for metadata manager
        op_results = list(op_results_by_pos.values())
        return self.metadata_manager.enhance_cloakmap_with_metadata(
            base_cloakmap, strategies, entities, op_results
        )

    # Overlap filtering delegated to EntityProcessor
    def _filter_overlapping_entities(
        self, entities: list[RecognizerResult]
    ) -> list[RecognizerResult]:
        """Filter out overlapping entities, keeping the highest confidence or longest match."""
        # Delegate to EntityProcessor
        if self.entity_processor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.entity_processor.filter_overlapping_entities(entities)

    # Batch processing delegated to EntityProcessor
    def _batch_process_entities(
        self, text: str, entities: list[RecognizerResult], strategies: dict[str, Strategy]
    ) -> list[OperatorResultLike]:
        """Process multiple entities in a single batch for efficiency."""
        # Delegate to EntityProcessor
        if self.entity_processor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.entity_processor.batch_process_entities(text, entities, strategies)

    def _apply_hash_strategy(
        self, text: str, entity_type: str, strategy: Strategy, confidence: float
    ) -> str:
        """Apply hash strategy with support for prefix and other parameters."""
        # Delegate to StrategyProcessor
        if self.strategy_processor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.strategy_processor.apply_hash_strategy(text, entity_type, strategy, confidence)

    def _apply_partial_strategy(
        self, text: str, entity_type: str, strategy: Strategy, confidence: float
    ) -> str:
        """Apply partial masking strategy with proper char count calculation."""
        # Delegate to StrategyProcessor
        if self.strategy_processor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.strategy_processor.apply_partial_strategy(text, entity_type, strategy, confidence)

    def _apply_custom_strategy(self, text: str, strategy: Strategy) -> str:
        """Apply a custom strategy using the provided callback."""
        # Delegate to StrategyProcessor
        if self.strategy_processor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.strategy_processor.apply_custom_strategy(text, strategy)

    def _apply_surrogate_strategy(self, text: str, entity_type: str, strategy: Strategy) -> str:
        """Apply surrogate strategy with high-quality fake data generation."""
        # Delegate to StrategyProcessor
        if self.strategy_processor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.strategy_processor.apply_surrogate_strategy(text, entity_type, strategy)

    def _fallback_redaction(self, text: str) -> str:
        """Simple fallback redaction when Presidio fails."""
        # Delegate to StrategyProcessor
        if self.strategy_processor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.strategy_processor._fallback_redaction(text)

    def _operator_result_to_dict(self, result: OperatorResultLike) -> dict[str, Any]:
        """Convert OperatorResult to dictionary for storage."""
        # Delegate to MetadataManager
        if self.metadata_manager is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.metadata_manager.operator_result_to_dict(result)

    def _create_synthetic_result(
        self, entity: RecognizerResult, strategy: Strategy, text: str
    ) -> SyntheticOperatorResult:
        """Create a synthetic OperatorResult for fallback scenarios."""
        # Delegate to EntityProcessor
        if self.entity_processor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.entity_processor.create_synthetic_result(entity, strategy, text)

    def _get_reversible_operators(self, strategies: dict[str, Strategy]) -> list[str]:
        """Identify which operators are reversible.

        Reversible operations are those that can be undone to restore the original text.
        Non-reversible operations like REDACT, HASH, and SURROGATE cannot be reversed.
        """
        # Delegate to MetadataManager
        if self.metadata_manager is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.metadata_manager.get_reversible_operators(strategies)

    def _find_segment_for_position(self, position: int, segments: list[TextSegment]) -> str | None:
        """Find the segment node_id for a given character position using binary search.

        Args:
            position: Character position in the full text
            segments: List of text segments with position information (assumed sorted)

        Returns:
            Node ID of the containing segment, or None if not found
        """
        # Delegate to TextProcessor
        if self.text_processor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.text_processor.find_segment_for_position(position, segments)

    def _update_table_cells(
        self,
        masked_document: DoclingDocument,
        text_segments: list[TextSegment],
        anchor_entries: list[AnchorEntry],
    ) -> None:
        """Update table cells with masked values based on anchors.

        Args:
            masked_document: The document with tables to update
            text_segments: Original text segments with metadata
            anchor_entries: Anchors containing masked values
        """
        # Delegate to DocumentReconstructor
        if self.document_reconstructor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        self.document_reconstructor.update_table_cells(
            masked_document, text_segments, anchor_entries
        )

    def _get_table_node_id(self, table_item: Any) -> str:
        """Get the node ID for a table item."""
        # Delegate to DocumentReconstructor
        if self.document_reconstructor is None:
            _ = self.anonymizer  # Ensure processor is initialized
        return self.document_reconstructor._get_table_node_id(table_item)

    def _cleanup_large_results(self, results: list[OperatorResultLike]) -> None:
        """Clean up large result sets for memory efficiency.

        For results exceeding memory thresholds:
        - Clear text fields from operator results after processing
        - Remove redundant metadata fields
        - Release references to large intermediate objects
        """
        # Delegate to MetadataManager
        if self.metadata_manager is None:
            _ = self.anonymizer  # Ensure processor is initialized
        self.metadata_manager.cleanup_large_results(results)
