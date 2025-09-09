"""Core MaskingEngine for orchestrating PII masking operations."""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from presidio_analyzer import RecognizerResult

from presidio_analyzer import RecognizerResult

from cloakpivot.core.types import DoclingDocument

from ..core.analyzer import EntityDetectionResult
from ..core.anchors import AnchorEntry
from ..core.cloakmap import CloakMap
from ..core.normalization import ConflictResolutionConfig, EntityNormalizer
from ..core.policies import MaskingPolicy
from ..core.strategies import Strategy
from ..document.extractor import TextSegment
from .applicator import StrategyApplicator
from .document_masker import DocumentMasker

logger = logging.getLogger(__name__)


@dataclass
class MaskingResult:
    """
    Result of a masking operation containing the masked document and CloakMap.

    Attributes:
        masked_document: The DoclingDocument with PII masked
        cloakmap: CloakMap containing anchor entries for reversibility
        stats: Statistics about the masking operation
    """

    masked_document: DoclingDocument
    cloakmap: CloakMap
    stats: dict[str, Any] | None = None


class MaskingEngine:
    """
    Core masking engine that orchestrates PII masking operations.

    This engine coordinates the masking process by:
    1. Resolving masking strategies for detected entities
    2. Generating masked replacement tokens
    3. Modifying the DoclingDocument with masked content
    4. Creating CloakMap entries for reversibility

    Examples:
        >>> engine = MaskingEngine()
        >>> result = engine.mask_document(
        ...     document=doc,
        ...     entities=detected_entities,
        ...     policy=masking_policy,
        ...     text_segments=segments
        ... )
        >>> print(f"Masked {len(result.cloakmap.anchors)} entities")
    """

    def __init__(
        self,
        resolve_conflicts: bool = False,
        conflict_resolution_config: ConflictResolutionConfig | None = None,
        store_original_text: bool = True,
        use_presidio_engine: bool | None = None,
    ) -> None:
        """Initialize the masking engine.

        Args:
            resolve_conflicts: Whether to resolve entity conflicts or raise errors (default: False for backward compatibility)
            conflict_resolution_config: Configuration for conflict resolution (uses defaults if None)
            store_original_text: Whether to store original text in metadata for round-trip functionality (default: True)
            use_presidio_engine: Whether to use Presidio engine instead of legacy. If None, checks environment variable (default: None)
        """
        # Determine which engine to use
        self.use_presidio = self._determine_presidio_usage(use_presidio_engine)

        # Initialize the appropriate masking adapter

        from .presidio_adapter import PresidioMaskingAdapter

        self.presidio_adapter: PresidioMaskingAdapter | None
        self.strategy_applicator: StrategyApplicator | None

        if self.use_presidio:
            self.presidio_adapter = PresidioMaskingAdapter()
            self.strategy_applicator = None  # Not used in Presidio mode
            logger.info("MaskingEngine using Presidio adapter")
        else:
            # Issue deprecation warning for legacy engine
            from ..migration import DeprecationManager
            DeprecationManager.warn_legacy_usage(
                "masking engine", 
                "set use_presidio_engine=True"
            )
            self.strategy_applicator = StrategyApplicator()
            self.presidio_adapter = None  # Not used in legacy mode
            logger.info("MaskingEngine using legacy StrategyApplicator")

        self.document_masker = DocumentMasker()
        self.resolve_conflicts = resolve_conflicts
        self.store_original_text = store_original_text
        self.entity_normalizer = (
            EntityNormalizer(conflict_resolution_config or ConflictResolutionConfig())
            if resolve_conflicts
            else None
        )
        logger.debug(
            f"MaskingEngine initialized with resolve_conflicts={resolve_conflicts}, "
            f"store_original_text={store_original_text}, use_presidio={self.use_presidio}"
        )

    def _determine_presidio_usage(self, explicit_flag: bool | None) -> bool:
        """Determine whether to use Presidio engine based on various sources.

        Priority order:
        1. Explicit parameter (if provided)
        2. Environment variable CLOAKPIVOT_USE_PRESIDIO_ENGINE
        3. Default to False (legacy engine for safety)

        Args:
            explicit_flag: Explicit flag from constructor parameter

        Returns:
            True if Presidio engine should be used, False otherwise
        """
        # 1. Explicit parameter takes precedence
        if explicit_flag is not None:
            logger.debug(f"Using explicit flag for Presidio engine: {explicit_flag}")
            return explicit_flag

        # 2. Check environment variable
        env_flag = os.getenv("CLOAKPIVOT_USE_PRESIDIO_ENGINE")
        if env_flag is not None:
            use_presidio = env_flag.strip().lower() in ("true", "1", "yes", "on")
            logger.debug(f"Using environment variable for Presidio engine: {use_presidio}")
            return use_presidio

        # 3. Default behavior (False for safety - use legacy engine)
        logger.debug("No Presidio engine preference found, defaulting to legacy engine")
        return False

    def mask_document(
        self,
        document: DoclingDocument,
        entities: list[RecognizerResult],
        policy: MaskingPolicy,
        text_segments: list[TextSegment],
        original_format: str | None = None,
    ) -> MaskingResult:
        """
        Mask PII entities in a document according to the given policy.

        Args:
            document: The DoclingDocument to mask
            entities: List of detected PII entities from Presidio
            policy: Masking policy defining strategies per entity type
            text_segments: Text segments extracted from the document
            original_format: Original document format for round-trip preservation

        Returns:
            MaskingResult containing masked document and CloakMap

        Raises:
            ValueError: If entities overlap or input validation fails
        """
        logger.info(f"Masking document {document.name} with {len(entities)} entities")

        # Validate inputs
        self._validate_inputs(document, entities, policy, text_segments)

        # Route to appropriate engine based on configuration
        if self.use_presidio:
            return self._mask_with_presidio(
                document, entities, policy, text_segments, original_format
            )
        else:
            return self._mask_with_legacy(
                document, entities, policy, text_segments, original_format
            )

    def _mask_with_presidio(
        self,
        document: DoclingDocument,
        entities: list[RecognizerResult],
        policy: MaskingPolicy,
        text_segments: list[TextSegment],
        original_format: str | None = None,
    ) -> MaskingResult:
        """Mask document using Presidio engine."""
        logger.debug("Using Presidio engine for masking")

        if self.presidio_adapter is None:
            raise RuntimeError("Presidio adapter not initialized")

        # Convert analyzer RecognizerResult to anonymizer RecognizerResult
        from presidio_anonymizer.entities import (
            RecognizerResult as AnonymizerRecognizerResult,
        )

        anonymizer_entities = [
            AnonymizerRecognizerResult(
                entity_type=entity.entity_type,
                start=entity.start,
                end=entity.end,
                score=entity.score
            )
            for entity in entities
        ]

        # Delegate to PresidioMaskingAdapter
        return self.presidio_adapter.mask_document(
            document=document,
            entities=anonymizer_entities,
            policy=policy,
            text_segments=text_segments,
            original_format=original_format
        )

    def _mask_with_legacy(
        self,
        document: DoclingDocument,
        entities: list[RecognizerResult],
        policy: MaskingPolicy,
        text_segments: list[TextSegment],
        original_format: str | None = None,
    ) -> MaskingResult:
        """Mask document using legacy StrategyApplicator engine."""
        logger.debug("Using legacy StrategyApplicator for masking")

        if self.strategy_applicator is None:
            raise RuntimeError("Strategy applicator not initialized for legacy mode")

        # Resolve entity conflicts or check for overlaps
        resolved_entities = self._resolve_entity_conflicts(entities, text_segments)

        # Generate masked replacements for each resolved entity
        anchor_entries = []
        for entity in resolved_entities:
            # Find the text segment containing this entity
            segment = self._find_segment_for_entity(entity, text_segments)
            if not segment:
                logger.warning(
                    f"No segment found for entity at {entity.start}-{entity.end}"
                )
                continue

            # Get strategy for this entity type
            strategy = policy.get_strategy_for_entity(entity.entity_type)

            # Extract original text
            global_start = entity.start
            global_end = entity.end
            relative_start = global_start - segment.start_offset
            relative_end = global_end - segment.start_offset
            original_text = segment.text[relative_start:relative_end]

            # Generate masked replacement
            masked_value = self.strategy_applicator.apply_strategy(
                original_text=original_text,
                entity_type=entity.entity_type,
                strategy=strategy,
                confidence=entity.score,
            )

            # Create anchor entry
            anchor = self._create_anchor_entry(
                segment=segment,
                entity=entity,
                original_text=original_text,
                masked_value=masked_value,
                strategy=strategy,
                relative_start=relative_start,
                relative_end=relative_end,
            )
            anchor_entries.append(anchor)

        # Create a copy of the document to modify
        masked_document = self._copy_document(document)

        # Apply masking to the document
        self.document_masker.apply_masking(
            document=masked_document, anchor_entries=anchor_entries
        )

        # Generate document hash for CloakMap
        doc_hash = self._compute_document_hash(document)

        # Create CloakMap with original format metadata
        metadata = {}
        if original_format:
            metadata["original_format"] = original_format

        cloakmap = CloakMap.create(
            doc_id=document.name or "unnamed_document",
            doc_hash=doc_hash,
            anchors=anchor_entries,
            policy=policy,
            metadata=metadata,
        )

        # Generate statistics
        stats = self._generate_stats(entities, anchor_entries, policy)

        logger.info(f"Masking completed: {len(anchor_entries)} entities masked")

        return MaskingResult(
            masked_document=masked_document, cloakmap=cloakmap, stats=stats
        )

    def _validate_inputs(
        self,
        document: DoclingDocument,
        entities: list[RecognizerResult],
        policy: MaskingPolicy,
        text_segments: list[TextSegment],
    ) -> None:
        """Validate input parameters."""
        if not isinstance(document, DoclingDocument):
            raise ValueError("document must be a DoclingDocument")

        if not isinstance(entities, list):
            raise ValueError("entities must be a list")

        for entity in entities:
            # Check for RecognizerResult-like object (duck typing for compatibility)
            if not hasattr(entity, 'entity_type') or not hasattr(entity, 'start') or not hasattr(entity, 'end') or not hasattr(entity, 'score'):
                raise ValueError("all entities must be RecognizerResult-like instances with entity_type, start, end, and score attributes")

        if not isinstance(policy, MaskingPolicy):
            raise ValueError("policy must be a MaskingPolicy")

        if not isinstance(text_segments, list):
            raise ValueError("text_segments must be a list")

        for segment in text_segments:
            if not isinstance(segment, TextSegment):
                raise ValueError("all text_segments must be TextSegment instances")

    def _resolve_entity_conflicts(
        self,
        entities: list[RecognizerResult],
        text_segments: list[TextSegment],
    ) -> list[RecognizerResult]:
        """Resolve entity conflicts using EntityNormalizer or validate no overlaps.

        This method handles entity conflicts in two modes based on the resolve_conflicts flag:

        1. Legacy mode (resolve_conflicts=False): Validates that no overlapping entities exist
           and raises ValueError if any are found, maintaining backward compatibility.

        2. Conflict resolution mode (resolve_conflicts=True): Uses EntityNormalizer to
           intelligently resolve overlapping and adjacent entities through merging,
           priority-based selection, or confidence-based resolution.

        Args:
            entities: List of RecognizerResult instances from Presidio analysis
            text_segments: List of TextSegment instances containing document text

        Returns:
            List of RecognizerResult instances with conflicts resolved (if enabled)
            or original entities (if validation-only mode)

        Raises:
            ValueError: If resolve_conflicts=False and overlapping entities are detected

        Note:
            The method converts between RecognizerResult and EntityDetectionResult types
            internally to leverage the EntityNormalizer's conflict resolution capabilities.
            Text extraction is performed relative to each segment's boundaries to ensure
            correct entity text content is preserved during the conversion process.
        """
        if not self.resolve_conflicts:
            # Legacy behavior: check for overlaps and raise error if found
            self._check_overlapping_entities(entities, text_segments)
            return entities

        if not entities:
            return entities

        # Convert RecognizerResult to EntityDetectionResult
        entity_detection_results = []
        text_by_segment = {}  # Cache text content by segment

        for entity in entities:
            segment = self._find_segment_for_entity(entity, text_segments)
            if not segment:
                logger.warning(
                    f"No segment found for entity at {entity.start}-{entity.end}, skipping"
                )
                continue

            # Get segment text for entity text extraction
            if segment.node_id not in text_by_segment:
                text_by_segment[segment.node_id] = segment.text

            segment_text = text_by_segment[segment.node_id]

            # Convert to relative positions within segment
            segment_relative_start = entity.start - segment.start_offset
            segment_relative_end = entity.end - segment.start_offset

            # Validate entity bounds relative to segment
            if segment_relative_start < 0:
                logger.warning(
                    f"Entity start {entity.start} before segment start {segment.start_offset}, "
                    f"adjusting to segment boundary"
                )
                segment_relative_start = 0

            if segment_relative_end > len(segment_text):
                logger.warning(
                    f"Entity end {entity.end} beyond segment text length {len(segment_text)}, "
                    f"adjusting to segment boundary"
                )
                segment_relative_end = len(segment_text)

            if segment_relative_start >= segment_relative_end:
                logger.warning(
                    f"Invalid entity bounds after adjustment: start={segment_relative_start}, "
                    f"end={segment_relative_end}, skipping entity"
                )
                continue

            # Extract entity text with bounds checking
            try:
                entity_text = segment_text[segment_relative_start:segment_relative_end]
                if not entity_text.strip():
                    logger.warning(
                        f"Empty or whitespace-only entity text at {entity.start}-{entity.end}, skipping"
                    )
                    continue

            except (IndexError, TypeError) as e:
                logger.warning(
                    f"Failed to extract entity text at {entity.start}-{entity.end}: {e}, skipping"
                )
                continue

            # Create detection result with error handling
            try:
                detection_result = EntityDetectionResult.from_presidio_result(
                    entity, entity_text
                )
                entity_detection_results.append(detection_result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to create EntityDetectionResult for entity at {entity.start}-{entity.end}: {e}, skipping"
                )
                continue

        # Use EntityNormalizer to resolve conflicts
        if self.entity_normalizer is None:
            raise ValueError(
                "Entity normalizer is not configured but conflict resolution was attempted"
            )
        normalization_result = self.entity_normalizer.normalize_entities(
            entity_detection_results
        )

        logger.info(
            f"Entity conflict resolution: {len(entities)} -> {len(normalization_result.normalized_entities)} entities, "
            f"{normalization_result.conflicts_resolved} conflicts resolved"
        )

        # Convert back to RecognizerResult
        resolved_entities = []
        for detection_result in normalization_result.normalized_entities:
            recognizer_result = RecognizerResult(
                entity_type=detection_result.entity_type,
                start=detection_result.start,
                end=detection_result.end,
                score=detection_result.confidence,
            )
            resolved_entities.append(recognizer_result)

        return resolved_entities

    def _check_overlapping_entities(
        self,
        entities: list[RecognizerResult],
        text_segments: list[TextSegment],
    ) -> None:
        """Check for overlapping entities and raise error if found."""
        # Group entities by their containing text segment
        segment_entities: dict[str, list[RecognizerResult]] = {}

        for entity in entities:
            segment = self._find_segment_for_entity(entity, text_segments)
            if segment:
                if segment.node_id not in segment_entities:
                    segment_entities[segment.node_id] = []
                segment_entities[segment.node_id].append(entity)

        # Check for overlaps within each segment
        for _node_id, node_entities in segment_entities.items():
            for i, entity1 in enumerate(node_entities):
                for entity2 in node_entities[i + 1 :]:
                    if self._entities_overlap(entity1, entity2):
                        raise ValueError(
                            f"Overlapping entities detected: "
                            f"{entity1.entity_type}({entity1.start}-"
                            f"{entity1.end}) and "
                            f"{entity2.entity_type}({entity2.start}-"
                            f"{entity2.end})"
                        )

    def _entities_overlap(
        self, entity1: RecognizerResult, entity2: RecognizerResult
    ) -> bool:
        """Check if two entities overlap."""
        return not (entity1.end <= entity2.start or entity2.end <= entity1.start)

    def _find_segment_for_entity(
        self, entity: RecognizerResult, text_segments: list[TextSegment]
    ) -> TextSegment | None:
        """Find the text segment that contains the given entity."""
        for segment in text_segments:
            if segment.contains_offset(entity.start) and segment.contains_offset(
                entity.end - 1
            ):
                return segment
        return None

    def _create_anchor_entry(
        self,
        segment: TextSegment,
        entity: RecognizerResult,
        original_text: str,
        masked_value: str,
        strategy: Strategy,
        relative_start: int,
        relative_end: int,
    ) -> AnchorEntry:
        """Create an AnchorEntry for a masked entity."""
        # Generate unique replacement ID
        replacement_id = f"repl_{uuid.uuid4().hex[:12]}"

        # Store metadata - include original text only if enabled
        metadata = {
            "segment_node_id": segment.node_id,
            "segment_node_type": segment.node_type,
            "global_start": entity.start,
            "global_end": entity.end,
        }

        if self.store_original_text:
            metadata["original_text"] = original_text

        # Use factory method to create anchor with salted checksum
        return AnchorEntry.create_from_detection(
            node_id=segment.node_id,
            start=relative_start,
            end=relative_end,
            entity_type=entity.entity_type,
            confidence=entity.score,
            original_text=original_text,
            masked_value=masked_value,
            strategy_used=strategy.kind.value,
            replacement_id=replacement_id,
            metadata=metadata,
        )

    def _copy_document(self, document: DoclingDocument) -> DoclingDocument:
        """Create a deep copy of the document for masking."""
        # For now, use simple copy approach
        # In a production implementation, we'd use a proper deep copy mechanism
        return copy.deepcopy(document)

    def _compute_document_hash(self, document: DoclingDocument) -> str:
        """Compute SHA-256 hash of the document content."""
        # Simple hash based on text content
        # In production, this would be more comprehensive
        content = ""
        for text_item in document.texts:
            if hasattr(text_item, "text") and text_item.text:
                content += text_item.text

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _generate_stats(
        self,
        entities: list[RecognizerResult],
        anchor_entries: list[AnchorEntry],
        policy: MaskingPolicy,
    ) -> dict[str, Any]:
        """Generate statistics about the masking operation."""
        entity_counts: dict[str, int] = {}
        strategy_counts: dict[str, int] = {}

        for anchor in anchor_entries:
            entity_counts[anchor.entity_type] = (
                entity_counts.get(anchor.entity_type, 0) + 1
            )
            strategy_counts[anchor.strategy_used] = (
                strategy_counts.get(anchor.strategy_used, 0) + 1
            )

        return {
            "total_entities_detected": len(entities),
            "total_entities_masked": len(anchor_entries),
            "entity_type_counts": entity_counts,
            "strategy_counts": strategy_counts,
            "timestamp": datetime.utcnow().isoformat(),
        }
