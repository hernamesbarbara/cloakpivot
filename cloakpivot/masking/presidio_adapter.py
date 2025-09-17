"""Presidio-based masking adapter for CloakPivot."""

import bisect
import copy
import hashlib
import logging
import threading
from typing import Any, cast

from presidio_anonymizer import AnonymizerEngine, OperatorResult
from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

try:
    import presidio_anonymizer

    PRESIDIO_VERSION = getattr(presidio_anonymizer, "__version__", "2.x.x")
except (ImportError, AttributeError):
    PRESIDIO_VERSION = "2.x.x"

from ..core.cloakmap import AnchorEntry, CloakMap
from ..core.cloakmap_enhancer import CloakMapEnhancer
from ..core.policies import MaskingPolicy
from ..core.presidio_mapper import StrategyToOperatorMapper
from ..core.strategies import Strategy, StrategyKind
from ..core.surrogate import SurrogateGenerator
from ..core.types import DoclingDocument
from ..document.extractor import TextSegment
from .engine import MaskingResult
from .protocols import (
    OperatorResultLike,
    SegmentBoundary,
    SyntheticOperatorResult,
)

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

        This method maintains compatibility with the existing MaskingEngine API
        while using Presidio for all masking operations.

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

        # Filter out overlapping entities to avoid conflicts
        filtered_entities = self._filter_overlapping_entities(entities)

        # Prepare strategies for each entity type
        strategies = {}
        for entity in filtered_entities:
            if entity.entity_type not in strategies:
                strategies[entity.entity_type] = policy.get_strategy_for_entity(entity.entity_type)

        # Build the full document text from segments (preserving structure)
        document_text = ""
        segment_boundaries: list[SegmentBoundary] = []
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
                document_text += SEGMENT_SEPARATOR  # Add separator between segments

        # Validate entity positions against document text length
        valid_entities = []
        for entity in filtered_entities:
            if entity.end <= len(document_text):
                valid_entities.append(entity)
            else:
                logger.warning(
                    f"Entity {entity.entity_type} at positions {entity.start}-{entity.end} "
                    f"exceeds document text length {len(document_text)}, skipping"
                )

        # Process all valid entities in batch
        operator_results = self._batch_process_entities(document_text, valid_entities, strategies)

        # Create anchor entries for CloakMap
        anchor_entries = []
        masked_text = document_text

        # Map operator results by position for correct matching
        op_results_by_pos: dict[tuple[int, int], OperatorResultLike] = {}
        for op_result in operator_results:
            key = (op_result.start, op_result.end)
            op_results_by_pos[key] = op_result

        # Process entities in reverse order to maintain positions
        sorted_entities = sorted(valid_entities, key=lambda x: x.start, reverse=True)
        entity_to_op_result = []

        for entity in sorted_entities:
            # Find matching operator result by position
            key = (entity.start, entity.end)
            matched_result: OperatorResultLike | None = op_results_by_pos.get(key)

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

            # Apply mask to text
            masked_text = masked_text[: entity.start] + masked_value + masked_text[entity.end :]

            # Create anchor entry with proper fields
            import base64
            import secrets
            import uuid

            # Generate secure salt using secrets module
            salt_bytes = secrets.token_bytes(8)  # 8 bytes = 64 bits of entropy
            salt = base64.b64encode(salt_bytes).decode()
            checksum_hash = hashlib.sha256(f"{salt}{original}".encode()).hexdigest()

            # Ensure masked value is not empty (for anchor validation)
            if not masked_value:
                masked_value = "*"  # Use single char as fallback

            # Find the segment containing this entity
            found_node_id = (
                self._find_segment_for_position(entity.start, text_segments)
                if text_segments
                else None
            )
            node_id = found_node_id if found_node_id is not None else "#/texts/0"

            anchor = AnchorEntry(
                node_id=node_id,
                start=entity.start,
                end=entity.start + len(masked_value),
                entity_type=entity.entity_type,
                masked_value=masked_value,
                confidence=entity.score,
                replacement_id=f"repl_{uuid.uuid4().hex[:12]}",
                original_checksum=checksum_hash,
                checksum_salt=salt,
                strategy_used=strategies[entity.entity_type].kind.value,
                timestamp=None,  # Will be set by CloakMap creation
                metadata={
                    "original_text": original,  # Store for reversibility in metadata
                    "presidio_operator": matched_result.operator if matched_result else "fallback",
                },
            )
            anchor_entries.append(anchor)
            # Track entity to operator result mapping for metadata
            entity_to_op_result.append((entity, matched_result))

        # Create masked document preserving original structure
        # Serialize the document to preserve all structure
        import json

        from docling_core.types.doc import DocItemLabel
        from docling_core.types.doc.document import TextItem

        doc_dict = json.loads(document.model_dump_json())

        # We'll update the texts in the dictionary later with masked versions
        # For now, create the masked document from the original structure
        masked_document = DoclingDocument.model_validate(doc_dict)

        # Instead of trying to split the masked text, apply masking to each segment individually
        if hasattr(document, "texts") and document.texts:
            from ..document.mapper import AnchorMapper

            AnchorMapper()

            masked_segments = []

            for i, original_item in enumerate(document.texts):
                if i < len(text_segments):
                    segment = text_segments[i]
                    segment_text = original_item.text

                    # Find entities that affect this segment
                    segment_entities: list[dict[str, Any]] = []
                    for entity in sorted_entities:
                        # Check if entity overlaps with this segment
                        if entity.start < segment.end_offset and entity.end > segment.start_offset:
                            # Calculate local positions within the segment
                            local_start = max(0, entity.start - segment.start_offset)
                            local_end = min(len(segment_text), entity.end - segment.start_offset)

                            # Only add if there's actual overlap
                            if local_start < local_end:
                                segment_entities.append(
                                    {
                                        "entity": entity,
                                        "local_start": local_start,
                                        "local_end": local_end,
                                    }
                                )

                    # Apply masks to this segment
                    masked_segment_text = segment_text
                    # Sort by position (reverse to maintain positions)
                    segment_entities.sort(key=lambda x: x["local_start"], reverse=True)

                    for entity_info in segment_entities:
                        entity = cast(RecognizerResult, entity_info["entity"])
                        local_start = cast(int, entity_info["local_start"])
                        local_end = cast(int, entity_info["local_end"])

                        # Find the anchor entry for this entity
                        matching_anchor: AnchorEntry | None = next(
                            (
                                a
                                for a in anchor_entries
                                if a.metadata
                                and a.metadata.get("original_text")
                                == document_text[entity.start : entity.end]
                            ),
                            None,
                        )

                        if matching_anchor:
                            # Apply the mask
                            masked_segment_text = (
                                masked_segment_text[:local_start]
                                + matching_anchor.masked_value
                                + masked_segment_text[local_end:]
                            )

                    # Create new text item preserving original structure
                    # Map labels that aren't valid for TextItem to TEXT
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

                    # Preserve other attributes if they exist
                    if hasattr(original_item, "prov"):
                        masked_text_item.prov = original_item.prov

                    masked_segments.append(masked_text_item)
                else:
                    # Fallback: copy original item if no boundary found
                    masked_segments.append(copy.deepcopy(original_item))

            # Update texts in place to preserve references
            for i, masked_item in enumerate(masked_segments):
                if i < len(masked_document.texts):
                    # Update the text content in place
                    masked_document.texts[i].text = masked_item.text
                    if hasattr(masked_item, "orig"):
                        masked_document.texts[i].orig = masked_item.orig

        # Also preserve _main_text for backward compatibility
        if hasattr(document, "_main_text"):
            # Set _main_text attribute for backward compatibility
            setattr(masked_document, "_main_text", masked_text)  # noqa: B010

        # Update table cells with masked values
        self._update_table_cells(masked_document, text_segments, anchor_entries)

        # Create base CloakMap
        base_cloakmap = CloakMap.create(
            doc_id=document.name,
            doc_hash=hashlib.sha256(document_text.encode()).hexdigest(),
            anchors=anchor_entries,
            policy=policy,
            metadata={"original_format": original_format} if original_format else {},
        )

        # Enhance with Presidio metadata - include original text only for reversible operations
        enhanced_operator_results = []
        reversible_operators = self._get_reversible_operators(strategies)

        for entity, op_result in entity_to_op_result:
            op_dict = self._operator_result_to_dict(op_result)

            # Only add original text for reversible operations
            operator = op_dict.get("operator", "")
            if operator in reversible_operators:
                op_dict["original_text"] = document_text[entity.start : entity.end]

            enhanced_operator_results.append(op_dict)

        # Only enhance with Presidio metadata if there are results
        if enhanced_operator_results:
            enhanced_cloakmap = self.cloakmap_enhancer.add_presidio_metadata(
                base_cloakmap,
                operator_results=enhanced_operator_results,
                engine_version=PRESIDIO_VERSION,
                reversible_operators=self._get_reversible_operators(strategies),
            )
        else:
            enhanced_cloakmap = base_cloakmap

        # Calculate statistics
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

    def _filter_overlapping_entities(
        self, entities: list[RecognizerResult]
    ) -> list[RecognizerResult]:
        """Filter out overlapping entities, keeping the highest confidence or longest match.

        Args:
            entities: List of detected entities that may overlap

        Returns:
            Filtered list with no overlapping entities
        """
        if not entities:
            return []

        # Sort by start position, then by score (descending), then by length (descending)
        sorted_entities = sorted(entities, key=lambda e: (e.start, -e.score, -(e.end - e.start)))

        filtered = []
        last_end = -1

        for entity in sorted_entities:
            # Skip if this entity overlaps with a previously selected one
            if entity.start >= last_end:
                filtered.append(entity)
                last_end = entity.end
            else:
                # Log that we're skipping an overlapping entity
                logger.debug(
                    f"Skipping overlapping entity: {entity.entity_type} at {entity.start}-{entity.end}"
                )

        return filtered

    def _batch_process_entities(
        self, text: str, entities: list[RecognizerResult], strategies: dict[str, Strategy]
    ) -> list[OperatorResultLike]:
        """
        Process multiple entities in a single batch for efficiency.

        Args:
            text: The text containing entities
            entities: List of entities to mask
            strategies: Mapping of entity types to strategies

        Returns:
            List of OperatorResult objects
        """
        try:
            # Separate SURROGATE entities from others
            surrogate_entities = []
            presidio_entities = []
            for entity in entities:
                entity_type = getattr(entity, "entity_type", "UNKNOWN")
                strategy = strategies.get(entity_type)
                if strategy and strategy.kind == StrategyKind.SURROGATE:
                    surrogate_entities.append(entity)
                else:
                    presidio_entities.append(entity)

            # Process SURROGATE entities manually
            text_result = text
            surrogate_results: list[OperatorResultLike] = []

            # Process surrogate entities in reverse order to maintain positions
            for entity in sorted(surrogate_entities, key=lambda x: x.start, reverse=True):
                entity_type = getattr(entity, "entity_type", "UNKNOWN")
                strategy = strategies.get(entity_type)
                if not strategy:
                    # This shouldn't happen as we filtered for SURROGATE entities
                    continue
                original_value = text[entity.start : entity.end]

                # Generate surrogate value
                surrogate_value = self._apply_surrogate_strategy(
                    original_value, entity_type, strategy
                )
                logger.debug(
                    f"SURROGATE: Replacing '{original_value}' with '{surrogate_value}' for {entity_type}"
                )

                # Replace in text
                text_result = (
                    text_result[: entity.start] + surrogate_value + text_result[entity.end :]
                )

                # Create a SyntheticOperatorResult for tracking
                op_result = SyntheticOperatorResult(
                    start=entity.start,
                    end=entity.end,
                    entity_type=entity_type,
                    text=surrogate_value,
                    operator="surrogate",
                )
                surrogate_results.append(op_result)

            # Process remaining entities with Presidio if any
            if presidio_entities:
                # Map strategies to operators for non-surrogate entities
                operators = {}
                for entity_type, strategy in strategies.items():
                    if strategy.kind != StrategyKind.SURROGATE:
                        if strategy.kind == StrategyKind.CUSTOM:
                            operators[entity_type] = OperatorConfig("custom")
                        else:
                            operators[entity_type] = self.operator_mapper.strategy_to_operator(
                                strategy
                            )

                # Use Presidio for non-surrogate entities
                result = self.anonymizer.anonymize(
                    text=text_result, analyzer_results=presidio_entities, operators=operators
                )

                # Combine results with explicit typing
                items_list: list[OperatorResultLike] = list(getattr(result, "items", []))
                return [*surrogate_results, *items_list]
            # Only surrogate entities were processed
            return surrogate_results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Create fallback results
            results: list[OperatorResultLike] = []
            for entity in entities:
                entity_type = getattr(entity, "entity_type", None) or UNKNOWN_ENTITY
                strategy = strategies.get(entity_type, Strategy(StrategyKind.REDACT, {"char": "*"}))
                results.append(self._create_synthetic_result(entity, strategy, text))
            return results

    def _apply_hash_strategy(
        self, text: str, entity_type: str, strategy: Strategy, confidence: float
    ) -> str:
        """Apply hash strategy with support for prefix and other parameters."""
        params = strategy.parameters or {}

        # Create entity for Presidio
        entity = RecognizerResult(entity_type=entity_type, start=0, end=len(text), score=confidence)

        # Use Presidio for the base hash
        operator_config = self.operator_mapper.strategy_to_operator(strategy)
        result = self.anonymizer.anonymize(
            text=text, analyzer_results=[entity], operators={entity_type: operator_config}
        )

        hashed_value = result.text

        # Add prefix if specified
        if "prefix" in params:
            hashed_value = params["prefix"] + hashed_value

        # Truncate if specified
        if "truncate" in params:
            truncate_length = params["truncate"]
            if isinstance(truncate_length, int) and truncate_length > 0:
                hashed_value = hashed_value[:truncate_length]

        return str(hashed_value)

    def _apply_partial_strategy(
        self, text: str, entity_type: str, strategy: Strategy, confidence: float
    ) -> str:
        """Apply partial masking strategy with proper char count calculation."""
        params = strategy.parameters or {}
        visible_chars = max(0, int(params.get("visible_chars", 4)))
        position = params.get("position", "end")
        mask_char = params.get("mask_char", "*")

        # Create entity for Presidio
        entity = RecognizerResult(entity_type=entity_type, start=0, end=len(text), score=confidence)

        # Calculate how many chars to mask based on text length
        text_length = len(text)
        visible_chars = min(visible_chars, text_length)  # Can't show more than we have

        if position == "end":
            # Show last N chars, mask the rest
            chars_to_mask = max(0, text_length - visible_chars)
            from_end = False
        elif position == "start":
            # Show first N chars, mask the rest
            chars_to_mask = max(0, text_length - visible_chars)
            from_end = True  # Mask from the end, leaving start visible
        else:
            # Default to end behavior
            chars_to_mask = max(0, text_length - visible_chars)
            from_end = False

        # Use Presidio's mask operator
        operator_config = OperatorConfig(
            "mask",
            {"masking_char": mask_char, "chars_to_mask": chars_to_mask, "from_end": from_end},
        )

        result = self.anonymizer.anonymize(
            text=text, analyzer_results=[entity], operators={entity_type: operator_config}
        )

        return str(result.text)

    def _apply_custom_strategy(self, text: str, strategy: Strategy) -> str:
        """Apply a custom strategy using the provided callback."""
        callback = strategy.parameters.get("callback") if strategy.parameters else None
        if callback and callable(callback):
            try:
                return cast(str, callback(text))
            except Exception as e:
                logger.error(f"Custom callback failed: {e}")
        return self._fallback_redaction(text)

    def _apply_surrogate_strategy(self, text: str, entity_type: str, strategy: Strategy) -> str:
        """Apply surrogate strategy with high-quality fake data generation."""
        try:
            # Check if strategy has a seed parameter
            seed = None
            if strategy.parameters and "seed" in strategy.parameters:
                seed = strategy.parameters["seed"]

            # If seed is different from current one, recreate generator
            if seed != getattr(self._surrogate_generator, "seed", None):
                self._surrogate_generator = SurrogateGenerator(seed=seed)
                logger.debug(f"Updated SurrogateGenerator with seed: {seed}")

            # Use the surrogate generator for quality fake data
            return self._surrogate_generator.generate_surrogate(text, entity_type)
        except Exception as e:
            logger.warning(f"Surrogate generation failed: {e}")
            # Fallback to simple replacement
            return f"[{entity_type}]"

    def _fallback_redaction(self, text: str) -> str:
        """Simple fallback redaction when Presidio fails."""
        return self._fallback_char * len(text)

    def _operator_result_to_dict(self, result: OperatorResultLike) -> dict[str, Any]:
        """Convert OperatorResult to dictionary for storage."""
        return {
            "entity_type": result.entity_type if hasattr(result, "entity_type") else UNKNOWN_ENTITY,
            "start": result.start if hasattr(result, "start") else 0,
            "end": result.end if hasattr(result, "end") else 0,
            "operator": result.operator if hasattr(result, "operator") else "unknown",
            "text": result.text if hasattr(result, "text") else "",
        }

    def _create_synthetic_result(
        self, entity: RecognizerResult, strategy: Strategy, text: str
    ) -> SyntheticOperatorResult:
        """Create a synthetic OperatorResult for fallback scenarios."""
        # Handle invalid entity positions
        start = getattr(entity, "start", 0) or 0
        end = getattr(entity, "end", len(text)) or len(text)
        entity_type = getattr(entity, "entity_type", UNKNOWN_ENTITY) or UNKNOWN_ENTITY

        # Ensure valid bounds
        if start < 0:
            start = 0
        if end > len(text):
            end = len(text)
        if start >= end:
            # Invalid range, use single char
            start = min(start, len(text) - 1)
            end = start + 1

        original = text[start:end] if start < end else "*"

        # Apply strategy manually
        params = strategy.parameters or {}
        if strategy.kind == StrategyKind.REDACT:
            masked = params.get("char", "*") * len(original)
        elif strategy.kind == StrategyKind.TEMPLATE:
            template = params.get("template", f"[{entity_type}]")
            # Replace {} with a unique ID if present
            if "{}" in template:
                import uuid

                unique_id = str(uuid.uuid4())[:8]
                masked = template.replace("{}", unique_id)
            else:
                masked = template
        elif strategy.kind == StrategyKind.HASH:
            algo = params.get("algorithm", "sha256")
            prefix = params.get("prefix", "")
            hash_obj = hashlib.new(algo)
            hash_obj.update(original.encode())
            masked = prefix + hash_obj.hexdigest()[:8]
        elif strategy.kind == StrategyKind.PARTIAL:
            visible = params.get("visible_chars", 4)
            position = params.get("position", "end")
            mask_char = params.get("mask_char", "*")
            if position == "end" and len(original) > visible:
                masked = mask_char * (len(original) - visible) + original[-visible:]
            elif position == "start" and len(original) > visible:
                masked = original[:visible] + mask_char * (len(original) - visible)
            else:
                masked = original  # Too short to partial mask
        else:
            masked = self._fallback_redaction(original)

        # Create synthetic result using type-safe dataclass
        return SyntheticOperatorResult(
            entity_type=entity_type,
            start=start,
            end=end,
            operator=strategy.kind.value,
            text=masked,
        )

    def _get_reversible_operators(self, strategies: dict[str, Strategy]) -> list[str]:
        """Identify which operators are reversible.

        Reversible operations are those that can be undone to restore the original text.
        Non-reversible operations like REDACT, HASH, and SURROGATE cannot be reversed.
        """
        reversible = set()
        for strategy in strategies.values():
            if strategy.kind in {StrategyKind.TEMPLATE, StrategyKind.CUSTOM}:
                try:
                    operator_config = self.operator_mapper.strategy_to_operator(strategy)
                    operator_name = operator_config.operator_name
                    # Only mark reversible if the operator preserves original in metadata
                    if operator_name in {"replace"}:  # Keep explicit and conservative
                        reversible.add(operator_name)
                except Exception:
                    # If we can't map the strategy, assume not reversible
                    pass
        return list(reversible)

    def _find_segment_for_position(self, position: int, segments: list[TextSegment]) -> str | None:
        """Find the segment node_id for a given character position using binary search.

        Args:
            position: Character position in the full text
            segments: List of text segments with position information (assumed sorted)

        Returns:
            Node ID of the containing segment, or None if not found
        """
        if not segments:
            return "#/texts/0"

        # Use binary search for O(log n) lookup
        starts = [s.start_offset for s in segments]
        idx = bisect.bisect_right(starts, position) - 1

        if 0 <= idx < len(segments):
            segment = segments[idx]
            if segment.start_offset <= position < segment.end_offset:
                # Return segment's node_id if available, otherwise construct one
                if hasattr(segment, "node_id"):
                    return segment.node_id
                if hasattr(segment, "segment_index"):
                    return f"#/texts/{segment.segment_index}"
                return f"#/texts/{idx}"

        return "#/texts/0"

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
        if not hasattr(masked_document, "tables") or not masked_document.tables:
            return

        # Create a mapping from node_id to masked value for table cells
        node_to_masked_value: dict[str, str] = {}
        for anchor in anchor_entries:
            # Only process anchors for table cells
            if "/cell_" in anchor.node_id:
                node_to_masked_value[anchor.node_id] = anchor.masked_value

        # Process each table
        for table_item in masked_document.tables:
            if not hasattr(table_item, "data") or not table_item.data:
                continue

            table_data = table_item.data

            # Get the base node ID for this table
            base_node_id = self._get_table_node_id(table_item)

            # If table uses grid (computed property from table_cells)
            if hasattr(table_data, "grid") and hasattr(table_data, "table_cells"):
                # First, get the grid to identify which cells exist
                grid = table_data.grid

                # If table_cells is empty, populate it from grid
                if not table_data.table_cells:
                    table_data.table_cells = []
                    for row in grid:
                        for cell in row:
                            if cell and hasattr(cell, "text") and cell.text:
                                table_data.table_cells.append(cell)

                # Create a map of positions to cells in table_cells
                cell_map: dict[tuple[int, int], Any] = {}
                for cell in table_data.table_cells:
                    if hasattr(cell, "start_row_offset_idx") and hasattr(
                        cell, "start_col_offset_idx"
                    ):
                        # Map by starting position
                        key = (cell.start_row_offset_idx, cell.start_col_offset_idx)
                        cell_map[key] = cell

                # Now check each position in the grid for masked values
                for row_idx in range(len(grid)):
                    for col_idx in range(len(grid[row_idx])):
                        cell_node_id = f"{base_node_id}/cell_{row_idx}_{col_idx}"

                        if cell_node_id in node_to_masked_value:
                            masked_value = node_to_masked_value[cell_node_id]

                            # Find the cell in table_cells that corresponds to this position
                            cell_key = (row_idx, col_idx)
                            if cell_key in cell_map:
                                # Update existing cell
                                cell = cell_map[cell_key]
                                if hasattr(cell, "text"):
                                    cell.text = masked_value
                                    logger.debug(
                                        f"Updated table cell ({row_idx}, {col_idx}) with masked value"
                                    )
                            else:
                                # Need to add a new cell to table_cells
                                from docling_core.types.doc.document import TableCell

                                new_cell = TableCell(
                                    text=masked_value,
                                    start_row_offset_idx=row_idx,
                                    end_row_offset_idx=row_idx + 1,
                                    start_col_offset_idx=col_idx,
                                    end_col_offset_idx=col_idx + 1,
                                )
                                table_data.table_cells.append(new_cell)
                                logger.debug(
                                    f"Added new table cell ({row_idx}, {col_idx}) with masked value"
                                )
            # Fallback for tables that only use table_cells (1D list)
            elif hasattr(table_data, "table_cells") and table_data.table_cells:
                # For old-style flat table_cells, update directly by matching node IDs
                for idx, cell in enumerate(table_data.table_cells):
                    if hasattr(cell, "text"):
                        # Try to determine row/col from cell properties or index
                        row_idx = (
                            cell.start_row_offset_idx
                            if hasattr(cell, "start_row_offset_idx")
                            else (
                                idx // table_data.num_cols
                                if hasattr(table_data, "num_cols") and table_data.num_cols > 0
                                else 0
                            )
                        )
                        col_idx = (
                            cell.start_col_offset_idx
                            if hasattr(cell, "start_col_offset_idx")
                            else (
                                idx % table_data.num_cols
                                if hasattr(table_data, "num_cols") and table_data.num_cols > 0
                                else idx
                            )
                        )
                        cell_node_id = f"{base_node_id}/cell_{row_idx}_{col_idx}"

                        if cell_node_id in node_to_masked_value:
                            cell.text = node_to_masked_value[cell_node_id]
                            logger.debug(f"Updated table cell at index {idx} with masked value")

    def _get_table_node_id(self, table_item: Any) -> str:
        """Get the node ID for a table item."""
        # Try to get from self_ref first
        if hasattr(table_item, "self_ref"):
            return str(table_item.self_ref)

        # Try to get from position in tables list
        # This would need access to the document but we don't have it here
        # Default fallback
        return "#/tables/0"

    def _cleanup_large_results(self, results: list[OperatorResult]) -> None:
        """Clean up large result sets for memory efficiency.

        For results exceeding memory thresholds:
        - Clear text fields from operator results after processing
        - Remove redundant metadata fields
        - Release references to large intermediate objects
        """
        # Define thresholds for memory management
        max_text_length = 10000  # Characters per text field
        max_results = 1000  # Maximum results to keep in memory

        if len(results) > max_results:
            # For very large result sets, clear text from older results
            # Keep only essential metadata for audit trail
            for result in results[:-100]:  # Keep last 100 results intact
                if hasattr(result, "text") and result.text and len(result.text) > max_text_length:
                    # Clear large text fields while preserving structure
                    result.text = f"[Text truncated - {len(result.text)} chars]"

                # Clear large metadata fields if present
                if (
                    hasattr(result, "operator_metadata")
                    and result.operator_metadata
                    and "original_text" in result.operator_metadata
                ):
                    orig_len = len(str(result.operator_metadata.get("original_text", "")))
                    if orig_len > max_text_length:
                        result.operator_metadata["original_text"] = (
                            f"[Truncated - {orig_len} chars]"
                        )
