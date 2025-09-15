"""Presidio-based masking adapter for CloakPivot."""

import copy
import hashlib
import logging
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

logger = logging.getLogger(__name__)


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
        self.operator_mapper = StrategyToOperatorMapper()
        self.cloakmap_enhancer = CloakMapEnhancer()
        self._fallback_char = "*"
        self.engine_config = engine_config or {}
        self._surrogate_generator = SurrogateGenerator()

        logger.debug(f"PresidioMaskingAdapter initialized with config: {self.engine_config}")

    @property
    def anonymizer(self) -> AnonymizerEngine:
        """Lazy-load the AnonymizerEngine on first access."""
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
                char = params.get("char", params.get("redact_char", "*"))
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

            return result.text

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
        segment_boundaries = []
        for i, segment in enumerate(text_segments):
            segment_boundaries.append(
                {
                    "segment_index": i,
                    "start": len(document_text),
                    "end": len(document_text) + len(segment.text),
                    "node_id": segment.node_id,
                }
            )
            document_text += segment.text
            if i < len(text_segments) - 1:
                document_text += "\n\n"  # Add separator between segments
                segment_boundaries[-1]["end"] += 2  # Adjust for separator

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
        op_results_by_pos = {}
        for op_result in operator_results:
            if hasattr(op_result, "start") and hasattr(op_result, "end"):
                key = (op_result.start, op_result.end)
                op_results_by_pos[key] = op_result

        # Process entities in reverse order to maintain positions
        sorted_entities = sorted(valid_entities, key=lambda x: x.start, reverse=True)
        entity_to_op_result = []

        for entity in sorted_entities:
            # Find matching operator result by position
            key = (entity.start, entity.end)
            op_result: OperatorResult | None = op_results_by_pos.get(key)

            # If no exact match, create a synthetic result
            if op_result is None:
                op_result = self._create_synthetic_result(
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
                op_result.text if hasattr(op_result, "text") else self._fallback_redaction(original)
            )

            # Apply mask to text
            masked_text = masked_text[: entity.start] + masked_value + masked_text[entity.end :]

            # Create anchor entry with proper fields
            import base64
            import uuid

            # Generate checksum and salt
            salt = base64.b64encode(
                hashlib.sha256(str(uuid.uuid4()).encode()).digest()[:8]
            ).decode()
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
                    "presidio_operator": (
                        op_result.operator if hasattr(op_result, "operator") else "fallback"
                    ),
                },
            )
            anchor_entries.append(anchor)
            # Track entity to operator result mapping for metadata
            entity_to_op_result.append((entity, op_result))

        # Create masked document preserving original structure
        from docling_core.types.doc.document import DocItemLabel, TextItem

        masked_document = DoclingDocument(
            name=document.name,
            texts=[],
            tables=copy.deepcopy(document.tables) if hasattr(document, "tables") else [],
            key_value_items=(
                copy.deepcopy(document.key_value_items)
                if hasattr(document, "key_value_items")
                else []
            ),
            origin=document.origin if hasattr(document, "origin") else None,
        )

        # Instead of trying to split the masked text, apply masking to each segment individually
        if hasattr(document, "texts") and document.texts:
            from ..document.mapper import AnchorMapper

            mapper = AnchorMapper()

            masked_segments = []

            for i, original_item in enumerate(document.texts):
                if i < len(text_segments):
                    segment = text_segments[i]
                    segment_text = original_item.text

                    # Find entities that affect this segment
                    segment_entities = []
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
                        entity = entity_info["entity"]
                        local_start = entity_info["local_start"]
                        local_end = entity_info["local_end"]

                        # Find the anchor entry for this entity
                        anchor = next(
                            (
                                a
                                for a in anchor_entries
                                if a.metadata.get("original_text")
                                == document_text[entity.start : entity.end]
                            ),
                            None,
                        )

                        if anchor:
                            # Apply the mask
                            masked_segment_text = (
                                masked_segment_text[:local_start]
                                + anchor.masked_value
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

            masked_document.texts = masked_segments

        # Also preserve _main_text for backward compatibility
        if hasattr(document, "_main_text"):
            masked_document._main_text = masked_text  # type: ignore[attr-defined]

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
    ) -> list[OperatorResult]:
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
            # Map strategies to operators
            operators = {}
            for entity_type, strategy in strategies.items():
                if strategy.kind == StrategyKind.CUSTOM:
                    # Handle custom strategies separately
                    operators[entity_type] = OperatorConfig("custom")
                elif strategy.kind == StrategyKind.SURROGATE:
                    # Use replace operator with fake data
                    operators[entity_type] = OperatorConfig("replace", {"new_value": "[SURROGATE]"})
                else:
                    operators[entity_type] = self.operator_mapper.strategy_to_operator(strategy)

            # Use Presidio to process all entities
            result = self.anonymizer.anonymize(
                text=text, analyzer_results=entities, operators=operators
            )

            # Extract operator results
            if hasattr(result, "items"):
                return result.items
            # Fallback: create synthetic results
            return [
                self._create_synthetic_result(entity, strategies[entity.entity_type], text)
                for entity in entities
            ]

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Create fallback results
            results = []
            for entity in entities:
                entity_type = getattr(entity, "entity_type", None) or "UNKNOWN"
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

        return hashed_value

    def _apply_partial_strategy(
        self, text: str, entity_type: str, strategy: Strategy, confidence: float
    ) -> str:
        """Apply partial masking strategy with proper char count calculation."""
        params = strategy.parameters or {}
        visible_chars = params.get("visible_chars", 4)
        position = params.get("position", "end")
        mask_char = params.get("mask_char", "*")

        # Create entity for Presidio
        entity = RecognizerResult(entity_type=entity_type, start=0, end=len(text), score=confidence)

        # Calculate how many chars to mask based on text length
        text_length = len(text)

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

        return result.text

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
            # Use the surrogate generator for quality fake data
            return self._surrogate_generator.generate_surrogate(text, entity_type)
        except Exception as e:
            logger.warning(f"Surrogate generation failed: {e}")
            # Fallback to simple replacement
            return f"[{entity_type}]"

    def _fallback_redaction(self, text: str) -> str:
        """Simple fallback redaction when Presidio fails."""
        return self._fallback_char * len(text)

    def _operator_result_to_dict(self, result: OperatorResult) -> dict[str, Any]:
        """Convert OperatorResult to dictionary for storage."""
        return {
            "entity_type": getattr(result, "entity_type", "UNKNOWN"),
            "start": getattr(result, "start", 0),
            "end": getattr(result, "end", 0),
            "operator": getattr(result, "operator", "unknown"),
            "text": getattr(result, "text", ""),
        }

    def _create_synthetic_result(
        self, entity: RecognizerResult, strategy: Strategy, text: str
    ) -> OperatorResult:
        """Create a synthetic OperatorResult for fallback scenarios."""
        # Handle invalid entity positions
        start = getattr(entity, "start", 0) or 0
        end = getattr(entity, "end", len(text)) or len(text)
        entity_type = getattr(entity, "entity_type", "UNKNOWN") or "UNKNOWN"

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

        # Create synthetic result
        result = type(
            "OperatorResult",
            (),
            {
                "entity_type": entity_type,
                "start": start,
                "end": end,
                "operator": strategy.kind.value,
                "text": masked,
            },
        )()

        return cast(OperatorResult, result)

    def _get_reversible_operators(self, strategies: dict[str, Strategy]) -> list[str]:
        """Identify which operators are reversible.

        Reversible operations are those that can be undone to restore the original text.
        Non-reversible operations like REDACT, HASH, and SURROGATE cannot be reversed.
        """
        reversible_kinds = {
            StrategyKind.TEMPLATE,  # Can store original
            # StrategyKind.CUSTOM could be reversible depending on implementation
            # PARTIAL is not reversible - it loses some information
            # REDACT is not reversible
            # HASH is not reversible
            # SURROGATE is not reversible
        }

        reversible = []
        for _entity_type, strategy in strategies.items():
            if strategy.kind in reversible_kinds:
                # Map strategy kind to Presidio operator name
                # Presidio typically uses "replace" for template operations
                reversible.append("replace")

        return list(set(reversible))  # Remove duplicates

    def _find_segment_for_position(
        self, position: int, segments: list[TextSegment]
    ) -> str | None:
        """Find the segment node_id for a given character position.

        Args:
            position: Character position in the full text
            segments: List of text segments with position information

        Returns:
            Node ID of the containing segment, or None if not found
        """
        for segment in segments:
            if hasattr(segment, "start") and hasattr(segment, "end"):
                if segment.start <= position < segment.end:
                    # Return segment's node_id if available, otherwise construct one
                    if hasattr(segment, "node_id"):
                        return segment.node_id
                    if hasattr(segment, "segment_index"):
                        return f"#/texts/{segment.segment_index}"
                    # Default to index in segment list
                    return f"#/texts/{segments.index(segment)}"

        # Fallback to default if no segment found
        return "#/texts/0"

    def _cleanup_large_results(self, results: list[OperatorResult]) -> None:
        """Clean up large result sets for memory efficiency.

        For results exceeding memory thresholds:
        - Clear text fields from operator results after processing
        - Remove redundant metadata fields
        - Release references to large intermediate objects
        """
        # Define thresholds for memory management
        MAX_TEXT_LENGTH = 10000  # Characters per text field
        MAX_RESULTS = 1000  # Maximum results to keep in memory

        if len(results) > MAX_RESULTS:
            # For very large result sets, clear text from older results
            # Keep only essential metadata for audit trail
            for result in results[:-100]:  # Keep last 100 results intact
                if hasattr(result, "text") and result.text and len(result.text) > MAX_TEXT_LENGTH:
                    # Clear large text fields while preserving structure
                    result.text = f"[Text truncated - {len(result.text)} chars]"

                # Clear large metadata fields if present
                if hasattr(result, "operator_metadata") and result.operator_metadata:
                    if "original_text" in result.operator_metadata:
                        orig_len = len(str(result.operator_metadata.get("original_text", "")))
                        if orig_len > MAX_TEXT_LENGTH:
                            result.operator_metadata["original_text"] = (
                                f"[Truncated - {orig_len} chars]"
                            )
