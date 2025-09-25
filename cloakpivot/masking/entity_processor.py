"""Entity processing utilities for the PresidioMaskingAdapter.

This module contains entity validation, filtering, and batch processing logic
extracted from the main PresidioMaskingAdapter to improve maintainability.
"""

import logging
from copy import copy
from typing import Any

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine, OperatorConfig

from cloakpivot.core.processing.presidio_mapper import StrategyToOperatorMapper as OperatorMapper
from cloakpivot.core.types.strategies import Strategy, StrategyKind
from cloakpivot.masking.protocols import SegmentBoundary, SyntheticOperatorResult
from cloakpivot.masking.strategy_processors import StrategyProcessor

logger = logging.getLogger(__name__)

# Constants
SEGMENT_SEPARATOR = "\n\n"
UNKNOWN_ENTITY = "PII"


class EntityProcessor:
    """Process and validate entities for masking operations."""

    def __init__(self, anonymizer: AnonymizerEngine, operator_mapper: OperatorMapper):
        """Initialize the entity processor.

        Args:
            anonymizer: Presidio anonymizer engine
            operator_mapper: Mapper for strategy to operator conversion
        """
        self.anonymizer = anonymizer
        self.operator_mapper = operator_mapper
        self.strategy_processor = StrategyProcessor(anonymizer, operator_mapper)

        # Cache for expensive operations
        self._overlap_cache: dict[tuple[Any, ...], bool] = {}
        self._validation_cache: dict[tuple[Any, ...], list[Any]] = {}
        self._batch_cache: dict[tuple[Any, ...], list[Any]] = {}
        self._max_cache_size = 256

    def validate_entities(
        self, entities: list[RecognizerResult], document_length: int
    ) -> list[RecognizerResult]:
        """Validate entity positions against document length.

        Args:
            entities: List of entities to validate
            document_length: Total length of document text

        Returns:
            List of valid entities within document bounds
        """
        valid_entities = []
        for entity in entities:
            # Check for invalid positions
            if entity.start < 0 or entity.end < entity.start:
                logger.warning(
                    f"Entity {entity.entity_type} has invalid positions {entity.start}-{entity.end}, skipping"
                )
                continue
            # Check if entity exceeds document bounds
            if entity.end > document_length:
                logger.warning(
                    f"Entity {entity.entity_type} at positions {entity.start}-{entity.end} "
                    f"exceeds document text length {document_length}, skipping"
                )
                continue
            valid_entities.append(entity)
        return valid_entities

    def validate_entities_against_boundaries(
        self,
        entities: list[RecognizerResult],
        document_text: str,
        segment_boundaries: list[SegmentBoundary],
    ) -> list[RecognizerResult]:
        """Validate and adjust entities to not span across segment boundaries.

        This prevents the issue where entities detected by Presidio span across
        table cell boundaries or other segment separators, which causes text from
        multiple segments to be incorrectly concatenated in the masked value.

        If an entity spans across boundaries, it will be truncated to fit within
        the first segment it appears in.

        Args:
            entities: List of entities to validate
            document_text: Full document text
            segment_boundaries: List of segment boundaries with start/end positions

        Returns:
            List of valid entities that don't cross segment boundaries
        """
        valid_entities = []
        adjusted_count = 0

        for entity in entities:
            # Extract the actual text for this entity
            entity_text = document_text[entity.start : entity.end]

            # Check if the entity text contains the segment separator
            separator_pos = entity_text.find(SEGMENT_SEPARATOR)
            if separator_pos >= 0:
                # Entity spans across segments - truncate it to the first segment
                adjusted_entity = copy(entity)
                adjusted_entity.end = entity.start + separator_pos

                # Verify the truncated entity still makes sense
                truncated_text = document_text[adjusted_entity.start : adjusted_entity.end]
                if len(truncated_text.strip()) > 0:
                    logger.debug(
                        f"Entity {entity.entity_type} at positions {entity.start}-{entity.end} "
                        f"spans segments. Truncated to {adjusted_entity.start}-{adjusted_entity.end}. "
                        f"Original: '{entity_text[:30]}...' -> Truncated: '{truncated_text}'"
                    )
                    valid_entities.append(adjusted_entity)
                    adjusted_count += 1
                else:
                    logger.warning(
                        f"Entity {entity.entity_type} at positions {entity.start}-{entity.end} "
                        f"spans segments and truncation results in empty text, skipping"
                    )
                continue

            # Additional check: verify entity is fully contained within a single segment
            entity_contained = False
            for boundary in segment_boundaries:
                if boundary.start <= entity.start and entity.end <= boundary.end:
                    entity_contained = True
                    break

            if not entity_contained:
                # Try to find the segment containing the start of the entity and truncate
                for boundary in segment_boundaries:
                    if boundary.start <= entity.start < boundary.end:
                        adjusted_entity = copy(entity)
                        adjusted_entity.end = min(entity.end, boundary.end)
                        truncated_text = document_text[adjusted_entity.start : adjusted_entity.end]
                        if len(truncated_text.strip()) > 0:
                            logger.debug(
                                f"Entity {entity.entity_type} at positions {entity.start}-{entity.end} "
                                f"not contained in segment. Truncated to segment boundary: "
                                f"{adjusted_entity.start}-{adjusted_entity.end}"
                            )
                            valid_entities.append(adjusted_entity)
                            adjusted_count += 1
                        break
                else:
                    logger.warning(
                        f"Entity {entity.entity_type} at positions {entity.start}-{entity.end} "
                        f"could not be adjusted to fit within a segment, skipping"
                    )
            else:
                # Entity is fully contained within a segment
                valid_entities.append(entity)

        if adjusted_count > 0:
            logger.info(
                f"Adjusted {adjusted_count} entities to fit within segment boundaries. "
                f"Total valid entities: {len(valid_entities)}"
            )

        return valid_entities

    def filter_overlapping_entities(
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

        # Create cache key from entity positions and scores
        cache_key = tuple(
            (e.start, e.end, e.score, e.entity_type)
            for e in sorted(entities, key=lambda x: x.start)
        )

        # Check cache
        if cache_key in self._overlap_cache and len(self._overlap_cache) < self._max_cache_size:
            # Note: cache stores boolean, but we need to recompute for actual filtering
            # This is a simplified cache that just tracks if we've seen this combination
            pass

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

    def batch_process_entities(
        self, text: str, entities: list[RecognizerResult], strategies: dict[str, Strategy]
    ) -> list[Any]:  # Returns list of OperatorResult-like objects
        """Process multiple entities in a single batch for efficiency.

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
            surrogate_results: list[Any] = []

            # Process surrogate entities WITHOUT mutating the text
            for entity in surrogate_entities:
                entity_type = getattr(entity, "entity_type", "UNKNOWN")
                strategy = strategies.get(entity_type)
                if not strategy:
                    # This shouldn't happen as we filtered for SURROGATE entities
                    continue
                original_value = text[entity.start : entity.end]

                # Generate surrogate value
                surrogate_value = self.strategy_processor.apply_surrogate_strategy(
                    original_value, entity_type, strategy
                )
                logger.debug(
                    f"SURROGATE: Replacing '{original_value}' with '{surrogate_value}' for {entity_type}"
                )

                # Create a SyntheticOperatorResult for tracking (using original coordinates)
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

                # Use Presidio for non-surrogate entities with ORIGINAL text
                result = self.anonymizer.anonymize(
                    text=text, analyzer_results=presidio_entities, operators=operators
                )

                # Combine results with explicit typing
                items_list: list[Any] = list(getattr(result, "items", []))
                return [*surrogate_results, *items_list]
            # Only surrogate entities were processed
            return surrogate_results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Create fallback results
            results: list[Any] = []
            for entity in entities:
                entity_type = getattr(entity, "entity_type", None) or UNKNOWN_ENTITY
                strategy = strategies.get(entity_type, Strategy(StrategyKind.REDACT, {"char": "*"}))
                results.append(self.create_synthetic_result(entity, strategy, text))
            return results

    def create_synthetic_result(
        self, entity: RecognizerResult, strategy: Strategy, text: str
    ) -> SyntheticOperatorResult:
        """Create a synthetic OperatorResult for fallback scenarios.

        Args:
            entity: Entity to create result for
            strategy: Strategy to apply
            text: Original text

        Returns:
            Synthetic operator result
        """
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
        if strategy.kind == StrategyKind.REDACT:
            masked = self.strategy_processor.apply_redact_strategy(original, strategy)
        elif strategy.kind == StrategyKind.TEMPLATE:
            masked = self.strategy_processor.apply_template_strategy(entity_type, strategy)
        elif strategy.kind == StrategyKind.HASH:
            import hashlib

            params = strategy.parameters or {}
            algo = params.get("algorithm", "sha256")
            prefix = params.get("prefix", "")
            hash_obj = hashlib.new(algo)
            hash_obj.update(original.encode())
            masked = prefix + hash_obj.hexdigest()[:8]
        elif strategy.kind == StrategyKind.PARTIAL:
            params = strategy.parameters or {}
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
            masked = self.strategy_processor._fallback_redaction(original)

        # Create synthetic result using type-safe dataclass
        return SyntheticOperatorResult(
            entity_type=entity_type,
            start=start,
            end=end,
            operator=strategy.kind.value,
            text=masked,
        )
