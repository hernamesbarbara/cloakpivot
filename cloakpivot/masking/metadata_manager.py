"""Metadata and anchor management utilities for the PresidioMaskingAdapter.

This module contains logic for creating anchor entries, managing cloakmap
metadata, and handling strategy-related operations.
"""

import base64
import hashlib
import logging
import secrets
import uuid
from typing import Any

from presidio_analyzer import RecognizerResult

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.cloakmap_enhancer import CloakMapEnhancer
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.protocols import OperatorResultLike, SyntheticOperatorResult
from cloakpivot.masking.text_processor import TextProcessor

logger = logging.getLogger(__name__)

UNKNOWN_ENTITY = "PII"


class MetadataManager:
    """Manage metadata, anchors, and cloakmap operations."""

    def __init__(self):
        """Initialize the metadata manager."""
        self.cloakmap_enhancer = CloakMapEnhancer()
        self.text_processor = TextProcessor()
        self._fallback_char = "*"

    def create_anchor_entries(
        self,
        document_text: str,
        entities: list[RecognizerResult],
        strategies: dict[str, Strategy],
        op_results_by_pos: dict[tuple[int, int], OperatorResultLike],
        text_segments: list[TextSegment],
        create_synthetic_result_func: Any = None,
    ) -> list[AnchorEntry]:
        """Create anchor entries for the CloakMap.

        Args:
            document_text: Full document text
            entities: List of entities being masked
            strategies: Masking strategies per entity type
            op_results_by_pos: Operator results mapped by position
            text_segments: Original text segments
            create_synthetic_result_func: Function to create synthetic results

        Returns:
            List of anchor entries for CloakMap
        """
        anchor_entries = []

        # Process entities in reverse order to maintain positions
        sorted_entities = sorted(entities, key=lambda x: x.start, reverse=True)

        for entity in sorted_entities:
            # Find matching operator result by position
            key = (entity.start, entity.end)
            matched_result = op_results_by_pos.get(key)

            # If no exact match, create a synthetic result
            if matched_result is None and create_synthetic_result_func:
                matched_result = create_synthetic_result_func(
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
                self.text_processor.find_segment_for_position(entity.start, text_segments)
                if text_segments
                else "#/texts/0"
            ) or "#/texts/0"

            anchor = AnchorEntry(
                node_id=node_id,
                start=entity.start,
                end=entity.end,
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

    def compute_replacements(
        self,
        entities: list[RecognizerResult],
        op_results: list[OperatorResultLike],
        document_text: str,
    ) -> list[tuple[int, int, str]]:
        """Compute replacement spans from entities and operator results.

        Args:
            entities: Original detected entities
            op_results: Operator results from masking
            document_text: Full document text

        Returns:
            List of (start, end, replacement) tuples
        """
        replacements = []

        # Map results by position
        op_results_by_pos = {(r.start, r.end): r for r in op_results}

        for entity in entities:
            key = (entity.start, entity.end)
            op_result = op_results_by_pos.get(key)

            if op_result:
                replacements.append((op_result.start, op_result.end, op_result.text))
            else:
                # Fallback if no matching result
                logger.warning(f"No operator result for entity at {entity.start}-{entity.end}")
                # Use fallback redaction
                original_text = document_text[entity.start : entity.end]
                replacements.append(
                    (entity.start, entity.end, self._fallback_redaction(original_text))
                )

        return replacements

    def prepare_strategies(
        self, entities: list[RecognizerResult], policy: MaskingPolicy
    ) -> dict[str, Strategy]:
        """Prepare masking strategies for each entity type.

        Args:
            entities: List of entities to prepare strategies for
            policy: Masking policy to use

        Returns:
            Mapping of entity types to strategies
        """
        # Debug type check
        if not isinstance(policy, MaskingPolicy):
            logger.error(f"prepare_strategies got wrong type for policy: {type(policy)}, value: {policy}")
            # Swapped parameters? Try to fix
            if isinstance(entities, MaskingPolicy) and isinstance(policy, list):
                entities, policy = policy, entities
                logger.warning("Parameters were swapped, correcting...")

        strategies = {}
        entity_types = set(entity.entity_type for entity in entities)

        for entity_type in entity_types:
            strategies[entity_type] = policy.get_strategy_for_entity(entity_type)

        return strategies

    def enhance_cloakmap_with_metadata(
        self,
        cloakmap: CloakMap,
        strategies: dict[str, Strategy],
        entities: list[RecognizerResult],
        op_results: list[OperatorResultLike] | None = None,
    ) -> CloakMap:
        """Enhance the CloakMap with additional metadata.

        Args:
            cloakmap: CloakMap to enhance
            strategies: Strategies used for masking
            entities: Original entities detected
            op_results: Optional operator results
        """
        # Add strategy metadata
        strategy_metadata = {}
        for entity_type, strategy in strategies.items():
            strategy_metadata[entity_type] = {
                "kind": strategy.kind.value,
                "parameters": strategy.parameters or {},
            }

        # Identify reversible operators
        reversible_operators = self.get_reversible_operators(strategies)

        # Create comprehensive metadata
        metadata = {
            "strategies_used": strategy_metadata,
            "reversible_operators": reversible_operators,
            "entity_types_found": list(set(e.entity_type for e in entities)),
            "entity_count": len(entities),
        }

        # If we have operator results, add statistics
        if op_results:
            metadata["operator_statistics"] = self._build_operator_statistics(op_results)

        # Use enhancer to add metadata
        # Convert operator results to dicts
        operator_result_dicts = []
        if op_results:
            for result in op_results:
                operator_result_dicts.append(self.operator_result_to_dict(result))

        # The add_presidio_metadata method returns an enhanced cloakmap
        return self.cloakmap_enhancer.add_presidio_metadata(
            cloakmap,
            operator_results=operator_result_dicts,
            engine_version="presidio",
            reversible_operators=reversible_operators
        )

    def get_reversible_operators(self, strategies: dict[str, Strategy]) -> list[str]:
        """Identify which operators are reversible.

        An operator is reversible if it preserves the original text in some form
        that can be recovered (e.g., stored in metadata, deterministic transformation).

        Args:
            strategies: Mapping of entity types to strategies

        Returns:
            List of reversible operator names
        """
        reversible = []
        for entity_type, strategy in strategies.items():
            # REDACT and TEMPLATE strategies lose information
            # HASH is one-way
            # PARTIAL loses some information
            # SURROGATE generates new data
            # CUSTOM depends on implementation
            if strategy.kind in [StrategyKind.REDACT, StrategyKind.TEMPLATE]:
                # These are reversible because we store the original in the cloakmap
                reversible.append(entity_type)
            elif strategy.kind == StrategyKind.HASH:
                # Hash is only reversible if we store the original
                reversible.append(entity_type)
            elif strategy.kind == StrategyKind.PARTIAL:
                # Partial is only reversible if we store the full original
                reversible.append(entity_type)
            elif strategy.kind == StrategyKind.SURROGATE:
                # Surrogate is reversible if we track mappings
                reversible.append(entity_type)

        return reversible

    def _build_operator_statistics(self, op_results: list[OperatorResultLike]) -> dict[str, Any]:
        """Build statistics from operator results.

        Args:
            op_results: List of operator results

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_masked": len(op_results),
            "operators_used": {},
        }

        for result in op_results:
            operator = getattr(result, "operator", "unknown")
            if operator not in stats["operators_used"]:
                stats["operators_used"][operator] = 0
            stats["operators_used"][operator] += 1

        return stats

    def _fallback_redaction(self, text: str) -> str:
        """Simple fallback redaction when other methods fail.

        Args:
            text: Text to redact

        Returns:
            Redacted text
        """
        return self._fallback_char * len(text)

    def operator_result_to_dict(self, result: OperatorResultLike) -> dict[str, Any]:
        """Convert OperatorResult to dictionary for storage.

        Args:
            result: Operator result to convert

        Returns:
            Dictionary representation
        """
        return {
            "entity_type": result.entity_type,
            "start": result.start,
            "end": result.end,
            "operator": result.operator,
            "text": result.text,
        }

    def cleanup_large_results(self, results: list[OperatorResultLike]) -> None:
        """Clean up large result objects to free memory.

        This is particularly important for large documents where
        operator results can accumulate significant memory usage.

        Args:
            results: List of operator results to clean up
        """
        for result in results:
            # Clear any large text fields we don't need anymore
            if hasattr(result, "_original_text"):
                delattr(result, "_original_text")
            if hasattr(result, "_context"):
                delattr(result, "_context")