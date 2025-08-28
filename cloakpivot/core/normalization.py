"""Entity normalization and conflict resolution for handling overlapping and adjacent entities."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .analyzer import EntityDetectionResult

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    """Strategy for resolving conflicts between overlapping entities."""

    HIGHEST_CONFIDENCE = "highest_confidence"
    LONGEST_ENTITY = "longest_entity"
    MOST_SPECIFIC = "most_specific"
    FIRST_DETECTED = "first_detected"
    MERGE_ADJACENT = "merge_adjacent"


class EntityPriority(Enum):
    """Priority levels for entity types in conflict resolution."""

    CRITICAL = 1  # SSN, Credit Card, etc.
    HIGH = 2  # Phone, Email, etc.
    MEDIUM = 3  # Person, Location, etc.
    LOW = 4  # URL, Date, etc.


@dataclass
class ConflictResolutionConfig:
    """Configuration for entity conflict resolution.

    Attributes:
        strategy: Primary strategy for resolving conflicts
        entity_priorities: Priority mapping for entity types
        confidence_threshold: Minimum confidence difference to prefer higher confidence
        merge_threshold_chars: Maximum distance between entities to consider merging
        allow_partial_overlaps: Whether to allow partial overlaps or resolve all
        preserve_high_confidence: Always preserve entities above this confidence
    """

    strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_CONFIDENCE
    entity_priorities: dict[str, EntityPriority] = field(default_factory=dict)
    confidence_threshold: float = 0.1
    merge_threshold_chars: int = 4
    allow_partial_overlaps: bool = False
    preserve_high_confidence: float = 0.95

    def __post_init__(self) -> None:
        """Set up default entity priorities if not provided."""
        if not self.entity_priorities:
            self.entity_priorities = {
                # Critical entities
                "US_SSN": EntityPriority.CRITICAL,
                "CREDIT_CARD": EntityPriority.CRITICAL,
                "US_PASSPORT": EntityPriority.CRITICAL,
                "US_DRIVER_LICENSE": EntityPriority.CRITICAL,
                # High priority entities
                "PHONE_NUMBER": EntityPriority.HIGH,
                "EMAIL_ADDRESS": EntityPriority.HIGH,
                "IBAN_CODE": EntityPriority.HIGH,
                "US_BANK_NUMBER": EntityPriority.HIGH,
                # Medium priority entities
                "PERSON": EntityPriority.MEDIUM,
                "LOCATION": EntityPriority.MEDIUM,
                "ORGANIZATION": EntityPriority.MEDIUM,
                "NRP": EntityPriority.MEDIUM,  # National Registration Person
                # Low priority entities
                "URL": EntityPriority.LOW,
                "IP_ADDRESS": EntityPriority.LOW,
                "DATE_TIME": EntityPriority.LOW,
                "MEDICAL_LICENSE": EntityPriority.LOW,
            }

    def get_entity_priority(self, entity_type: str) -> EntityPriority:
        """Get priority for an entity type."""
        return self.entity_priorities.get(entity_type, EntityPriority.MEDIUM)


@dataclass
class NormalizationResult:
    """Result of entity normalization process.

    Attributes:
        normalized_entities: Final list of normalized entities
        conflicts_resolved: Number of conflicts that were resolved
        entities_merged: Number of entities that were merged
        entities_removed: Number of entities that were removed
        resolution_details: Detailed information about each resolution
    """

    normalized_entities: list[EntityDetectionResult] = field(default_factory=list)
    conflicts_resolved: int = 0
    entities_merged: int = 0
    entities_removed: int = 0
    resolution_details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        """Total number of changes made during normalization."""
        return self.conflicts_resolved + self.entities_merged + self.entities_removed


@dataclass
class EntityGroup:
    """Group of overlapping or adjacent entities for conflict resolution."""

    entities: list[EntityDetectionResult] = field(default_factory=list)
    start_pos: int = 0
    end_pos: int = 0

    def __post_init__(self) -> None:
        """Calculate group boundaries."""
        if self.entities:
            self.start_pos = min(e.start for e in self.entities)
            self.end_pos = max(e.end for e in self.entities)

    def add_entity(self, entity: EntityDetectionResult) -> None:
        """Add entity to group and update boundaries."""
        self.entities.append(entity)
        self.start_pos = (
            min(self.start_pos, entity.start) if self.entities else entity.start
        )
        self.end_pos = max(self.end_pos, entity.end) if self.entities else entity.end

    def overlaps_with(self, entity: EntityDetectionResult) -> bool:
        """Check if group overlaps with an entity."""
        return not (self.end_pos <= entity.start or entity.end <= self.start_pos)

    def is_adjacent_to(self, entity: EntityDetectionResult, threshold: int = 5) -> bool:
        """Check if group is adjacent to an entity within threshold."""
        gap_start = abs(entity.start - self.end_pos)
        gap_end = abs(self.start_pos - entity.end)
        return min(gap_start, gap_end) <= threshold


class EntityNormalizer:
    """Entity normalizer for resolving conflicts and merging adjacent entities."""

    def __init__(self, config: Optional[ConflictResolutionConfig] = None):
        """Initialize entity normalizer.

        Args:
            config: Configuration for conflict resolution (uses defaults if None)
        """
        self.config = config or ConflictResolutionConfig()
        logger.info(
            f"EntityNormalizer initialized with strategy: {self.config.strategy}"
        )

    def normalize_entities(
        self, entities: list[EntityDetectionResult]
    ) -> NormalizationResult:
        """Normalize a list of entities by resolving conflicts and merging adjacent entities.

        Args:
            entities: List of EntityDetectionResult objects to normalize

        Returns:
            NormalizationResult with normalized entities and statistics
        """
        if not entities:
            return NormalizationResult()

        logger.info(f"Normalizing {len(entities)} entities")

        # Sort entities for deterministic processing
        sorted_entities = sorted(entities)

        # Group overlapping and adjacent entities
        groups = self._group_entities(sorted_entities)
        logger.debug(f"Created {len(groups)} entity groups")

        # Resolve conflicts within each group
        result = NormalizationResult()

        for i, group in enumerate(groups):
            if len(group.entities) == 1:
                # No conflicts in single-entity group
                result.normalized_entities.extend(group.entities)
            else:
                # Resolve conflicts in multi-entity group
                resolved = self._resolve_group_conflicts(group, i)
                result.normalized_entities.extend(resolved.entities)
                result.conflicts_resolved += len(group.entities) - len(
                    resolved.entities
                )

                # Add resolution details
                if len(group.entities) > len(resolved.entities):
                    result.resolution_details.append(
                        {
                            "group_index": i,
                            "original_count": len(group.entities),
                            "resolved_count": len(resolved.entities),
                            "strategy_used": self.config.strategy.value,
                            "entities_involved": [
                                {
                                    "type": e.entity_type,
                                    "text": e.text,
                                    "confidence": e.confidence,
                                }
                                for e in group.entities
                            ],
                        }
                    )

        # Sort final results
        result.normalized_entities.sort()

        logger.info(
            f"Normalization complete: {len(result.normalized_entities)} entities, "
            f"{result.conflicts_resolved} conflicts resolved"
        )

        return result

    def _group_entities(
        self, entities: list[EntityDetectionResult]
    ) -> list[EntityGroup]:
        """Group overlapping and adjacent entities.

        Args:
            entities: Sorted list of entities

        Returns:
            List of EntityGroup objects
        """
        if not entities:
            return []

        groups: list[EntityGroup] = []

        for entity in entities:
            # Find groups that this entity overlaps with or is adjacent to
            matching_groups = []

            for group in groups:
                if group.overlaps_with(entity):
                    matching_groups.append(group)
                elif group.is_adjacent_to(entity, self.config.merge_threshold_chars):
                    matching_groups.append(group)

            if not matching_groups:
                # Create new group
                new_group = EntityGroup([entity])
                groups.append(new_group)
            elif len(matching_groups) == 1:
                # Add to existing group
                matching_groups[0].add_entity(entity)
            else:
                # Merge multiple groups
                merged_group = EntityGroup([entity])
                for group in matching_groups:
                    merged_group.entities.extend(group.entities)
                    groups.remove(group)

                merged_group.__post_init__()  # Recalculate boundaries
                groups.append(merged_group)

        return groups

    def _resolve_group_conflicts(
        self, group: EntityGroup, group_index: int
    ) -> EntityGroup:
        """Resolve conflicts within a single entity group.

        Args:
            group: EntityGroup with potentially conflicting entities
            group_index: Index of the group for logging

        Returns:
            EntityGroup with conflicts resolved
        """
        if len(group.entities) <= 1:
            return group

        logger.debug(
            f"Resolving conflicts in group {group_index} with {len(group.entities)} entities"
        )

        # Apply resolution strategy
        if self.config.strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            resolved_entities = self._resolve_by_confidence(group.entities)
        elif self.config.strategy == ConflictResolutionStrategy.LONGEST_ENTITY:
            resolved_entities = self._resolve_by_length(group.entities)
        elif self.config.strategy == ConflictResolutionStrategy.MOST_SPECIFIC:
            resolved_entities = self._resolve_by_specificity(group.entities)
        elif self.config.strategy == ConflictResolutionStrategy.FIRST_DETECTED:
            resolved_entities = self._resolve_by_position(group.entities)
        elif self.config.strategy == ConflictResolutionStrategy.MERGE_ADJACENT:
            resolved_entities = self._resolve_by_merging(group.entities)
        else:
            # Default to confidence-based resolution
            resolved_entities = self._resolve_by_confidence(group.entities)

        return EntityGroup(resolved_entities)

    def _resolve_by_confidence(
        self, entities: list[EntityDetectionResult]
    ) -> list[EntityDetectionResult]:
        """Resolve conflicts by preferring highest confidence entities."""
        if not entities:
            return []

        # Always preserve entities above preserve_high_confidence threshold
        preserved = [
            e for e in entities if e.confidence >= self.config.preserve_high_confidence
        ]

        if preserved:
            logger.debug(f"Preserved {len(preserved)} high-confidence entities")
            return preserved  # Keep all preserved entities even if they overlap

        # Sort by confidence descending
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)

        # For entities in the same group (all conflicting), keep only the highest confidence
        if len(entities) > 1:
            return [sorted_entities[0]]  # Keep only the best entity

        return sorted_entities

    def _resolve_by_length(
        self, entities: list[EntityDetectionResult]
    ) -> list[EntityDetectionResult]:
        """Resolve conflicts by preferring longest entities."""
        sorted_entities = sorted(entities, key=lambda e: len(e.text), reverse=True)

        result = []
        for entity in sorted_entities:
            if not any(entity.overlaps_with(existing) for existing in result):
                result.append(entity)

        return result

    def _resolve_by_specificity(
        self, entities: list[EntityDetectionResult]
    ) -> list[EntityDetectionResult]:
        """Resolve conflicts by preferring more specific entity types."""
        # Sort by priority (lower number = higher priority)
        sorted_entities = sorted(
            entities,
            key=lambda e: (
                self.config.get_entity_priority(e.entity_type).value,
                -e.confidence,
            ),
        )

        result = []
        for entity in sorted_entities:
            if not any(entity.overlaps_with(existing) for existing in result):
                result.append(entity)

        return result

    def _resolve_by_position(
        self, entities: list[EntityDetectionResult]
    ) -> list[EntityDetectionResult]:
        """Resolve conflicts by keeping entities in order of appearance."""
        sorted_entities = sorted(entities, key=lambda e: (e.start, e.end))

        result = []
        for entity in sorted_entities:
            if not any(entity.overlaps_with(existing) for existing in result):
                result.append(entity)

        return result

    def _resolve_by_merging(
        self, entities: list[EntityDetectionResult]
    ) -> list[EntityDetectionResult]:
        """Resolve conflicts by merging adjacent entities of same type."""
        if not entities:
            return []

        # Group by entity type
        by_type: dict[str, list[EntityDetectionResult]] = {}
        for entity in entities:
            if entity.entity_type not in by_type:
                by_type[entity.entity_type] = []
            by_type[entity.entity_type].append(entity)

        result = []

        for entity_type, type_entities in by_type.items():
            # Sort by position
            type_entities.sort(key=lambda e: e.start)

            merged = []
            current_group = [type_entities[0]]

            for i in range(1, len(type_entities)):
                entity = type_entities[i]
                last_in_group = current_group[-1]

                # Check if adjacent (within merge threshold)
                gap = entity.start - last_in_group.end
                if gap <= self.config.merge_threshold_chars:
                    current_group.append(entity)
                else:
                    # Finalize current group and start new one
                    if len(current_group) > 1:
                        merged_entity = self._merge_entity_group(
                            current_group, entity_type
                        )
                        merged.append(merged_entity)
                    else:
                        merged.extend(current_group)

                    current_group = [entity]

            # Handle final group
            if len(current_group) > 1:
                merged_entity = self._merge_entity_group(current_group, entity_type)
                merged.append(merged_entity)
            else:
                merged.extend(current_group)

            result.extend(merged)

        return sorted(result)

    def _merge_entity_group(
        self, entities: list[EntityDetectionResult], entity_type: str
    ) -> EntityDetectionResult:
        """Merge a group of adjacent entities into a single entity."""
        if not entities:
            raise ValueError("Cannot merge empty entity group")

        if len(entities) == 1:
            return entities[0]

        # Calculate merged boundaries
        start_pos = min(e.start for e in entities)
        end_pos = max(e.end for e in entities)

        # Use highest confidence
        max_confidence = max(e.confidence for e in entities)

        # Reconstruct text (this is approximate since we don't have full text context)
        merged_text = f"[MERGED_{entity_type}]"

        logger.debug(
            f"Merged {len(entities)} {entity_type} entities into single entity "
            f"at positions {start_pos}-{end_pos}"
        )

        return EntityDetectionResult(
            entity_type=entity_type,
            start=start_pos,
            end=end_pos,
            confidence=max_confidence,
            text=merged_text,
        )

    def _remove_overlaps(
        self, entities: list[EntityDetectionResult]
    ) -> list[EntityDetectionResult]:
        """Remove overlapping entities, keeping the first in sorted order."""
        if not entities:
            return []

        sorted_entities = sorted(entities)
        result = [sorted_entities[0]]

        for entity in sorted_entities[1:]:
            if not any(entity.overlaps_with(existing) for existing in result):
                result.append(entity)

        return result

    def validate_normalization(
        self,
        original: list[EntityDetectionResult],
        normalized: list[EntityDetectionResult],
    ) -> dict[str, Any]:
        """Validate that normalization preserved important properties.

        Args:
            original: Original entity list before normalization
            normalized: Normalized entity list

        Returns:
            Dictionary with validation results
        """
        validation = {
            "original_count": len(original),
            "normalized_count": len(normalized),
            "entities_removed": len(original) - len(normalized),
            "sorted_correctly": normalized == sorted(normalized),
            "no_overlaps": True,
            "high_confidence_preserved": True,
            "warnings": [],
        }

        # Check for overlaps in normalized entities
        for i, entity1 in enumerate(normalized):
            for j, entity2 in enumerate(normalized[i + 1 :], i + 1):
                if entity1.overlaps_with(entity2):
                    validation["no_overlaps"] = False
                    validation["warnings"].append(
                        f"Overlap found between entities at indices {i} and {j}"
                    )

        # Check if high-confidence entities were preserved
        high_conf_original = {
            (e.entity_type, e.start, e.end)
            for e in original
            if e.confidence >= self.config.preserve_high_confidence
        }

        high_conf_normalized = {
            (e.entity_type, e.start, e.end)
            for e in normalized
            if e.confidence >= self.config.preserve_high_confidence
        }

        if not high_conf_original.issubset(high_conf_normalized):
            validation["high_confidence_preserved"] = False
            missing = high_conf_original - high_conf_normalized
            validation["warnings"].append(
                f"High confidence entities were removed: {missing}"
            )

        return validation
