"""Tests for entity normalization and conflict resolution."""


import pytest

from cloakpivot.core.analyzer import EntityDetectionResult
from cloakpivot.core.normalization import (
    ConflictResolutionConfig,
    ConflictResolutionStrategy,
    EntityGroup,
    EntityNormalizer,
    EntityPriority,
    NormalizationResult,
)


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        EntityDetectionResult("PERSON", 0, 8, 0.9, "John Doe"),
        EntityDetectionResult("EMAIL_ADDRESS", 20, 37, 0.95, "john@example.com"),
        EntityDetectionResult("PHONE_NUMBER", 50, 62, 0.85, "555-123-4567"),
    ]


@pytest.fixture
def overlapping_entities():
    """Create overlapping entities for conflict testing."""
    return [
        EntityDetectionResult("PERSON", 0, 8, 0.9, "John Doe"),
        EntityDetectionResult("PERSON", 2, 10, 0.7, "hn Doe X"),  # Overlaps with first
        EntityDetectionResult("EMAIL_ADDRESS", 15, 32, 0.95, "john@example.com"),
        EntityDetectionResult("URL", 15, 35, 0.6, "http://john@example.com"),  # Overlaps with email
    ]


@pytest.fixture
def adjacent_entities():
    """Create adjacent entities for merging tests."""
    return [
        EntityDetectionResult("PHONE_NUMBER", 0, 12, 0.9, "555-123-4567"),
        EntityDetectionResult("PHONE_NUMBER", 15, 27, 0.85, "555-987-6543"),  # 3 chars gap
        EntityDetectionResult("PERSON", 40, 48, 0.8, "John Doe"),
        EntityDetectionResult("PERSON", 50, 60, 0.75, "Jane Smith"),  # 2 chars gap
    ]


class TestConflictResolutionConfig:
    """Test conflict resolution configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConflictResolutionConfig()

        assert config.strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        assert config.confidence_threshold == 0.1
        assert config.merge_threshold_chars == 4
        assert not config.allow_partial_overlaps
        assert config.preserve_high_confidence == 0.95

        # Check default priorities are set
        assert config.get_entity_priority("US_SSN") == EntityPriority.CRITICAL
        assert config.get_entity_priority("PHONE_NUMBER") == EntityPriority.HIGH
        assert config.get_entity_priority("PERSON") == EntityPriority.MEDIUM
        assert config.get_entity_priority("URL") == EntityPriority.LOW

    def test_custom_priorities(self):
        """Test custom entity priorities."""
        custom_priorities = {
            "CUSTOM_ENTITY": EntityPriority.CRITICAL,
            "PHONE_NUMBER": EntityPriority.LOW  # Override default
        }

        config = ConflictResolutionConfig(entity_priorities=custom_priorities)

        assert config.get_entity_priority("CUSTOM_ENTITY") == EntityPriority.CRITICAL
        assert config.get_entity_priority("PHONE_NUMBER") == EntityPriority.LOW
        # Unknown entity should default to MEDIUM
        assert config.get_entity_priority("UNKNOWN") == EntityPriority.MEDIUM


class TestEntityGroup:
    """Test entity grouping functionality."""

    def test_group_initialization(self, sample_entities):
        """Test entity group initialization and boundary calculation."""
        group = EntityGroup(sample_entities[:2])  # PERSON and EMAIL

        assert len(group.entities) == 2
        assert group.start_pos == 0   # Start of PERSON
        assert group.end_pos == 37    # End of EMAIL

    def test_add_entity(self, sample_entities):
        """Test adding entities to a group."""
        group = EntityGroup([sample_entities[0]])  # Start with PERSON

        assert group.start_pos == 0
        assert group.end_pos == 8

        group.add_entity(sample_entities[1])  # Add EMAIL

        assert len(group.entities) == 2
        assert group.start_pos == 0   # Still starts with PERSON
        assert group.end_pos == 37    # Now ends with EMAIL

    def test_overlap_detection(self, overlapping_entities):
        """Test overlap detection between groups and entities."""
        # Create group with first PERSON entity (0-8)
        group = EntityGroup([overlapping_entities[0]])

        # Second PERSON (2-10) should overlap
        assert group.overlaps_with(overlapping_entities[1])

        # EMAIL (15-32) should not overlap
        assert not group.overlaps_with(overlapping_entities[2])

    def test_adjacency_detection(self, adjacent_entities):
        """Test adjacency detection between groups and entities."""
        # Create group with first PHONE (0-12)
        group = EntityGroup([adjacent_entities[0]])

        # Second PHONE (15-27) has 3 char gap - should be adjacent with threshold 5
        assert group.is_adjacent_to(adjacent_entities[1], threshold=5)
        assert not group.is_adjacent_to(adjacent_entities[1], threshold=2)

        # PERSON (40-48) has larger gap - should not be adjacent
        assert not group.is_adjacent_to(adjacent_entities[2], threshold=5)


class TestEntityNormalizer:
    """Test entity normalization functionality."""

    def test_normalizer_initialization(self):
        """Test normalizer initialization."""
        normalizer = EntityNormalizer()

        assert normalizer.config.strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE

        # Test with custom config
        config = ConflictResolutionConfig(strategy=ConflictResolutionStrategy.LONGEST_ENTITY)
        normalizer = EntityNormalizer(config)

        assert normalizer.config.strategy == ConflictResolutionStrategy.LONGEST_ENTITY

    def test_normalize_no_conflicts(self, sample_entities):
        """Test normalization with no conflicts (should return unchanged)."""
        normalizer = EntityNormalizer()
        result = normalizer.normalize_entities(sample_entities)

        assert len(result.normalized_entities) == 3
        assert result.conflicts_resolved == 0
        assert result.entities_merged == 0
        assert result.entities_removed == 0

        # Entities should be sorted
        assert result.normalized_entities == sorted(sample_entities)

    def test_normalize_empty_list(self):
        """Test normalization with empty entity list."""
        normalizer = EntityNormalizer()
        result = normalizer.normalize_entities([])

        assert len(result.normalized_entities) == 0
        assert result.total_changes == 0

    def test_highest_confidence_resolution(self, overlapping_entities):
        """Test conflict resolution by highest confidence."""
        config = ConflictResolutionConfig(strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE)
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(overlapping_entities)

        # Should keep highest confidence entities from each overlap group
        assert len(result.normalized_entities) == 2
        assert result.conflicts_resolved == 2  # 2 entities removed

        # Should keep first PERSON (0.9) over second (0.7)
        person_entities = [e for e in result.normalized_entities if e.entity_type == "PERSON"]
        assert len(person_entities) == 1
        assert person_entities[0].confidence == 0.9

        # Should keep EMAIL (0.95) over URL (0.6)
        email_entities = [e for e in result.normalized_entities if e.entity_type == "EMAIL_ADDRESS"]
        assert len(email_entities) == 1
        assert email_entities[0].confidence == 0.95

    def test_longest_entity_resolution(self):
        """Test conflict resolution by entity length."""
        entities = [
            EntityDetectionResult("PERSON", 0, 8, 0.7, "John Doe"),      # 8 chars
            EntityDetectionResult("PERSON", 2, 15, 0.6, "hn Doe Smith"),  # 13 chars, overlaps
        ]

        config = ConflictResolutionConfig(strategy=ConflictResolutionStrategy.LONGEST_ENTITY)
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(entities)

        assert len(result.normalized_entities) == 1
        assert result.normalized_entities[0].text == "hn Doe Smith"  # Longer entity kept

    def test_most_specific_resolution(self):
        """Test conflict resolution by entity specificity."""
        entities = [
            EntityDetectionResult("PERSON", 0, 8, 0.8, "John Doe"),
            EntityDetectionResult("US_SSN", 2, 10, 0.7, "23-45-67"),  # Critical priority, overlaps
        ]

        config = ConflictResolutionConfig(strategy=ConflictResolutionStrategy.MOST_SPECIFIC)
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(entities)

        # Should keep US_SSN because it has CRITICAL priority
        assert len(result.normalized_entities) == 1
        assert result.normalized_entities[0].entity_type == "US_SSN"

    def test_merge_adjacent_strategy(self, adjacent_entities):
        """Test merging adjacent entities of same type."""
        config = ConflictResolutionConfig(
            strategy=ConflictResolutionStrategy.MERGE_ADJACENT,
            merge_threshold_chars=5
        )
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(adjacent_entities)

        # Should merge adjacent PHONE_NUMBER entities (gap=3, threshold=5)
        phone_entities = [e for e in result.normalized_entities if e.entity_type == "PHONE_NUMBER"]
        assert len(phone_entities) == 1

        # Should merge adjacent PERSON entities (gap=2, threshold=5)
        person_entities = [e for e in result.normalized_entities if e.entity_type == "PERSON"]
        assert len(person_entities) == 1

        # Total should be 2 merged entities
        assert len(result.normalized_entities) == 2

    def test_preserve_high_confidence(self):
        """Test preservation of high-confidence entities."""
        entities = [
            EntityDetectionResult("PERSON", 0, 8, 0.96, "John Doe"),      # Above preserve threshold
            EntityDetectionResult("PERSON", 2, 10, 0.98, "hn Doe X"),     # Above preserve threshold, overlaps
        ]

        config = ConflictResolutionConfig(preserve_high_confidence=0.95)
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(entities)

        # Both should be preserved despite overlap due to high confidence
        assert len(result.normalized_entities) == 2

    def test_group_creation(self, overlapping_entities):
        """Test entity grouping for conflict resolution."""
        normalizer = EntityNormalizer()
        groups = normalizer._group_entities(sorted(overlapping_entities))

        # Should create 2 groups: one for overlapping PERSON entities, one for EMAIL/URL overlap
        assert len(groups) == 2

        # First group should have 2 PERSON entities
        person_group = next(g for g in groups if any(e.entity_type == "PERSON" for e in g.entities))
        assert len(person_group.entities) == 2

        # Second group should have EMAIL and URL
        email_group = next(g for g in groups if any(e.entity_type == "EMAIL_ADDRESS" for e in g.entities))
        assert len(email_group.entities) == 2

    def test_validation(self, overlapping_entities):
        """Test normalization validation."""
        normalizer = EntityNormalizer()
        result = normalizer.normalize_entities(overlapping_entities)

        validation = normalizer.validate_normalization(overlapping_entities, result.normalized_entities)

        assert validation["original_count"] == 4
        assert validation["normalized_count"] == 2
        assert validation["entities_removed"] == 2
        assert validation["sorted_correctly"]
        assert validation["no_overlaps"]
        assert not validation["warnings"]  # No warnings for this case


class TestNormalizationResult:
    """Test normalization result functionality."""

    def test_result_properties(self, sample_entities):
        """Test result properties and statistics."""
        result = NormalizationResult(
            normalized_entities=sample_entities,
            conflicts_resolved=2,
            entities_merged=1,
            entities_removed=0
        )

        assert result.total_changes == 3  # 2 + 1 + 0
        assert len(result.normalized_entities) == 3

    def test_empty_result(self):
        """Test empty normalization result."""
        result = NormalizationResult()

        assert len(result.normalized_entities) == 0
        assert result.total_changes == 0
        assert len(result.resolution_details) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_identical_entities(self):
        """Test handling of identical entities."""
        entities = [
            EntityDetectionResult("PERSON", 0, 8, 0.9, "John Doe"),
            EntityDetectionResult("PERSON", 0, 8, 0.9, "John Doe"),  # Identical
        ]

        normalizer = EntityNormalizer()
        result = normalizer.normalize_entities(entities)

        # Should keep only one entity
        assert len(result.normalized_entities) == 1
        assert result.conflicts_resolved == 1

    def test_complex_overlaps(self):
        """Test complex overlapping scenarios."""
        entities = [
            EntityDetectionResult("PERSON", 0, 10, 0.9, "John Smith"),
            EntityDetectionResult("PERSON", 5, 15, 0.8, "Smith Jane"),   # Overlaps with first
            EntityDetectionResult("PERSON", 12, 20, 0.85, "Jane Doe"),   # Overlaps with second
        ]

        normalizer = EntityNormalizer()
        result = normalizer.normalize_entities(entities)

        # All three overlap, so should resolve to best entity (highest confidence)
        assert len(result.normalized_entities) == 1
        assert result.normalized_entities[0].confidence == 0.9
        assert result.conflicts_resolved == 2

    def test_no_merge_different_types(self):
        """Test that different entity types are not merged even if adjacent."""
        entities = [
            EntityDetectionResult("PERSON", 0, 8, 0.9, "John Doe"),
            EntityDetectionResult("PHONE_NUMBER", 10, 22, 0.85, "555-123-4567"),  # Adjacent but different type
        ]

        config = ConflictResolutionConfig(
            strategy=ConflictResolutionStrategy.MERGE_ADJACENT,
            merge_threshold_chars=5
        )
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(entities)

        # Should not merge different entity types
        assert len(result.normalized_entities) == 2
        assert result.entities_merged == 0
