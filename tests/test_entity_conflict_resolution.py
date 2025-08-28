"""Tests for enhanced entity conflict resolution functionality."""

import pytest
from docling_core.types import DoclingDocument
from presidio_analyzer import RecognizerResult

from cloakpivot.core.analyzer import EntityDetectionResult
from cloakpivot.core.normalization import (
    ConflictResolutionConfig,
    ConflictResolutionStrategy,
    EntityNormalizer,
    EntityPriority,
)
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.engine import MaskingEngine


class TestEntityConflictResolution:
    """Test suite for entity conflict resolution functionality."""

    @pytest.fixture
    def simple_document(self) -> DoclingDocument:
        """Create a simple test document."""
        from docling_core.types.doc.document import TextItem

        doc = DoclingDocument(name="test_doc")
        text_item = TextItem(
            text="Call John Smith at 555-123-4567 or email john.smith@example.com in New York",
            self_ref="#/texts/0",
            label="text",
            orig="Call John Smith at 555-123-4567 or email john.smith@example.com in New York",
        )
        doc.texts = [text_item]
        return doc

    @pytest.fixture
    def text_segments(self, simple_document) -> list[TextSegment]:
        """Create text segments for the document."""
        return [
            TextSegment(
                node_id="#/texts/0",
                text=simple_document.texts[0].text,
                start_offset=0,
                end_offset=len(simple_document.texts[0].text),
                node_type="TextItem",
            )
        ]

    @pytest.fixture
    def overlapping_entities(self) -> list[EntityDetectionResult]:
        """Create overlapping entity detections for testing."""
        base_text = "Call John Smith at 555-123-4567 or email john.smith@example.com in New York"
        
        presidio_results = [
            # "John Smith" as PERSON
            RecognizerResult(entity_type="PERSON", start=5, end=15, score=0.90),
            # "Smith" as part of a longer pattern (overlaps with PERSON)
            RecognizerResult(entity_type="ORGANIZATION", start=10, end=15, score=0.60),
            # Phone number
            RecognizerResult(entity_type="PHONE_NUMBER", start=19, end=31, score=0.95),
            # Email
            RecognizerResult(entity_type="EMAIL_ADDRESS", start=41, end=65, score=0.85),
            # Location
            RecognizerResult(entity_type="LOCATION", start=69, end=77, score=0.75),
        ]
        
        return [
            EntityDetectionResult.from_presidio_result(r, base_text[r.start:r.end])
            for r in presidio_results
        ]

    @pytest.fixture
    def adjacent_entities(self) -> list[EntityDetectionResult]:
        """Create adjacent entity detections for merging tests."""
        base_text = "Call John Smith at 555-123-4567 or email john.smith@example.com in New York"
        
        presidio_results = [
            # Sequential phone number parts that should merge
            RecognizerResult(entity_type="PHONE_NUMBER", start=19, end=22, score=0.80),  # "555"
            RecognizerResult(entity_type="PHONE_NUMBER", start=23, end=26, score=0.85),  # "123" 
            RecognizerResult(entity_type="PHONE_NUMBER", start=27, end=31, score=0.90),  # "4567"
        ]
        
        return [
            EntityDetectionResult.from_presidio_result(r, base_text[r.start:r.end])
            for r in presidio_results
        ]

    @pytest.fixture
    def basic_policy(self) -> MaskingPolicy:
        """Create a basic masking policy."""
        return MaskingPolicy(
            default_strategy=Strategy(kind=StrategyKind.REDACT),
            per_entity={
                "PHONE_NUMBER": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[PHONE]"}),
                "EMAIL_ADDRESS": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[EMAIL]"}),
                "PERSON": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[PERSON]"}),
            },
        )

    def test_confidence_based_resolution(self, overlapping_entities):
        """Test conflict resolution based on confidence scores."""
        config = ConflictResolutionConfig(
            strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
            merge_threshold_chars=0  # Only resolve actual overlaps, not adjacent entities
        )
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(overlapping_entities)

        # Should keep highest confidence entities and resolve overlaps
        assert len(result.normalized_entities) == 4  # One overlap resolved
        assert result.conflicts_resolved == 1

        # Verify highest confidence entity was kept (PERSON vs ORGANIZATION overlap)
        person_entities = [e for e in result.normalized_entities if e.entity_type == "PERSON"]
        org_entities = [e for e in result.normalized_entities if e.entity_type == "ORGANIZATION"]
        
        assert len(person_entities) == 1  # PERSON kept (higher confidence)
        assert len(org_entities) == 0     # ORGANIZATION removed (lower confidence)

    def test_priority_based_resolution(self, overlapping_entities):
        """Test conflict resolution based on entity type priorities."""
        config = ConflictResolutionConfig(
            strategy=ConflictResolutionStrategy.MOST_SPECIFIC,
            entity_priorities={
                "PHONE_NUMBER": EntityPriority.CRITICAL,
                "PERSON": EntityPriority.HIGH,
                "ORGANIZATION": EntityPriority.MEDIUM,
                "LOCATION": EntityPriority.LOW,
            }
        )
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(overlapping_entities)

        # Should resolve conflicts based on priority
        assert result.conflicts_resolved >= 1
        assert len(result.normalized_entities) <= len(overlapping_entities)

    def test_adjacent_entity_merging(self, adjacent_entities):
        """Test merging of adjacent entities of the same type."""
        config = ConflictResolutionConfig(
            strategy=ConflictResolutionStrategy.MERGE_ADJACENT,
            merge_threshold_chars=2
        )
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(adjacent_entities)

        # Should merge adjacent PHONE_NUMBER entities
        assert len(result.normalized_entities) == 1  # All merged into one
        merged_entity = result.normalized_entities[0]
        assert merged_entity.entity_type == "PHONE_NUMBER"
        assert merged_entity.start == 19  # Start of first entity
        assert merged_entity.end == 31    # End of last entity
        assert merged_entity.confidence == 0.90  # Highest confidence

    def test_masking_engine_integration(
        self, simple_document, basic_policy, text_segments
    ):
        """Test that MaskingEngine integrates with conflict resolution."""
        # Create overlapping RecognizerResult entities for MaskingEngine
        overlapping_recognizer_results = [
            # "John Smith" as PERSON
            RecognizerResult(entity_type="PERSON", start=5, end=15, score=0.90),
            # "Smith" as part of a longer pattern (overlaps with PERSON)
            RecognizerResult(entity_type="ORGANIZATION", start=10, end=15, score=0.60),
            # Phone number
            RecognizerResult(entity_type="PHONE_NUMBER", start=19, end=31, score=0.95),
            # Email
            RecognizerResult(entity_type="EMAIL_ADDRESS", start=41, end=65, score=0.85),
            # Location
            RecognizerResult(entity_type="LOCATION", start=69, end=77, score=0.75),
        ]
        
        engine = MaskingEngine(resolve_conflicts=True)

        # This should not raise an error when conflicts are resolved
        result = engine.mask_document(
            document=simple_document,
            entities=overlapping_recognizer_results,
            policy=basic_policy,
            text_segments=text_segments,
        )

        # Should successfully produce masked document
        assert result.masked_document is not None
        assert len(result.cloakmap.anchors) > 0

        # Verify overlaps were resolved
        anchor_positions = [(a.start, a.end) for a in result.cloakmap.anchors]
        for i, (start1, end1) in enumerate(anchor_positions):
            for start2, end2 in anchor_positions[i + 1:]:
                assert not (start1 < end2 and start2 < end1), f"Overlapping anchors: ({start1}, {end1}) and ({start2}, {end2})"

    def test_deterministic_resolution(self, overlapping_entities):
        """Test that conflict resolution produces deterministic results."""
        config = ConflictResolutionConfig(
            strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        )
        normalizer = EntityNormalizer(config)

        # Run normalization multiple times
        results = []
        for _ in range(5):
            result = normalizer.normalize_entities(overlapping_entities.copy())
            results.append(result.normalized_entities)

        # All results should be identical
        for result in results[1:]:
            assert len(result) == len(results[0])
            for i, entity in enumerate(result):
                expected = results[0][i]
                assert entity.entity_type == expected.entity_type
                assert entity.start == expected.start
                assert entity.end == expected.end
                assert entity.confidence == expected.confidence

    def test_high_confidence_preservation(self, overlapping_entities):
        """Test that very high confidence entities are always preserved."""
        # Modify one entity to have very high confidence
        high_conf_entities = overlapping_entities.copy()
        base_text = "Call John Smith at 555-123-4567 or email john.smith@example.com in New York"
        presidio_result = RecognizerResult(
            entity_type="ORGANIZATION", start=10, end=15, score=0.98  # Very high confidence
        )
        high_conf_entities[1] = EntityDetectionResult.from_presidio_result(
            presidio_result, base_text[presidio_result.start:presidio_result.end]
        )

        config = ConflictResolutionConfig(
            strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
            preserve_high_confidence=0.97
        )
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(high_conf_entities)

        # Should preserve the high-confidence entity even if it overlaps
        high_conf_preserved = any(
            e.confidence >= 0.97 for e in result.normalized_entities
        )
        assert high_conf_preserved

    def test_validation_after_normalization(self, overlapping_entities):
        """Test validation of normalization results."""
        config = ConflictResolutionConfig(
            strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        )
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(overlapping_entities)
        
        validation = normalizer.validate_normalization(
            original=overlapping_entities,
            normalized=result.normalized_entities
        )

        # Should pass all validation checks
        assert validation["sorted_correctly"] is True
        assert validation["no_overlaps"] is True
        assert len(validation["warnings"]) == 0

    def test_conflict_resolution_logging(self, overlapping_entities, caplog):
        """Test that conflict resolution produces appropriate log messages."""
        import logging
        caplog.set_level(logging.DEBUG)

        config = ConflictResolutionConfig(
            strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        )
        normalizer = EntityNormalizer(config)

        normalizer.normalize_entities(overlapping_entities)

        # Should have debug logs about conflict resolution
        assert any("Normalizing" in record.message for record in caplog.records)
        assert any("entity groups" in record.message for record in caplog.records)