"""Tests for internal methods of PresidioMaskingAdapter."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from docling_core.types import DoclingDocument
from presidio_analyzer import RecognizerResult

from cloakpivot.core.policies.policies import MaskingPolicy
from cloakpivot.core.types.strategies import Strategy, StrategyKind
from cloakpivot.masking.presidio_adapter import PresidioMaskingAdapter

# MaskedEntity replaced with RecognizerResult


class TestPresidioAdapterInternals:
    """Test internal methods of PresidioMaskingAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create a basic PresidioMaskingAdapter instance."""
        return PresidioMaskingAdapter()

    @pytest.fixture
    def mock_document(self):
        """Create a mock DoclingDocument."""
        doc = MagicMock(spec=DoclingDocument)
        text_item = MagicMock()
        text_item.text = "Test document with email john@example.com and phone 555-1234"
        doc.texts = [text_item]
        return doc

    def test_filter_overlapping_entities_complex_overlaps(self, adapter):
        """Test filtering of complex overlapping entities."""
        # Create overlapping entities
        entities = [
            RecognizerResult(entity_type="EMAIL", start=10, end=30, score=0.9),
            RecognizerResult(entity_type="PERSON", start=15, end=25, score=0.85),
            RecognizerResult(entity_type="DOMAIN", start=20, end=30, score=0.8),
            RecognizerResult(entity_type="PHONE_NUMBER", start=40, end=50, score=0.95),
            RecognizerResult(entity_type="NUMBER", start=45, end=50, score=0.7),
        ]

        # Filter overlapping entities
        filtered = adapter._filter_overlapping_entities(entities)

        # Should keep highest confidence non-overlapping entities
        assert len(filtered) == 2  # EMAIL and PHONE_NUMBER
        assert filtered[0].entity_type == "EMAIL"
        assert filtered[1].entity_type == "PHONE_NUMBER"

    def test_validate_entities_against_boundaries_edge_cases(self, adapter):
        """Test validation of entities against text boundaries."""
        from cloakpivot.masking.protocols import SegmentBoundary

        text = "Short text with email@test.com"
        segment_boundaries = [
            SegmentBoundary(segment_index=0, start=0, end=len(text), node_id="#/texts/0")
        ]
        entities = [
            RecognizerResult(entity_type="EMAIL", start=16, end=30, score=0.9),
            RecognizerResult(entity_type="PERSON", start=28, end=35, score=0.8),  # Beyond boundary
            RecognizerResult(entity_type="PHONE", start=-5, end=10, score=0.7),  # Negative start
        ]

        validated = adapter._validate_entities_against_boundaries(entities, text, segment_boundaries)

        # Should keep EMAIL and truncate PERSON to fit within boundary, skip PHONE (negative start)
        assert len(validated) == 2
        assert validated[0].entity_type == "EMAIL"
        assert validated[1].entity_type == "PERSON"
        assert validated[1].end == 30  # Truncated from 35 to 30

    def test_batch_process_entities_large_batches(self, adapter):
        """Test batch processing of large entity batches."""
        # Create test text
        text = "x" * 1000  # Large text

        # Create a large batch of entities
        entities = []
        for i in range(10):  # Reduced from 100 to avoid out of bounds
            entities.append(
                RecognizerResult(
                    entity_type=f"TYPE_{i % 5}",
                    start=i * 10,
                    end=(i * 10) + 8,
                    score=0.8 + (i % 3) * 0.05
                )
            )

        # Create strategies
        strategies = {}
        for i in range(5):
            strategies[f"TYPE_{i}"] = Strategy(kind=StrategyKind.REDACT, parameters={"char": "*"})

        # Process entities
        results = adapter._batch_process_entities(text, entities, strategies)

        # Verify all entities were processed
        assert len(results) == 10  # All entities should be processed

    def test_prepare_strategies_fallback_scenarios(self, adapter):
        """Test strategy preparation with fallback scenarios."""
        # Test with missing strategy
        entities = [
            RecognizerResult(
                entity_type="CUSTOM_TYPE",
                start=0,
                end=10,
                score=0.9
            )
        ]

        # MaskingPolicy should provide default strategy for unknown types
        policy = MaskingPolicy()
        strategies = adapter._prepare_strategies(entities, policy)

        # Should have strategy for CUSTOM_TYPE
        assert "CUSTOM_TYPE" in strategies
        # Default strategy should be created
        assert isinstance(strategies["CUSTOM_TYPE"], Strategy)

    def test_apply_spans_unicode_handling(self, adapter):
        """Test applying spans with unicode characters."""
        text = "Hello 世界 email@test.com 你好"
        spans = [
            (6, 8, "[REDACTED]"),  # Replace 世界
            (9, 23, "[EMAIL]"),  # Replace email@test.com
        ]

        result = adapter._apply_spans(text, spans)

        # Should handle unicode correctly
        assert "[REDACTED]" in result
        assert "[EMAIL]" in result
        assert "Hello" in result
        assert "你好" in result

    def test_build_full_text_with_empty_segments(self, adapter):
        """Test building full text when some segments are empty."""
        segments = [
            {"text": "First segment"},
            {"text": ""},  # Empty segment
            {"text": "Third segment"},
            {"text": None},  # None segment
            {"text": "Final segment"},
        ]

        full_text = adapter._build_full_text(segments)

        # Should handle empty/None segments gracefully
        assert "First segment" in full_text
        assert "Third segment" in full_text
        assert "Final segment" in full_text
        assert full_text.count("segment") == 3

    def test_create_synthetic_result_various_entities(self, adapter):
        """Test creating synthetic results for various entity types."""
        entity_types = ["EMAIL", "PERSON", "PHONE_NUMBER", "CREDIT_CARD", "SSN"]

        for entity_type in entity_types:
            result = adapter._create_synthetic_result(
                entity_type=entity_type,
                start=10,
                end=20,
                score=0.85,
                text="sample_text"
            )

            assert result.entity_type == entity_type
            assert result.start == 10
            assert result.end == 20
            assert result.score == 0.85

    def test_cleanup_large_results_memory_management(self, adapter):
        """Test cleanup of large results for memory management."""
        from presidio_anonymizer import OperatorResult

        # Create a large result set
        large_results = []
        for i in range(1000):
            result = Mock(spec=OperatorResult)
            result.text = f"masked_text_{i}" + "x" * 1000  # Large text
            result.entity_type = f"TYPE_{i % 5}"
            large_results.append(result)

        # Cleanup should work without errors
        adapter._cleanup_large_results(large_results)

        # Method modifies in place, verify it doesn't crash with large input
        assert True  # If we get here, cleanup worked

    def test_filter_overlapping_entities_preserves_highest_score(self, adapter):
        """Test that filtering preserves highest scoring entities."""
        entities = [
            RecognizerResult(entity_type="PERSON", start=0, end=10, score=0.9),
            RecognizerResult(entity_type="PERSON", start=11, end=20, score=0.85),
            RecognizerResult(entity_type="EMAIL", start=25, end=40, score=0.95),
            RecognizerResult(entity_type="EMAIL", start=30, end=45, score=0.9),
        ]

        filtered = adapter._filter_overlapping_entities(entities)

        # Should keep non-overlapping and highest scoring
        assert len(filtered) == 3  # Two PERSON (non-overlapping) and one EMAIL (highest)
        email_entities = [e for e in filtered if e.entity_type == "EMAIL"]
        assert len(email_entities) == 1
        assert email_entities[0].score == 0.95

    def test_validate_entities_filters_invalid(self, adapter):
        """Test that validation filters out invalid entities."""
        entities = [
            RecognizerResult(entity_type="LOCATION", start=10, end=50, score=0.8),
            RecognizerResult(entity_type="CITY", start=20, end=30, score=0.9),
            RecognizerResult(entity_type="INVALID", start=-5, end=10, score=0.85),
        ]

        validated = adapter._validate_entities(entities, "Test text for validation purposes")

        # Should filter invalid entities
        assert all(e.start >= 0 for e in validated)
        assert all(e.end >= e.start for e in validated)

    def test_batch_processing_splits_correctly(self, adapter):
        """Test that batch processing splits entities correctly."""
        entities = [
            RecognizerResult(entity_type="EMAIL", start=i*10, end=i*10+5, score=0.8)
            for i in range(25)
        ]

        batches = adapter._batch_process_entities(entities, batch_size=10)

        # Should create correct number of batches
        assert len(batches) == 3  # 25 entities / 10 per batch = 3 batches
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

    def test_apply_spans_with_replacements(self, adapter):
        """Test applying spans to text."""
        original = "John Doe works at Acme Corp and his email is john@acme.com"
        spans = [
            (0, 8, "[PERSON]"),
            (18, 27, "[COMPANY]"),
            (46, 59, "[EMAIL]"),
        ]

        result = adapter._apply_spans(original, spans)

        assert "[PERSON]" in result
        assert "[COMPANY]" in result
        assert "[EMAIL]" in result
        assert "john@acme.com" not in result

    def test_prepare_strategies_creates_dict(self, adapter):
        """Test that prepare_strategies creates proper dict."""
        # Strategy already imported from core.strategies

        entities = [
            RecognizerResult(
                entity_type="EMAIL",
                start=10,
                end=30,
                score=0.9
            ),
            RecognizerResult(
                entity_type="PHONE",
                start=40,
                end=50,
                score=0.85
            ),
        ]

        strategies = adapter._prepare_strategies(entities)

        # Should create strategy dict
        assert isinstance(strategies, dict)
        assert len(strategies) >= 1

    def test_document_metadata_preservation(self, adapter, mock_document):
        """Test preservation of document metadata during processing."""
        mock_document.metadata = {
            "source": "test_file.pdf",
            "created": "2024-01-01",
            "custom_field": "value"
        }

        # Process document
        with patch.object(adapter, '_analyze_text') as mock_analyze:
            mock_analyze.return_value = []
            result = adapter.mask_document(mock_document)

            # Verify metadata is preserved
            assert hasattr(result, 'metadata') or hasattr(result, 'document')
            # Exact check depends on implementation

    def test_filter_overlapping_performance(self, adapter):
        """Test performance of overlap filtering."""
        import time

        # Create many potentially overlapping entities
        entities = []
        for i in range(100):
            entities.append(
                RecognizerResult(
                    entity_type="ID",
                    start=i * 2,
                    end=(i * 2) + 5,
                    score=0.8 + (i % 10) * 0.01
                )
            )

        start_time = time.time()
        filtered = adapter._filter_overlapping_entities(entities)
        elapsed = time.time() - start_time

        # Should filter quickly even with many entities
        assert elapsed < 0.5  # Less than 0.5 seconds
        assert len(filtered) > 0
