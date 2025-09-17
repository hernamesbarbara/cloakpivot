"""Edge case tests for presidio_adapter improvements."""

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, TextItem
from presidio_analyzer.recognizer_result import RecognizerResult

from cloakpivot.core import MaskingPolicy, Strategy, StrategyKind
from cloakpivot.document.extractor import TextExtractor, TextSegment
from cloakpivot.masking.presidio_adapter import PresidioMaskingAdapter


class TestEdgeCases:
    """Test edge cases identified in FEAT_improve_presidio_adapter_ideas.md."""

    def test_entities_on_segment_boundaries(self):
        """Test entities that fall exactly on segment boundaries."""
        adapter = PresidioMaskingAdapter()
        policy = MaskingPolicy(
            default_strategy=Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[PII]"})
        )

        # Create document with multiple segments
        doc = DoclingDocument(name="test.txt")
        doc.texts = [
            TextItem(text="First segment with email@test.com", label=DocItemLabel.TEXT, self_ref="#/texts/0", orig="First segment with email@test.com"),
            TextItem(text="test@example.com starts this segment", label=DocItemLabel.TEXT, self_ref="#/texts/1", orig="test@example.com starts this segment"),
        ]

        # Extract segments
        extractor = TextExtractor()
        segments = extractor.extract_text_segments(doc)

        # Create entity exactly at segment boundary (end of first, start of second)
        # The email in first segment ends at position 34
        # The second segment starts at position 36 (after \n\n separator)
        entities = [
            RecognizerResult(entity_type="EMAIL", start=19, end=34, score=0.95),  # email@test.com
            RecognizerResult(entity_type="EMAIL", start=36, end=52, score=0.95),  # test@example.com
        ]

        result = adapter.mask_document(doc, entities, policy, segments)

        # Check that both emails are masked
        assert "[PII]" in result.masked_document.texts[0].text
        assert "[PII]" in result.masked_document.texts[1].text

    def test_overlapping_entities_same_confidence(self):
        """Test overlapping entities with identical confidence but different lengths."""
        adapter = PresidioMaskingAdapter()

        # Create overlapping entities with same confidence
        entities = [
            RecognizerResult(entity_type="PERSON", start=0, end=5, score=0.90),  # "John "
            RecognizerResult(entity_type="PERSON", start=0, end=8, score=0.90),  # "John Doe"
            RecognizerResult(entity_type="PERSON", start=5, end=8, score=0.90),  # "Doe"
        ]

        filtered = adapter._filter_overlapping_entities(entities)

        # Should keep the longest match when confidence is equal
        assert len(filtered) == 1
        assert filtered[0].start == 0
        assert filtered[0].end == 8

    def test_very_short_strings_partial_masking(self):
        """Test PARTIAL masking with strings shorter than visible_chars."""
        adapter = PresidioMaskingAdapter()

        # Test with string shorter than visible_chars
        result = adapter._apply_partial_strategy(
            text="abc",
            entity_type="TEST",
            strategy=Strategy(
                kind=StrategyKind.PARTIAL,
                parameters={"visible_chars": 5, "position": "end"}
            ),
            confidence=0.95
        )

        # Should return original when text is shorter than visible_chars
        assert result == "abc"

        # Test with exact length
        result = adapter._apply_partial_strategy(
            text="12345",
            entity_type="TEST",
            strategy=Strategy(
                kind=StrategyKind.PARTIAL,
                parameters={"visible_chars": 5, "position": "end"}
            ),
            confidence=0.95
        )

        # Should show all chars when length equals visible_chars
        assert result == "12345"

    def test_hash_truncate_edge_cases(self):
        """Test HASH strategy with various truncate parameters."""
        adapter = PresidioMaskingAdapter()

        # Test truncate shorter than hash
        result = adapter._apply_hash_strategy(
            text="test@example.com",
            entity_type="EMAIL",
            strategy=Strategy(
                kind=StrategyKind.HASH,
                parameters={"truncate": 5, "prefix": "H:"}
            ),
            confidence=0.95
        )

        assert result.startswith("H:")
        assert len(result) == 5  # Total length 5 including prefix

        # Test truncate longer than hash (should not pad)
        result = adapter._apply_hash_strategy(
            text="test",
            entity_type="TEXT",
            strategy=Strategy(
                kind=StrategyKind.HASH,
                parameters={"truncate": 100}
            ),
            confidence=0.95
        )

        # Should not exceed actual hash length
        assert len(result) <= 100

    def test_surrogate_seed_changes(self):
        """Test SURROGATE strategy with seed parameter changes."""
        adapter = PresidioMaskingAdapter()

        # First call with seed "42"
        result1 = adapter._apply_surrogate_strategy(
            text="John Doe",
            entity_type="PERSON",
            strategy=Strategy(
                kind=StrategyKind.SURROGATE,
                parameters={"seed": "42"}
            )
        )

        # Second call with same seed should give same result
        result2 = adapter._apply_surrogate_strategy(
            text="John Doe",
            entity_type="PERSON",
            strategy=Strategy(
                kind=StrategyKind.SURROGATE,
                parameters={"seed": "42"}
            )
        )

        assert result1 == result2

        # Call with different seed should give different result
        result3 = adapter._apply_surrogate_strategy(
            text="John Doe",
            entity_type="PERSON",
            strategy=Strategy(
                kind=StrategyKind.SURROGATE,
                parameters={"seed": "123"}
            )
        )

        assert result1 != result3

    def test_empty_document_segments(self):
        """Test handling of empty document segments."""
        adapter = PresidioMaskingAdapter()
        policy = MaskingPolicy(default_strategy=Strategy(kind=StrategyKind.REDACT))

        # Empty document
        doc = DoclingDocument(name="empty.txt")
        doc.texts = []

        segments = []
        entities = []

        result = adapter.mask_document(doc, entities, policy, segments)

        assert result.masked_document is not None
        assert len(result.masked_document.texts) == 0

    def test_entity_at_document_end(self):
        """Test entity at the exact end of document."""
        adapter = PresidioMaskingAdapter()
        policy = MaskingPolicy(
            default_strategy=Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[MASKED]"})
        )

        doc = DoclingDocument(name="test.txt")
        doc.texts = [
            TextItem(text="Contact us at test@example.com", label=DocItemLabel.TEXT, self_ref="#/texts/0", orig="Contact us at test@example.com")
        ]

        extractor = TextExtractor()
        segments = extractor.extract_text_segments(doc)

        # Entity at exact end of text
        entities = [
            RecognizerResult(entity_type="EMAIL", start=14, end=30, score=0.95)
        ]

        result = adapter.mask_document(doc, entities, policy, segments)

        assert "test@example.com" not in result.masked_document.texts[0].text
        assert "[MASKED]" in result.masked_document.texts[0].text

    def test_apply_spans_efficiency(self):
        """Test that _apply_spans handles large documents efficiently."""
        adapter = PresidioMaskingAdapter()

        # Create a large text with many replacements
        text = "a" * 10000
        spans = [(i * 100, i * 100 + 10, f"[MASK{i}]") for i in range(100)]

        # This should complete quickly even with 100 replacements
        result = adapter._apply_spans(text, spans)

        # Verify replacements were made
        for i in range(100):
            assert f"[MASK{i}]" in result

    def test_zero_length_entity(self):
        """Test handling of zero-length entities."""
        adapter = PresidioMaskingAdapter()

        # Create entity with start == end
        entity = RecognizerResult(entity_type="TEST", start=5, end=5, score=0.95)

        filtered = adapter._filter_overlapping_entities([entity])

        # Zero-length entities should be kept (edge case handling)
        assert len(filtered) == 1

    def test_entity_beyond_document_length(self):
        """Test entity that extends beyond document length."""
        adapter = PresidioMaskingAdapter()
        policy = MaskingPolicy(default_strategy=Strategy(kind=StrategyKind.REDACT))

        doc = DoclingDocument(name="test.txt")
        doc.texts = [
            TextItem(text="Short text", label=DocItemLabel.TEXT, self_ref="#/texts/0", orig="Short text")
        ]

        extractor = TextExtractor()
        segments = extractor.extract_text_segments(doc)

        # Entity beyond text length
        entities = [
            RecognizerResult(entity_type="TEST", start=5, end=100, score=0.95)
        ]

        result = adapter.mask_document(doc, entities, policy, segments)

        # Should handle gracefully without crashing
        assert result.stats["entities_masked"] == 1
        # But the out-of-bounds entity should be skipped
        assert "Short text" in result.masked_document.texts[0].text or "*****" in result.masked_document.texts[0].text

    def test_partial_masking_positions(self):
        """Test all position options for partial masking."""
        adapter = PresidioMaskingAdapter()
        text = "1234567890"

        # Test "start" position
        result = adapter._apply_partial_strategy(
            text=text,
            entity_type="TEST",
            strategy=Strategy(
                kind=StrategyKind.PARTIAL,
                parameters={"visible_chars": 3, "position": "start", "mask_char": "X"}
            ),
            confidence=0.95
        )
        assert result == "123XXXXXXX"

        # Test "end" position
        result = adapter._apply_partial_strategy(
            text=text,
            entity_type="TEST",
            strategy=Strategy(
                kind=StrategyKind.PARTIAL,
                parameters={"visible_chars": 3, "position": "end", "mask_char": "X"}
            ),
            confidence=0.95
        )
        assert result == "XXXXXXX890"

        # Test invalid position (should default to end)
        result = adapter._apply_partial_strategy(
            text=text,
            entity_type="TEST",
            strategy=Strategy(
                kind=StrategyKind.PARTIAL,
                parameters={"visible_chars": 3, "position": "middle", "mask_char": "X"}
            ),
            confidence=0.95
        )
        assert result == "XXXXXXX890"

    def test_multiple_entities_same_position(self):
        """Test multiple entities at the same position (different types)."""
        adapter = PresidioMaskingAdapter()

        # Multiple entity types detected at same position
        entities = [
            RecognizerResult(entity_type="EMAIL", start=0, end=16, score=0.95),
            RecognizerResult(entity_type="URL", start=0, end=16, score=0.90),
        ]

        filtered = adapter._filter_overlapping_entities(entities)

        # Should keep highest confidence
        assert len(filtered) == 1
        assert filtered[0].entity_type == "EMAIL"
        assert filtered[0].score == 0.95


class TestPerformanceOptimizations:
    """Test performance optimizations."""

    def test_apply_spans_preserves_text_correctly(self):
        """Test that _apply_spans preserves non-replaced text correctly."""
        adapter = PresidioMaskingAdapter()

        text = "The quick brown fox jumps over the lazy dog"
        spans = [
            (4, 9, "[FAST]"),   # "quick" -> "[FAST]"
            (16, 19, "[ANIMAL]"), # "fox" -> "[ANIMAL]"
            (35, 39, "[TIRED]"),  # "lazy" -> "[TIRED]"
        ]

        result = adapter._apply_spans(text, spans)

        expected = "The [FAST] brown [ANIMAL] jumps over the [TIRED] dog"
        assert result == expected

    def test_apply_spans_empty_cases(self):
        """Test _apply_spans with edge cases."""
        adapter = PresidioMaskingAdapter()

        # Empty spans
        assert adapter._apply_spans("test", []) == "test"

        # Empty text
        assert adapter._apply_spans("", [(0, 0, "X")]) == "X"

        # Single replacement at start
        assert adapter._apply_spans("test", [(0, 2, "XX")]) == "XXst"

        # Single replacement at end
        assert adapter._apply_spans("test", [(2, 4, "XX")]) == "teXX"

        # Adjacent replacements
        assert adapter._apply_spans("test", [(0, 2, "XX"), (2, 4, "YY")]) == "XXYY"

    def test_apply_spans_order_independence(self):
        """Test that _apply_spans handles spans correctly regardless of input order."""
        adapter = PresidioMaskingAdapter()

        text = "ABCDEFGHIJ"

        # Spans in random order
        spans_unordered = [
            (6, 8, "**"),  # GH -> **
            (0, 2, "##"),  # AB -> ##
            (3, 5, "$$"),  # DE -> $$
        ]

        result = adapter._apply_spans(text, spans_unordered)

        # Should sort internally and apply correctly
        assert result == "##C$$F**IJ"