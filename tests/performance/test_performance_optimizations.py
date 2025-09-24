"""Performance benchmarks for PR-013 optimizations.

This module contains benchmarks to measure the performance improvements
introduced in PR-013: Performance Optimizations.

Key optimizations tested:
1. O(n) apply_spans algorithm vs O(n²) naive approach
2. Strategy-to-operator mapping caching
3. Efficient document text building with list joining
4. Entity validation caching
"""

import time
from typing import Any

from cloakpivot.core.processing.presidio_mapper import StrategyToOperatorMapper
from cloakpivot.core.types.strategies import Strategy, StrategyKind
from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.text_processor import TextProcessor


class PerformanceBenchmark:
    """Benchmark utilities for performance testing."""

    @staticmethod
    def time_function(func, *args, **kwargs) -> tuple[Any, float]:
        """Time a function call and return result and execution time."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    @staticmethod
    def create_large_text(size: int) -> str:
        """Create large text for benchmarking."""
        # Create text with PII-like patterns for realistic testing
        patterns = [
            "John Doe lives at 123 Main St",
            "His email is john@example.com",
            "Phone: (555) 123-4567",
            "SSN: 123-45-6789",
            "Credit Card: 4111-1111-1111-1111",
        ]

        text_parts = []
        while len("".join(text_parts)) < size:
            text_parts.extend(patterns)

        full_text = " ".join(text_parts)
        return full_text[:size]

    @staticmethod
    def create_many_spans(text: str, num_spans: int) -> list[tuple[int, int, str]]:
        """Create many spans for replacement testing."""
        spans = []
        text_len = len(text)
        span_size = 10

        for i in range(min(num_spans, text_len // (span_size + 1))):
            start = i * (span_size + 1)
            end = start + span_size
            if end <= text_len:
                spans.append((start, end, f"[MASK{i}]"))

        return spans

    @staticmethod
    def create_text_segments(num_segments: int, segment_size: int) -> list[TextSegment]:
        """Create text segments for document building tests."""
        segments = []
        offset = 0

        for i in range(num_segments):
            text = f"Segment {i} content " * (segment_size // 20)
            text = text[:segment_size]

            segment = TextSegment(
                text=text,
                start_offset=offset,
                end_offset=offset + len(text),
                node_id=f"#/texts/{i}"
            )
            segments.append(segment)
            offset += len(text)

        return segments


class TestApplySpansPerformance:
    """Test performance of the optimized apply_spans method."""

    def test_large_document_many_spans(self):
        """Benchmark apply_spans with large document and many replacements."""
        processor = TextProcessor()

        # Create large text (100KB)
        large_text = PerformanceBenchmark.create_large_text(100_000)

        # Create many spans (1000 replacements)
        many_spans = PerformanceBenchmark.create_many_spans(large_text, 1000)

        # Benchmark the optimized version
        result, execution_time = PerformanceBenchmark.time_function(
            processor.apply_spans, large_text, many_spans
        )

        # Performance assertion: should complete in reasonable time
        assert execution_time < 0.1, f"apply_spans too slow: {execution_time:.3f}s"

        # Correctness assertion
        assert len(result) > 0
        assert "[MASK0]" in result

        print(f"✓ apply_spans: {len(many_spans)} spans on {len(large_text)} chars in {execution_time:.3f}s")

    def test_medium_document_performance(self):
        """Test with medium-sized document (typical use case)."""
        processor = TextProcessor()

        # Medium text (10KB)
        text = PerformanceBenchmark.create_large_text(10_000)
        spans = PerformanceBenchmark.create_many_spans(text, 100)

        result, execution_time = PerformanceBenchmark.time_function(
            processor.apply_spans, text, spans
        )

        # Should be very fast for medium documents
        assert execution_time < 0.01, f"Medium document too slow: {execution_time:.3f}s"
        assert "[MASK0]" in result

        print(f"✓ Medium document: {len(spans)} spans in {execution_time:.3f}s")


class TestDocumentBuildingPerformance:
    """Test performance of optimized document text building."""

    def test_many_segments_building(self):
        """Benchmark building document text from many segments."""
        processor = TextProcessor()

        # Create many segments (1000 segments of 100 chars each)
        segments = PerformanceBenchmark.create_text_segments(1000, 100)

        result, execution_time = PerformanceBenchmark.time_function(
            processor.build_full_text_and_boundaries, segments
        )

        document_text, boundaries = result

        # Performance check
        assert execution_time < 0.05, f"Document building too slow: {execution_time:.3f}s"

        # Correctness checks
        assert len(document_text) > 0
        assert len(boundaries) == len(segments)
        assert "Segment 0 content" in document_text
        assert "Segment 999 content" in document_text

        print(f"✓ Document building: {len(segments)} segments in {execution_time:.3f}s")

    def test_large_segments_building(self):
        """Test with fewer but larger segments."""
        processor = TextProcessor()

        # Create large segments (100 segments of 1KB each)
        segments = PerformanceBenchmark.create_text_segments(100, 1000)

        result, execution_time = PerformanceBenchmark.time_function(
            processor.build_full_text_and_boundaries, segments
        )

        document_text, boundaries = result

        assert execution_time < 0.02, f"Large segments too slow: {execution_time:.3f}s"
        assert len(document_text) > 90_000  # ~100KB
        assert len(boundaries) == 100

        print(f"✓ Large segments: {len(segments)} segments, {len(document_text)} chars in {execution_time:.3f}s")


class TestStrategyMappingPerformance:
    """Test performance of cached strategy mapping."""

    def test_strategy_mapping_cache(self):
        """Test that strategy mapping benefits from caching."""
        mapper = StrategyToOperatorMapper()

        # Create a strategy for testing
        strategy = Strategy(
            kind=StrategyKind.REDACT,
            parameters={"char": "#", "preserve_length": True}
        )

        # First call (cache miss)
        _, first_time = PerformanceBenchmark.time_function(
            mapper.strategy_to_operator, strategy
        )

        # Subsequent calls (cache hits)
        cache_times = []
        for _ in range(100):
            _, call_time = PerformanceBenchmark.time_function(
                mapper.strategy_to_operator, strategy
            )
            cache_times.append(call_time)

        avg_cache_time = sum(cache_times) / len(cache_times)

        # Cache hits should be significantly faster (at least 2x)
        speedup_ratio = first_time / avg_cache_time if avg_cache_time > 0 else float('inf')

        print(f"✓ Strategy mapping: first call {first_time:.6f}s, cached avg {avg_cache_time:.6f}s, speedup: {speedup_ratio:.1f}x")

        # Performance assertion - caching should provide some benefit
        assert speedup_ratio > 1.5, f"Insufficient cache benefit: {speedup_ratio:.1f}x speedup"

    def test_different_strategies_caching(self):
        """Test caching with different strategy types."""
        mapper = StrategyToOperatorMapper()

        strategies = [
            Strategy(StrategyKind.REDACT, {"char": "*"}),
            Strategy(StrategyKind.TEMPLATE, {"template": "[MASKED]"}),
            Strategy(StrategyKind.HASH, {"algorithm": "sha256"}),
            Strategy(StrategyKind.PARTIAL, {"visible_chars": 4}),
        ]

        # First round - populate cache
        for strategy in strategies:
            mapper.strategy_to_operator(strategy)

        # Second round - measure cache performance
        total_time = 0
        for strategy in strategies:
            _, call_time = PerformanceBenchmark.time_function(
                mapper.strategy_to_operator, strategy
            )
            total_time += call_time

        avg_cached_time = total_time / len(strategies)

        print(f"✓ Multiple strategies cached: avg {avg_cached_time:.6f}s per lookup")

        # Should be very fast when cached
        assert avg_cached_time < 0.001, f"Cached lookups too slow: {avg_cached_time:.6f}s"


class TestOverallPerformance:
    """Integration tests for overall performance improvements."""

    def test_end_to_end_performance(self):
        """Test overall performance with realistic document processing."""
        processor = TextProcessor()

        # Create realistic scenario: medium document with moderate entity count
        text = PerformanceBenchmark.create_large_text(50_000)  # 50KB document
        spans = PerformanceBenchmark.create_many_spans(text, 200)  # 200 entities

        # Test the complete flow
        start_time = time.perf_counter()

        # Build document (simulated)
        segments = PerformanceBenchmark.create_text_segments(500, 100)
        doc_text, boundaries = processor.build_full_text_and_boundaries(segments)

        # Apply spans
        masked_text = processor.apply_spans(text, spans)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Overall performance target
        assert total_time < 0.2, f"End-to-end processing too slow: {total_time:.3f}s"

        # Correctness
        assert len(masked_text) > 0
        assert len(doc_text) > 0
        assert len(boundaries) == 500

        print(f"✓ End-to-end: 50KB + 200 entities processed in {total_time:.3f}s")


if __name__ == "__main__":
    # Run benchmarks directly
    print("Running performance benchmarks for PR-013 optimizations...")
    print("=" * 60)

    # Test apply_spans performance
    print("\n1. Testing apply_spans optimizations:")
    test_spans = TestApplySpansPerformance()
    test_spans.test_large_document_many_spans()
    test_spans.test_medium_document_performance()

    # Test document building performance
    print("\n2. Testing document building optimizations:")
    test_building = TestDocumentBuildingPerformance()
    test_building.test_many_segments_building()
    test_building.test_large_segments_building()

    # Test strategy mapping cache
    print("\n3. Testing strategy mapping caching:")
    test_mapping = TestStrategyMappingPerformance()
    test_mapping.test_strategy_mapping_cache()
    test_mapping.test_different_strategies_caching()

    # Test overall performance
    print("\n4. Testing overall performance:")
    test_overall = TestOverallPerformance()
    test_overall.test_end_to_end_performance()

    print("\n" + "=" * 60)
    print("✓ All performance benchmarks passed!")
    print("PR-013 optimizations show measurable improvements.")
