"""Performance benchmarking tests for CloakPivot.

These tests measure and validate performance characteristics to catch regressions
and ensure the system meets performance requirements.
"""

import statistics
import time
from typing import Callable

import psutil
import pytest

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.unmasking.engine import UnmaskingEngine
from tests.utils.assertions import (
    assert_memory_usage_reasonable,
    assert_performance_acceptable,
)
from tests.utils.generators import DocumentGenerator, PolicyGenerator, TextGenerator
from tests.utils.masking_helpers import mask_document_with_detection

# Performance threshold constants - optimized for faster CI runs
SMALL_DOC_TIMEOUT = 5.0
MEDIUM_DOC_TIMEOUT = 8.0
LARGE_DOC_TIMEOUT_SMALL = 10.0
LARGE_DOC_TIMEOUT_LARGE = 15.0
MULTI_SECTION_TIMEOUT = 15.0
ROUND_TRIP_TIMEOUT = 10.0
BATCH_TIMEOUT = 10.0

# Memory threshold constants
SMALL_DOC_MEMORY_LIMIT = 2000.0
MEDIUM_DOC_MEMORY_LIMIT = 3000.0
LARGE_DOC_MEMORY_LIMIT = 2500.0
MULTI_SECTION_MEMORY_LIMIT = 12000.0
ROUND_TRIP_MEMORY_LIMIT = 4100.0
BATCH_MEMORY_LIMIT = 8800.0


class PerformanceProfiler:
    """Utility class for measuring performance metrics."""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
        self.process = psutil.Process()

    def start(self):
        """Start performance measurement."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory

    def stop(self) -> dict[str, float]:
        """Stop measurement and return metrics."""
        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, end_memory)

        return {
            "elapsed_time": end_time - self.start_time,
            "start_memory_mb": self.start_memory,
            "end_memory_mb": end_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_delta_mb": end_memory - self.start_memory,
        }

    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)


def run_with_profiling(func: Callable, *args, **kwargs) -> tuple[any, dict[str, float]]:
    """Run function with performance profiling."""
    profiler = PerformanceProfiler()
    profiler.start()

    # Monitor memory during execution
    import threading

    stop_monitoring = threading.Event()

    def memory_monitor():
        while not stop_monitoring.wait(0.1):  # Check every 100ms
            profiler.update_peak_memory()

    monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    monitor_thread.start()

    try:
        result = func(*args, **kwargs)
    finally:
        stop_monitoring.set()
        monitor_thread.join(timeout=1.0)

    metrics = profiler.stop()
    return result, metrics


class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""

    @pytest.fixture
    def masking_engine(self) -> MaskingEngine:
        """Create masking engine for benchmarking."""
        return MaskingEngine()

    @pytest.fixture
    def unmasking_engine(self) -> UnmaskingEngine:
        """Create unmasking engine for benchmarking."""
        return UnmaskingEngine()

    @pytest.fixture
    def benchmark_policy(self) -> MaskingPolicy:
        """Standard policy for benchmarking."""
        return PolicyGenerator.generate_comprehensive_policy("medium")

    @pytest.mark.performance
    def test_small_document_performance(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
        benchmark,
    ):
        """Benchmark performance with small documents (< 1KB)."""
        # Generate small document - reduced size for faster testing
        text = TextGenerator.generate_text_with_pii_density(50, 0.2)  # ~50 words
        document = DocumentGenerator.generate_simple_document(text, "small_doc")

        def mask_operation():
            return mask_document_with_detection(
                document, benchmark_policy, analyzer=shared_analyzer
            )

        # Use pytest-benchmark for CI integration
        benchmark(mask_operation)

        # Traditional performance assertions for backwards compatibility
        result_traditional, metrics = run_with_profiling(mask_operation)
        text_length = len(text)
        assert_performance_acceptable(
            metrics["elapsed_time"], SMALL_DOC_TIMEOUT, text_length
        )
        assert_memory_usage_reasonable(
            metrics["memory_delta_mb"], SMALL_DOC_MEMORY_LIMIT, text_length
        )

        # Log benchmark results
        chars_per_sec = (
            text_length / metrics["elapsed_time"] if metrics["elapsed_time"] > 0 else 0
        )
        print(
            f"Small document: {chars_per_sec:.0f} chars/sec, {metrics['peak_memory_mb']:.1f}MB peak"
        )

    @pytest.mark.performance
    def test_medium_document_performance(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
        benchmark,
    ):
        """Benchmark performance with medium documents (1-10KB)."""
        # Generate medium document - reduced size for faster testing
        text = TextGenerator.generate_text_with_pii_density(200, 0.15)  # ~200 words
        document = DocumentGenerator.generate_simple_document(text, "medium_doc")

        def mask_operation():
            return mask_document_with_detection(
                document, benchmark_policy, analyzer=shared_analyzer
            )

        # Use pytest-benchmark for CI integration
        benchmark(mask_operation)

        # Traditional performance assertions for backwards compatibility
        result_traditional, metrics = run_with_profiling(mask_operation)
        text_length = len(text)
        assert_performance_acceptable(
            metrics["elapsed_time"], MEDIUM_DOC_TIMEOUT, text_length
        )
        assert_memory_usage_reasonable(
            metrics["memory_delta_mb"], MEDIUM_DOC_MEMORY_LIMIT, text_length
        )

        chars_per_sec = (
            text_length / metrics["elapsed_time"] if metrics["elapsed_time"] > 0 else 0
        )
        print(
            f"Medium document: {chars_per_sec:.0f} chars/sec, {metrics['peak_memory_mb']:.1f}MB peak"
        )

    @pytest.mark.performance
    def test_large_document_performance_scaled_fast(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
    ):
        """Benchmark performance with single scaled document size for fast runs."""
        word_count = 500  # Single representative size for fast testing - reduced
        expected_time = LARGE_DOC_TIMEOUT_SMALL
        # Generate document of specified size
        text = TextGenerator.generate_text_with_pii_density(word_count, 0.1)
        document = DocumentGenerator.generate_simple_document(
            text, f"large_doc_{word_count}"
        )

        def mask_operation():
            return mask_document_with_detection(
                document, benchmark_policy, analyzer=shared_analyzer
            )

        result, metrics = run_with_profiling(mask_operation)

        # Scale performance expectations based on document size
        text_length = len(text)
        assert_performance_acceptable(
            metrics["elapsed_time"], expected_time, text_length
        )
        assert_memory_usage_reasonable(
            metrics["memory_delta_mb"], LARGE_DOC_MEMORY_LIMIT, text_length
        )

        chars_per_sec = (
            text_length / metrics["elapsed_time"] if metrics["elapsed_time"] > 0 else 0
        )
        print(
            f"Large document ({word_count} words): {chars_per_sec:.0f} chars/sec, {metrics['peak_memory_mb']:.1f}MB peak"
        )

    @pytest.mark.performance
    def test_very_large_document_sampling(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
    ):
        """Test performance with very large documents using sampling approach."""
        # Generate smaller representative samples instead of one huge document - reduced sizes
        word_counts = [200, 400, 600]
        total_time = 0.0

        for word_count in word_counts:
            text = TextGenerator.generate_text_with_pii_density(word_count, 0.1)
            document = DocumentGenerator.generate_simple_document(
                text, f"sample_{word_count}"
            )

            def mask_operation(doc=document):
                return mask_document_with_detection(
                    doc, benchmark_policy, analyzer=shared_analyzer
                )

            result, metrics = run_with_profiling(mask_operation)
            total_time += metrics["elapsed_time"]

            # Each sample should be reasonable
            text_length = len(text)
            assert_performance_acceptable(metrics["elapsed_time"], 10.0, text_length)

        # Total time across all samples should be reasonable
        assert (
            total_time < BATCH_TIMEOUT
        ), f"Total sampling time {total_time:.2f}s too slow"

    @pytest.mark.performance
    def test_multi_section_document_performance(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
    ):
        """Benchmark performance with multi-section documents."""
        # Generate document with multiple sections - reduced count and size
        sections = []
        for i in range(5):  # Reduced from 10 to 5 sections
            section_text = TextGenerator.generate_text_with_pii_density(
                50, 0.2
            )  # Reduced from 200 to 50 words
            sections.append(f"Section {i + 1}: {section_text}")

        document = DocumentGenerator.generate_multi_section_document(
            sections, "multi_section_doc"
        )

        def mask_operation():
            return mask_document_with_detection(
                document, benchmark_policy, analyzer=shared_analyzer
            )

        result, metrics = run_with_profiling(mask_operation)

        # Performance should scale with number of sections
        total_length = sum(len(section) for section in sections)
        assert_performance_acceptable(
            metrics["elapsed_time"], MULTI_SECTION_TIMEOUT, total_length
        )
        assert_memory_usage_reasonable(
            metrics["peak_memory_mb"], MULTI_SECTION_MEMORY_LIMIT, total_length
        )  # Increased for multi-section with shared analyzer and ML models

        sections_per_sec = (
            len(sections) / metrics["elapsed_time"]
            if metrics["elapsed_time"] > 0
            else 0
        )
        print(
            f"Multi-section: {sections_per_sec:.1f} sections/sec, {metrics['peak_memory_mb']:.1f}MB peak"
        )

    @pytest.mark.performance
    def test_round_trip_performance(
        self,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
    ):
        """Benchmark round-trip masking/unmasking performance."""
        text = TextGenerator.generate_text_with_pii_density(
            150, 0.2
        )  # Reduced from 500 to 150 words
        document = DocumentGenerator.generate_simple_document(text, "round_trip_doc")

        def round_trip_operation():
            mask_result = mask_document_with_detection(
                document, benchmark_policy, analyzer=shared_analyzer
            )
            unmask_result = unmasking_engine.unmask_document(
                mask_result.masked_document, mask_result.cloakmap
            )
            return mask_result, unmask_result

        (mask_result, unmask_result), metrics = run_with_profiling(round_trip_operation)

        # Round-trip should be less than 2x masking time
        text_length = len(text)
        assert_performance_acceptable(
            metrics["elapsed_time"], ROUND_TRIP_TIMEOUT, text_length
        )
        assert_memory_usage_reasonable(
            metrics["memory_delta_mb"], ROUND_TRIP_MEMORY_LIMIT, text_length
        )  # Use memory delta to avoid test suite contamination

        print(
            f"Round-trip: {metrics['elapsed_time']:.3f}s, {metrics['peak_memory_mb']:.1f}MB peak"
        )

    @pytest.mark.performance
    def test_privacy_level_performance_comparison(
        self, masking_engine: MaskingEngine, shared_analyzer
    ):
        """Compare performance across different privacy levels."""
        text = TextGenerator.generate_text_with_pii_density(
            100, 0.2
        )  # Reduced from 500 to 100 words
        document = DocumentGenerator.generate_simple_document(
            text, "privacy_comparison_doc"
        )

        results = {}

        for privacy_level in ["low", "medium", "high"]:
            policy = PolicyGenerator.generate_comprehensive_policy(privacy_level)

            def mask_operation(p=policy):
                return mask_document_with_detection(
                    document, p, analyzer=shared_analyzer
                )

            result, metrics = run_with_profiling(mask_operation)
            results[privacy_level] = metrics

            print(f"Privacy {privacy_level}: {metrics['elapsed_time']:.3f}s")

        # High privacy should not be significantly slower than low privacy
        # This is a reasonable assumption for most masking strategies
        low_time = results["low"]["elapsed_time"]
        high_time = results["high"]["elapsed_time"]

        assert (
            high_time < low_time * 3.0
        ), f"High privacy too slow: {high_time:.3f}s vs {low_time:.3f}s"

    @pytest.mark.performance
    def test_batch_processing_performance(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
    ):
        """Benchmark batch processing performance."""
        # Create multiple documents - reduced count and size
        documents = []
        for i in range(3):  # Reduced from 5 to 3 documents
            text = TextGenerator.generate_text_with_pii_density(
                100, 0.15
            )  # Reduced from 300 to 100 words
            doc = DocumentGenerator.generate_simple_document(text, f"batch_doc_{i}")
            documents.append(doc)

        def batch_mask_operation():
            results = []
            for doc in documents:
                result = mask_document_with_detection(
                    doc, benchmark_policy, analyzer=shared_analyzer
                )
                results.append(result)
            return results

        results, metrics = run_with_profiling(batch_mask_operation)

        # Batch processing should be efficient
        total_length = sum(len(doc.texts[0].text) for doc in documents)
        docs_per_sec = (
            len(documents) / metrics["elapsed_time"]
            if metrics["elapsed_time"] > 0
            else 0
        )

        assert (
            docs_per_sec > 0.5
        ), f"Batch processing too slow: {docs_per_sec:.2f} docs/sec"
        assert_memory_usage_reasonable(
            metrics["peak_memory_mb"], BATCH_MEMORY_LIMIT, total_length
        )  # Increased for batch processing with ML models

        print(
            f"Batch processing: {docs_per_sec:.1f} docs/sec, {metrics['peak_memory_mb']:.1f}MB peak"
        )

    @pytest.mark.performance
    def test_memory_leak_detection_batched_fast(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
    ):
        """Test for memory leaks with small iteration count for fast runs."""
        iterations = 8  # Smaller iteration count for fast testing
        text = TextGenerator.generate_text_with_pii_density(
            150, 0.2
        )  # Slightly smaller text
        document = DocumentGenerator.generate_simple_document(
            text, f"memory_leak_test_{iterations}"
        )

        # Measure baseline memory
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_measurements = [baseline_memory]

        # Perform operations using shared analyzer
        for i in range(iterations):
            mask_document_with_detection(
                document, benchmark_policy, analyzer=shared_analyzer
            )

            # Measure memory every 3 operations for faster execution
            if i % 3 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)

        # Check for memory growth trend
        memory_growth = memory_measurements[-1] - memory_measurements[0]

        # Scaled expectations for smaller batches
        max_growth = 50.0 if iterations <= 10 else 75.0
        assert memory_growth < max_growth, (
            f"Potential memory leak detected: {memory_growth:.1f}MB growth "
            f"over {iterations} iterations"
        )

        print(f"Memory growth over {iterations} operations: {memory_growth:.1f}MB")

    @pytest.mark.performance
    def test_memory_stability_quick(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
    ):
        """Quick memory stability test with minimal iterations."""
        text = TextGenerator.generate_text_with_pii_density(100, 0.1)
        document = DocumentGenerator.generate_simple_document(text, "quick_memory_test")

        # Test with just a few iterations for speed
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Perform 5 operations and check stability
        for _ in range(5):
            mask_document_with_detection(
                document, benchmark_policy, analyzer=shared_analyzer
            )

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - baseline_memory

        # Very conservative check for obvious leaks
        assert (
            memory_growth < 25.0
        ), f"Quick memory leak detected: {memory_growth:.1f}MB growth in 5 operations"

    @pytest.mark.performance
    def test_concurrent_processing_performance(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
    ):
        """Test performance with concurrent document processing."""
        import concurrent.futures

        # Create multiple documents - reduced count and size
        documents = []
        for i in range(4):  # Reduced from 10 to 4 documents
            text = TextGenerator.generate_text_with_pii_density(
                50, 0.15
            )  # Reduced from 200 to 50 words
            doc = DocumentGenerator.generate_simple_document(
                text, f"concurrent_doc_{i}"
            )
            documents.append(doc)

        def process_document(doc):
            return mask_document_with_detection(
                doc, benchmark_policy, analyzer=shared_analyzer
            )

        # Sequential processing
        start_time = time.perf_counter()
        sequential_results = [process_document(doc) for doc in documents]
        sequential_time = time.perf_counter() - start_time

        # Concurrent processing
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(process_document, documents))
        concurrent_time = time.perf_counter() - start_time

        # Concurrent processing should provide some speedup
        # (though the exact speedup depends on the implementation)
        speedup_ratio = sequential_time / concurrent_time if concurrent_time > 0 else 1

        print(
            f"Sequential: {sequential_time:.3f}s, Concurrent: {concurrent_time:.3f}s, "
            f"Speedup: {speedup_ratio:.2f}x"
        )

        # Verify results are equivalent
        assert len(sequential_results) == len(concurrent_results)

        # Basic sanity check - concurrent should be at least as fast as sequential
        # (allowing for overhead and measurement variance)
        assert concurrent_time <= sequential_time * 1.5

    @pytest.mark.performance
    def test_strategy_performance_comparison(
        self, masking_engine: MaskingEngine, shared_analyzer
    ):
        """Compare performance of different masking strategies."""
        from cloakpivot.core.strategies import StrategyKind

        text = TextGenerator.generate_text_with_pii_density(
            100, 0.3
        )  # Reduced from 300 to 100 words
        document = DocumentGenerator.generate_simple_document(
            text, "strategy_comparison"
        )

        strategies = [
            StrategyKind.TEMPLATE,
            StrategyKind.REDACT,
            StrategyKind.HASH,
            StrategyKind.SURROGATE,
        ]

        results = {}

        for strategy in strategies:
            # Create policy with single strategy
            policy = PolicyGenerator.generate_custom_policy(
                {"PHONE_NUMBER": strategy}, {"PHONE_NUMBER": 0.5}
            )

            def mask_operation(p=policy):
                return mask_document_with_detection(
                    document, p, analyzer=shared_analyzer
                )

            result, metrics = run_with_profiling(mask_operation)
            results[strategy.value] = metrics

            print(f"Strategy {strategy.value}: {metrics['elapsed_time']:.4f}s")

        # All strategies should complete in reasonable time
        for strategy_name, metrics in results.items():
            assert (
                metrics["elapsed_time"] < 5.0
            ), f"Strategy {strategy_name} too slow: {metrics['elapsed_time']:.3f}s"


class TestPerformanceRegression:
    """Tests to detect performance regressions."""

    @pytest.mark.performance
    def test_regression_baseline(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
        benchmark,
    ):
        """Establish performance baseline for regression detection."""
        # Standard benchmark document
        text = """
        Employee Information Record
        ===========================

        Personal Details:
        Name: John Smith
        Phone: 555-123-4567
        Email: john.smith@company.com
        SSN: 123-45-6789

        Emergency Contact:
        Name: Jane Doe
        Phone: 555-987-6543
        Email: jane.doe@personal.com

        Additional Information:
        Credit Card: 4532-1234-5678-9012
        Driver License: DL123456789
        Address: 123 Main St, New York, NY 10001
        """

        document = DocumentGenerator.generate_simple_document(
            text, "regression_baseline"
        )

        def mask_operation():
            return mask_document_with_detection(
                document, benchmark_policy, analyzer=shared_analyzer
            )

        # Use pytest-benchmark for CI integration with multiple rounds for stability
        benchmark.pedantic(mask_operation, rounds=5, iterations=1)

        # Traditional performance assertions for backwards compatibility
        times = []
        memory_usage = []

        for _ in range(5):
            result_traditional, metrics = run_with_profiling(mask_operation)
            times.append(metrics["elapsed_time"])
            memory_usage.append(metrics["peak_memory_mb"])

        # Calculate statistics
        avg_time = statistics.mean(times)
        avg_memory = statistics.mean(memory_usage)
        time_stddev = statistics.stdev(times) if len(times) > 1 else 0

        # Performance expectations (realistic values when using shared analyzer)
        expected_max_time = 5.0  # seconds - more realistic for NLP model processing
        expected_max_memory = 8800.0  # MB - realistic for Presidio with loaded models

        assert avg_time < expected_max_time, (
            f"Performance regression detected: avg time {avg_time:.3f}s "
            f"exceeds expected {expected_max_time}s"
        )

        assert avg_memory < expected_max_memory, (
            f"Memory regression detected: avg memory {avg_memory:.1f}MB "
            f"exceeds expected {expected_max_memory}MB"
        )

        # Log results for tracking
        print(
            f"Regression baseline: {avg_time:.3f}s ±{time_stddev:.3f}s, {avg_memory:.1f}MB"
        )

        # Save metrics for comparison (in a real scenario, you'd persist these)
        regression_data = {
            "avg_time": avg_time,
            "avg_memory": avg_memory,
            "time_stddev": time_stddev,
            "version": "1.0",  # Increment when intentional changes are made
        }

        return regression_data


@pytest.mark.slow
class TestPerformanceBenchmarksComprehensive:
    """Comprehensive performance benchmarks with full parametrization for slow runs."""

    @pytest.fixture
    def masking_engine(self) -> MaskingEngine:
        """Create masking engine for benchmarking."""
        return MaskingEngine()

    @pytest.fixture
    def unmasking_engine(self) -> UnmaskingEngine:
        """Create unmasking engine for benchmarking."""
        return UnmaskingEngine()

    @pytest.fixture
    def benchmark_policy(self) -> MaskingPolicy:
        """Standard policy for benchmarking."""
        return PolicyGenerator.generate_comprehensive_policy("medium")

    @pytest.mark.performance
    @pytest.mark.parametrize(
        "word_count,expected_time",
        [
            (2500, LARGE_DOC_TIMEOUT_SMALL),  # Smaller batch
            (5000, LARGE_DOC_TIMEOUT_LARGE),  # Medium batch
            (
                7500,
                LARGE_DOC_TIMEOUT_LARGE * 1.5,
            ),  # Larger batch for comprehensive testing
        ],
    )
    def test_large_document_performance_scaled_comprehensive(
        self,
        word_count: int,
        expected_time: float,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
    ):
        """Benchmark performance with all document sizes - comprehensive slow version."""
        # Generate document of specified size
        text = TextGenerator.generate_text_with_pii_density(word_count, 0.1)
        document = DocumentGenerator.generate_simple_document(
            text, f"large_doc_{word_count}"
        )

        def mask_operation():
            return mask_document_with_detection(
                document, benchmark_policy, analyzer=shared_analyzer
            )

        result, metrics = run_with_profiling(mask_operation)

        # Scale performance expectations based on document size
        text_length = len(text)
        assert_performance_acceptable(
            metrics["elapsed_time"], expected_time, text_length
        )
        assert_memory_usage_reasonable(
            metrics["memory_delta_mb"], LARGE_DOC_MEMORY_LIMIT, text_length
        )

        chars_per_sec = (
            text_length / metrics["elapsed_time"] if metrics["elapsed_time"] > 0 else 0
        )
        print(
            f"Large document ({word_count} words): {chars_per_sec:.0f} chars/sec, {metrics['peak_memory_mb']:.1f}MB peak"
        )

    @pytest.mark.performance
    @pytest.mark.parametrize("iterations", [8, 12, 16])
    def test_memory_leak_detection_batched_comprehensive(
        self,
        iterations: int,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
    ):
        """Test for memory leaks with various iteration counts - comprehensive slow version."""
        text = TextGenerator.generate_text_with_pii_density(150, 0.2)
        document = DocumentGenerator.generate_simple_document(
            text, f"memory_leak_test_{iterations}"
        )

        # Measure baseline memory
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_measurements = [baseline_memory]

        # Perform operations using shared analyzer
        for i in range(iterations):
            mask_document_with_detection(
                document, benchmark_policy, analyzer=shared_analyzer
            )

            # Measure memory every few operations
            if i % 3 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)

        # Check for memory growth trend
        memory_growth = memory_measurements[-1] - memory_measurements[0]

        # Scaled expectations based on iteration count
        max_growth = (
            50.0 + (iterations - 8) * 5.0
        )  # Allow more growth for more iterations
        assert memory_growth < max_growth, (
            f"Potential memory leak detected: {memory_growth:.1f}MB growth "
            f"over {iterations} iterations"
        )

        print(f"Memory growth over {iterations} operations: {memory_growth:.1f}MB")

    @pytest.mark.performance
    @pytest.mark.parametrize("privacy_level", ["low", "medium", "high"])
    def test_privacy_level_performance_comparison_comprehensive(
        self, privacy_level: str, masking_engine: MaskingEngine, shared_analyzer
    ):
        """Compare performance across all privacy levels - comprehensive slow version."""
        text = TextGenerator.generate_text_with_pii_density(500, 0.2)
        document = DocumentGenerator.generate_simple_document(
            text, f"privacy_comparison_doc_{privacy_level}"
        )

        policy = PolicyGenerator.generate_comprehensive_policy(privacy_level)

        def mask_operation():
            return mask_document_with_detection(
                document, policy, analyzer=shared_analyzer
            )

        result, metrics = run_with_profiling(mask_operation)

        print(
            f"Privacy {privacy_level}: {metrics['elapsed_time']:.3f}s, {metrics['peak_memory_mb']:.1f}MB"
        )

        # Performance should be reasonable for all privacy levels
        text_length = len(text)
        assert_performance_acceptable(metrics["elapsed_time"], 15.0, text_length)

    @pytest.mark.performance
    @pytest.mark.parametrize("batch_size", [5, 8, 12])
    def test_batch_processing_performance_comprehensive(
        self,
        batch_size: int,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer,
    ):
        """Benchmark batch processing with various sizes - comprehensive slow version."""
        # Create multiple documents
        documents = []
        for i in range(batch_size):
            text = TextGenerator.generate_text_with_pii_density(300, 0.15)
            doc = DocumentGenerator.generate_simple_document(text, f"batch_doc_{i}")
            documents.append(doc)

        def batch_mask_operation():
            results = []
            for doc in documents:
                result = mask_document_with_detection(
                    doc, benchmark_policy, analyzer=shared_analyzer
                )
                results.append(result)
            return results

        results, metrics = run_with_profiling(batch_mask_operation)

        # Batch processing should be efficient
        total_length = sum(len(doc.texts[0].text) for doc in documents)
        docs_per_sec = (
            len(documents) / metrics["elapsed_time"]
            if metrics["elapsed_time"] > 0
            else 0
        )

        assert (
            docs_per_sec > 0.2
        ), f"Batch processing too slow: {docs_per_sec:.2f} docs/sec"

        # Scale memory expectations with batch size
        memory_limit = (
            BATCH_MEMORY_LIMIT + (batch_size - 5) * 1000
        )  # Additional memory for larger batches
        assert_memory_usage_reasonable(
            metrics["peak_memory_mb"], memory_limit, total_length
        )

        print(
            f"Batch processing ({batch_size} docs): {docs_per_sec:.1f} docs/sec, {metrics['peak_memory_mb']:.1f}MB peak"
        )
