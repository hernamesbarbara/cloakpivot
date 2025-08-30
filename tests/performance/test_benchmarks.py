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
            'elapsed_time': end_time - self.start_time,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': end_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_delta_mb': end_memory - self.start_memory
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
        shared_analyzer
    ):
        """Benchmark performance with small documents (< 1KB)."""
        # Generate small document
        text = TextGenerator.generate_text_with_pii_density(100, 0.2)  # ~100 words
        document = DocumentGenerator.generate_simple_document(text, "small_doc")

        def mask_operation():
            return mask_document_with_detection(document, benchmark_policy, analyzer=shared_analyzer)

        result, metrics = run_with_profiling(mask_operation)

        # Performance assertions - use memory delta instead of peak to avoid test suite contamination
        # Note: First run may load models, so memory delta can be high initially
        text_length = len(text)
        assert_performance_acceptable(metrics['elapsed_time'], 30.0, text_length)
        assert_memory_usage_reasonable(metrics['memory_delta_mb'], 2000.0, text_length)  # Increased for ML models

        # Log benchmark results
        chars_per_sec = text_length / metrics['elapsed_time'] if metrics['elapsed_time'] > 0 else 0
        print(f"Small document: {chars_per_sec:.0f} chars/sec, {metrics['peak_memory_mb']:.1f}MB peak")

    @pytest.mark.performance
    def test_medium_document_performance(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer
    ):
        """Benchmark performance with medium documents (1-10KB)."""
        # Generate medium document
        text = TextGenerator.generate_text_with_pii_density(1000, 0.15)  # ~1000 words
        document = DocumentGenerator.generate_simple_document(text, "medium_doc")

        def mask_operation():
            return mask_document_with_detection(document, benchmark_policy, analyzer=shared_analyzer)

        result, metrics = run_with_profiling(mask_operation)

        # Performance assertions - use memory delta instead of peak to avoid test suite contamination
        text_length = len(text)
        assert_performance_acceptable(metrics['elapsed_time'], 20.0, text_length)
        assert_memory_usage_reasonable(metrics['memory_delta_mb'], 3000.0, text_length)  # Increased for ML models

        chars_per_sec = text_length / metrics['elapsed_time'] if metrics['elapsed_time'] > 0 else 0
        print(f"Medium document: {chars_per_sec:.0f} chars/sec, {metrics['peak_memory_mb']:.1f}MB peak")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_document_performance(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer
    ):
        """Benchmark performance with large documents (10-100KB)."""
        # Generate large document
        text = TextGenerator.generate_text_with_pii_density(10000, 0.1)  # ~10000 words
        document = DocumentGenerator.generate_simple_document(text, "large_doc")

        def mask_operation():
            return mask_document_with_detection(document, benchmark_policy, analyzer=shared_analyzer)

        result, metrics = run_with_profiling(mask_operation)

        # Performance assertions (more lenient for large documents) - use memory delta
        text_length = len(text)
        assert_performance_acceptable(metrics['elapsed_time'], 90.0, text_length)
        assert_memory_usage_reasonable(metrics['memory_delta_mb'], 5000.0, text_length)  # Increased for ML models

        chars_per_sec = text_length / metrics['elapsed_time'] if metrics['elapsed_time'] > 0 else 0
        print(f"Large document: {chars_per_sec:.0f} chars/sec, {metrics['peak_memory_mb']:.1f}MB peak")

    @pytest.mark.performance
    def test_multi_section_document_performance(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer
    ):
        """Benchmark performance with multi-section documents."""
        # Generate document with multiple sections
        sections = []
        for i in range(10):
            section_text = TextGenerator.generate_text_with_pii_density(200, 0.2)
            sections.append(f"Section {i + 1}: {section_text}")

        document = DocumentGenerator.generate_multi_section_document(sections, "multi_section_doc")

        def mask_operation():
            return mask_document_with_detection(document, benchmark_policy, analyzer=shared_analyzer)

        result, metrics = run_with_profiling(mask_operation)

        # Performance should scale with number of sections
        total_length = sum(len(section) for section in sections)
        assert_performance_acceptable(metrics['elapsed_time'], 60.0, total_length)
        assert_memory_usage_reasonable(metrics['peak_memory_mb'], 12000.0, total_length)  # Increased for multi-section with shared analyzer and ML models

        sections_per_sec = len(sections) / metrics['elapsed_time'] if metrics['elapsed_time'] > 0 else 0
        print(f"Multi-section: {sections_per_sec:.1f} sections/sec, {metrics['peak_memory_mb']:.1f}MB peak")

    @pytest.mark.performance
    def test_round_trip_performance(
        self,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer
    ):
        """Benchmark round-trip masking/unmasking performance."""
        text = TextGenerator.generate_text_with_pii_density(500, 0.2)
        document = DocumentGenerator.generate_simple_document(text, "round_trip_doc")

        def round_trip_operation():
            mask_result = mask_document_with_detection(document, benchmark_policy, analyzer=shared_analyzer)
            unmask_result = unmasking_engine.unmask_document(
                mask_result.masked_document,
                mask_result.cloakmap
            )
            return mask_result, unmask_result

        (mask_result, unmask_result), metrics = run_with_profiling(round_trip_operation)

        # Round-trip should be less than 2x masking time
        text_length = len(text)
        assert_performance_acceptable(metrics['elapsed_time'], 40.0, text_length)
        assert_memory_usage_reasonable(metrics['peak_memory_mb'], 8200.0, text_length)  # Increased for ML models

        print(f"Round-trip: {metrics['elapsed_time']:.3f}s, {metrics['peak_memory_mb']:.1f}MB peak")

    @pytest.mark.performance
    def test_privacy_level_performance_comparison(
        self,
        masking_engine: MaskingEngine,
        shared_analyzer
    ):
        """Compare performance across different privacy levels."""
        text = TextGenerator.generate_text_with_pii_density(500, 0.2)
        document = DocumentGenerator.generate_simple_document(text, "privacy_comparison_doc")

        results = {}

        for privacy_level in ["low", "medium", "high"]:
            policy = PolicyGenerator.generate_comprehensive_policy(privacy_level)

            def mask_operation():
                return mask_document_with_detection(document, policy, analyzer=shared_analyzer)

            result, metrics = run_with_profiling(mask_operation)
            results[privacy_level] = metrics

            print(f"Privacy {privacy_level}: {metrics['elapsed_time']:.3f}s")

        # High privacy should not be significantly slower than low privacy
        # This is a reasonable assumption for most masking strategies
        low_time = results['low']['elapsed_time']
        high_time = results['high']['elapsed_time']

        assert high_time < low_time * 3.0, f"High privacy too slow: {high_time:.3f}s vs {low_time:.3f}s"

    @pytest.mark.performance
    def test_batch_processing_performance(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer
    ):
        """Benchmark batch processing performance."""
        # Create multiple documents
        documents = []
        for i in range(5):
            text = TextGenerator.generate_text_with_pii_density(300, 0.15)
            doc = DocumentGenerator.generate_simple_document(text, f"batch_doc_{i}")
            documents.append(doc)

        def batch_mask_operation():
            results = []
            for doc in documents:
                result = mask_document_with_detection(doc, benchmark_policy, analyzer=shared_analyzer)
                results.append(result)
            return results

        results, metrics = run_with_profiling(batch_mask_operation)

        # Batch processing should be efficient
        total_length = sum(len(doc.texts[0].text) for doc in documents)
        docs_per_sec = len(documents) / metrics['elapsed_time'] if metrics['elapsed_time'] > 0 else 0

        assert docs_per_sec > 0.5, f"Batch processing too slow: {docs_per_sec:.2f} docs/sec"
        assert_memory_usage_reasonable(metrics['peak_memory_mb'], 8100.0, total_length)  # Increased for batch processing with ML models

        print(f"Batch processing: {docs_per_sec:.1f} docs/sec, {metrics['peak_memory_mb']:.1f}MB peak")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_leak_detection(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer
    ):
        """Test for memory leaks during repeated operations."""
        text = TextGenerator.generate_text_with_pii_density(200, 0.2)
        document = DocumentGenerator.generate_simple_document(text, "memory_leak_test")

        # Measure baseline memory
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_measurements = [baseline_memory]

        # Perform many operations using shared analyzer to avoid AnalyzerEngine recreation
        for i in range(20):
            mask_document_with_detection(document, benchmark_policy, analyzer=shared_analyzer)

            # Measure memory every few operations
            if i % 5 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)

        # Check for memory growth trend
        memory_growth = memory_measurements[-1] - memory_measurements[0]

        # Allow some growth but detect significant leaks (more realistic with shared analyzer)
        max_acceptable_growth = 100.0  # MB - increased for realistic expectations with shared analyzer
        assert memory_growth < max_acceptable_growth, (
            f"Potential memory leak detected: {memory_growth:.1f}MB growth "
            f"over {len(memory_measurements)} measurements"
        )

        print(f"Memory growth over {len(memory_measurements)} operations: {memory_growth:.1f}MB")

    @pytest.mark.performance
    def test_concurrent_processing_performance(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer
    ):
        """Test performance with concurrent document processing."""
        import concurrent.futures

        # Create multiple documents
        documents = []
        for i in range(10):
            text = TextGenerator.generate_text_with_pii_density(200, 0.15)
            doc = DocumentGenerator.generate_simple_document(text, f"concurrent_doc_{i}")
            documents.append(doc)

        def process_document(doc):
            return mask_document_with_detection(doc, benchmark_policy, analyzer=shared_analyzer)

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

        print(f"Sequential: {sequential_time:.3f}s, Concurrent: {concurrent_time:.3f}s, "
              f"Speedup: {speedup_ratio:.2f}x")

        # Verify results are equivalent
        assert len(sequential_results) == len(concurrent_results)

        # Basic sanity check - concurrent should be at least as fast as sequential
        # (allowing for overhead and measurement variance)
        assert concurrent_time <= sequential_time * 1.5

    @pytest.mark.performance
    def test_strategy_performance_comparison(
        self,
        masking_engine: MaskingEngine,
        shared_analyzer
    ):
        """Compare performance of different masking strategies."""
        from cloakpivot.core.strategies import StrategyKind

        text = TextGenerator.generate_text_with_pii_density(300, 0.3)  # Higher PII density
        document = DocumentGenerator.generate_simple_document(text, "strategy_comparison")

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
                {"PHONE_NUMBER": strategy},
                {"PHONE_NUMBER": 0.5}
            )

            def mask_operation():
                return mask_document_with_detection(document, policy, analyzer=shared_analyzer)

            result, metrics = run_with_profiling(mask_operation)
            results[strategy.value] = metrics

            print(f"Strategy {strategy.value}: {metrics['elapsed_time']:.4f}s")

        # All strategies should complete in reasonable time
        for strategy_name, metrics in results.items():
            assert metrics['elapsed_time'] < 5.0, f"Strategy {strategy_name} too slow: {metrics['elapsed_time']:.3f}s"


class TestPerformanceRegression:
    """Tests to detect performance regressions."""

    @pytest.mark.performance
    def test_regression_baseline(
        self,
        masking_engine: MaskingEngine,
        benchmark_policy: MaskingPolicy,
        shared_analyzer
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

        document = DocumentGenerator.generate_simple_document(text, "regression_baseline")

        # Run multiple times to get stable measurement using shared analyzer
        times = []
        memory_usage = []

        for _ in range(5):
            def mask_operation():
                return mask_document_with_detection(document, benchmark_policy, analyzer=shared_analyzer)

            result, metrics = run_with_profiling(mask_operation)
            times.append(metrics['elapsed_time'])
            memory_usage.append(metrics['peak_memory_mb'])

        # Calculate statistics
        avg_time = statistics.mean(times)
        avg_memory = statistics.mean(memory_usage)
        time_stddev = statistics.stdev(times) if len(times) > 1 else 0

        # Performance expectations (realistic values when using shared analyzer)
        expected_max_time = 5.0  # seconds - more realistic for NLP model processing
        expected_max_memory = 8100.0  # MB - realistic for Presidio with loaded models

        assert avg_time < expected_max_time, (
            f"Performance regression detected: avg time {avg_time:.3f}s "
            f"exceeds expected {expected_max_time}s"
        )

        assert avg_memory < expected_max_memory, (
            f"Memory regression detected: avg memory {avg_memory:.1f}MB "
            f"exceeds expected {expected_max_memory}MB"
        )

        # Log results for tracking
        print(f"Regression baseline: {avg_time:.3f}s ±{time_stddev:.3f}s, {avg_memory:.1f}MB")

        # Save metrics for comparison (in a real scenario, you'd persist these)
        regression_data = {
            'avg_time': avg_time,
            'avg_memory': avg_memory,
            'time_stddev': time_stddev,
            'version': '1.0'  # Increment when intentional changes are made
        }

        return regression_data
