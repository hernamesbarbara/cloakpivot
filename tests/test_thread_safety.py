"""Comprehensive thread safety tests for singleton loaders.

This module provides stress testing and validation of thread-safe access patterns
for the singleton loaders implemented in cloakpivot.loaders. Tests concurrent
access scenarios, cache consistency, deadlock prevention, and performance under load.
"""

import concurrent.futures
import threading
import time
from typing import Any, Callable

import psutil
import pytest

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.loaders import (
    clear_all_caches,
    get_cache_info,
    get_detection_pipeline,
    get_detection_pipeline_from_policy,
    get_document_processor,
    get_presidio_analyzer,
)


class TestConcurrentAccess:
    """Test concurrent access patterns for all loader functions."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    @pytest.mark.performance
    def test_stress_analyzer_concurrent_access(self):
        """Stress test analyzer creation with high concurrent load."""
        thread_count = 50
        iterations_per_thread = 20
        results = []
        exceptions = []

        def worker():
            try:
                thread_results = []
                for _ in range(iterations_per_thread):
                    analyzer = get_presidio_analyzer()
                    thread_results.append(id(analyzer))
                results.extend(thread_results)
            except Exception as e:
                exceptions.append(e)

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(worker) for _ in range(thread_count)]

            for future in concurrent.futures.as_completed(futures, timeout=30):
                future.result()  # Wait for completion and propagate exceptions

        execution_time = time.time() - start_time

        # Validate results
        assert len(exceptions) == 0, f"Exceptions during concurrent access: {exceptions}"
        assert len(results) == thread_count * iterations_per_thread

        # All should return the same instance (singleton pattern)
        unique_instances = set(results)
        assert len(unique_instances) == 1, f"Expected 1 unique instance, got {len(unique_instances)}"

        # Performance validation - should complete in reasonable time
        assert execution_time < 10.0, f"Stress test took too long: {execution_time:.2f}s"

        print(f"Stress test completed: {thread_count} threads × {iterations_per_thread} iterations "
              f"in {execution_time:.2f}s ({len(results)/execution_time:.1f} ops/sec)")

    @pytest.mark.performance
    def test_stress_processor_concurrent_access(self):
        """Stress test processor creation with high concurrent load."""
        thread_count = 30
        iterations_per_thread = 15
        results = []
        exceptions = []

        def worker():
            try:
                thread_results = []
                for _ in range(iterations_per_thread):
                    processor = get_document_processor()
                    thread_results.append(id(processor))
                results.extend(thread_results)
            except Exception as e:
                exceptions.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(worker) for _ in range(thread_count)]

            for future in concurrent.futures.as_completed(futures, timeout=20):
                future.result()

        # Validate results
        assert len(exceptions) == 0, f"Exceptions during concurrent access: {exceptions}"
        assert len(results) == thread_count * iterations_per_thread

        # All should return the same instance
        unique_instances = set(results)
        assert len(unique_instances) == 1, f"Expected 1 unique instance, got {len(unique_instances)}"

    @pytest.mark.performance
    def test_stress_pipeline_concurrent_access(self):
        """Stress test pipeline creation with high concurrent load."""
        thread_count = 25
        iterations_per_thread = 10
        results = []
        exceptions = []

        def worker():
            try:
                thread_results = []
                for _ in range(iterations_per_thread):
                    pipeline = get_detection_pipeline()
                    thread_results.append(id(pipeline))
                results.extend(thread_results)
            except Exception as e:
                exceptions.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(worker) for _ in range(thread_count)]

            for future in concurrent.futures.as_completed(futures, timeout=15):
                future.result()

        # Validate results
        assert len(exceptions) == 0, f"Exceptions during concurrent access: {exceptions}"
        assert len(results) == thread_count * iterations_per_thread

        # All should return the same instance
        unique_instances = set(results)
        assert len(unique_instances) == 1, f"Expected 1 unique instance, got {len(unique_instances)}"

    @pytest.mark.performance
    def test_mixed_loader_concurrent_access(self):
        """Test concurrent access to different loader types simultaneously."""
        thread_count_per_loader = 10
        results = {"analyzer": [], "processor": [], "pipeline": []}
        exceptions = []

        def analyzer_worker():
            try:
                for _ in range(5):
                    analyzer = get_presidio_analyzer()
                    results["analyzer"].append(id(analyzer))
            except Exception as e:
                exceptions.append(("analyzer", e))

        def processor_worker():
            try:
                for _ in range(5):
                    processor = get_document_processor()
                    results["processor"].append(id(processor))
            except Exception as e:
                exceptions.append(("processor", e))

        def pipeline_worker():
            try:
                for _ in range(5):
                    pipeline = get_detection_pipeline()
                    results["pipeline"].append(id(pipeline))
            except Exception as e:
                exceptions.append(("pipeline", e))

        all_workers = (
            [analyzer_worker] * thread_count_per_loader +
            [processor_worker] * thread_count_per_loader +
            [pipeline_worker] * thread_count_per_loader
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(worker) for worker in all_workers]

            for future in concurrent.futures.as_completed(futures, timeout=20):
                future.result()

        # Validate results
        assert len(exceptions) == 0, f"Exceptions during mixed concurrent access: {exceptions}"

        # Each loader type should return singleton instances
        for loader_type, instance_ids in results.items():
            assert len(instance_ids) == thread_count_per_loader * 5
            unique_instances = set(instance_ids)
            assert len(unique_instances) == 1, f"{loader_type} violated singleton pattern"


class TestConfigurationVariations:
    """Test concurrent access with different configurations."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    @pytest.mark.performance
    def test_concurrent_different_analyzer_configs(self):
        """Test concurrent requests for analyzers with different configurations."""
        configs = [
            ("en", 0.5, "spacy"),
            ("es", 0.7, "spacy"),
            ("fr", 0.6, "spacy"),
            ("en", 0.8, "spacy"),
        ]

        results = {}
        exceptions = []

        def create_analyzer_for_config(lang: str, conf: float, engine: str):
            try:
                analyzer = get_presidio_analyzer(language=lang, min_confidence=conf, nlp_engine_name=engine)
                config_key = (lang, conf, engine)
                if config_key not in results:
                    results[config_key] = []
                results[config_key].append(id(analyzer))
            except Exception as e:
                exceptions.append((lang, conf, engine, e))

        # Create workers for each configuration
        workers = []
        for config in configs:
            for _ in range(10):  # 10 threads per config
                workers.append(lambda c=config: create_analyzer_for_config(*c))

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for worker in workers]

            for future in concurrent.futures.as_completed(futures, timeout=15):
                future.result()

        # Validate results
        assert len(exceptions) == 0, f"Exceptions during config variations: {exceptions}"

        # Should have created distinct instances for each config
        assert len(results) == len(configs), f"Expected {len(configs)} distinct configs, got {len(results)}"

        # Each config should return singleton instances
        for config_key, instance_ids in results.items():
            assert len(instance_ids) == 10, f"Config {config_key} missing instances"
            unique_instances = set(instance_ids)
            assert len(unique_instances) == 1, f"Config {config_key} violated singleton pattern"

    @pytest.mark.performance
    def test_concurrent_processor_configurations(self):
        """Test concurrent requests for processors with different configurations."""
        results = {}
        exceptions = []

        def create_processor_with_chunking(enable_chunked: bool):
            try:
                processor = get_document_processor(enable_chunked=enable_chunked)
                if enable_chunked not in results:
                    results[enable_chunked] = []
                results[enable_chunked].append(id(processor))
            except Exception as e:
                exceptions.append((enable_chunked, e))

        # Create workers for both chunked configurations
        workers = []
        for chunked in [True, False]:
            for _ in range(15):  # 15 threads per config
                workers.append(lambda c=chunked: create_processor_with_chunking(c))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for worker in workers]

            for future in concurrent.futures.as_completed(futures, timeout=10):
                future.result()

        # Validate results
        assert len(exceptions) == 0, f"Exceptions during processor config variations: {exceptions}"
        assert len(results) == 2  # Should have both True and False configurations

        # Each config should return singleton instances
        for chunked_setting, instance_ids in results.items():
            assert len(instance_ids) == 15
            unique_instances = set(instance_ids)
            assert len(unique_instances) == 1, f"Chunked={chunked_setting} violated singleton pattern"

    @pytest.mark.performance
    def test_concurrent_policy_based_pipelines(self):
        """Test concurrent creation of pipelines from different policies."""
        policies = [
            MaskingPolicy(locale="en", thresholds={"EMAIL": 0.8}),
            MaskingPolicy(locale="es", thresholds={"PERSON": 0.7}),
            MaskingPolicy(locale="fr", thresholds={"PHONE": 0.9}),
        ]

        results = []
        exceptions = []
        policy_results = {i: [] for i in range(len(policies))}

        def create_pipeline_from_policy(policy_index: int, policy: MaskingPolicy):
            try:
                pipeline = get_detection_pipeline_from_policy(policy)
                results.append(id(pipeline))
                policy_results[policy_index].append(id(pipeline))
            except Exception as e:
                exceptions.append((policy_index, e))

        # Create workers for each policy
        workers = []
        for i, policy in enumerate(policies):
            for _ in range(8):  # 8 threads per policy
                workers.append(lambda idx=i, pol=policy: create_pipeline_from_policy(idx, pol))

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(worker) for worker in workers]

            for future in concurrent.futures.as_completed(futures, timeout=15):
                future.result()

        # Validate results
        assert len(exceptions) == 0, f"Exceptions during policy-based pipeline creation: {exceptions}"
        assert len(results) == len(policies) * 8

        # Note: Policy-based pipelines create new instances each time by design,
        # but should use cached analyzers internally. We validate no exceptions occurred.
        for policy_index, pipeline_ids in policy_results.items():
            assert len(pipeline_ids) == 8, f"Policy {policy_index} missing pipeline instances"


class TestCacheConsistency:
    """Test cache consistency under concurrent load."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    @pytest.mark.performance
    def test_cache_hit_rate_under_concurrent_load(self):
        """Test that cache hit rates remain high under concurrent access."""
        # Pre-populate cache with single call
        get_presidio_analyzer()
        initial_cache_info = get_cache_info()

        thread_count = 40
        iterations_per_thread = 25
        exceptions = []

        def worker():
            try:
                for _ in range(iterations_per_thread):
                    get_presidio_analyzer()  # Should hit cache every time
            except Exception as e:
                exceptions.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(worker) for _ in range(thread_count)]

            for future in concurrent.futures.as_completed(futures, timeout=15):
                future.result()

        final_cache_info = get_cache_info()

        # Validate no exceptions
        assert len(exceptions) == 0, f"Exceptions during cache consistency test: {exceptions}"

        # Calculate hit rate
        total_new_accesses = thread_count * iterations_per_thread
        new_hits = final_cache_info["analyzer"]["hits"] - initial_cache_info["analyzer"]["hits"]
        new_misses = final_cache_info["analyzer"]["misses"] - initial_cache_info["analyzer"]["misses"]

        assert new_hits == total_new_accesses, f"Expected {total_new_accesses} hits, got {new_hits}"
        assert new_misses == 0, f"Expected 0 new misses, got {new_misses}"

        # Hit rate should be 100% since all access same cached instance
        hit_rate = new_hits / total_new_accesses if total_new_accesses > 0 else 0
        assert hit_rate >= 0.95, f"Cache hit rate {hit_rate:.2%} below target 95%"

    @pytest.mark.performance
    def test_cache_eviction_under_concurrent_load(self):
        """Test cache behavior when maxsize is exceeded under concurrent load."""
        maxsize = 8  # From @lru_cache(maxsize=8) on get_presidio_analyzer

        # Create more configurations than cache can hold
        configs = [("en", 0.1 + i * 0.01) for i in range(maxsize + 3)]

        results = {config: [] for config in configs}
        exceptions = []

        def create_analyzer_for_config(lang: str, confidence: float):
            try:
                analyzer = get_presidio_analyzer(language=lang, min_confidence=confidence)
                config_key = (lang, confidence)
                results[config_key].append(id(analyzer))
            except Exception as e:
                exceptions.append((lang, confidence, e))

        # Create concurrent workers for all configurations
        workers = []
        for config in configs:
            for _ in range(3):  # 3 threads per config
                workers.append(lambda c=config: create_analyzer_for_config(*c))

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for worker in workers]

            for future in concurrent.futures.as_completed(futures, timeout=15):
                future.result()

        # Validate results
        assert len(exceptions) == 0, f"Exceptions during cache eviction test: {exceptions}"

        final_cache_info = get_cache_info()

        # Cache should be at maxsize due to eviction
        assert final_cache_info["analyzer"]["currsize"] <= maxsize

        # Each config that made it into final cache should have singleton behavior
        for config_key, instance_ids in results.items():
            assert len(instance_ids) == 3, f"Config {config_key} missing instances"
            unique_instances = set(instance_ids)
            assert len(unique_instances) == 1, f"Config {config_key} violated singleton pattern"


class TestDeadlockPrevention:
    """Test deadlock prevention and timeout scenarios."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    @pytest.mark.performance
    @pytest.mark.timeout(20)
    def test_no_deadlock_mixed_operations(self):
        """Test that mixed cache operations don't cause deadlocks."""
        operations_completed = []
        exceptions = []

        def mixed_operations():
            try:
                # Mix of different operations that could potentially deadlock
                get_presidio_analyzer()
                get_document_processor()
                get_detection_pipeline()
                clear_all_caches()
                get_presidio_analyzer(language="es")
                get_cache_info()
                operations_completed.append(threading.current_thread().ident)
            except Exception as e:
                exceptions.append(e)

        thread_count = 15

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(mixed_operations) for _ in range(thread_count)]

            # If there's a deadlock, this will timeout due to pytest-timeout
            for future in concurrent.futures.as_completed(futures, timeout=15):
                future.result()

        # Validate all operations completed
        assert len(exceptions) == 0, f"Exceptions during mixed operations: {exceptions}"
        assert len(operations_completed) == thread_count, "Some threads didn't complete"

    @pytest.mark.performance
    @pytest.mark.timeout(10)
    def test_no_deadlock_rapid_cache_clear(self):
        """Test rapid cache clearing doesn't cause deadlocks."""
        operations_completed = []
        exceptions = []

        def rapid_cache_operations():
            try:
                for _ in range(5):
                    get_presidio_analyzer()
                    clear_all_caches()
                    get_presidio_analyzer(min_confidence=0.6)
                operations_completed.append(threading.current_thread().ident)
            except Exception as e:
                exceptions.append(e)

        thread_count = 10

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(rapid_cache_operations) for _ in range(thread_count)]

            for future in concurrent.futures.as_completed(futures, timeout=8):
                future.result()

        assert len(exceptions) == 0, f"Exceptions during rapid cache operations: {exceptions}"
        assert len(operations_completed) == thread_count


class TestPerformanceUnderLoad:
    """Test performance characteristics under concurrent load."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    @pytest.mark.performance
    def test_response_time_scaling(self):
        """Test that response times scale appropriately with thread count."""
        thread_counts = [1, 5, 10, 20, 30]
        performance_results = []

        for thread_count in thread_counts:
            clear_all_caches()

            # Pre-populate cache to focus on concurrent access performance
            get_presidio_analyzer()

            start_time = time.time()
            exceptions = []

            def worker(exc_list=exceptions):
                try:
                    get_presidio_analyzer()
                except Exception as e:
                    exc_list.append(e)

            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(worker) for _ in range(thread_count)]

                for future in concurrent.futures.as_completed(futures, timeout=10):
                    future.result()

            execution_time = time.time() - start_time

            assert len(exceptions) == 0, f"Exceptions with {thread_count} threads: {exceptions}"

            avg_time_per_thread = execution_time / thread_count
            performance_results.append((thread_count, execution_time, avg_time_per_thread))

        # Validate performance scaling
        # Average time per thread should remain reasonable
        for thread_count, total_time, avg_time in performance_results:
            assert avg_time < 0.1, f"Average time per thread too high: {avg_time:.3f}s with {thread_count} threads"
            assert total_time < 5.0, f"Total execution time too high: {total_time:.2f}s with {thread_count} threads"

        print("Performance scaling results:")
        for thread_count, total_time, avg_time in performance_results:
            print(f"  {thread_count:2d} threads: {total_time:.3f}s total, {avg_time:.4f}s avg/thread")

    @pytest.mark.performance
    def test_memory_usage_under_load(self):
        """Test memory usage remains stable under concurrent load."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        thread_count = 30
        iterations = 10
        exceptions = []

        def memory_intensive_worker():
            try:
                for _ in range(iterations):
                    # Create multiple loader instances to stress memory
                    get_presidio_analyzer()
                    get_document_processor()
                    get_detection_pipeline()
            except Exception as e:
                exceptions.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(memory_intensive_worker) for _ in range(thread_count)]

            for future in concurrent.futures.as_completed(futures, timeout=20):
                future.result()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        memory_increase_percentage = (memory_increase / initial_memory) * 100

        assert len(exceptions) == 0, f"Exceptions during memory test: {exceptions}"

        # Memory increase should be reasonable (target: <50MB or <20% increase)
        assert memory_increase < 50.0, f"Memory increase too high: {memory_increase:.1f}MB"
        assert memory_increase_percentage < 20.0, f"Memory increase percentage too high: {memory_increase_percentage:.1f}%"

        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
              f"(+{memory_increase:.1f}MB, +{memory_increase_percentage:.1f}%)")


class TestErrorHandlingConcurrency:
    """Test error handling in concurrent scenarios."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    @pytest.mark.performance
    def test_exception_isolation_across_threads(self):
        """Test that exceptions in one thread don't affect others."""
        successful_results = []
        caught_exceptions = []

        def worker_with_invalid_config():
            try:
                # This should raise ConfigurationError
                get_presidio_analyzer(language="", min_confidence=1.5)
            except Exception as e:
                caught_exceptions.append(e)

        def worker_with_valid_config():
            try:
                analyzer = get_presidio_analyzer()
                successful_results.append(id(analyzer))
            except Exception as e:
                caught_exceptions.append(e)

        # Mix of valid and invalid workers
        workers = [worker_with_valid_config] * 15 + [worker_with_invalid_config] * 5

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for worker in workers]

            # Don't propagate exceptions from futures - we handle them in workers
            for future in concurrent.futures.as_completed(futures, timeout=10):
                try:
                    future.result()
                except Exception:
                    pass  # Expected for invalid config workers

        # Should have successful results from valid workers
        assert len(successful_results) == 15, f"Expected 15 successful results, got {len(successful_results)}"

        # Should have caught exceptions from invalid workers
        assert len(caught_exceptions) == 5, f"Expected 5 caught exceptions, got {len(caught_exceptions)}"

        # All successful results should be the same instance
        unique_successful = set(successful_results)
        assert len(unique_successful) == 1, "Successful workers should return same cached instance"

    @pytest.mark.performance
    def test_recovery_after_initialization_failure(self):
        """Test system recovers properly after initialization failures."""
        # This test simulates what happens if initialization fails and then succeeds

        results_after_recovery = []
        exceptions = []

        # First, try to create with invalid config (should fail)
        def failing_worker():
            try:
                get_presidio_analyzer(language="invalid_language_code_that_is_too_long")
            except Exception as e:
                exceptions.append(e)

        # Then create with valid config (should succeed)
        def recovery_worker():
            try:
                analyzer = get_presidio_analyzer()
                results_after_recovery.append(id(analyzer))
            except Exception as e:
                exceptions.append(e)

        # Run failing workers first
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(failing_worker) for _ in range(5)]
            for future in concurrent.futures.as_completed(futures, timeout=5):
                try:
                    future.result()
                except Exception:
                    pass  # Expected failures

        # Then run recovery workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(recovery_worker) for _ in range(10)]
            for future in concurrent.futures.as_completed(futures, timeout=5):
                future.result()

        # Should have recovered successfully
        assert len(results_after_recovery) == 10, "Recovery workers should succeed"

        # All recovery results should be the same instance
        unique_recovery = set(results_after_recovery)
        assert len(unique_recovery) == 1, "Recovery should create consistent singleton"


def stress_test_loader_function(
    loader_func: Callable[[], Any],
    thread_count: int = 20,
    iterations: int = 10,
    timeout: int = 10
) -> tuple[float, list[int], list[Exception]]:
    """Generic stress test helper for any loader function.

    Args:
        loader_func: Function to stress test
        thread_count: Number of concurrent threads
        iterations: Iterations per thread
        timeout: Timeout in seconds

    Returns:
        Tuple of (execution_time, result_ids, exceptions)
    """
    results = []
    exceptions = []

    def worker():
        try:
            thread_results = []
            for _ in range(iterations):
                instance = loader_func()
                thread_results.append(id(instance))
            results.extend(thread_results)
        except Exception as e:
            exceptions.append(e)

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [executor.submit(worker) for _ in range(thread_count)]

        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            future.result()

    execution_time = time.time() - start_time

    return execution_time, results, exceptions
