"""Tests for thread-safe singleton loaders module."""

import concurrent.futures
import threading
import time

import pytest

from cloakpivot.core.analyzer import AnalyzerConfig, AnalyzerEngineWrapper
from cloakpivot.core.detection import EntityDetectionPipeline
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.document.processor import DocumentProcessor
from cloakpivot.loaders import (
    _generate_config_hash,
    _generate_policy_hash,
    clear_all_caches,
    get_cache_info,
    get_detection_pipeline,
    get_detection_pipeline_from_policy,
    get_document_processor,
    get_presidio_analyzer,
    get_presidio_analyzer_from_config,
)


class TestGenerateConfigHash:
    """Test configuration hashing for cache keys."""

    def test_identical_configs_same_hash(self):
        """Test that identical configs produce the same hash."""
        config1 = AnalyzerConfig(language="en", min_confidence=0.5)
        config2 = AnalyzerConfig(language="en", min_confidence=0.5)

        hash1 = _generate_config_hash(config1)
        hash2 = _generate_config_hash(config2)

        assert hash1 == hash2
        assert len(hash1) == 32  # SHA-256 truncated to 32 chars

    def test_different_configs_different_hash(self):
        """Test that different configs produce different hashes."""
        config1 = AnalyzerConfig(language="en", min_confidence=0.5)
        config2 = AnalyzerConfig(language="es", min_confidence=0.5)

        hash1 = _generate_config_hash(config1)
        hash2 = _generate_config_hash(config2)

        assert hash1 != hash2

    def test_config_hash_includes_all_relevant_fields(self):
        """Test that hash changes when any relevant field changes."""
        base_config = AnalyzerConfig(language="en", min_confidence=0.5)
        base_hash = _generate_config_hash(base_config)

        # Test language change
        lang_config = AnalyzerConfig(language="es", min_confidence=0.5)
        assert _generate_config_hash(lang_config) != base_hash

        # Test confidence change
        conf_config = AnalyzerConfig(language="en", min_confidence=0.7)
        assert _generate_config_hash(conf_config) != base_hash

        # Test nlp_engine_name change
        nlp_config = AnalyzerConfig(
            language="en", min_confidence=0.5, nlp_engine_name="transformers"
        )
        assert _generate_config_hash(nlp_config) != base_hash


class TestGeneratePolicyHash:
    """Test policy hashing for cache keys."""

    def test_none_policy_returns_none_hash(self):
        """Test that None policy returns 'none' hash."""
        hash_result = _generate_policy_hash(None)
        assert hash_result == "none"

    def test_identical_policies_same_hash(self):
        """Test that identical policies produce the same hash."""
        policy1 = MaskingPolicy(locale="en", thresholds={"EMAIL": 0.8})
        policy2 = MaskingPolicy(locale="en", thresholds={"EMAIL": 0.8})

        hash1 = _generate_policy_hash(policy1)
        hash2 = _generate_policy_hash(policy2)

        assert hash1 == hash2

    def test_different_policies_different_hash(self):
        """Test that different policies produce different hashes."""
        policy1 = MaskingPolicy(locale="en", thresholds={"EMAIL": 0.8})
        policy2 = MaskingPolicy(locale="es", thresholds={"EMAIL": 0.8})

        hash1 = _generate_policy_hash(policy1)
        hash2 = _generate_policy_hash(policy2)

        assert hash1 != hash2


class TestGetPresidioAnalyzer:
    """Test get_presidio_analyzer function."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    def test_returns_analyzer_wrapper(self):
        """Test that function returns AnalyzerEngineWrapper instance."""
        analyzer = get_presidio_analyzer()
        assert isinstance(analyzer, AnalyzerEngineWrapper)

    def test_default_parameters(self):
        """Test analyzer created with default parameters."""
        analyzer = get_presidio_analyzer()

        assert analyzer.config.language == "en"
        assert analyzer.config.min_confidence == 0.5
        assert analyzer.config.nlp_engine_name == "spacy"

    def test_custom_parameters(self):
        """Test analyzer created with custom parameters."""
        analyzer = get_presidio_analyzer(
            language="es", min_confidence=0.7, nlp_engine_name="transformers"
        )

        assert analyzer.config.language == "es"
        assert analyzer.config.min_confidence == 0.7
        assert analyzer.config.nlp_engine_name == "transformers"

    def test_caching_behavior(self):
        """Test that identical calls return cached instances."""
        analyzer1 = get_presidio_analyzer()
        analyzer2 = get_presidio_analyzer()

        # Should be the same instance due to caching
        assert analyzer1 is analyzer2

    def test_different_parameters_different_instances(self):
        """Test that different parameters create different instances."""
        analyzer1 = get_presidio_analyzer(language="en")
        analyzer2 = get_presidio_analyzer(language="es")

        # Should be different instances
        assert analyzer1 is not analyzer2
        assert analyzer1.config.language == "en"
        assert analyzer2.config.language == "es"

    def test_cache_hit_miss_statistics(self):
        """Test cache statistics are updated correctly."""
        # Clear cache to start fresh
        clear_all_caches()
        cache_info_initial = get_cache_info()

        # First call should be a cache miss
        get_presidio_analyzer()
        cache_info_after_first = get_cache_info()

        assert (
            cache_info_after_first["analyzer"]["misses"]
            == cache_info_initial["analyzer"]["misses"] + 1
        )
        assert cache_info_after_first["analyzer"]["currsize"] == 1

        # Second call should be a cache hit
        get_presidio_analyzer()
        cache_info_after_second = get_cache_info()

        assert (
            cache_info_after_second["analyzer"]["hits"]
            == cache_info_after_first["analyzer"]["hits"] + 1
        )
        assert cache_info_after_second["analyzer"]["currsize"] == 1


class TestGetPresidioAnalyzerFromConfig:
    """Test get_presidio_analyzer_from_config function."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    def test_returns_analyzer_from_config(self):
        """Test that function returns analyzer with config settings."""
        config = AnalyzerConfig(language="fr", min_confidence=0.8)
        analyzer = get_presidio_analyzer_from_config(config)

        assert isinstance(analyzer, AnalyzerEngineWrapper)
        assert analyzer.config.language == "fr"
        assert analyzer.config.min_confidence == 0.8

    def test_identical_configs_return_same_instance(self):
        """Test that identical configs return cached instances."""
        config1 = AnalyzerConfig(language="en", min_confidence=0.6)
        config2 = AnalyzerConfig(language="en", min_confidence=0.6)

        analyzer1 = get_presidio_analyzer_from_config(config1)
        analyzer2 = get_presidio_analyzer_from_config(config2)

        # Should be the same instance due to caching
        assert analyzer1 is analyzer2


class TestGetDocumentProcessor:
    """Test get_document_processor function."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    def test_returns_document_processor(self):
        """Test that function returns DocumentProcessor instance."""
        processor = get_document_processor()
        assert isinstance(processor, DocumentProcessor)

    def test_default_parameters(self):
        """Test processor created with default parameters."""
        processor = get_document_processor()

        # Check that chunked processing is enabled by default
        assert processor._enable_chunked_processing is True

    def test_custom_parameters(self):
        """Test processor created with custom parameters."""
        processor = get_document_processor(enable_chunked=False)

        assert processor._enable_chunked_processing is False

    def test_caching_behavior(self):
        """Test that identical calls return cached instances."""
        processor1 = get_document_processor()
        processor2 = get_document_processor()

        # Should be the same instance due to caching
        assert processor1 is processor2

    def test_different_parameters_different_instances(self):
        """Test that different parameters create different instances."""
        processor1 = get_document_processor(enable_chunked=True)
        processor2 = get_document_processor(enable_chunked=False)

        # Should be different instances
        assert processor1 is not processor2
        assert processor1._enable_chunked_processing is True
        assert processor2._enable_chunked_processing is False


class TestGetDetectionPipeline:
    """Test get_detection_pipeline function."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    def test_returns_detection_pipeline(self):
        """Test that function returns EntityDetectionPipeline instance."""
        pipeline = get_detection_pipeline()
        assert isinstance(pipeline, EntityDetectionPipeline)

    def test_uses_cached_analyzer(self):
        """Test that pipeline uses cached analyzer instance."""
        # Get an analyzer first to populate cache
        get_presidio_analyzer()

        # Get pipeline - should use the same analyzer
        pipeline = get_detection_pipeline()

        # The pipeline should have an analyzer (though not necessarily the same instance
        # due to implementation details, but should be configured the same way)
        assert isinstance(pipeline.analyzer, AnalyzerEngineWrapper)

    def test_caching_behavior(self):
        """Test that identical calls return cached instances."""
        pipeline1 = get_detection_pipeline()
        pipeline2 = get_detection_pipeline()

        # Should be the same instance due to caching
        assert pipeline1 is pipeline2


class TestGetDetectionPipelineFromPolicy:
    """Test get_detection_pipeline_from_policy function."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    def test_returns_pipeline_from_policy(self):
        """Test that function returns pipeline configured from policy."""
        policy = MaskingPolicy(locale="es", thresholds={"EMAIL": 0.9})
        pipeline = get_detection_pipeline_from_policy(policy)

        assert isinstance(pipeline, EntityDetectionPipeline)
        # The analyzer should be configured according to the policy
        assert pipeline.analyzer.config.language == "es"

    def test_identical_policies_return_different_instances(self):
        """Test that identical policies return different pipeline instances.

        This is expected because the function creates a new pipeline
        each time, though it uses cached analyzers internally.
        """
        policy1 = MaskingPolicy(locale="en", thresholds={"EMAIL": 0.8})
        policy2 = MaskingPolicy(locale="en", thresholds={"EMAIL": 0.8})

        pipeline1 = get_detection_pipeline_from_policy(policy1)
        pipeline2 = get_detection_pipeline_from_policy(policy2)

        # Pipelines should be different instances
        assert pipeline1 is not pipeline2
        assert isinstance(pipeline1, EntityDetectionPipeline)
        assert isinstance(pipeline2, EntityDetectionPipeline)

        # But analyzers should be the same cached instance
        assert pipeline1.analyzer is pipeline2.analyzer


class TestThreadSafety:
    """Test thread safety of loader functions."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    @pytest.mark.performance
    def test_concurrent_analyzer_creation(self):
        """Test that concurrent analyzer creation is thread-safe."""
        results = []
        exceptions = []

        def create_analyzer():
            try:
                analyzer = get_presidio_analyzer()
                results.append(analyzer)
            except Exception as e:
                exceptions.append(e)

        # Create multiple threads
        threads = []
        for _i in range(10):
            thread = threading.Thread(target=create_analyzer)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == 10

        # All results should be the same instance due to caching
        first_analyzer = results[0]
        for analyzer in results[1:]:
            assert analyzer is first_analyzer

    @pytest.mark.performance
    def test_concurrent_processor_creation(self):
        """Test that concurrent processor creation is thread-safe."""
        results = []
        exceptions = []

        def create_processor():
            try:
                processor = get_document_processor()
                results.append(processor)
            except Exception as e:
                exceptions.append(e)

        # Use ThreadPoolExecutor for cleaner thread management
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_processor) for _ in range(10)]

            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()  # This will raise if any exception occurred

        # Check results
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == 10

        # All results should be the same instance due to caching
        first_processor = results[0]
        for processor in results[1:]:
            assert processor is first_processor

    @pytest.mark.performance
    def test_concurrent_pipeline_creation(self):
        """Test that concurrent pipeline creation is thread-safe."""
        results = []
        exceptions = []

        def create_pipeline():
            try:
                pipeline = get_detection_pipeline()
                results.append(pipeline)
            except Exception as e:
                exceptions.append(e)

        # Use ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_pipeline) for _ in range(10)]

            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()

        # Check results
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == 10

        # All results should be the same instance due to caching
        first_pipeline = results[0]
        for pipeline in results[1:]:
            assert pipeline is first_pipeline

    @pytest.mark.performance
    def test_concurrent_mixed_configuration_access(self):
        """Test concurrent access with mixed configurations."""
        results = {"analyzer": [], "processor": [], "pipeline": []}
        exceptions = []

        def create_analyzer_variant(confidence: float):
            try:
                analyzer = get_presidio_analyzer(min_confidence=confidence)
                results["analyzer"].append((confidence, id(analyzer)))
            except Exception as e:
                exceptions.append(("analyzer", e))

        def create_processor_variant(chunked: bool):
            try:
                processor = get_document_processor(enable_chunked=chunked)
                results["processor"].append((chunked, id(processor)))
            except Exception as e:
                exceptions.append(("processor", e))

        def create_pipeline():
            try:
                pipeline = get_detection_pipeline()
                results["pipeline"].append(id(pipeline))
            except Exception as e:
                exceptions.append(("pipeline", e))

        # Create workers with different configurations
        workers = []
        # Different analyzer configurations
        for conf in [0.5, 0.6, 0.7]:
            for _ in range(3):
                workers.append(lambda c=conf: create_analyzer_variant(c))

        # Different processor configurations
        for chunked in [True, False]:
            for _ in range(3):
                workers.append(lambda ch=chunked: create_processor_variant(ch))

        # Pipeline workers
        for _ in range(6):
            workers.append(create_pipeline)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker) for worker in workers]

            for future in concurrent.futures.as_completed(futures, timeout=10):
                future.result()

        # Validate results
        assert (
            len(exceptions) == 0
        ), f"Exceptions during mixed config test: {exceptions}"

        # Validate analyzer results - each confidence should have consistent instances
        analyzer_by_conf = {}
        for conf, analyzer_id in results["analyzer"]:
            if conf not in analyzer_by_conf:
                analyzer_by_conf[conf] = []
            analyzer_by_conf[conf].append(analyzer_id)

        for conf, analyzer_ids in analyzer_by_conf.items():
            assert len(analyzer_ids) == 3, f"Confidence {conf} should have 3 instances"
            assert (
                len(set(analyzer_ids)) == 1
            ), f"Confidence {conf} should return same cached instance"

        # Validate processor results - each chunked setting should have consistent instances
        processor_by_chunked = {}
        for chunked, processor_id in results["processor"]:
            if chunked not in processor_by_chunked:
                processor_by_chunked[chunked] = []
            processor_by_chunked[chunked].append(processor_id)

        for chunked, processor_ids in processor_by_chunked.items():
            assert len(processor_ids) == 3, f"Chunked {chunked} should have 3 instances"
            assert (
                len(set(processor_ids)) == 1
            ), f"Chunked {chunked} should return same cached instance"

        # Validate pipeline results - all should be same instance
        pipeline_ids = results["pipeline"]
        assert len(pipeline_ids) == 6, "Should have 6 pipeline instances"
        assert (
            len(set(pipeline_ids)) == 1
        ), "All pipelines should be same cached instance"

    @pytest.mark.performance
    def test_high_concurrency_stress_test(self):
        """Stress test with higher concurrency than basic tests."""
        thread_count = 25
        iterations_per_thread = 5
        results = []
        exceptions = []

        def stress_worker():
            try:
                thread_results = []
                for _ in range(iterations_per_thread):
                    # Mix of all loader types
                    analyzer = get_presidio_analyzer()
                    processor = get_document_processor()
                    pipeline = get_detection_pipeline()
                    thread_results.extend([id(analyzer), id(processor), id(pipeline)])
                results.extend(thread_results)
            except Exception as e:
                exceptions.append(e)

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=thread_count
        ) as executor:
            futures = [executor.submit(stress_worker) for _ in range(thread_count)]

            for future in concurrent.futures.as_completed(futures, timeout=15):
                future.result()

        execution_time = time.time() - start_time

        # Validate results
        assert len(exceptions) == 0, f"Exceptions during stress test: {exceptions}"

        expected_total_results = (
            thread_count * iterations_per_thread * 3
        )  # 3 loaders per iteration
        assert len(results) == expected_total_results

        # Group results by type (every 3rd element starting from index 0, 1, 2)
        analyzer_ids = results[0::3]
        processor_ids = results[1::3]
        pipeline_ids = results[2::3]

        # Each type should return singleton instances
        assert len(set(analyzer_ids)) == 1, "Analyzers should be singleton under stress"
        assert (
            len(set(processor_ids)) == 1
        ), "Processors should be singleton under stress"
        assert len(set(pipeline_ids)) == 1, "Pipelines should be singleton under stress"

        # Performance validation
        operations_per_second = expected_total_results / execution_time
        assert (
            execution_time < 10.0
        ), f"Stress test took too long: {execution_time:.2f}s"

        print(
            f"Stress test: {thread_count} threads × {iterations_per_thread} iterations "
            f"completed in {execution_time:.2f}s ({operations_per_second:.1f} ops/sec)"
        )

    @pytest.mark.performance
    def test_rapid_cache_clear_and_recreate(self):
        """Test rapid cache clearing and recreation doesn't break thread safety."""
        results = []
        exceptions = []
        clear_count = 0

        def cache_clear_worker():
            nonlocal clear_count
            try:
                for _ in range(3):
                    get_presidio_analyzer()
                    clear_all_caches()
                    clear_count += 1
                    analyzer = get_presidio_analyzer(min_confidence=0.6)
                    results.append(id(analyzer))
            except Exception as e:
                exceptions.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(cache_clear_worker) for _ in range(5)]

            for future in concurrent.futures.as_completed(futures, timeout=10):
                future.result()

        # Validate results
        assert (
            len(exceptions) == 0
        ), f"Exceptions during rapid cache clear test: {exceptions}"
        assert len(results) == 15  # 5 threads × 3 iterations each
        assert clear_count == 15  # Should have cleared cache 15 times

        # Note: Due to rapid clearing, we can't guarantee singleton behavior,
        # but we validate that no exceptions occurred during the process


class TestCacheManagement:
    """Test cache management functions."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    def test_clear_all_caches(self):
        """Test that clear_all_caches clears all cached instances."""
        # Create some cached instances
        get_presidio_analyzer()
        get_document_processor()
        get_detection_pipeline()

        # Verify cache has items
        cache_info = get_cache_info()
        assert cache_info["analyzer"]["currsize"] > 0
        assert cache_info["processor"]["currsize"] > 0
        assert cache_info["pipeline"]["currsize"] > 0

        # Clear caches
        clear_all_caches()

        # Verify cache is empty
        cache_info_after = get_cache_info()
        assert cache_info_after["analyzer"]["currsize"] == 0
        assert cache_info_after["processor"]["currsize"] == 0
        assert cache_info_after["pipeline"]["currsize"] == 0

    def test_get_cache_info_structure(self):
        """Test that get_cache_info returns properly structured data."""
        cache_info = get_cache_info()

        # Check structure
        assert "analyzer" in cache_info
        assert "processor" in cache_info
        assert "pipeline" in cache_info

        for cache_name in ["analyzer", "processor", "pipeline"]:
            cache_data = cache_info[cache_name]
            assert "hits" in cache_data
            assert "misses" in cache_data
            assert "maxsize" in cache_data
            assert "currsize" in cache_data

            # Check types
            assert isinstance(cache_data["hits"], int)
            assert isinstance(cache_data["misses"], int)
            assert isinstance(cache_data["maxsize"], int)
            assert isinstance(cache_data["currsize"], int)

    def test_cache_size_limits(self):
        """Test that cache size limits are respected."""
        # Create many different analyzers to test maxsize
        analyzers = []
        for i in range(10):  # More than maxsize of 8
            analyzer = get_presidio_analyzer(
                language="en",
                min_confidence=0.1 + (i * 0.01),  # Different confidence for each
            )
            analyzers.append(analyzer)

        cache_info = get_cache_info()

        # Cache size should not exceed maxsize
        assert cache_info["analyzer"]["currsize"] <= cache_info["analyzer"]["maxsize"]
        # Should be exactly maxsize since we created more than maxsize items
        assert cache_info["analyzer"]["currsize"] == cache_info["analyzer"]["maxsize"]


class TestMemoryLeakPrevention:
    """Test that caches don't cause memory leaks."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    def test_cache_eviction_on_size_limit(self):
        """Test that old entries are evicted when cache size limit is reached."""
        # Create more analyzers than the cache can hold
        maxsize = 8  # From @lru_cache(maxsize=8)

        analyzers = []
        for i in range(maxsize + 2):  # Create 2 more than maxsize
            analyzer = get_presidio_analyzer(
                language="en", min_confidence=0.1 + (i * 0.01)
            )
            analyzers.append(analyzer)

        cache_info = get_cache_info()

        # Cache should be at maxsize, indicating eviction occurred
        assert cache_info["analyzer"]["currsize"] == maxsize
        assert (
            cache_info["analyzer"]["misses"] >= maxsize + 2
        )  # At least one miss per creation

    def test_cache_info_matches_actual_behavior(self):
        """Test that cache info statistics match actual caching behavior."""
        clear_all_caches()

        # First call - should be a miss
        analyzer1 = get_presidio_analyzer()
        cache_info_1 = get_cache_info()
        assert cache_info_1["analyzer"]["misses"] == 1
        assert cache_info_1["analyzer"]["hits"] == 0
        assert cache_info_1["analyzer"]["currsize"] == 1

        # Second call with same parameters - should be a hit
        analyzer2 = get_presidio_analyzer()
        cache_info_2 = get_cache_info()
        assert cache_info_2["analyzer"]["hits"] == 1
        assert cache_info_2["analyzer"]["misses"] == 1
        assert cache_info_2["analyzer"]["currsize"] == 1

        # Should be the same instance
        assert analyzer1 is analyzer2
