"""Tests for parallel test execution functionality.

This module tests the parallel execution capabilities added to the CloakPivot
test suite, including worker isolation, session fixture compatibility, and
load balancing.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from .parallel_support import (
    ParallelTestSupport,
    WorkerResourceManager,
    get_worker_resource_manager,
    setup_parallel_test_environment,
    teardown_parallel_test_environment,
)

PYTEST_XDIST_AVAILABLE = importlib.util.find_spec("pytest_xdist") is not None


class TestParallelTestSupport:
    """Test the ParallelTestSupport utility class."""

    def test_is_main_worker_detection(self) -> None:
        """Test worker detection logic."""
        # In single-threaded mode, should be main worker
        if not PYTEST_XDIST_AVAILABLE:
            assert ParallelTestSupport.is_main_worker()
        else:
            # Result depends on actual execution mode
            result = ParallelTestSupport.is_main_worker()
            assert isinstance(result, bool)

    def test_get_worker_count_default(self) -> None:
        """Test default worker count detection."""
        # Clear environment variables
        for var in [
            "PYTEST_XDIST_WORKER_COUNT",
            "PYTEST_WORKERS",
            "PYTEST_NUM_WORKERS",
        ]:
            if var in os.environ:
                del os.environ[var]

        count = ParallelTestSupport.get_worker_count()
        assert count >= 1
        assert isinstance(count, int)

    def test_get_worker_count_from_env(self) -> None:
        """Test worker count from environment variables."""
        with patch.dict(os.environ, {"PYTEST_WORKERS": "4"}):
            count = ParallelTestSupport.get_worker_count()
            assert count == 4

        with patch.dict(os.environ, {"PYTEST_XDIST_WORKER_COUNT": "8"}):
            count = ParallelTestSupport.get_worker_count()
            assert count == 8

    def test_get_worker_id(self) -> None:
        """Test worker ID detection."""
        worker_id = ParallelTestSupport.get_worker_id()
        assert isinstance(worker_id, str)
        assert len(worker_id) > 0

        # Should be 'main' in single-threaded mode
        if not PYTEST_XDIST_AVAILABLE or ParallelTestSupport.is_main_worker():
            assert worker_id == "main"

    def test_setup_worker_environment(self) -> None:
        """Test worker environment setup."""
        # Test with explicit worker ID
        env_vars = ParallelTestSupport.setup_worker_environment("test_worker")

        assert "PYTEST_WORKER_RANDOM_SEED" in env_vars
        assert "WORKER_TEMP_DIR" in env_vars
        assert "PYTEST_CURRENT_WORKER" in env_vars

        assert env_vars["PYTEST_CURRENT_WORKER"] == "test_worker"
        assert os.environ["PYTEST_CURRENT_WORKER"] == "test_worker"

        # Check temp directory path was configured (but not created yet due to lazy creation)
        temp_dir = Path(env_vars["WORKER_TEMP_DIR"])
        assert not temp_dir.exists()  # Should not exist until first access

        # Access the temp dir to trigger lazy creation
        actual_temp_dir = ParallelTestSupport.get_worker_temp_dir()
        assert actual_temp_dir.exists()
        assert actual_temp_dir.is_dir()
        assert str(actual_temp_dir) == env_vars["WORKER_TEMP_DIR"]

        # Cleanup
        import shutil

        shutil.rmtree(actual_temp_dir, ignore_errors=True)

    def test_get_worker_temp_dir(self) -> None:
        """Test worker-specific temporary directory creation."""
        # Setup environment first
        ParallelTestSupport.setup_worker_environment("test_temp_worker")

        temp_dir = ParallelTestSupport.get_worker_temp_dir()
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert "test_temp_worker" in str(temp_dir)

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_is_parallel_execution(self) -> None:
        """Test parallel execution detection."""
        result = ParallelTestSupport.is_parallel_execution()
        assert isinstance(result, bool)

        # Mock parallel execution
        with patch("tests.parallel_support.is_xdist_worker", return_value=True):
            assert ParallelTestSupport.is_parallel_execution()

    def test_get_optimal_worker_count(self) -> None:
        """Test optimal worker count calculation."""
        count = ParallelTestSupport.get_optimal_worker_count()
        multiprocessing.cpu_count()

        assert isinstance(count, int)
        assert count >= 1
        assert count <= 8  # Should be capped at 8

        # Test with explicit configuration
        with patch.dict(os.environ, {"PYTEST_WORKERS": "3"}):
            assert ParallelTestSupport.get_optimal_worker_count() == 3

        # Test with resource limits
        with patch.dict(
            os.environ, {"PYTEST_MAX_WORKERS": "2", "PYTEST_MIN_WORKERS": "1"}
        ):
            count = ParallelTestSupport.get_optimal_worker_count()
            assert 1 <= count <= 2


class TestWorkerResourceManager:
    """Test the WorkerResourceManager class."""

    def test_initialization(self) -> None:
        """Test resource manager initialization."""
        manager = WorkerResourceManager("test_manager")
        assert manager.worker_id == "test_manager"
        assert len(manager._temp_dirs) == 0
        assert len(manager._cleanup_callbacks) == 0

    def test_create_temp_dir(self) -> None:
        """Test temporary directory creation."""
        manager = WorkerResourceManager("test_create_temp")

        # Setup worker environment for this test
        ParallelTestSupport.setup_worker_environment("test_create_temp")

        temp_dir = manager.create_temp_dir("test_prefix_")

        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert "test_prefix_" in str(temp_dir)
        assert len(manager._temp_dirs) == 1

        # Create another temp dir
        temp_dir2 = manager.create_temp_dir("another_")
        assert len(manager._temp_dirs) == 2
        assert temp_dir != temp_dir2

        # Cleanup
        manager.cleanup()
        # Also cleanup the base worker environment
        ParallelTestSupport.cleanup_worker_environment()

    def test_register_cleanup(self) -> None:
        """Test cleanup callback registration."""
        manager = WorkerResourceManager("test_cleanup")

        callback_called = []

        def test_callback() -> None:
            callback_called.append(True)

        manager.register_cleanup(test_callback)
        assert len(manager._cleanup_callbacks) == 1

        # Cleanup should call the callback
        manager.cleanup()
        assert len(callback_called) == 1

    def test_cleanup_with_exception(self) -> None:
        """Test cleanup handles exceptions gracefully."""
        manager = WorkerResourceManager("test_exception")

        def failing_callback() -> None:
            raise RuntimeError("Test exception")

        manager.register_cleanup(failing_callback)

        # Should not raise exception
        manager.cleanup()


class TestGlobalResourceManager:
    """Test the global resource manager functionality."""

    def test_get_worker_resource_manager(self) -> None:
        """Test global resource manager retrieval."""
        manager1 = get_worker_resource_manager()
        manager2 = get_worker_resource_manager()

        # Should return the same instance
        assert manager1 is manager2
        assert isinstance(manager1, WorkerResourceManager)

    def test_setup_teardown_environment(self) -> None:
        """Test environment setup and teardown."""
        # Setup should not raise exceptions
        setup_parallel_test_environment()

        # Should create resource manager
        manager = get_worker_resource_manager()
        assert isinstance(manager, WorkerResourceManager)

        # Teardown should not raise exceptions
        teardown_parallel_test_environment()


@pytest.mark.integration
class TestParallelSessionFixtures:
    """Test session fixtures work correctly in parallel execution."""

    def test_worker_id_fixture(self, worker_id) -> None:
        """Test worker ID fixture provides consistent ID."""
        assert isinstance(worker_id, str)
        assert len(worker_id) > 0

    def test_shared_temp_dir_fixture(self, shared_temp_dir) -> None:
        """Test shared temporary directory fixture."""
        assert isinstance(shared_temp_dir, Path)
        assert shared_temp_dir.exists()
        assert shared_temp_dir.is_dir()

        # Should be writable
        test_file = shared_temp_dir / "test_file.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_parallel_shared_analyzer_fixture(self, parallel_shared_analyzer) -> None:
        """Test parallel analyzer fixture creates proper analyzer."""
        from presidio_analyzer import AnalyzerEngine

        assert isinstance(parallel_shared_analyzer, AnalyzerEngine)

        # Should be functional
        result = parallel_shared_analyzer.analyze(
            text="John Doe's phone is 555-123-4567", language="en"
        )
        assert isinstance(result, list)

    def test_shared_analyzer_uses_parallel_version(self, shared_analyzer) -> None:
        """Test that shared_analyzer fixture uses parallel-compatible version."""
        from presidio_analyzer import AnalyzerEngine

        assert isinstance(shared_analyzer, AnalyzerEngine)

        # Should work the same as parallel_shared_analyzer
        result = shared_analyzer.analyze(
            text="Test user at test@example.com", language="en"
        )
        assert isinstance(result, list)


@pytest.mark.performance
class TestParallelPerformance:
    """Test performance aspects of parallel execution."""

    def test_worker_isolation(self, worker_id, shared_temp_dir) -> None:
        """Test that workers are properly isolated."""
        # Create a file in worker's temp directory
        test_file = shared_temp_dir / f"worker_{worker_id}_test.txt"
        test_content = f"Content from worker {worker_id} at {time.time()}"

        test_file.write_text(test_content)

        # File should exist and have correct content
        assert test_file.exists()
        assert test_file.read_text() == test_content

        # Content should be unique to this worker
        assert worker_id in test_file.read_text()

    def test_analyzer_instance_isolation(
        self, parallel_shared_analyzer, worker_id
    ) -> None:
        """Test that analyzer instances don't interfere between workers."""
        # Each worker should have its own analyzer
        analyzer = parallel_shared_analyzer

        # Analyzer should be functional
        results = analyzer.analyze(
            text=f"Worker {worker_id} testing John Doe at john@example.com",
            language="en",
        )

        assert isinstance(results, list)
        # Should detect some entities in the test text
        if results:  # May be empty depending on available recognizers
            for result in results:
                assert hasattr(result, "entity_type")
                assert hasattr(result, "start")
                assert hasattr(result, "end")

    @pytest.mark.slow
    def test_concurrent_analysis_performance(self, shared_analyzer) -> None:
        """Test analyzer performance under concurrent usage."""
        import concurrent.futures

        test_texts = [
            "John Doe lives at john.doe@example.com and his phone is 555-123-4567",
            "Jane Smith can be reached at jane.smith@company.org or (555) 987-6543",
            "Contact Bob Johnson at bob.johnson@email.com, SSN: 123-45-6789",
            "Alice Williams, phone: 555-111-2222, credit card: 4532-1234-5678-9012",
        ]

        def analyze_text(text: str) -> list:
            return shared_analyzer.analyze(text=text, language="en")

        # Simulate concurrent analysis
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_text, text) for text in test_texts]
            results = [future.result() for future in futures]

        execution_time = time.time() - start_time

        # All analyses should complete
        assert len(results) == len(test_texts)
        for result in results:
            assert isinstance(result, list)

        # Should complete reasonably quickly (adjust threshold as needed)
        assert execution_time < 30.0  # 30 seconds max for concurrent analysis


@pytest.mark.property
class TestParallelPropertyTests:
    """Property-based tests for parallel execution."""

    @pytest.mark.parametrize("worker_count", [1, 2, 4])
    def test_worker_count_scaling(self, worker_count) -> None:
        """Test that worker count configuration works correctly."""
        with patch.dict(os.environ, {"PYTEST_WORKERS": str(worker_count)}):
            detected_count = ParallelTestSupport.get_worker_count()
            assert detected_count == worker_count

    @pytest.mark.parametrize(
        "worker_id", ["worker_1", "worker_2", "main", "gw0", "gw1"]
    )
    def test_worker_environment_setup_consistency(self, worker_id) -> None:
        """Test worker environment setup for different worker IDs."""
        env_vars = ParallelTestSupport.setup_worker_environment(worker_id)

        # Should always set these variables
        required_vars = [
            "PYTEST_WORKER_RANDOM_SEED",
            "WORKER_TEMP_DIR",
            "PYTEST_CURRENT_WORKER",
        ]
        for var in required_vars:
            assert var in env_vars
            assert env_vars[var] is not None

        # Worker ID should be preserved
        assert env_vars["PYTEST_CURRENT_WORKER"] == worker_id

        # Temp directory should be worker-specific
        temp_dir = Path(env_vars["WORKER_TEMP_DIR"])
        assert worker_id in str(temp_dir)

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


# Integration test to verify the complete parallel testing flow
@pytest.mark.integration
def test_complete_parallel_flow(shared_analyzer, shared_temp_dir, worker_id) -> None:
    """Integration test for complete parallel testing workflow."""
    # This test verifies that all components work together

    # 1. Worker identification should work
    assert isinstance(worker_id, str)

    # 2. Worker-specific temp directory should be available
    assert shared_temp_dir.exists()

    # 3. Analyzer should be functional
    results = shared_analyzer.analyze(
        text="Integration test for worker {worker_id}: contact@example.com",
        language="en",
    )
    assert isinstance(results, list)

    # 4. Worker isolation - create unique file
    test_file = shared_temp_dir / f"integration_test_{worker_id}.json"
    import json

    test_data = {
        "worker_id": worker_id,
        "temp_dir": str(shared_temp_dir),
        "analysis_results": len(results),
        "timestamp": time.time(),
    }

    with open(test_file, "w") as f:
        json.dump(test_data, f)

    # Verify file was created correctly
    assert test_file.exists()
    with open(test_file) as f:
        loaded_data = json.load(f)

    assert loaded_data["worker_id"] == worker_id
    assert loaded_data["analysis_results"] == len(results)
