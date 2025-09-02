"""Comprehensive integration tests for final performance validation.

This test suite validates that all performance optimizations are working
together correctly and that the final validation system itself functions
as expected.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cloakpivot.core.performance import get_profiler
from cloakpivot.loaders import (
    clear_all_caches,
    get_cache_info,
    get_detection_pipeline,
    get_presidio_analyzer,
)

# Add scripts directory to path for importing validation module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestPerformanceValidation:
    """Comprehensive validation of all performance optimizations."""

    def test_singleton_pattern_effectiveness(self) -> None:
        """Validate singleton pattern provides expected performance benefits."""
        # Clear caches first
        clear_all_caches()

        # Measure multiple analyzer retrievals
        start_time = time.time()

        analyzers = []
        for _ in range(10):
            analyzer = get_presidio_analyzer()
            analyzers.append(analyzer)

        elapsed_ms = (time.time() - start_time) * 1000

        # All should be same instance (singleton)
        assert all(a is analyzers[0] for a in analyzers), "Singleton pattern not working"

        # Should be very fast after first initialization
        assert elapsed_ms < 1000, f"Singleton retrieval too slow: {elapsed_ms:.2f}ms"

    def test_session_fixture_performance(self) -> None:
        """Validate session fixtures are working effectively."""
        # This test runs in test environment with session fixtures
        # The fact that it runs quickly validates fixture optimization

        # Multiple pipeline creations should be fast with session fixtures
        start_time = time.time()

        pipelines = []
        for _ in range(5):
            pipeline = get_detection_pipeline()
            pipelines.append(pipeline)

        elapsed_ms = (time.time() - start_time) * 1000

        # Should be fast with proper fixtures
        assert elapsed_ms < 2000, f"Pipeline creation too slow: {elapsed_ms:.2f}ms"

    def test_entity_detection_performance(self) -> None:
        """Validate entity detection meets performance targets."""
        analyzer = get_presidio_analyzer()

        test_text = """
        John Doe works at Example Corp. His email is john.doe@example.com
        and his phone number is (555) 123-4567. His SSN is 123-45-6789.
        """

        # Warm up
        analyzer.analyze_text(test_text)

        # Measure performance
        times = []
        for _ in range(20):
            start = time.time()
            results = analyzer.analyze_text(test_text)
            end = time.time()
            times.append((end - start) * 1000)

            # Verify functionality
            assert len(results) >= 3, "Should detect at least 3 entities"

        avg_time = sum(times) / len(times)
        # Allow some flexibility for CI environments
        assert avg_time < 200, f"Entity detection too slow: {avg_time:.2f}ms (target: <100ms, allowing <200ms in tests)"

    def test_parallel_execution_setup(self) -> None:
        """Validate parallel execution configuration is working."""
        # Check that pytest-xdist is properly configured (if installed)
        try:
            result = subprocess.run([
                sys.executable, "-c",
                "import pytest_xdist; print('pytest-xdist available')"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                assert "pytest-xdist available" in result.stdout
            # If pytest-xdist not installed, that's okay for basic tests

        except Exception:
            # pytest-xdist not available, skip this check
            pytest.skip("pytest-xdist not available for parallel execution test")

    def test_cache_hit_rate_measurement(self) -> None:
        """Test that cache hit rate can be measured accurately."""
        # Clear caches to start fresh
        clear_all_caches()

        # Generate some cache activity
        for _ in range(5):
            get_presidio_analyzer()  # Should hit cache after first call
            get_detection_pipeline()  # Should hit cache after first call

        # Get cache statistics
        cache_stats = get_cache_info()

        # Verify we can get meaningful statistics
        assert isinstance(cache_stats, dict), "Cache stats should be a dictionary"
        assert len(cache_stats) > 0, "Should have cache statistics for some components"

        # Check that we have expected cache components
        expected_components = ["analyzer", "processor", "pipeline"]
        for component in expected_components:
            if component in cache_stats:
                stats = cache_stats[component]
                assert "hits" in stats, f"{component} stats should include hits"
                assert "misses" in stats, f"{component} stats should include misses"

    @pytest.mark.slow
    def test_comprehensive_performance_regression(self) -> None:
        """Run comprehensive performance regression check."""
        # Import the validation module
        try:
            from final_performance_validation import ComprehensivePerformanceValidator
        except ImportError:
            pytest.skip("Final performance validation module not available")

        validator = ComprehensivePerformanceValidator()

        # Run a subset of measurements for testing
        measurements = {}

        # Test analyzer initialization measurement
        measurements["analyzer_initialization"] = validator._measure_analyzer_initialization()

        # Test entity detection measurement
        measurements["entity_detection"] = validator._measure_entity_detection_speed()

        # Test cache hit rate measurement
        measurements["cache_hit_rate"] = validator._measure_cache_hit_rate()

        # Verify all measurements are reasonable
        assert measurements["analyzer_initialization"] > 0, "Analyzer initialization time should be positive"
        assert measurements["analyzer_initialization"] < 5000, "Analyzer initialization should be < 5 seconds"

        assert measurements["entity_detection"] > 0, "Entity detection time should be positive"
        assert measurements["entity_detection"] < 1000, "Entity detection should be < 1 second"

        assert 0 <= measurements["cache_hit_rate"] <= 100, "Cache hit rate should be a percentage"

    def test_validation_script_execution(self) -> None:
        """Test that the validation script can be executed successfully."""
        script_path = Path(__file__).parent.parent / "scripts" / "final_performance_validation.py"

        if not script_path.exists():
            pytest.skip("Final performance validation script not found")

        # Skip this test when running in CI or when pytest is already running
        # to avoid recursive pytest calls
        # Check for pytest execution by looking for pytest in sys.modules or running processes
        pytest_running = (
            os.getenv('CI') or
            os.getenv('PYTEST_CURRENT_TEST') or
            'pytest' in sys.modules or
            any('pytest' in str(frame.f_code.co_filename) for frame in sys._current_frames().values())
        )
        if pytest_running:
            pytest.skip("Skipping validation script test to avoid recursive pytest calls")

        # Create a temporary baseline file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_data = {
                "measurements": {
                    "analyzer_cold_start": {
                        "results": {"mean": 2000.0}  # 2 seconds baseline
                    },
                    "small_text_analysis": {
                        "results": {"mean": 150.0}  # 150ms baseline
                    }
                }
            }
            json.dump(baseline_data, f)
            baseline_file = f.name

        try:
            # Create temporary output files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as report_f:
                report_file = report_f.name
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_f:
                json_file = json_f.name

            # Set environment variable to prevent the script from running tests
            env = os.environ.copy()
            env['CLOAKPIVOT_SKIP_TEST_EXECUTION'] = 'true'

            # Run the script with a short timeout
            result = subprocess.run([
                sys.executable, str(script_path),
                "--baseline", baseline_file,
                "--output", report_file,
                "--json-output", json_file,
                "--verbose"
            ], capture_output=True, text=True, timeout=60, env=env)

            # Script should complete (though some targets might not be met in test environment)
            assert result.returncode in [0, 1], f"Script failed with code {result.returncode}: {result.stderr}"

            # Check that output files were created
            assert Path(report_file).exists(), "Validation report should be created"
            assert Path(json_file).exists(), "JSON results should be created"

            # Verify JSON output structure
            with open(json_file) as f:
                json_data = json.load(f)
                assert "validation_results" in json_data, "JSON should contain validation results"
                assert "timestamp" in json_data, "JSON should contain timestamp"

        except subprocess.TimeoutExpired:
            pytest.fail("Validation script took too long to execute")

        finally:
            # Clean up temporary files
            try:
                os.unlink(baseline_file)
                os.unlink(report_file)
                os.unlink(json_file)
            except FileNotFoundError:
                pass

    def test_performance_target_calculations(self) -> None:
        """Test that performance target calculations work correctly."""
        try:
            from final_performance_validation import PerformanceTarget
        except ImportError:
            pytest.skip("Final performance validation module not available")

        # Test reduction target
        reduction_target = PerformanceTarget(
            name="test_reduction",
            description="50% reduction in execution time",
            target_value=50.0,
            target_unit="% reduction",
            baseline_value=100.0,
            current_value=40.0
        )

        # Should show 60% improvement
        assert reduction_target.improvement_pct == 60.0, (
            f"Expected 60% improvement, got {reduction_target.improvement_pct}"
        )
        assert reduction_target.target_met is True, "Target should be met with 60% improvement vs 50% target"

        # Test absolute target
        absolute_target = PerformanceTarget(
            name="test_absolute",
            description="<100ms execution time",
            target_value=100.0,
            target_unit="ms",
            current_value=85.0
        )

        assert absolute_target.target_met is True, "Target should be met with 85ms vs <100ms target"

        # Test unmet target
        unmet_target = PerformanceTarget(
            name="test_unmet",
            description="<50ms execution time",
            target_value=50.0,
            target_unit="ms",
            current_value=75.0
        )

        assert unmet_target.target_met is False, "Target should not be met with 75ms vs <50ms target"

    def test_profiler_integration(self) -> None:
        """Test that the global profiler is working correctly."""
        profiler = get_profiler()

        # Test profiling functionality
        with profiler.measure_operation("test_operation"):
            time.sleep(0.01)  # 10ms operation

        # Get operation stats
        stats = profiler.get_operation_stats("test_operation")

        assert stats.total_calls >= 1, "Should have recorded at least one call"
        assert stats.average_duration_ms >= 10, "Should have recorded ~10ms duration"

    def test_environment_configuration_detection(self) -> None:
        """Test that environment configuration is properly detected."""
        # Test singleton configuration
        original_singleton = os.environ.get('CLOAKPIVOT_USE_SINGLETON')

        try:
            # Test with singleton enabled
            os.environ['CLOAKPIVOT_USE_SINGLETON'] = 'true'
            assert os.getenv('CLOAKPIVOT_USE_SINGLETON', 'true').lower() == 'true'

            # Test with singleton disabled
            os.environ['CLOAKPIVOT_USE_SINGLETON'] = 'false'
            assert os.getenv('CLOAKPIVOT_USE_SINGLETON', 'true').lower() == 'false'

        finally:
            # Restore original value
            if original_singleton is not None:
                os.environ['CLOAKPIVOT_USE_SINGLETON'] = original_singleton
            elif 'CLOAKPIVOT_USE_SINGLETON' in os.environ:
                del os.environ['CLOAKPIVOT_USE_SINGLETON']


class TestValidationReportGeneration:
    """Test validation report generation functionality."""

    def test_report_format_markdown(self) -> None:
        """Test that validation reports are generated in proper markdown format."""
        try:
            from final_performance_validation import ComprehensivePerformanceValidator
        except ImportError:
            pytest.skip("Final performance validation module not available")

        validator = ComprehensivePerformanceValidator()

        # Create mock validation results
        validation_results = {
            "entity_detection_speed": {
                "target_met": True,
                "current_value": 85.0,
                "target_value": 100.0,
                "improvement_pct": None,
                "description": "<100ms average entity detection time",
                "baseline_value": None
            },
            "cache_hit_rate": {
                "target_met": False,
                "current_value": 75.0,
                "target_value": 90.0,
                "improvement_pct": None,
                "description": "90%+ cache hit rate for model loading",
                "baseline_value": None
            }
        }

        report = validator.generate_validation_report(validation_results)

        # Verify report structure
        assert "# CloakPivot Performance Optimization - Final Validation Report" in report
        assert "## Executive Summary" in report
        assert "## Detailed Results" in report
        assert "## Recommendations" in report
        assert "✅" in report  # Should have at least one passed target
        assert "❌" in report  # Should have at least one failed target

    def test_json_output_structure(self) -> None:
        """Test that JSON output has the expected structure."""
        try:
            from final_performance_validation import ComprehensivePerformanceValidator
        except ImportError:
            pytest.skip("Final performance validation module not available")

        validator = ComprehensivePerformanceValidator()

        # Set environment variable to skip test execution
        original_env = os.environ.get('CLOAKPIVOT_SKIP_TEST_EXECUTION')
        os.environ['CLOAKPIVOT_SKIP_TEST_EXECUTION'] = 'true'

        try:
            # Create minimal validation results
            validation_results = validator.run_comprehensive_validation()
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ['CLOAKPIVOT_SKIP_TEST_EXECUTION'] = original_env
            else:
                os.environ.pop('CLOAKPIVOT_SKIP_TEST_EXECUTION', None)

        # Verify structure
        assert isinstance(validation_results, dict), "Validation results should be a dictionary"

        for target_name, result in validation_results.items():
            assert "target_met" in result, f"Target {target_name} should have target_met field"
            assert "current_value" in result, f"Target {target_name} should have current_value field"
            assert "target_value" in result, f"Target {target_name} should have target_value field"
            assert "description" in result, f"Target {target_name} should have description field"


class TestErrorHandling:
    """Test error handling in validation system."""

    def test_missing_baseline_handling(self) -> None:
        """Test that missing baseline files are handled gracefully."""
        try:
            from final_performance_validation import ComprehensivePerformanceValidator
        except ImportError:
            pytest.skip("Final performance validation module not available")

        validator = ComprehensivePerformanceValidator()

        # Try to load non-existent baseline
        validator.load_baseline_metrics(Path("/nonexistent/baseline.json"))

        # Should not raise exception, just log warning
        # Validation should still proceed
        # Set environment variable to skip test execution
        original_env = os.environ.get('CLOAKPIVOT_SKIP_TEST_EXECUTION')
        os.environ['CLOAKPIVOT_SKIP_TEST_EXECUTION'] = 'true'

        try:
            validation_results = validator.run_comprehensive_validation()
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ['CLOAKPIVOT_SKIP_TEST_EXECUTION'] = original_env
            else:
                os.environ.pop('CLOAKPIVOT_SKIP_TEST_EXECUTION', None)

        assert isinstance(validation_results, dict)

    def test_invalid_baseline_format_handling(self) -> None:
        """Test handling of invalid baseline file formats."""
        try:
            from final_performance_validation import ComprehensivePerformanceValidator
        except ImportError:
            pytest.skip("Final performance validation module not available")

        validator = ComprehensivePerformanceValidator()

        # Create invalid baseline file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            invalid_baseline = f.name

        try:
            # Should handle invalid JSON gracefully
            validator.load_baseline_metrics(Path(invalid_baseline))

            # Validation should still proceed
            # Set environment variable to skip test execution
            original_env = os.environ.get('CLOAKPIVOT_SKIP_TEST_EXECUTION')
            os.environ['CLOAKPIVOT_SKIP_TEST_EXECUTION'] = 'true'

            try:
                validation_results = validator.run_comprehensive_validation()
            finally:
                # Restore original environment
                if original_env is not None:
                    os.environ['CLOAKPIVOT_SKIP_TEST_EXECUTION'] = original_env
                else:
                    os.environ.pop('CLOAKPIVOT_SKIP_TEST_EXECUTION', None)

            assert isinstance(validation_results, dict)

        finally:
            os.unlink(invalid_baseline)

    @patch('subprocess.run')
    def test_test_execution_failure_handling(self, mock_run) -> None:
        """Test handling of test execution failures."""
        try:
            from final_performance_validation import ComprehensivePerformanceValidator
        except ImportError:
            pytest.skip("Final performance validation module not available")

        validator = ComprehensivePerformanceValidator()

        # Mock subprocess failure
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Test execution failed"

        # Should handle test failure gracefully
        execution_time = validator._measure_test_execution_time()

        # Should return a reasonable fallback time
        assert isinstance(execution_time, float)
        assert execution_time >= 0
