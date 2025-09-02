"""Performance baseline validation tests.

These tests validate that baseline measurement infrastructure works correctly
and that baseline measurements produce consistent, meaningful results.
"""

import json
import statistics
import time
from pathlib import Path

import pytest

from benchmarks.baseline_config import (
    BaselineConfig,
    ScenarioConfig,
    get_default_config,
    get_quick_config,
    validate_against_prd_targets,
)
from cloakpivot.core.analyzer import AnalyzerEngineWrapper
from cloakpivot.core.performance import (
    PerformanceProfiler,
    get_profiler,
    profile_method,
)


class TestBaselineConfiguration:
    """Test baseline configuration functionality."""

    def test_scenario_config_creation(self):
        """Test creation of scenario configuration."""
        config = ScenarioConfig(
            description="Test scenario",
            iterations=10,
            target_max_ms=100.0,
            test_func="test_function",
        )

        assert config.description == "Test scenario"
        assert config.iterations == 10
        assert config.target_max_ms == 100.0
        assert config.test_func == "test_function"
        assert config.enabled is True

    def test_scenario_config_validation(self):
        """Test scenario configuration validation."""
        # Test invalid iterations
        with pytest.raises(ValueError, match="iterations must be positive"):
            ScenarioConfig(
                description="Test", iterations=0, target_max_ms=100.0, test_func="test"
            )

        # Test invalid target time
        with pytest.raises(ValueError, match="target_max_ms must be positive"):
            ScenarioConfig(
                description="Test", iterations=5, target_max_ms=0.0, test_func="test"
            )

    def test_default_config_creation(self):
        """Test creation of default baseline configuration."""
        config = get_default_config()

        assert isinstance(config, BaselineConfig)
        assert len(config.scenarios) > 0
        assert "analyzer_cold_start" in config.scenarios
        assert "analyzer_warm_start" in config.scenarios
        assert "small_text_analysis" in config.scenarios

        # Validate all scenarios
        errors = config.validate()
        assert len(errors) == 0, f"Configuration validation errors: {errors}"

    def test_quick_config_creation(self):
        """Test creation of quick baseline configuration."""
        config = get_quick_config()
        default_config = get_default_config()

        # Quick config should have reduced iterations
        for name, scenario in config.scenarios.items():
            default_scenario = default_config.scenarios[name]
            assert scenario.iterations <= default_scenario.iterations

    def test_enabled_scenarios_filtering(self):
        """Test filtering of enabled scenarios."""
        config = get_default_config()

        # Disable a scenario
        config.scenarios["small_text_analysis"].enabled = False

        enabled = config.get_enabled_scenarios()
        assert "small_text_analysis" not in enabled
        assert "analyzer_cold_start" in enabled  # Should still be enabled


class TestPerformanceProfilerIntegration:
    """Test integration with PerformanceProfiler."""

    def test_profile_method_decorator(self):
        """Test that @profile_method decorator works correctly."""
        from cloakpivot.core.performance import get_profiler

        # Use the global profiler that the decorator uses
        profiler = get_profiler()

        @profile_method("test_operation")
        def test_function(duration_ms: float = 10):
            time.sleep(duration_ms / 1000)  # Convert to seconds
            return "result"

        # Clear any previous metrics
        profiler.reset_metrics()

        # Execute function
        result = test_function(20)
        assert result == "result"

        # Check that profiling captured the operation
        stats = profiler.get_operation_stats("test_operation")
        assert stats.total_calls == 1
        assert (
            stats.total_duration_ms >= 15
        )  # Should be at least 15ms (allowing for timing variance)

    def test_profiler_context_manager(self):
        """Test PerformanceProfiler context manager."""
        profiler = PerformanceProfiler()
        profiler.reset_metrics()

        with profiler.measure_operation("test_context") as metric:
            time.sleep(0.01)  # 10ms

        assert metric.duration_ms >= 8  # Allow for timing variance
        assert metric.operation == "test_context"

        # Check profiler stats
        stats = profiler.get_operation_stats("test_context")
        assert stats.total_calls == 1

    def test_analyzer_profiling_integration(self):
        """Test that analyzer methods are properly profiled."""
        # Create analyzer (will use profiling decorators)
        analyzer = AnalyzerEngineWrapper()

        # Get global profiler to check stats
        profiler = get_profiler()
        profiler.reset_metrics()

        # Perform analysis (should trigger profiling)
        analyzer.analyze_text("Contact john@example.com for more info")

        # Check that profiling captured analyzer operations
        all_stats = profiler.get_operation_stats()

        # Should have both initialization and analysis profiling
        profiling_captured = False
        for operation_name in all_stats.keys():
            if "analyzer" in operation_name or "entity" in operation_name:
                profiling_captured = True
                break

        assert profiling_captured, (
            f"No analyzer profiling found in operations: {list(all_stats.keys())}"
        )


class TestBaselineMeasurement:
    """Test baseline measurement functionality."""

    def test_basic_timing_measurement(self):
        """Test basic timing measurement functionality."""

        def measure_operation(iterations: int = 5) -> dict[str, float]:
            times = []
            for _ in range(iterations):
                start_time = time.perf_counter()
                time.sleep(0.001)  # 1ms operation
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)

            return {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                "min": min(times),
                "max": max(times),
                "count": len(times),
            }

        results = measure_operation(10)

        assert results["count"] == 10
        assert results["mean"] >= 0.5  # Should be at least 0.5ms
        assert results["min"] >= 0.0
        assert results["max"] >= results["mean"]
        assert results["std_dev"] >= 0.0

    def test_analyzer_cold_start_measurement(self):
        """Test analyzer cold start measurement."""

        def measure_analyzer_cold_start(iterations: int = 3) -> dict[str, float]:
            times = []

            for _i in range(iterations):
                start_time = time.perf_counter()

                analyzer = AnalyzerEngineWrapper()
                # Force initialization
                analyzer.analyze_text("test")

                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                times.append(duration_ms)

            return {
                "mean": statistics.mean(times),
                "min": min(times),
                "max": max(times),
                "count": len(times),
            }

        results = measure_analyzer_cold_start(2)  # Use small number for test speed

        assert results["count"] == 2
        assert results["mean"] > 0  # Should take some time
        assert results["min"] <= results["mean"] <= results["max"]

    def test_analyzer_warm_start_measurement(self):
        """Test analyzer warm start measurement."""
        # Pre-create and initialize analyzer
        analyzer = AnalyzerEngineWrapper()
        analyzer.analyze_text("warmup")

        def measure_warm_analysis(iterations: int = 5) -> dict[str, float]:
            times = []

            for _i in range(iterations):
                start_time = time.perf_counter()
                analyzer.analyze_text("test analysis text")
                end_time = time.perf_counter()

                duration_ms = (end_time - start_time) * 1000
                times.append(duration_ms)

            return {
                "mean": statistics.mean(times),
                "min": min(times),
                "max": max(times),
            }

        results = measure_warm_analysis(3)

        assert results["mean"] > 0
        assert results["min"] <= results["mean"] <= results["max"]


class TestBaselineReporting:
    """Test baseline reporting and validation."""

    def test_baseline_report_structure(self):
        """Test structure of baseline measurement report."""
        # Create a minimal report structure
        report = {
            "timestamp": "2025-01-01T00:00:00Z",
            "version": "test_baseline",
            "system_info": {
                "python_version": "3.11.0",
                "platform": "test_platform",
                "cpu_count": 4,
                "memory_gb": 8.0,
            },
            "measurements": {
                "test_scenario": {
                    "description": "Test scenario",
                    "target_max_ms": 100.0,
                    "iterations": 5,
                    "results": {
                        "mean": 50.0,
                        "median": 48.0,
                        "std_dev": 5.0,
                        "min": 45.0,
                        "max": 58.0,
                        "count": 5,
                    },
                }
            },
        }

        # Validate report structure
        assert "timestamp" in report
        assert "version" in report
        assert "system_info" in report
        assert "measurements" in report

        # Validate system info
        sys_info = report["system_info"]
        assert "python_version" in sys_info
        assert "platform" in sys_info
        assert "cpu_count" in sys_info
        assert "memory_gb" in sys_info

        # Validate measurements
        measurements = report["measurements"]
        assert len(measurements) == 1

        test_measurement = measurements["test_scenario"]
        assert "description" in test_measurement
        assert "target_max_ms" in test_measurement
        assert "results" in test_measurement

        results = test_measurement["results"]
        required_stats = ["mean", "median", "std_dev", "min", "max", "count"]
        for stat in required_stats:
            assert stat in results

    def test_prd_target_validation(self):
        """Test validation against PRD performance targets."""
        # Create mock measurement data
        measurements = {
            "analyzer_cold_start": {
                "results": {
                    "mean": 2000.0  # 2 second current baseline
                }
            },
            "small_text_analysis": {
                "results": {
                    "mean": 150.0  # 150ms current baseline
                }
            },
        }

        # Validate against PRD targets
        validation_results = validate_against_prd_targets(measurements)

        assert "analyzer_initialization_improvement" in validation_results
        assert "entity_detection_max_ms" in validation_results

        # With these baselines, improvements should be achievable
        assert validation_results["analyzer_initialization_improvement"] is True
        # 150ms is above 100ms target, but within 2x allowance
        assert validation_results["entity_detection_max_ms"] is True

    def test_baseline_report_serialization(self):
        """Test that baseline reports can be serialized to JSON."""
        report = {
            "timestamp": "2025-01-01T00:00:00Z",
            "measurements": {"test": {"results": {"mean": 123.45, "count": 10}}},
        }

        # Should serialize without errors
        json_str = json.dumps(report)
        assert len(json_str) > 0

        # Should deserialize back to same structure
        loaded_report = json.loads(json_str)
        assert loaded_report["timestamp"] == report["timestamp"]
        assert loaded_report["measurements"]["test"]["results"]["mean"] == 123.45


class TestBaselineScriptIntegration:
    """Test integration with baseline measurement script."""

    def test_script_importable(self):
        """Test that baseline measurement script can be imported."""
        # This tests that the script structure is correct
        import sys

        # Add scripts directory to path
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))

        try:
            # Should be able to import the measurement classes
            from measure_baseline import BASELINE_SCENARIOS, BaselineMeasurement

            assert BaselineMeasurement is not None
            assert isinstance(BASELINE_SCENARIOS, dict)
            assert len(BASELINE_SCENARIOS) > 0

        except ImportError as e:
            pytest.fail(f"Could not import baseline measurement script: {e}")
        finally:
            sys.path.remove(str(scripts_dir))

    def test_baseline_scenario_definitions(self):
        """Test that baseline scenarios are properly defined."""
        import sys

        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))

        try:
            from measure_baseline import BASELINE_SCENARIOS

            # Check required scenarios exist
            required_scenarios = [
                "analyzer_cold_start",
                "analyzer_warm_start",
                "small_text_analysis",
                "medium_text_analysis",
            ]

            for scenario in required_scenarios:
                assert scenario in BASELINE_SCENARIOS, f"Missing scenario: {scenario}"

                config = BASELINE_SCENARIOS[scenario]
                assert "description" in config
                assert "iterations" in config
                assert "target_max_ms" in config
                assert "test_func" in config

                # Validate values are reasonable
                assert config["iterations"] > 0
                assert config["target_max_ms"] > 0

        finally:
            sys.path.remove(str(scripts_dir))


@pytest.mark.performance
class TestBaselinePerformanceValidation:
    """Performance validation tests for baseline measurement."""

    def test_measurement_repeatability(self):
        """Test that measurements are reasonably repeatable."""

        def simple_measurement() -> float:
            start = time.perf_counter()
            time.sleep(0.005)  # 5ms operation
            end = time.perf_counter()
            return (end - start) * 1000

        # Take multiple measurements
        measurements = [simple_measurement() for _ in range(10)]

        mean_time = statistics.mean(measurements)
        std_dev = statistics.stdev(measurements)

        # Should be around 5ms with low variance
        assert 3.0 <= mean_time <= 8.0  # Allow for system variance
        assert std_dev <= 2.0  # Standard deviation should be reasonable

    def test_profiler_overhead(self):
        """Test that profiler overhead is minimal."""

        def test_operation():
            # Simple operation
            result = sum(range(1000))
            return result

        # Measure without profiling
        start = time.perf_counter()
        for _ in range(100):
            test_operation()
        end = time.perf_counter()
        baseline_time = end - start

        # Measure with profiling - use lightweight profiler without memory tracking
        profiler = PerformanceProfiler(
            enable_memory_tracking=False,
            enable_detailed_logging=False,
            auto_report_threshold_ms=10000.0,  # High threshold to avoid logging overhead
        )

        @profiler.timing_decorator("test_op")
        def profiled_operation():
            return test_operation()

        start = time.perf_counter()
        for _ in range(100):
            profiled_operation()
        end = time.perf_counter()
        profiled_time = end - start

        # Profiler overhead should be reasonable (less than 50% overhead)
        overhead_ratio = profiled_time / baseline_time
        assert overhead_ratio <= 1.5, (
            f"Profiler overhead too high: {overhead_ratio:.2f}x"
        )

    def test_baseline_measurement_performance(self):
        """Test that baseline measurement itself doesn't take too long."""

        # Measure time to run a small baseline scenario
        start_time = time.perf_counter()

        # Simulate a baseline measurement
        analyzer = AnalyzerEngineWrapper()
        for _ in range(3):  # Small number of iterations
            analyzer.analyze_text("Test text with email test@example.com")

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000

        # Baseline measurement should complete reasonably quickly
        assert total_time <= 10000, (
            f"Baseline measurement took too long: {total_time}ms"
        )
