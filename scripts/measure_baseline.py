#!/usr/bin/env python3
"""Baseline performance measurement script for CloakPivot operations.

This script measures current performance characteristics across key operations
to establish baselines for validating improvements from singleton integration.
"""

import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cloakpivot.core.analyzer import AnalyzerEngineWrapper
from cloakpivot.core.detection import EntityDetectionPipeline
from cloakpivot.core.performance import PerformanceProfiler, get_profiler
from cloakpivot.core.policies import MaskingPolicy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Baseline configuration
BASELINE_SCENARIOS = {
    "analyzer_cold_start": {
        "description": "First analyzer initialization",
        "iterations": 5,
        "target_max_ms": 2000,
        "test_func": "measure_analyzer_cold_start"
    },
    "analyzer_warm_start": {
        "description": "Subsequent analyzer initializations",
        "iterations": 10,
        "target_max_ms": 500,
        "test_func": "measure_analyzer_warm_start"
    },
    "small_text_analysis": {
        "description": "Analyze <1KB text",
        "iterations": 100,
        "target_max_ms": 50,
        "test_func": "measure_small_text_analysis"
    },
    "medium_text_analysis": {
        "description": "Analyze 1-10KB text",
        "iterations": 50,
        "target_max_ms": 200,
        "test_func": "measure_medium_text_analysis"
    },
    "pipeline_creation": {
        "description": "Create EntityDetectionPipeline",
        "iterations": 20,
        "target_max_ms": 100,
        "test_func": "measure_pipeline_creation"
    }
}

# Test data
SMALL_TEXT = """
Contact John Doe at john.doe@example.com or call (555) 123-4567.
His SSN is 123-45-6789 and credit card is 4532-1234-5678-9012.
"""

MEDIUM_TEXT = """
Dear Customer,

We are writing to inform you about an important security update regarding your account.

Your account details:
- Name: Jane Smith
- Email: jane.smith@company.org
- Phone: (555) 987-6543
- Account ID: ACC-2023-789456
- SSN: 987-65-4321

Recent transactions:
1. Purchase at Store ABC on 2024-01-15 using card **** **** **** 5678
2. Online payment to Service XYZ on 2024-01-16 for $129.99
3. ATM withdrawal on 2024-01-17 at location 123 Main St

If you have any questions, please contact our support team at:
- Email: support@company.org
- Phone: 1-800-SUPPORT (1-800-786-7678)
- Address: 456 Corporate Blvd, Business City, BC 12345

Important security reminder:
- Never share your credentials with anyone
- Your temporary PIN is 7834 (expires in 24 hours)
- Report suspicious activity immediately

This communication contains sensitive information. Please handle accordingly.

Thank you for your attention to this matter.

Sincerely,
Security Team
Company Inc.
""" * 3  # Make it roughly 1-10KB


class BaselineMeasurement:
    """Class to measure and record baseline performance metrics."""

    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        """Initialize baseline measurement.

        Args:
            profiler: Optional PerformanceProfiler instance
        """
        self.profiler = profiler or get_profiler()
        self.results: dict[str, Any] = {}

    def measure_analyzer_cold_start(self, iterations: int) -> dict[str, float]:
        """Measure analyzer cold start initialization performance."""
        logger.info(f"Measuring analyzer cold start ({iterations} iterations)")
        times = []

        for i in range(iterations):
            # Create fresh analyzer instance each time
            start_time = time.perf_counter()

            analyzer = AnalyzerEngineWrapper()
            # Force initialization by calling analyze
            analyzer.analyze_text("test")

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            times.append(duration_ms)

            logger.debug(f"Cold start iteration {i+1}: {duration_ms:.1f}ms")

        return self._calculate_stats(times)

    def measure_analyzer_warm_start(self, iterations: int) -> dict[str, float]:
        """Measure analyzer warm start performance (reusing existing instance)."""
        logger.info(f"Measuring analyzer warm start ({iterations} iterations)")

        # Create and initialize once
        analyzer = AnalyzerEngineWrapper()
        analyzer.analyze_text("warmup test")

        times = []
        for i in range(iterations):
            start_time = time.perf_counter()

            # Just measure analysis time on warm instance
            analyzer.analyze_text("test text for warm analysis")

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            times.append(duration_ms)

            logger.debug(f"Warm start iteration {i+1}: {duration_ms:.1f}ms")

        return self._calculate_stats(times)

    def measure_small_text_analysis(self, iterations: int) -> dict[str, float]:
        """Measure small text analysis performance."""
        logger.info(f"Measuring small text analysis ({iterations} iterations)")

        analyzer = AnalyzerEngineWrapper()
        # Warmup
        analyzer.analyze_text("warmup")

        times = []
        for i in range(iterations):
            start_time = time.perf_counter()

            results = analyzer.analyze_text(SMALL_TEXT)

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            times.append(duration_ms)

            if i == 0:
                logger.debug(f"Small text detected {len(results)} entities")

        return self._calculate_stats(times)

    def measure_medium_text_analysis(self, iterations: int) -> dict[str, float]:
        """Measure medium text analysis performance."""
        logger.info(f"Measuring medium text analysis ({iterations} iterations)")

        analyzer = AnalyzerEngineWrapper()
        # Warmup
        analyzer.analyze_text("warmup")

        times = []
        for i in range(iterations):
            start_time = time.perf_counter()

            results = analyzer.analyze_text(MEDIUM_TEXT)

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            times.append(duration_ms)

            if i == 0:
                logger.debug(f"Medium text detected {len(results)} entities, text length: {len(MEDIUM_TEXT)}")

        return self._calculate_stats(times)

    def measure_pipeline_creation(self, iterations: int) -> dict[str, float]:
        """Measure EntityDetectionPipeline creation performance."""
        logger.info(f"Measuring pipeline creation ({iterations} iterations)")

        # Create a basic policy for pipeline
        policy = MaskingPolicy(
            name="baseline_test",
            entity_types=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"],
            locale="en"
        )

        times = []
        for _i in range(iterations):
            start_time = time.perf_counter()

            try:
                EntityDetectionPipeline(policy)
            except Exception as e:
                # Pipeline might not be fully implemented yet
                logger.warning(f"Pipeline creation failed: {e}")
                # Fallback to analyzer creation
                AnalyzerEngineWrapper.from_policy(policy)

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            times.append(duration_ms)

        return self._calculate_stats(times)

    def _calculate_stats(self, times: list[float]) -> dict[str, float]:
        """Calculate statistics from timing measurements."""
        if not times:
            return {"mean": 0, "median": 0, "std_dev": 0, "min": 0, "max": 0}

        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
            "count": len(times)
        }

    def run_all_scenarios(self) -> dict[str, Any]:
        """Run all baseline measurement scenarios."""
        logger.info("Starting baseline performance measurement")

        start_time = time.perf_counter()
        measurements = {}

        for scenario_name, config in BASELINE_SCENARIOS.items():
            logger.info(f"Running scenario: {scenario_name}")

            try:
                # Get the measurement function
                test_func = getattr(self, config["test_func"])

                # Run the measurement
                scenario_results = test_func(config["iterations"])
                measurements[scenario_name] = {
                    "description": config["description"],
                    "target_max_ms": config["target_max_ms"],
                    "iterations": config["iterations"],
                    "results": scenario_results
                }

                logger.info(
                    f"Completed {scenario_name}: "
                    f"mean={scenario_results['mean']:.1f}ms, "
                    f"target={config['target_max_ms']}ms"
                )

            except Exception as e:
                logger.error(f"Failed to run scenario {scenario_name}: {e}")
                measurements[scenario_name] = {
                    "description": config["description"],
                    "target_max_ms": config["target_max_ms"],
                    "iterations": config["iterations"],
                    "error": str(e)
                }

        total_time = time.perf_counter() - start_time

        # Generate system info
        system_info = self._get_system_info()

        # Compile final report
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "before_singleton",
            "system_info": system_info,
            "measurements": measurements,
            "total_measurement_time_s": total_time,
            "profiler_stats": self._get_profiler_summary()
        }

        logger.info(f"Baseline measurement completed in {total_time:.1f}s")
        return report

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for baseline context."""
        import platform

        import psutil

        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "cpu_freq_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else None
        }

    def _get_profiler_summary(self) -> dict[str, Any]:
        """Get summary from the profiler."""
        try:
            stats = self.profiler.get_operation_stats()
            return {
                "total_operations": sum(s.total_calls for s in stats.values()),
                "operation_count": len(stats),
                "total_time_ms": sum(s.total_duration_ms for s in stats.values())
            }
        except Exception:
            return {"error": "Unable to get profiler stats"}


def save_baseline_report(report: dict[str, Any], output_dir: str = "benchmarks/baseline_reports") -> str:
    """Save baseline report to file."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_report_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Save report
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Baseline report saved to: {filepath}")
    return filepath


def main():
    """Main function to run baseline measurements."""
    logger.info("CloakPivot Baseline Performance Measurement")
    logger.info("=" * 50)

    # Initialize measurement
    measurement = BaselineMeasurement()

    try:
        # Run all scenarios
        report = measurement.run_all_scenarios()

        # Save report
        report_path = save_baseline_report(report)

        # Print summary
        print("\nBaseline Measurement Summary:")
        print("=" * 40)

        for scenario_name, data in report["measurements"].items():
            if "error" in data:
                print(f"❌ {scenario_name}: ERROR - {data['error']}")
            else:
                results = data["results"]
                target = data["target_max_ms"]
                mean_time = results["mean"]

                status = "✅" if mean_time <= target else "⚠️"
                print(f"{status} {scenario_name}: {mean_time:.1f}ms (target: {target}ms)")

        print(f"\nReport saved: {report_path}")
        print(f"Total measurement time: {report['total_measurement_time_s']:.1f}s")

        return 0

    except Exception as e:
        logger.error(f"Baseline measurement failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
