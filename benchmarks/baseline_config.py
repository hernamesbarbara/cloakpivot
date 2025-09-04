"""Configuration for baseline performance measurements.

This module defines the scenarios, targets, and test configurations used for
establishing performance baselines in CloakPivot operations.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ScenarioConfig:
    """Configuration for a single performance measurement scenario.

    Attributes:
        description: Human-readable description of what is being measured
        iterations: Number of times to run the measurement for statistical validity
        target_max_ms: Target maximum time in milliseconds (performance goal)
        test_func: Name of the test function to execute
        enabled: Whether this scenario should be included in measurements
        metadata: Additional configuration specific to this scenario
    """

    description: str
    iterations: int
    target_max_ms: float
    test_func: str
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.iterations <= 0:
            raise ValueError(f"iterations must be positive, got {self.iterations}")
        if self.target_max_ms <= 0:
            raise ValueError(
                f"target_max_ms must be positive, got {self.target_max_ms}"
            )


@dataclass
class BaselineConfig:
    """Complete configuration for baseline performance measurements.

    Attributes:
        scenarios: Dictionary of scenario configurations by name
        system_requirements: Minimum system requirements for valid measurements
        measurement_settings: Global settings for measurement execution
        report_settings: Configuration for report generation
    """

    scenarios: dict[str, ScenarioConfig] = field(default_factory=dict)
    system_requirements: dict[str, Any] = field(default_factory=dict)
    measurement_settings: dict[str, Any] = field(default_factory=dict)
    report_settings: dict[str, Any] = field(default_factory=dict)

    def get_enabled_scenarios(self) -> dict[str, ScenarioConfig]:
        """Get only the enabled scenarios."""
        return {
            name: config for name, config in self.scenarios.items() if config.enabled
        }

    def validate(self) -> list[str]:
        """Validate the configuration and return any validation errors."""
        errors = []

        if not self.scenarios:
            errors.append("No scenarios defined")

        for name, scenario in self.scenarios.items():
            try:
                scenario.__post_init__()
            except ValueError as e:
                errors.append(f"Scenario '{name}': {e}")

        return errors


# Default baseline scenarios based on the PRD requirements
DEFAULT_BASELINE_SCENARIOS = {
    "analyzer_cold_start": ScenarioConfig(
        description="First analyzer initialization (cold start)",
        iterations=5,
        target_max_ms=2000,  # 2 seconds for cold start is reasonable
        test_func="measure_analyzer_cold_start",
        metadata={
            "category": "initialization",
            "priority": "high",
            "notes": "Measures first-time analyzer creation including model loading",
        },
    ),
    "analyzer_warm_start": ScenarioConfig(
        description="Subsequent analyzer initializations (warm start)",
        iterations=10,
        target_max_ms=500,  # 0.5 seconds for warm start
        test_func="measure_analyzer_warm_start",
        metadata={
            "category": "initialization",
            "priority": "high",
            "notes": "Measures analyzer reuse performance",
        },
    ),
    "small_text_analysis": ScenarioConfig(
        description="Analyze small text (<1KB)",
        iterations=100,
        target_max_ms=50,  # 50ms for small text
        test_func="measure_small_text_analysis",
        metadata={
            "category": "analysis",
            "priority": "high",
            "text_size_bytes": 500,
            "notes": "Measures analysis performance on typical small documents",
        },
    ),
    "medium_text_analysis": ScenarioConfig(
        description="Analyze medium text (1-10KB)",
        iterations=50,
        target_max_ms=200,  # 200ms for medium text
        test_func="measure_medium_text_analysis",
        metadata={
            "category": "analysis",
            "priority": "high",
            "text_size_bytes": 5000,
            "notes": "Measures analysis performance on medium-sized documents",
        },
    ),
    "large_text_analysis": ScenarioConfig(
        description="Analyze large text (10-100KB)",
        iterations=20,
        target_max_ms=1000,  # 1 second for large text
        test_func="measure_large_text_analysis",
        enabled=False,  # Disabled by default, enable when test function is implemented
        metadata={
            "category": "analysis",
            "priority": "medium",
            "text_size_bytes": 50000,
            "notes": "Measures analysis performance on large documents",
        },
    ),
    "pipeline_creation": ScenarioConfig(
        description="Create EntityDetectionPipeline",
        iterations=20,
        target_max_ms=100,  # 100ms for pipeline creation
        test_func="measure_pipeline_creation",
        metadata={
            "category": "initialization",
            "priority": "medium",
            "notes": "Measures pipeline setup time",
        },
    ),
    "batch_processing": ScenarioConfig(
        description="Process batch of small documents",
        iterations=10,
        target_max_ms=2000,  # 2 seconds for batch processing
        test_func="measure_batch_processing",
        enabled=False,  # Disabled by default until implemented
        metadata={
            "category": "batch",
            "priority": "medium",
            "batch_size": 10,
            "notes": "Measures batch processing performance",
        },
    ),
    "memory_usage_analysis": ScenarioConfig(
        description="Memory usage during analysis",
        iterations=5,
        target_max_ms=1000,  # Time is less important for memory tests
        test_func="measure_memory_usage",
        enabled=False,  # Disabled by default until implemented
        metadata={
            "category": "memory",
            "priority": "low",
            "notes": "Measures memory consumption patterns",
        },
    ),
}

# System requirements for valid baseline measurements
DEFAULT_SYSTEM_REQUIREMENTS = {
    "min_memory_gb": 4,
    "min_cpu_cores": 2,
    "python_version_min": "3.8",
    "required_packages": ["presidio-analyzer", "spacy", "psutil"],
}

# Default measurement settings
DEFAULT_MEASUREMENT_SETTINGS = {
    "warmup_iterations": 2,
    "gc_between_measurements": True,
    "measure_memory": True,
    "detailed_logging": False,
    "fail_on_target_miss": False,
    "timeout_per_scenario_s": 300,  # 5 minutes max per scenario
    "statistical_confidence": 0.95,
}

# Default report settings
DEFAULT_REPORT_SETTINGS = {
    "include_system_info": True,
    "include_profiler_stats": True,
    "include_raw_measurements": False,  # Don't include all raw times by default
    "generate_charts": False,  # Disabled until chart generation is implemented
    "output_formats": ["json"],  # Could add "html", "csv" later
    "decimal_places": 2,
}


def get_default_config() -> BaselineConfig:
    """Get the default baseline configuration.

    Returns:
        BaselineConfig instance with default scenarios and settings
    """
    return BaselineConfig(
        scenarios=DEFAULT_BASELINE_SCENARIOS,
        system_requirements=DEFAULT_SYSTEM_REQUIREMENTS,
        measurement_settings=DEFAULT_MEASUREMENT_SETTINGS,
        report_settings=DEFAULT_REPORT_SETTINGS,
    )


def create_custom_config(
    scenarios: Optional[dict[str, ScenarioConfig]] = None,
    system_requirements: Optional[dict[str, Any]] = None,
    measurement_settings: Optional[dict[str, Any]] = None,
    report_settings: Optional[dict[str, Any]] = None,
) -> BaselineConfig:
    """Create a custom baseline configuration.

    Args:
        scenarios: Custom scenarios (uses defaults if None)
        system_requirements: Custom system requirements (uses defaults if None)
        measurement_settings: Custom measurement settings (uses defaults if None)
        report_settings: Custom report settings (uses defaults if None)

    Returns:
        BaselineConfig instance with custom settings
    """
    config = get_default_config()

    if scenarios is not None:
        config.scenarios = scenarios

    if system_requirements is not None:
        config.system_requirements.update(system_requirements)

    if measurement_settings is not None:
        config.measurement_settings.update(measurement_settings)

    if report_settings is not None:
        config.report_settings.update(report_settings)

    return config


def get_quick_config() -> BaselineConfig:
    """Get a configuration for quick baseline measurements.

    Uses fewer iterations and shorter timeouts for faster execution.

    Returns:
        BaselineConfig optimized for quick measurements
    """
    config = get_default_config()

    # Reduce iterations for quick measurements
    for scenario in config.scenarios.values():
        scenario.iterations = max(3, scenario.iterations // 3)

    # Shorter timeout
    config.measurement_settings["timeout_per_scenario_s"] = 60

    return config


def get_comprehensive_config() -> BaselineConfig:
    """Get a configuration for comprehensive baseline measurements.

    Uses more iterations and enables additional scenarios for thorough measurement.

    Returns:
        BaselineConfig optimized for comprehensive measurements
    """
    config = get_default_config()

    # Increase iterations for more statistical validity
    for scenario in config.scenarios.values():
        scenario.iterations = scenario.iterations * 2

    # Enable additional scenarios
    config.scenarios["large_text_analysis"].enabled = True
    config.scenarios["batch_processing"].enabled = True

    # Enable detailed logging and raw measurements
    config.measurement_settings["detailed_logging"] = True
    config.report_settings["include_raw_measurements"] = True

    return config


# Performance targets based on PRD requirements
PRD_PERFORMANCE_TARGETS = {
    "analyzer_initialization_improvement": 0.8,  # 80% improvement target
    "entity_detection_max_ms": 100,  # <100ms target
    "test_suite_improvement": 0.5,  # 50% improvement target
    "memory_reduction_target": 0.3,  # 30% memory reduction target
}


def validate_against_prd_targets(measurements: dict[str, Any]) -> dict[str, bool]:
    """Validate measurements against PRD performance targets.

    Args:
        measurements: Baseline measurement results

    Returns:
        Dictionary mapping target names to whether they are achievable
    """
    results = {}

    # Check analyzer initialization target (80% improvement)
    if "analyzer_cold_start" in measurements:
        current_time = measurements["analyzer_cold_start"]["results"]["mean"]
        target_time = current_time * (
            1 - PRD_PERFORMANCE_TARGETS["analyzer_initialization_improvement"]
        )
        results["analyzer_initialization_improvement"] = (
            target_time > 100
        )  # Reasonable minimum

    # Check entity detection target (<100ms)
    if "small_text_analysis" in measurements:
        current_time = measurements["small_text_analysis"]["results"]["mean"]
        results["entity_detection_max_ms"] = (
            current_time
            <= PRD_PERFORMANCE_TARGETS["entity_detection_max_ms"]
            * 2  # Allow 2x current
        )

    return results
