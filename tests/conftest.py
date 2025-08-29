"""Test configuration and shared fixtures."""

import os

import pytest
from hypothesis import HealthCheck, Phase, settings
from presidio_analyzer import AnalyzerEngine


@pytest.fixture(scope="session")
def shared_analyzer() -> AnalyzerEngine:
    """Create a shared AnalyzerEngine for the entire test session.

    This fixture prevents per-test initialization overhead by reusing
    the same analyzer across all property-based tests.
    """
    return AnalyzerEngine()


# Register Hypothesis profiles for different test environments
settings.register_profile(
    "fast",
    max_examples=10,
    deadline=None,  # No deadline for fast local development
    suppress_health_check=[
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
        HealthCheck.too_slow,
    ],
    phases=[Phase.generate],  # Skip shrinking for speed
    database=None,  # Disable example database for faster startup
)

settings.register_profile(
    "ci",
    max_examples=12,
    deadline=3000,  # 3 second deadline for CI stability
    suppress_health_check=[
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
    phases=[Phase.generate, Phase.shrink],  # Allow limited shrinking in CI
    database=None,  # Disable example database to avoid I/O overhead
)

settings.register_profile(
    "thorough",
    max_examples=100,
    deadline=30000,  # 30 second deadline for thorough testing
    suppress_health_check=[
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
    # All phases enabled for comprehensive testing
    database=None,
)

# Load profile based on environment variable
profile = os.getenv("HYPOTHESIS_PROFILE", "fast")
settings.load_profile(profile)


# Update pytest configuration to add markers for performance tests
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "property: marks property-based tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify test collection to add markers based on test names and patterns."""
    for item in items:
        # Add property marker to tests using Hypothesis
        if hasattr(item, "function") and hasattr(item.function, "_hypothesis_internal_test"):
            item.add_marker(pytest.mark.property)

        # Add slow marker to tests with "slow" in the name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)

        # Add performance marker to tests with "performance" in the name
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
