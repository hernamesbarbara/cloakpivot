"""Global pytest configuration and shared fixtures for CloakPivot tests.

Fixture Dependency Relationships:
The fixture hierarchy is designed to minimize resource creation while maintaining test isolation:

Core Dependencies:
- worker_id → shared_temp_dir, parallel_shared_analyzer, performance_profiler
- parallel_shared_analyzer → shared_analyzer → shared_detection_pipeline
- temp_dir (independent session fixture for basic temp directory needs)

Document Fixtures:
- sample_text_with_pii → simple_document, large_document, simple_text_segments
- complex_document → complex_text_segments (independent complex document structure)

Policy Fixtures (all independent):
- basic_masking_policy, strict_masking_policy, benchmark_policy

Performance Fixtures:
- shared_analyzer → shared_detection_pipeline
- shared_document_processor (independent)
- performance_test_configs (independent configuration dictionary)

Mocking Fixtures:
- mock_presidio_analyzer, mock_presidio_anonymizer (independent mocks)
- mock_analyzer_results (independent test data)

Parametrized Fixtures:
- privacy_level, strategy_kind (depend on fast/slow mode environment variable)
- privacy_level_slow, strategy_kind_slow (always use full parameter sets)

Fast/Slow Mode Configuration:
The test suite supports multiple execution modes to balance speed and coverage:

1. Fast Mode (default): Uses minimal parametrization for quick CI runs
   - Single privacy level ("medium")
   - Single strategy kind (TEMPLATE)
   - Reduced iteration counts and batch sizes
   - Excludes slow, performance, and property-based tests
   - Usage: pytest -m "not (slow or performance or property)"
   - Hypothesis profile: Use -o hypothesis-profile=ci_fast or HYPOTHESIS_PROFILE=ci_fast

2. Slow Mode: Full parametrization for comprehensive testing
   - All privacy levels ("low", "medium", "high")
   - All strategy kinds (TEMPLATE, REDACT, HASH, SURROGATE, PARTIAL)
   - Full iteration counts and batch sizes
   - Includes all test types
   - Usage: PYTEST_FAST_MODE=0 pytest or pytest -m "slow"

Parallel Test Execution:
The test suite supports parallel execution using pytest-xdist:
- Automatic worker count detection based on CPU cores
- Worker-specific session fixtures to avoid conflicts
- Load balancing for optimal test distribution
- Usage: pytest -n auto (or set PYTEST_WORKERS environment variable)

Additional markers:
- @pytest.mark.slow: Tests that only run in comprehensive mode
- @pytest.mark.performance: Performance benchmarks and timing tests
- @pytest.mark.property: Property-based tests using Hypothesis
- Fast tests run in both modes, slow/performance/property tests need explicit inclusion
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem
from hypothesis import Verbosity, settings
from presidio_analyzer import RecognizerResult

from cloakpivot.core.analyzer import AnalyzerConfig
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.document.extractor import TextSegment

# Import parallel test support utilities
from .parallel_support import (
    ParallelTestSupport,
    get_worker_resource_manager,
    setup_parallel_test_environment,
    teardown_parallel_test_environment,
)

# Configure Hypothesis profiles for different test environments
settings.register_profile(
    "ci_fast",
    max_examples=3,
    deadline=2000,  # 2 seconds
    verbosity=Verbosity.quiet,
    suppress_health_check=[
        # Suppress health checks that can slow down CI
        # These are generally safe to ignore for property testing in CI
    ],
)

settings.register_profile(
    "default",
    max_examples=10,
    deadline=5000,  # 5 seconds
)

settings.register_profile(
    "comprehensive",
    max_examples=100,
    deadline=30000,  # 30 seconds
    verbosity=Verbosity.verbose,
)


@pytest.fixture(scope="session")
def temp_dir() -> Path:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def worker_id(request) -> str:
    """Get the worker ID for xdist parallel execution."""
    # Primary method: Check for xdist worker input from pytest-xdist
    # When running with pytest -n auto, each worker process gets a unique workerid
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]

    # Fallback method: Use parallel support utility for non-xdist environments
    # This handles cases where tests are run without pytest-xdist but may still need worker identification
    return ParallelTestSupport.get_worker_id()


@pytest.fixture(scope="session")
def shared_temp_dir(worker_id: str) -> Path:
    """Worker-specific temporary directory for parallel execution."""
    # Use worker resource manager for cleanup tracking
    resource_manager = get_worker_resource_manager()
    temp_dir = resource_manager.create_temp_dir(f"shared_temp_{worker_id}_")

    yield temp_dir

    # Cleanup is handled by the resource manager


@pytest.fixture(scope="session")
def parallel_shared_analyzer(worker_id: str):
    """Session-scoped analyzer compatible with parallel execution."""
    from presidio_analyzer import AnalyzerEngine

    # Each worker gets its own analyzer instance to avoid conflicts
    analyzer = AnalyzerEngine()
    return analyzer


@pytest.fixture(scope="session")
def sample_text_with_pii() -> str:
    """Sample text containing various PII types for testing."""
    return (
        "Contact John Doe at 555-123-4567 or john.doe@example.com. "
        "His SSN is 123-45-6789 and credit card is 4532-1234-5678-9012. "
        "Address: 123 Main St, New York, NY 10001. "
        "License: DL123456789 expires 12/31/2025."
    )


@pytest.fixture(scope="session")
def simple_document(sample_text_with_pii: str) -> DoclingDocument:
    """Create a simple DoclingDocument with PII content."""
    doc = DoclingDocument(name="test_document")
    text_item = TextItem(
        text=sample_text_with_pii,
        self_ref="#/texts/0",
        label="text",
        orig=sample_text_with_pii,
    )
    doc.texts = [text_item]
    return doc


@pytest.fixture(scope="session")
def complex_document() -> DoclingDocument:
    """Create a complex document with multiple text items and structures."""
    doc = DoclingDocument(name="complex_test_document")

    # Header
    header = TextItem(
        text="Employee Information Report",
        self_ref="#/texts/0",
        label="text",
        orig="Employee Information Report",
    )

    # Content with PII
    content1 = TextItem(
        text="Employee: Alice Smith, SSN: 987-65-4321, Phone: 555-987-6543",
        self_ref="#/texts/1",
        label="text",
        orig="Employee: Alice Smith, SSN: 987-65-4321, Phone: 555-987-6543",
    )

    content2 = TextItem(
        text="Emergency Contact: Bob Johnson at bob.johnson@company.com or 555-123-9876",
        self_ref="#/texts/2",
        label="text",
        orig="Emergency Contact: Bob Johnson at bob.johnson@company.com or 555-123-9876",
    )

    doc.texts = [header, content1, content2]
    return doc


@pytest.fixture(scope="session")
def detected_entities() -> list[RecognizerResult]:
    """Sample detected PII entities for testing."""
    return [
        RecognizerResult(entity_type="PHONE_NUMBER", start=20, end=32, score=0.95),
        RecognizerResult(entity_type="EMAIL_ADDRESS", start=36, end=56, score=0.88),
        RecognizerResult(entity_type="US_SSN", start=71, end=82, score=0.92),
        RecognizerResult(entity_type="CREDIT_CARD", start=102, end=121, score=0.85),
    ]


@pytest.fixture(scope="session")
def basic_masking_policy() -> MaskingPolicy:
    """Create a basic masking policy for testing with reversible strategies."""
    return MaskingPolicy(
        locale="en",
        per_entity={
            "PHONE_NUMBER": Strategy(
                kind=StrategyKind.SURROGATE, parameters={"format_type": "phone"}
            ),
            "EMAIL_ADDRESS": Strategy(
                kind=StrategyKind.SURROGATE, parameters={"format_type": "email"}
            ),
            "US_SSN": Strategy(
                kind=StrategyKind.SURROGATE, parameters={"format_type": "ssn"}
            ),
            "CREDIT_CARD": Strategy(
                kind=StrategyKind.SURROGATE, parameters={"format_type": "credit_card"}
            ),
        },
        thresholds={
            "PHONE_NUMBER": 0.7,
            "EMAIL_ADDRESS": 0.8,
            "US_SSN": 0.9,
            "CREDIT_CARD": 0.8,
        },
    )


@pytest.fixture(scope="session")
def strict_masking_policy() -> MaskingPolicy:
    """Create a strict masking policy for testing."""
    return MaskingPolicy(
        locale="en",
        per_entity={
            "PHONE_NUMBER": Strategy(
                kind=StrategyKind.HASH,
                parameters={"algorithm": "sha256", "truncate": 8},
            ),
            "EMAIL_ADDRESS": Strategy(
                kind=StrategyKind.HASH,
                parameters={"algorithm": "sha256", "truncate": 8},
            ),
            "US_SSN": Strategy(
                kind=StrategyKind.HASH,
                parameters={"algorithm": "sha256", "truncate": 8},
            ),
            "CREDIT_CARD": Strategy(
                kind=StrategyKind.HASH,
                parameters={"algorithm": "sha256", "truncate": 8},
            ),
            "PERSON": Strategy(
                kind=StrategyKind.HASH,
                parameters={"algorithm": "sha256", "truncate": 8},
            ),
            "LOCATION": Strategy(
                kind=StrategyKind.HASH,
                parameters={"algorithm": "sha256", "truncate": 8},
            ),
        },
        thresholds={
            "PHONE_NUMBER": 0.5,
            "EMAIL_ADDRESS": 0.5,
            "US_SSN": 0.8,
            "CREDIT_CARD": 0.7,
            "PERSON": 0.8,
            "LOCATION": 0.7,
        },
    )


@pytest.fixture(scope="session")
def mock_analyzer_results() -> list[RecognizerResult]:
    """Mock analyzer results for various PII types."""
    return [
        RecognizerResult(entity_type="PHONE_NUMBER", start=0, end=12, score=0.95),
        RecognizerResult(entity_type="EMAIL_ADDRESS", start=20, end=35, score=0.88),
        RecognizerResult(entity_type="PERSON", start=40, end=49, score=0.85),
        RecognizerResult(entity_type="US_SSN", start=55, end=66, score=0.92),
    ]


@pytest.fixture(scope="session")
def test_files_dir() -> Path:
    """Directory containing test fixture files."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def golden_files_dir() -> Path:
    """Directory containing golden files for regression testing."""
    return Path(__file__).parent / "fixtures" / "golden_files"


@pytest.fixture(scope="session")
def sample_policies_dir() -> Path:
    """Directory containing sample policy files for testing."""
    return Path(__file__).parent / "fixtures" / "policies"


@pytest.fixture
def reset_registries():
    """Reset plugin and storage registries before and after test to ensure isolation."""
    from cloakpivot.plugins.registry import reset_plugin_registry
    from cloakpivot.storage.registry import reset_storage_registry

    # Reset before test
    reset_plugin_registry()
    reset_storage_registry()

    yield

    # Reset after test
    reset_plugin_registry()
    reset_storage_registry()


@pytest.fixture
def mock_presidio_analyzer():
    """Mock Presidio AnalyzerEngine for controlled testing."""
    mock = Mock()
    mock.analyze.return_value = []
    return mock


@pytest.fixture
def mock_presidio_anonymizer():
    """Mock Presidio AnonymizerEngine for controlled testing."""
    mock = Mock()
    mock.anonymize.return_value = Mock(text="anonymized text", items=[])
    return mock


@pytest.fixture(scope="session")
def simple_text_segments(sample_text_with_pii: str) -> list[TextSegment]:
    """Create text segments for simple document testing."""
    return [
        TextSegment(
            node_id="#/texts/0",
            text=sample_text_with_pii,
            start_offset=0,
            end_offset=len(sample_text_with_pii),
            node_type="TextItem",
        )
    ]


@pytest.fixture(scope="session")
def complex_text_segments() -> list[TextSegment]:
    """Create text segments for complex document testing."""
    return [
        TextSegment(
            node_id="#/texts/0",
            text="Employee Information Report",
            start_offset=0,
            end_offset=27,
            node_type="TextItem",
        ),
        TextSegment(
            node_id="#/texts/1",
            text="Employee: Alice Smith, SSN: 987-65-4321, Phone: 555-987-6543",
            start_offset=0,
            end_offset=60,
            node_type="TextItem",
        ),
        TextSegment(
            node_id="#/texts/2",
            text="Emergency Contact: Bob Johnson at bob.johnson@company.com or 555-123-9876",
            start_offset=0,
            end_offset=73,
            node_type="TextItem",
        ),
    ]


# Performance testing fixtures
@pytest.fixture(scope="session")
def large_document(sample_text_with_pii: str) -> DoclingDocument:
    """Create a large document for performance testing."""
    doc = DoclingDocument(name="large_test_document")

    # Create 100 text items with PII content
    text_items = []
    for i in range(100):
        text_item = TextItem(
            text=f"Section {i}: {sample_text_with_pii}",
            self_ref=f"#/texts/{i}",
            label="text",
            orig=f"Section {i}: {sample_text_with_pii}",
        )
        text_items.append(text_item)

    doc.texts = text_items
    return doc


# Parametrized fixtures with fast/slow mode support
def _get_privacy_levels():
    """Get privacy levels based on test execution mode."""
    # Check if we're running in fast mode (default) or slow mode
    import os

    fast_mode = os.environ.get("PYTEST_FAST_MODE", "1") == "1"

    if fast_mode:
        return ["medium"]  # Single representative value for fast runs
    else:
        return ["low", "medium", "high"]  # Full coverage for slow runs


def _get_strategy_kinds():
    """Get strategy kinds based on test execution mode."""
    import os

    fast_mode = os.environ.get("PYTEST_FAST_MODE", "1") == "1"

    if fast_mode:
        return [StrategyKind.TEMPLATE]  # Single representative strategy for fast runs
    else:
        return [
            StrategyKind.TEMPLATE,
            StrategyKind.REDACT,
            StrategyKind.HASH,
            StrategyKind.SURROGATE,
            StrategyKind.PARTIAL,
        ]  # Full coverage for slow runs


@pytest.fixture(params=_get_privacy_levels())
def privacy_level(request) -> str:
    """Parametrized privacy level with fast/slow mode support."""
    return request.param


@pytest.fixture(params=_get_strategy_kinds())
def strategy_kind(request) -> StrategyKind:
    """Parametrized strategy kind with fast/slow mode support."""
    return request.param


# Slow-only fixtures for comprehensive testing
@pytest.fixture(params=["low", "medium", "high"])
def privacy_level_slow(request) -> str:
    """Parametrized privacy level for slow comprehensive testing."""
    return request.param


@pytest.fixture(
    params=[
        StrategyKind.TEMPLATE,
        StrategyKind.REDACT,
        StrategyKind.HASH,
        StrategyKind.SURROGATE,
        StrategyKind.PARTIAL,
    ]
)
def strategy_kind_slow(request) -> StrategyKind:
    """Parametrized strategy kind for slow comprehensive strategy testing."""
    return request.param


# Fixtures for masking engines
@pytest.fixture(scope="session")
def masking_engine():
    """Create a MaskingEngine instance for testing.

    Uses session scope for maximum performance optimization since MaskingEngine
    is stateless and safe to reuse across all tests in the session.
    This reduces repeated construction costs while maintaining test isolation
    through separate input documents and policies.
    """
    from cloakpivot.masking.engine import MaskingEngine

    return MaskingEngine()


@pytest.fixture(scope="session")
def benchmark_policy() -> MaskingPolicy:
    """Create a benchmark masking policy for performance testing."""
    return MaskingPolicy(
        locale="en",
        per_entity={
            "PHONE_NUMBER": Strategy(
                kind=StrategyKind.TEMPLATE, parameters={"template": "[PHONE]"}
            ),
            "EMAIL_ADDRESS": Strategy(
                kind=StrategyKind.TEMPLATE, parameters={"template": "[EMAIL]"}
            ),
            "US_SSN": Strategy(
                kind=StrategyKind.SURROGATE, parameters={"format_type": "ssn"}
            ),
            "CREDIT_CARD": Strategy(
                kind=StrategyKind.SURROGATE, parameters={"format_type": "credit_card"}
            ),
            "PERSON": Strategy(
                kind=StrategyKind.TEMPLATE, parameters={"template": "[PERSON]"}
            ),
        },
        thresholds={
            "PHONE_NUMBER": 0.7,
            "EMAIL_ADDRESS": 0.7,
            "US_SSN": 0.8,
            "CREDIT_CARD": 0.7,
            "PERSON": 0.8,
        },
    )


@pytest.fixture(scope="session")
def shared_analyzer(parallel_shared_analyzer):
    """Shared AnalyzerEngine instance for performance testing.

    This fixture creates a single AnalyzerEngine instance that can be reused
    across multiple tests to avoid the overhead of repeatedly initializing
    the engine and loading language models.

    In parallel execution mode, each worker gets its own analyzer instance
    to avoid conflicts while maintaining performance benefits.
    """
    # Use the parallel-compatible analyzer
    return parallel_shared_analyzer


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure pytest with custom markers and parallel execution settings."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line(
        "markers", "golden: marks tests as golden file regression tests"
    )
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running tests")
    config.addinivalue_line(
        "markers", "property: marks tests as property-based tests using Hypothesis"
    )

    # Configure parallel execution if not explicitly configured
    _configure_parallel_execution(config)


def _configure_parallel_execution(config):
    """Configure pytest-xdist parallel execution settings."""
    # Only configure if pytest-xdist is available and not already configured
    try:
        import pytest_xdist  # noqa: F401
    except ImportError:
        # pytest-xdist not available, skip parallel configuration
        return

    # Check if parallel execution is already configured via command line
    numprocesses_option = config.getoption("--numprocesses", default=None)
    dist_option = config.getoption("--dist", default=None)

    # If -n/--numprocesses not specified, set optimal worker count
    if numprocesses_option is None and not hasattr(config.option, "numprocesses"):
        worker_count = ParallelTestSupport.get_optimal_worker_count()
        if worker_count > 1:
            config.option.numprocesses = worker_count

            # Set distribution strategy if not specified
            if dist_option is None:
                config.option.dist = os.getenv("PYTEST_DIST", "loadfile")


def pytest_collection_modifyitems(config, items):
    """Optimize test distribution for better parallel performance."""

    # Sort tests by estimated execution time (longest first) for better load balancing
    def get_test_weight(item):
        """Estimate test execution time based on markers and name."""
        weight = 1  # Base weight

        # Performance tests are typically longer
        if item.get_closest_marker("performance"):
            weight += 10

        # Integration tests are usually slower than unit tests
        if item.get_closest_marker("integration"):
            weight += 5
        elif item.get_closest_marker("e2e"):
            weight += 15

        # Property-based tests can be variable
        if item.get_closest_marker("property"):
            weight += 3

        # Tests with 'slow' marker
        if item.get_closest_marker("slow"):
            weight += 8

        # Golden file tests might be slower due to file I/O
        if item.get_closest_marker("golden"):
            weight += 2

        return weight

    # Sort items by weight (heaviest first) for better load distribution
    items.sort(key=get_test_weight, reverse=True)


def pytest_sessionstart(session):
    """Set up parallel test environment at session start."""
    setup_parallel_test_environment()

    # Set up shared analyzer for masking helpers to reduce resource usage
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        # presidio_analyzer not available, skip shared analyzer setup
        pass
    else:
        try:
            from tests.utils.masking_helpers import set_test_shared_analyzer

            shared_analyzer = AnalyzerEngine()
            set_test_shared_analyzer(shared_analyzer)
        except ImportError:
            # masking_helpers module not available, skip shared analyzer setup
            pass
        except Exception as e:
            # Log unexpected errors but don't fail test setup
            import logging

            logging.warning(f"Failed to setup shared analyzer: {e}")


def pytest_sessionfinish(session, exitstatus):
    """Tear down parallel test environment at session end."""
    # Clear shared analyzer
    try:
        from tests.utils.masking_helpers import clear_test_shared_analyzer

        clear_test_shared_analyzer()
    except ImportError:
        pass

    teardown_parallel_test_environment()


# Session-scoped fixtures for performance optimization
@pytest.fixture(scope="session")
def shared_document_processor():
    """Shared DocumentProcessor instance for performance testing.

    Creates a single DocumentProcessor that can be reused across tests
    to avoid the overhead of DocPivot initialization.
    """
    from cloakpivot.document.processor import DocumentProcessor

    processor = DocumentProcessor(enable_chunked_processing=True)
    # Pre-warm the processor by checking format support
    try:
        processor.supports_format("test.json")  # Check for JSON support
    except AttributeError:
        # supports_format may not exist, that's okay
        pass

    return processor


@pytest.fixture(scope="session")
def shared_detection_pipeline(shared_analyzer):
    """Shared EntityDetectionPipeline for performance testing.

    Creates a pipeline using the shared analyzer to maximize reuse.
    Avoids singleton loader to prevent test hanging issues.
    """
    from cloakpivot.core.analyzer import AnalyzerEngineWrapper
    from cloakpivot.core.detection import EntityDetectionPipeline

    # Always use direct creation to avoid singleton loader hanging
    # The singleton loader may cause blocking when used in test environments
    try:
        # Try to create wrapper with shared analyzer
        wrapper = AnalyzerEngineWrapper()
        wrapper._engine = shared_analyzer
        wrapper._is_initialized = True
        pipeline = EntityDetectionPipeline(analyzer=wrapper)
    except AttributeError as e:
        # Wrapper doesn't support direct engine assignment, create pipeline normally
        import logging

        logging.debug(f"Direct analyzer assignment failed: {e}")
        pipeline = EntityDetectionPipeline()
    except (TypeError, ValueError) as e:
        # Pipeline creation failed with wrapper, try without analyzer parameter
        import logging

        logging.debug(f"Pipeline creation with wrapper failed: {e}")
        pipeline = EntityDetectionPipeline()

    return pipeline


@pytest.fixture(scope="function")
def performance_profiler(worker_id: str):
    """Worker-specific PerformanceProfiler for test metrics collection.
    Changed from session-scoped to function-scoped to avoid shared state issues
    in parallel test execution. Each test gets its own profiler instance.
    """
    from cloakpivot.core.performance import PerformanceProfiler

    profiler = PerformanceProfiler()

    # Configure for test environment
    profiler.enable_memory_tracking = True
    profiler.enable_detailed_logging = False  # Reduce noise in tests

    yield profiler

    # Function teardown: save performance metrics for this specific test/worker
    try:
        stats = profiler.get_operation_stats()
        if stats:
            import json
            import os
            from datetime import datetime

            # Create reports directory with proper error handling
            try:
                os.makedirs("test_reports", exist_ok=True)
            except PermissionError:
                import logging

                logging.warning("Permission denied creating test_reports directory")
                return
            except OSError as e:
                import logging

                logging.warning(f"Failed to create test_reports directory: {e}")
                return

            timestamp = (
                datetime.now().isoformat().replace(":", "-")
            )  # Safe for filenames

            # Convert stats to JSON-serializable format with error handling
            serializable_stats = {}
            try:
                for op_name, op_stats in stats.items():
                    serializable_stats[op_name] = {
                        "operation": op_stats.operation,
                        "total_calls": op_stats.total_calls,
                        "total_duration_ms": op_stats.total_duration_ms,
                        "average_duration_ms": op_stats.average_duration_ms,
                        "min_duration_ms": op_stats.min_duration_ms,
                        "max_duration_ms": op_stats.max_duration_ms,
                        "success_rate": op_stats.success_rate,
                        "failure_count": op_stats.failure_count,
                    }
            except (AttributeError, TypeError) as e:
                import logging

                logging.warning(f"Failed to serialize performance stats: {e}")
                return

            # Write metrics file with specific error handling
            filename = f"test_reports/performance_metrics_{worker_id}_{timestamp}.json"
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
            except PermissionError:
                import logging

                logging.warning(f"Permission denied writing to {filename}")
            except OSError as e:
                import logging

                logging.warning(
                    f"Failed to write performance metrics file {filename}: {e}"
                )
            except json.JSONEncodeError as e:
                import logging

                logging.warning(f"Failed to JSON encode performance metrics: {e}")
    except Exception as e:
        # Catch-all for unexpected errors during teardown - should not fail the test
        import logging

        logging.warning(f"Unexpected error in performance profiler teardown: {e}")


# @pytest.fixture(scope="session")
# def cached_analyzer_wrapper(shared_analyzer) -> AnalyzerEngineWrapper:
#     """AnalyzerEngineWrapper using the shared analyzer instance."""
#     from cloakpivot.core.analyzer import AnalyzerConfig, AnalyzerEngineWrapper
#
#     # Create wrapper that uses the shared engine
#     wrapper = AnalyzerEngineWrapper(config=AnalyzerConfig())
#     try:
#         wrapper._engine = shared_analyzer
#         wrapper._is_initialized = True
#     except AttributeError:
#         # If internal attributes don't exist, just return the wrapper
#         pass
#
#     return wrapper


@pytest.fixture(scope="session")
def performance_test_configs() -> dict[str, AnalyzerConfig]:
    """Various analyzer configurations for performance testing."""

    return {
        "minimal": AnalyzerConfig(language="en", min_confidence=0.7),
        "standard": AnalyzerConfig(language="en", min_confidence=0.5),
        "comprehensive": AnalyzerConfig(language="en", min_confidence=0.3),
    }


@pytest.fixture(scope="session")
def sample_documents() -> dict[str, str]:
    """Pre-loaded sample documents for testing."""
    return {
        "small_text": "Test User lives at test.user@example.com and his phone is 555-0123.",
        "medium_text": """
        Test User is a software engineer living in Example City.
        His contact information includes:
        - Email: test.user@example.com
        - Phone: (555) 012-3456
        - SSN: 000-12-3456
        - Credit Card: 4000-0000-0000-0002
        """,
        "large_text": """
        Test User is a software engineer living in Example City.
        His contact information includes:
        - Email: test.user@example.com
        - Phone: (555) 012-3456
        - SSN: 000-12-3456
        - Credit Card: 4000-0000-0000-0002

        Emergency contact: Test Contact at test.contact@example.org or (555) 098-7654.
        She works at 456 Test Street, Example City, EX 12345.

        Additional information:
        - Driver License: TEST123456789
        - Passport: TEST87654321
        - Bank Account: 0000111122223333
        - Medical Record: MR000000001

        This document contains synthetic PII data for testing purposes only.
        All information is fake and used for validation of masking operations.
        """
        * 3,  # Simulate larger document
    }
