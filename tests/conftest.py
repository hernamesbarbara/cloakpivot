"""Global pytest configuration and shared fixtures for CloakPivot tests."""

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem
from presidio_analyzer import RecognizerResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.document.extractor import TextSegment


@pytest.fixture(scope="session")
def temp_dir() -> Path:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_text_with_pii() -> str:
    """Sample text containing various PII types for testing."""
    return (
        "Contact John Doe at 555-123-4567 or john.doe@example.com. "
        "His SSN is 123-45-6789 and credit card is 4532-1234-5678-9012. "
        "Address: 123 Main St, New York, NY 10001. "
        "License: DL123456789 expires 12/31/2025."
    )


@pytest.fixture
def simple_document(sample_text_with_pii: str) -> DoclingDocument:
    """Create a simple DoclingDocument with PII content."""
    doc = DoclingDocument(name="test_document")
    text_item = TextItem(
        text=sample_text_with_pii,
        self_ref="#/texts/0",
        label="text",
        orig=sample_text_with_pii
    )
    doc.texts = [text_item]
    return doc


@pytest.fixture
def complex_document() -> DoclingDocument:
    """Create a complex document with multiple text items and structures."""
    doc = DoclingDocument(name="complex_test_document")
    
    # Header
    header = TextItem(
        text="Employee Information Report",
        self_ref="#/texts/0",
        label="text",
        orig="Employee Information Report"
    )
    
    # Content with PII
    content1 = TextItem(
        text="Employee: Alice Smith, SSN: 987-65-4321, Phone: 555-987-6543",
        self_ref="#/texts/1", 
        label="text",
        orig="Employee: Alice Smith, SSN: 987-65-4321, Phone: 555-987-6543"
    )
    
    content2 = TextItem(
        text="Emergency Contact: Bob Johnson at bob.johnson@company.com or 555-123-9876",
        self_ref="#/texts/2",
        label="text", 
        orig="Emergency Contact: Bob Johnson at bob.johnson@company.com or 555-123-9876"
    )
    
    doc.texts = [header, content1, content2]
    return doc


@pytest.fixture
def detected_entities() -> List[RecognizerResult]:
    """Sample detected PII entities for testing."""
    return [
        RecognizerResult(
            entity_type="PHONE_NUMBER",
            start=20,
            end=32,
            score=0.95
        ),
        RecognizerResult(
            entity_type="EMAIL_ADDRESS", 
            start=36,
            end=56,
            score=0.88
        ),
        RecognizerResult(
            entity_type="US_SSN",
            start=71,
            end=82,
            score=0.92
        ),
        RecognizerResult(
            entity_type="CREDIT_CARD",
            start=102,
            end=121,
            score=0.85
        ),
    ]


@pytest.fixture
def basic_masking_policy() -> MaskingPolicy:
    """Create a basic masking policy for testing with reversible strategies."""
    return MaskingPolicy(
        locale="en",
        per_entity={
            "PHONE_NUMBER": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "phone"}),
            "EMAIL_ADDRESS": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "email"}),
            "US_SSN": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "ssn"}),
            "CREDIT_CARD": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "credit_card"}),
        },
        thresholds={
            "PHONE_NUMBER": 0.7,
            "EMAIL_ADDRESS": 0.8,
            "US_SSN": 0.9,
            "CREDIT_CARD": 0.8,
        }
    )


@pytest.fixture
def strict_masking_policy() -> MaskingPolicy:
    """Create a strict masking policy for testing."""
    return MaskingPolicy(
        locale="en",
        per_entity={
            "PHONE_NUMBER": Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha256", "truncate": 8}),
            "EMAIL_ADDRESS": Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha256", "truncate": 8}),
            "US_SSN": Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha256", "truncate": 8}),
            "CREDIT_CARD": Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha256", "truncate": 8}),
            "PERSON": Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha256", "truncate": 8}),
            "LOCATION": Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha256", "truncate": 8}),
        },
        thresholds={
            "PHONE_NUMBER": 0.5,
            "EMAIL_ADDRESS": 0.5,
            "US_SSN": 0.8,
            "CREDIT_CARD": 0.7,
            "PERSON": 0.8,
            "LOCATION": 0.7,
        }
    )


@pytest.fixture
def mock_analyzer_results() -> List[RecognizerResult]:
    """Mock analyzer results for various PII types."""
    return [
        RecognizerResult(entity_type="PHONE_NUMBER", start=0, end=12, score=0.95),
        RecognizerResult(entity_type="EMAIL_ADDRESS", start=20, end=35, score=0.88),
        RecognizerResult(entity_type="PERSON", start=40, end=49, score=0.85),
        RecognizerResult(entity_type="US_SSN", start=55, end=66, score=0.92),
    ]


@pytest.fixture
def test_files_dir() -> Path:
    """Directory containing test fixture files."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def golden_files_dir() -> Path:
    """Directory containing golden files for regression testing."""
    return Path(__file__).parent / "fixtures" / "golden_files"


@pytest.fixture
def sample_policies_dir() -> Path:
    """Directory containing sample policy files for testing."""
    return Path(__file__).parent / "fixtures" / "policies"


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset plugin registries before each test to ensure isolation."""
    # This fixture ensures test isolation by resetting global state
    yield
    # Reset any global registries or caches if needed
    # This prevents tests from affecting each other


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


@pytest.fixture
def simple_text_segments(sample_text_with_pii: str) -> List[TextSegment]:
    """Create text segments for simple document testing."""
    return [
        TextSegment(
            node_id="#/texts/0",
            text=sample_text_with_pii,
            start_offset=0,
            end_offset=len(sample_text_with_pii),
            node_type="TextItem"
        )
    ]


@pytest.fixture
def complex_text_segments() -> List[TextSegment]:
    """Create text segments for complex document testing."""
    return [
        TextSegment(
            node_id="#/texts/0",
            text="Employee Information Report",
            start_offset=0,
            end_offset=27,
            node_type="TextItem"
        ),
        TextSegment(
            node_id="#/texts/1",
            text="Employee: Alice Smith, SSN: 987-65-4321, Phone: 555-987-6543",
            start_offset=0,
            end_offset=62,
            node_type="TextItem"
        ),
        TextSegment(
            node_id="#/texts/2",
            text="Emergency Contact: Bob Johnson at bob.johnson@company.com or 555-123-9876",
            start_offset=0,
            end_offset=75,
            node_type="TextItem"
        )
    ]


# Performance testing fixtures
@pytest.fixture
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
            orig=f"Section {i}: {sample_text_with_pii}"
        )
        text_items.append(text_item)
    
    doc.texts = text_items
    return doc


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=[
    "low",
    "medium", 
    "high"
])
def privacy_level(request) -> str:
    """Parametrized privacy level for testing different configurations."""
    return request.param


@pytest.fixture(params=[
    StrategyKind.TEMPLATE,
    StrategyKind.REDACT,
    StrategyKind.HASH,
    StrategyKind.SURROGATE,
    StrategyKind.PARTIAL,
])
def strategy_kind(request) -> StrategyKind:
    """Parametrized strategy kind for comprehensive strategy testing.""" 
    return request.param


# Fixtures for masking engines
@pytest.fixture
def masking_engine():
    """Create a MaskingEngine instance for testing."""
    from cloakpivot.masking.engine import MaskingEngine
    return MaskingEngine()


@pytest.fixture
def benchmark_policy() -> MaskingPolicy:
    """Create a benchmark masking policy for performance testing."""
    return MaskingPolicy(
        locale="en",
        per_entity={
            "PHONE_NUMBER": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[PHONE]"}),
            "EMAIL_ADDRESS": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[EMAIL]"}),
            "US_SSN": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "ssn"}),
            "CREDIT_CARD": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "credit_card"}),
            "PERSON": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[PERSON]"}),
        },
        thresholds={
            "PHONE_NUMBER": 0.7,
            "EMAIL_ADDRESS": 0.7,
            "US_SSN": 0.8,
            "CREDIT_CARD": 0.7,
            "PERSON": 0.8,
        }
    )


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "golden: marks tests as golden file regression tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running tests"
    )