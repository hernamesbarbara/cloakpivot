"""Shared fixtures for CloakPivot tests."""

import json
import random
from collections.abc import Generator
from pathlib import Path

import pytest
from docling_core.types import DoclingDocument

from cloakpivot import CloakEngine, CloakEngineBuilder


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def pdf_dir(test_data_dir: Path) -> Path:
    """Path to PDF test files."""
    return test_data_dir / "pdf"


@pytest.fixture
def json_dir(test_data_dir: Path) -> Path:
    """Path to JSON test files."""
    return test_data_dir / "json"


@pytest.fixture
def email_docling_path(json_dir: Path) -> Path:
    """Path to email.docling.json test file."""
    return json_dir / "email.docling.json"


@pytest.fixture
def pdf_styles_docling_path(json_dir: Path) -> Path:
    """Path to PDF styles docling JSON test file."""
    return json_dir / "2025-07-03-Test-PDF-Styles.docling.json"


@pytest.fixture
def email_docling_document(email_docling_path: Path) -> DoclingDocument:
    """Load email document as DoclingDocument."""
    with email_docling_path.open() as f:
        data = json.load(f)
    return DoclingDocument.model_validate(data)


@pytest.fixture
def pdf_styles_docling_document(pdf_styles_docling_path: Path) -> DoclingDocument:
    """Load PDF styles document as DoclingDocument."""
    with pdf_styles_docling_path.open() as f:
        data = json.load(f)
    return DoclingDocument.model_validate(data)


@pytest.fixture
def basic_engine() -> CloakEngine:
    """Create a basic CloakEngine with default settings."""
    return CloakEngine()


@pytest.fixture
def conservative_engine() -> CloakEngine:
    """Create a CloakEngine with conservative settings."""
    from cloakpivot.core import MaskingPolicy, Strategy, StrategyKind

    policy = MaskingPolicy(default_strategy=Strategy(StrategyKind.REDACT))
    return CloakEngineBuilder().with_custom_policy(policy).with_confidence_threshold(0.9).build()


@pytest.fixture
def custom_engine() -> CloakEngine:
    """Create a CloakEngine with custom configuration using builder."""
    return CloakEngineBuilder().with_confidence_threshold(0.7).with_languages(["en", "es"]).build()


@pytest.fixture
def sample_text() -> str:
    """Sample text with PII for testing."""
    return """
    John Doe can be reached at john.doe@example.com or at 555-123-4567.
    He lives in San Francisco, CA and works at Example Corp.
    His social security number should not be shared.
    """


@pytest.fixture
def sample_markdown() -> str:
    """Sample markdown document with PII."""
    return """# Contact Information

## Personal Details
- **Name**: Jane Smith
- **Email**: jane.smith@company.com
- **Phone**: (415) 555-9876
- **Location**: New York, NY

## Professional Background
Jane works as a Senior Engineer at TechCorp Inc.
She can be reached during business hours at her office number: 212-555-1234.

## Emergency Contact
In case of emergency, contact Bob Smith at bob@family.com or 555-emergency.
"""


@pytest.fixture(autouse=True)
def isolate_test_state() -> Generator[None, None, None]:
    """
    Reset global/singleton/cached state between tests to guarantee isolation.

    This fixture addresses test isolation issues where tests pass individually
    but fail when run as part of the full test suite.
    """
    # Reset RNG for reproducibility
    random.seed(42)

    # Reset Faker seed if available
    try:
        from faker import Faker

        Faker.seed(42)
    except ImportError:
        pass

    # Clean up any Mock patches on DoclingDocument
    from unittest.mock import Mock

    from docling_core.types import DoclingDocument

    # Store original model_validate if it exists
    original_model_validate = None
    if hasattr(DoclingDocument, "model_validate"):
        original_model_validate = DoclingDocument.model_validate
        # Check if it's been mocked
        if isinstance(original_model_validate, Mock):
            # Remove the mock
            delattr(DoclingDocument, "model_validate")

    yield

    # Clean up after test
    # Reset DoclingDocument.model_validate if it was mocked during the test
    if hasattr(DoclingDocument, "model_validate"):
        current_model_validate = DoclingDocument.model_validate
        if isinstance(current_model_validate, Mock):
            # Remove the mock
            delattr(DoclingDocument, "model_validate")


@pytest.fixture(autouse=True)
def cleanup() -> Generator[None, None, None]:
    """Clean up any temporary files after tests."""
    yield
    # Clean up any temp files created during tests
    import shutil
    import tempfile

    temp_dir = Path(tempfile.gettempdir())
    for temp_file in temp_dir.glob("test_*"):
        if temp_file.is_file():
            temp_file.unlink(missing_ok=True)
        elif temp_file.is_dir():
            shutil.rmtree(temp_file, ignore_errors=True)
