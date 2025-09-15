"""Minimal test configuration and fixtures for CloakPivot tests."""

import json
from pathlib import Path

import pytest
from docling_core.types import DoclingDocument

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.defaults import get_default_policy
from cloakpivot.engine import CloakEngine


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Get the test data directory containing real PDFs and JSON files."""
    return project_root / "data"


@pytest.fixture(scope="session")
def pdf_dir(test_data_dir: Path) -> Path:
    """Get the directory containing test PDFs."""
    return test_data_dir / "pdf"


@pytest.fixture(scope="session")
def json_dir(test_data_dir: Path) -> Path:
    """Get the directory containing test JSON files."""
    return test_data_dir / "json"


@pytest.fixture
def email_docling_path(json_dir: Path) -> Path:
    """Path to email document in Docling format."""
    return json_dir / "email.docling.json"


@pytest.fixture
def pdf_styles_docling_path(json_dir: Path) -> Path:
    """Path to PDF styles test document in Docling format."""
    return json_dir / "2025-07-03-Test-PDF-Styles.docling.json"


@pytest.fixture
def email_docling_document(email_docling_path: Path) -> DoclingDocument:
    """Load email document as DoclingDocument."""
    with open(email_docling_path) as f:
        data = json.load(f)
    # Workaround: Change version to 1.6.0 for docling-core compatibility
    # Our code handles 1.7.0 features, but docling-core v2.47.0 still validates against 1.6.0
    data["version"] = "1.6.0"
    return DoclingDocument.model_validate(data)


@pytest.fixture
def pdf_styles_docling_document(pdf_styles_docling_path: Path) -> DoclingDocument:
    """Load PDF styles document as DoclingDocument."""
    with open(pdf_styles_docling_path) as f:
        data = json.load(f)
    # Workaround: Change version to 1.6.0 for docling-core compatibility
    data["version"] = "1.6.0"
    return DoclingDocument.model_validate(data)


@pytest.fixture
def simple_text_document() -> DoclingDocument:
    """Create a simple DoclingDocument with known PII for testing."""
    return DoclingDocument(
        version="1.0.0",
        name="test_document",
        text="John Doe's email is john.doe@example.com and his phone is 555-123-4567.",
    )


@pytest.fixture
def default_masking_policy() -> MaskingPolicy:
    """Get default masking policy."""
    return get_default_policy()


@pytest.fixture
def cloak_engine() -> CloakEngine:
    """Create a CloakEngine instance with default configuration."""
    return CloakEngine()


@pytest.fixture
def sample_pii_text() -> str:
    """Sample text containing various PII types."""
    return """
    Contact Information:
    Name: Sarah Johnson
    Email: sarah.johnson@company.com
    Phone: (555) 987-6543
    SSN: 123-45-6789
    Credit Card: 4111-1111-1111-1111
    Address: 123 Main St, Springfield, IL 62701
    Medical License: MD-2024-78901
    """