# CloakPivot New Test Structure Plan

Generated: 2025-09-15
**Updated: 2025-09-15 - IMPLEMENTED**

## Overview

This document outlines a new test structure for CloakPivot based on the current refactored codebase. The plan focuses on testing actual functionality that exists post-refactoring, rather than updating obsolete tests.

**Status: ✅ IMPLEMENTED - New test structure created and initial tests written**

## Core Functionality Analysis

Based on code review, CloakPivot currently provides:

1. **PII Detection & Masking**: Using Presidio for entity recognition
2. **Document Processing**: Supporting DoclingDocument formats (v1.2.0 - v1.7.0+)
3. **Reversible Masking**: CloakMap-based unmasking system
4. **Policy System**: Flexible masking strategies and rules
5. **CLI Interface**: Simple mask/unmask commands
6. **Format Support**: JSON, Markdown, Text output formats

## Proposed Test Structure

```
tests/
├── README.md                         # Test documentation and running instructions
├── conftest.py                       # Shared fixtures and test configuration
├── test_data/                       # Test documents and expected outputs
│   ├── documents/
│   │   ├── simple_text.txt
│   │   ├── email_sample.md
│   │   ├── phone_numbers.json
│   │   └── mixed_pii.pdf
│   ├── policies/
│   │   ├── conservative.yaml
│   │   ├── template_based.yaml
│   │   └── custom_rules.yaml
│   └── expected/
│       └── [expected output files]
│
├── unit/                            # Unit tests for individual components
│   ├── core/
│   │   ├── test_analyzer.py        # AnalyzerConfig, entity detection
│   │   ├── test_policies.py        # MaskingPolicy, validation
│   │   ├── test_strategies.py      # Strategy implementations
│   │   ├── test_cloakmap.py       # CloakMap operations
│   │   ├── test_anchors.py        # Anchor system
│   │   └── test_detection.py      # Entity detection logic
│   │
│   ├── masking/
│   │   ├── test_engine.py         # MaskingEngine core
│   │   ├── test_applicator.py     # Strategy application
│   │   └── test_presidio_adapter.py # Presidio integration
│   │
│   ├── unmasking/
│   │   ├── test_engine.py         # UnmaskingEngine core
│   │   ├── test_anchor_resolver.py # Anchor resolution
│   │   └── test_cloakmap_loader.py # CloakMap loading
│   │
│   ├── document/
│   │   ├── test_processor.py      # Document processing
│   │   ├── test_extractor.py      # Text extraction
│   │   └── test_mapper.py         # Document mapping
│   │
│   └── formats/
│       └── test_serialization.py  # Format conversion
│
├── integration/                     # Integration tests
│   ├── test_mask_unmask_roundtrip.py
│   ├── test_presidio_integration.py
│   ├── test_policy_inheritance.py
│   ├── test_docling_versions.py   # Test v1.2.0 - v1.7.0 compatibility
│   ├── test_entity_conflicts.py
│   └── test_format_preservation.py
│
├── e2e/                            # End-to-end tests
│   ├── test_cli_mask_unmask.py
│   ├── test_batch_processing.py   # If batch feature is kept
│   ├── test_real_documents.py     # Real-world document scenarios
│   └── test_error_scenarios.py
│
└── performance/                    # Performance tests (optional)
    ├── test_large_documents.py
    └── test_memory_usage.py
```

## Test Categories and Coverage

### 1. Unit Tests (tests/unit/)

#### Core Components
- **test_analyzer.py**
  ```python
  # Test cases:
  - AnalyzerConfig initialization and validation
  - Language configuration
  - Confidence threshold settings
  - Custom recognizer registration
  ```

- **test_policies.py**
  ```python
  # Test cases:
  - MaskingPolicy creation and validation
  - Strategy assignment per entity type
  - Threshold configuration
  - Allow/deny list functionality
  - Context rules application
  - Privacy level settings
  ```

- **test_strategies.py**
  ```python
  # Test cases:
  - REDACT strategy
  - TEMPLATE strategy with auto-generation
  - PARTIAL masking with format preservation
  - HASH strategy with deterministic output
  - SURROGATE generation
  - Custom callback strategies
  ```

- **test_cloakmap.py**
  ```python
  # Test cases:
  - CloakMap creation and serialization
  - Entry addition and retrieval
  - Integrity validation
  - Merge operations
  - Version compatibility
  ```

#### Masking System
- **test_masking_engine.py**
  ```python
  # Test cases:
  - Document masking with various policies
  - Entity detection and replacement
  - Conflict resolution
  - Performance metrics collection
  ```

#### Unmasking System
- **test_unmasking_engine.py**
  ```python
  # Test cases:
  - Document restoration from CloakMap
  - Anchor-based position mapping
  - Format preservation
  - Error handling for corrupted maps
  ```

### 2. Integration Tests (tests/integration/)

- **test_mask_unmask_roundtrip.py**
  ```python
  # Test complete masking and unmasking cycle:
  - Text documents
  - Structured documents (tables, lists)
  - Multi-language content
  - Various entity types
  ```

- **test_presidio_integration.py**
  ```python
  # Test Presidio components:
  - AnalyzerEngine configuration
  - Custom recognizer integration
  - Language model support
  - Entity recognition accuracy
  ```

- **test_docling_versions.py**
  ```python
  # Test DoclingDocument compatibility:
  - v1.2.0 format
  - v1.7.0 segment-local charspan changes
  - Version detection and adaptation
  ```

### 3. End-to-End Tests (tests/e2e/)

- **test_cli_mask_unmask.py**
  ```python
  # Test CLI commands:
  - mask command with various options
  - unmask command
  - Policy file loading
  - Output format selection
  - Error messages and help
  ```

- **test_real_documents.py**
  ```python
  # Test with realistic documents:
  - PDF processing
  - Word documents
  - HTML content
  - Mixed format documents
  ```

## Test Data Strategy

### Sample Documents
Create minimal, focused test documents for each scenario:
```
test_data/documents/
├── minimal/           # Single entity type per file
│   ├── email_only.txt
│   ├── phone_only.txt
│   └── ssn_only.txt
├── complex/           # Multiple entity types
│   ├── business_letter.md
│   └── medical_record.json
└── edge_cases/       # Problematic scenarios
    ├── overlapping_entities.txt
    └── malformed_data.txt
```

### Policy Files
Test various policy configurations:
```yaml
# test_data/policies/strict.yaml
version: "1.0"
name: "Strict Test Policy"
default_strategy:
  kind: "redact"
  parameters:
    redact_char: "X"
thresholds:
  PHONE_NUMBER: 0.9
  EMAIL_ADDRESS: 0.9
  CREDIT_CARD: 0.95
```

## Test Implementation Guidelines

### 1. Fixture Design
```python
# conftest.py
@pytest.fixture
def simple_document():
    """Provide a minimal DoclingDocument for testing."""
    return create_test_document("John Doe's email is john@example.com")

@pytest.fixture
def masking_engine():
    """Provide configured MaskingEngine instance."""
    return MaskingEngine(policy=get_test_policy())

@pytest.fixture
def test_cloakmap():
    """Provide a pre-populated CloakMap."""
    return create_test_cloakmap()
```

### 2. Test Patterns
```python
class TestMaskingEngine:
    """Unit tests for MaskingEngine."""

    def test_mask_email_with_template(self, simple_document):
        """Test email masking with template strategy."""
        # Arrange
        policy = MaskingPolicy(
            per_entity={"EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE)}
        )
        engine = MaskingEngine(policy=policy)

        # Act
        result = engine.mask(simple_document)

        # Assert
        assert "[EMAIL]" in result.document.text
        assert "john@example.com" not in result.document.text
        assert len(result.cloakmap.entries) == 1
```

### 3. Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(
    text=st.text(min_size=10, max_size=1000),
    confidence=st.floats(min_value=0.0, max_value=1.0)
)
def test_masking_is_reversible(text, confidence):
    """Property: Masking followed by unmasking returns original."""
    doc = create_document(text)
    masked = mask_document(doc, confidence_threshold=confidence)
    unmasked = unmask_document(masked.document, masked.cloakmap)
    assert unmasked.text == doc.text
```

## Test Execution Strategy

### Phase 1: Core Unit Tests
Focus on testing individual components in isolation:
1. Policy system
2. Strategy implementations
3. CloakMap operations
4. Basic masking/unmasking

### Phase 2: Integration Tests
Test component interactions:
1. Presidio integration
2. Document processing pipeline
3. Round-trip masking/unmasking
4. Policy inheritance

### Phase 3: End-to-End Tests
Test complete workflows:
1. CLI commands
2. Real document processing
3. Error scenarios
4. Performance benchmarks

## Success Criteria

### Coverage Goals
- Unit test coverage: >80%
- Integration test coverage: >70%
- E2E test coverage: Key user workflows

### Quality Metrics
- All tests pass consistently
- Tests run in <30 seconds (excluding E2E)
- Clear test names and documentation
- No flaky tests

### Maintainability
- Tests are independent (no shared state)
- Fixtures are reusable
- Test data is version controlled
- Clear separation of test levels

## Migration Strategy

### From Old to New Tests
1. **Identify salvageable tests**: Review existing tests for reusable logic
2. **Extract test data**: Preserve valuable test documents and scenarios
3. **Rewrite incrementally**: Start with core unit tests, then integration
4. **Parallel execution**: Run old and new tests during transition
5. **Deprecate gradually**: Remove old tests as new ones prove stable

### Priority Order
1. Core masking/unmasking tests (critical path)
2. Policy and strategy tests (configuration)
3. Presidio integration tests (external dependency)
4. CLI tests (user interface)
5. Performance tests (optimization)

## Testing Tools and Dependencies

### Required Packages
```toml
[tool.poetry.group.test.dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
pytest-xdist = "^3.3.0"  # Parallel execution
pytest-timeout = "^2.1.0"
hypothesis = "^6.82.0"  # Property-based testing
faker = "^19.2.0"  # Test data generation
```

### CI/CD Configuration
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Unit Tests
        run: pytest tests/unit/ -v
      - name: Run Integration Tests
        run: pytest tests/integration/ -v
      - name: Run E2E Tests
        run: pytest tests/e2e/ -v --timeout=300
```

## Implementation Status (2025-09-15)

### ✅ Completed:
1. **Created test directories** - New structure implemented
2. **Archived old tests** - Moved to tests_old/ for reference
3. **Wrote core unit tests** - `tests/unit/test_cloak_engine.py`
4. **Leveraged existing test data** - Using real PDFs and JSON from `data/`
5. **Implemented fixtures** - New minimal `conftest.py` created
6. **Integration tests** - `test_real_documents.py`, `test_roundtrip.py`
7. **E2E tests** - `test_cli.py` for CLI testing

### 📝 What Was Created:
```
tests/
├── conftest.py                          # ✅ Minimal fixtures using real data
├── unit/
│   └── test_cloak_engine.py            # ✅ Core engine tests
├── integration/
│   ├── test_real_documents.py          # ✅ Tests with actual PDFs/JSON
│   └── test_roundtrip.py               # ✅ Comprehensive roundtrip tests
└── e2e/
    └── test_cli.py                      # ✅ CLI command tests
```

### 🔍 Issues Discovered:
1. **DoclingDocument structure** - Can't be created with simple text, needs proper segments
2. **Version mismatch** - JSON files are v1.7.0 but docling-core expects v1.6.0
3. **Missing CLI module** - `__main__.py` doesn't exist
4. **Broken imports** - Fixed during cleanup

### 🎯 Key Insights:
- Starting fresh was the right decision - avoided hours of fixing obsolete tests
- Using real test data from `data/` directory provides better coverage
- Clean structure immediately revealed actual issues in the codebase

This implementation provides a solid foundation for ongoing test development.