# CloakPivot Testing Guide

This document provides comprehensive guidance for running and contributing to the CloakPivot test suite.

## Quick Start

```bash
# Setup development environment (includes test dependencies)
make dev

# Run all tests with coverage
make test

# Run full CI/CD pipeline locally
make all
```

## Development Workflow

The project uses a centralized Makefile for all testing operations:

```bash
# Show all available commands
make help

# Quick validation before committing
make check          # Format + lint

# Run different test types
make test           # All tests with coverage
make test-unit      # Unit tests only
make test-integration # Integration tests only
make test-e2e       # End-to-end tests only
make test-fast      # Tests without coverage (faster)
make test-verbose   # Tests with verbose output

# Generate coverage reports
make coverage-html  # HTML report in htmlcov/
```

## Test Suite Architecture

### CloakEngine Test Suite

The test suite has been refactored to focus on the simplified CloakEngine API:

#### Core CloakEngine Tests
- `test_cloak_engine_simple.py`: Basic masking/unmasking functionality
- `test_cloak_engine_builder.py`: Builder pattern and configuration tests
- `test_cloak_engine_examples.py`: Specification and documentation examples
- `test_defaults.py`: Default configuration and policy presets

#### Functional Tests
- `test_masking_engine.py`: Masking functionality via CloakEngine
- `test_unmasking_engine.py`: Unmasking and round-trip tests
- `test_masking_integration.py`: End-to-end integration tests
- `test_property_masking.py`: Property-based testing with Hypothesis

#### Performance Tests
- **Markers**: `@pytest.mark.performance`, `@pytest.mark.slow`
- **Purpose**: Benchmark CloakEngine performance
- **Coverage**: Processing speed, memory usage, engine reuse benefits

### Test Infrastructure

#### Fixtures (`tests/conftest.py`)
Global fixtures for test data, mock objects, and test environment setup.

#### Test Utilities (`tests/utils/`)
- `assertions.py`: Custom assertion helpers for domain-specific validation
- `generators.py`: Test data generators using Hypothesis strategies
- `fixtures.py`: Complex fixture builders for specialized test scenarios

#### Test Data (`tests/fixtures/`)
- `documents/`: Sample documents for testing various formats
- `policies/`: Test policy configurations
- `golden_files/`: Expected outputs for regression testing

## Test Configuration

All test configuration is centralized in `pyproject.toml`:

### Pytest Configuration
```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
markers = [
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (component interaction)",
    "e2e: End-to-end tests (full workflow)",
    "slow: Slow running tests (> 5 seconds)",
]
```

### Coverage Configuration
```toml
[tool.coverage.run]
source = ["cloakpivot"]
branch = true
parallel = true

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

## Running Tests

### Using Make Commands (Recommended)

```bash
# Basic test execution
make test           # Run all tests with coverage
make test-fast      # Run without coverage (faster)

# Specific test categories
make test-unit      # Unit tests only
make test-integration # Integration tests
make test-e2e       # End-to-end tests

# Coverage reports
make coverage-html  # Generate HTML report
open htmlcov/index.html  # View report
```

### Direct pytest Usage

```bash
# Run specific test files
pytest tests/test_analyzer.py -v

# Run tests with markers
pytest -m "unit and not slow" -v

# Run with coverage
pytest --cov=cloakpivot --cov-report=html tests/

# Parallel execution
pytest -n auto tests/
```

### Test Markers

Use pytest markers to run specific test subsets:

```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m "not slow"    # Exclude slow tests
pytest -m e2e           # End-to-end tests
```

## Test Development Guidelines

### Writing Unit Tests

```python
import pytest
from cloakpivot.core.policies import MaskingPolicy

class TestMaskingPolicy:
    def test_policy_creation(self):
        """Test basic policy creation."""
        policy = MaskingPolicy(locale="en", privacy_level="MEDIUM")
        assert policy.locale == "en"

    @pytest.mark.parametrize("privacy_level", ["LOW", "MEDIUM", "HIGH"])
    def test_privacy_levels(self, privacy_level):
        """Test all privacy levels."""
        policy = MaskingPolicy(privacy_level=privacy_level)
        assert policy.privacy_level.value == privacy_level
```

### Writing Integration Tests

```python
@pytest.mark.integration
def test_document_processing_with_cloakengine(simple_document):
    """Test complete document processing workflow."""
    from cloakpivot.engine import CloakEngine

    engine = CloakEngine()
    result = engine.mask_document(simple_document)

    # Verify masking
    assert result.entities_found > 0
    assert result.entities_masked > 0
    assert len(result.document.texts) == len(simple_document.texts)

    # Test round-trip
    unmasked = engine.unmask_document(result.document, result.cloakmap)
    assert unmasked.texts[0].text == simple_document.texts[0].text
```

### Writing Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=10), st.floats(min_value=0.1, max_value=0.9))
def test_threshold_property(text, threshold):
    """Property: threshold values should control detection sensitivity."""
    # Test implementation
    pass
```

### Writing Performance Tests

```python
@pytest.mark.slow
def test_cloakengine_performance():
    """Benchmark CloakEngine processing performance."""
    from cloakpivot.engine import CloakEngine
    import time

    engine = CloakEngine()
    document = create_test_document()

    start_time = time.perf_counter()
    result = engine.mask_document(document)
    processing_time = time.perf_counter() - start_time

    assert processing_time < 5.0  # Should complete within 5 seconds
    assert result.entities_masked > 0
```

## Code Quality

The project enforces code quality through automated tools:

### Before Committing

```bash
# Quick validation
make check  # Runs format + lint

# Or run individually
make format  # Black formatting
make lint    # Ruff linting
make type    # MyPy type checking
```

### CI/CD Pipeline

```bash
# Run complete pipeline locally
make all  # format + lint + type + test

# This is what CI runs
make ci-test
```

## Coverage Requirements

The project maintains the following coverage standards:

- **Minimum Coverage**: 60% (configured in pyproject.toml)
- **Target Coverage**: 80%+ for core modules
- **Branch Coverage**: Enabled for all tests

```bash
# Check current coverage
make test

# Generate detailed HTML report
make coverage-html
open htmlcov/index.html

# View coverage in terminal
pytest --cov=cloakpivot --cov-report=term-missing
```

## Continuous Integration

### Local CI Simulation

```bash
# Run exactly what CI runs
make ci-install  # Setup CI environment
make ci-test     # Run CI test suite
```

### Pre-commit Validation

```bash
# Before committing
make pre-commit  # Same as 'make check'

# Before pushing
make pre-push    # Same as 'make all'
```

## Debugging Test Failures

### Common Issues

1. **Import Errors**: Ensure development environment is set up
   ```bash
   make dev
   ```

2. **Slow Tests**: Identify slow tests
   ```bash
   pytest --durations=10
   ```

3. **Flaky Tests**: Re-run failed tests
   ```bash
   pytest --lf  # Run last failed
   ```

### Debugging Commands

```bash
# Run single test with output
pytest tests/test_analyzer.py::TestAnalyzer::test_basic -v -s

# Debug with pdb
pytest tests/test_analyzer.py::TestAnalyzer::test_basic --pdb

# Show coverage for specific module
pytest --cov=cloakpivot.core.analyzer --cov-report=term-missing tests/
```

## Project Configuration

All testing tools are configured in `pyproject.toml`:

- **Black**: line-length=100, target-version=py311
- **Ruff**: Comprehensive linting with integrated isort
- **MyPy**: Gradual typing with per-module overrides
- **Pytest**: Coverage integration, markers, and test discovery
- **Coverage**: Branch coverage, multiple report formats

All configuration details are in `pyproject.toml`.

## Contributing Test Guidelines

### Test Quality Standards

1. **Descriptive Names**: Test names should describe the scenario
2. **Single Assertion**: Each test verifies one behavior
3. **Independent**: Tests don't depend on other tests
4. **Fast**: Unit tests complete in < 1 second
5. **Deterministic**: Consistent results every run

### Adding New Tests

1. Choose appropriate test category (unit/integration/e2e)
2. Use existing fixtures and utilities where possible
3. Add appropriate markers for test categorization
4. Ensure tests pass with `make test`
5. Check coverage hasn't decreased

### Code Review Checklist

- [ ] Tests cover new functionality
- [ ] Tests include edge cases
- [ ] Tests pass locally with `make test`
- [ ] Coverage maintained or improved
- [ ] Test documentation updated if needed

## Maintenance Commands

```bash
# Clean all test artifacts
make clean

# Verify tool configuration
make verify

# Check for outdated dependencies
make deps-check

# Update dependencies
make deps-update
```

## Getting Help

```bash
# Show all available commands
make help

# Show project information
make info

# Verify tools are installed
make verify
```

For more details on the project configuration and development workflow, see:
- [README.md](README.md) - Project overview and usage
- [Makefile](Makefile) - All available commands
- `pyproject.toml` - Complete configuration details

This testing infrastructure ensures CloakPivot maintains high quality, performance, and reliability across all supported use cases.