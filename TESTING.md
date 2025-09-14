# CloakPivot Testing Guide

This document provides comprehensive guidance for running and contributing to the CloakPivot test suite.

## Quick Start

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests/

# Run tests with coverage
pytest --cov=cloakpivot --cov-report=html tests/

# Run specific test files
pytest tests/test_cloak_engine_simple.py -v
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

## Running Tests

### Using the Test Runner

The `run_tests.py` script provides convenient access to all test categories:

```bash
# Basic usage
python run_tests.py <test_type> [options]

# Examples
python run_tests.py unit --verbose --coverage
python run_tests.py integration
python run_tests.py fast  # All tests except slow/performance
python run_tests.py all   # Complete test suite
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
pytest -m golden        # Golden file regression tests
pytest -m performance   # Performance benchmarks
pytest -m property      # Property-based tests
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
@pytest.mark.performance
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

## Golden File Testing

Golden files contain expected outputs for regression testing. When outputs change:

1. **Review Changes**: Ensure changes are intentional
2. **Update Golden Files**: Delete old golden files to regenerate them
3. **Commit Updates**: Include golden file updates in your commits

```bash
# Regenerate golden files
rm tests/fixtures/golden_files/*.json
python run_tests.py golden
```

## Coverage Requirements

- **Unit Tests**: > 90% line coverage
- **Integration Tests**: > 85% branch coverage
- **Overall Project**: > 80% combined coverage

```bash
# Generate detailed coverage report
pytest --cov=cloakpivot --cov-report=html --cov-report=term-missing
open htmlcov/index.html  # View detailed report
```

## Performance Benchmarking

Performance tests establish baselines and detect regressions:

```bash
# Run performance benchmarks
python run_tests.py performance

# Run with benchmark comparison
pytest --benchmark-compare tests/performance/
```

### Performance Expectations

| Document Size | Max Processing Time | Max Memory Usage |
|---------------|-------------------|------------------|
| Small (< 1KB) | 2 seconds        | 50 MB           |
| Medium (1-10KB) | 5 seconds      | 100 MB          |
| Large (10-100KB) | 30 seconds     | 500 MB          |

## Continuous Integration

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### CI Pipeline

The CI pipeline runs:

1. **Code Quality**: Black, Ruff, MyPy
2. **Fast Tests**: Unit and integration tests
3. **Coverage**: Minimum 80% coverage requirement
4. **Performance**: Regression detection

## Debugging Test Failures

### Common Issues

1. **Flaky Tests**: Use `pytest --lf` to run last failed tests
2. **Slow Tests**: Use `pytest --durations=10` to identify slow tests
3. **Memory Issues**: Use `pytest --tb=short` for concise tracebacks

### Debugging Commands

```bash
# Run single test with full output
pytest tests/test_analyzer.py::TestAnalyzer::test_basic -v -s

# Debug with pdb
pytest tests/test_analyzer.py::TestAnalyzer::test_basic --pdb

# Show test coverage for specific module
pytest --cov=cloakpivot.core.analyzer --cov-report=term-missing tests/
```

## Contributing Test Guidelines

### Test Quality Standards

1. **Descriptive Names**: Test names should describe the scenario being tested
2. **Single Assertion**: Each test should verify one specific behavior
3. **Independent Tests**: Tests should not depend on other tests
4. **Fast Execution**: Unit tests should complete in < 1 second
5. **Deterministic**: Tests should produce consistent results

### Code Review Checklist

- [ ] Tests cover all new functionality
- [ ] Tests include edge cases and error conditions
- [ ] Performance impact is measured and acceptable
- [ ] Golden files are updated if outputs change
- [ ] Documentation is updated for new test categories

### Adding New Test Categories

To add a new test category:

1. Create marker in `pyproject.toml`
2. Add category to `run_tests.py`
3. Update this documentation
4. Add examples and guidelines

## Troubleshooting

### Common Test Errors

#### Import Errors
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -e .
```

#### Timeout Errors
```bash
# Increase timeout for slow tests
pytest --timeout=600 tests/
```

#### Memory Errors
```bash
# Run tests with memory monitoring
pytest --memory-profile tests/
```

#### Permission Errors
```bash
# Fix file permissions in test fixtures
chmod -R 644 tests/fixtures/
```

### Getting Help

- Check existing test examples in the codebase
- Review test utility functions in `tests/utils/`
- Consult pytest documentation for advanced features
- Ask questions in code reviews or team discussions

## Performance Monitoring

### Tracking Test Performance

Monitor test execution time to prevent test suite slowdown:

```bash
# Track slowest tests
pytest --durations=0 tests/

# Benchmark test execution
pytest --benchmark-autosave tests/performance/
```

### Memory Monitoring

Track memory usage during test execution:

```bash
# Profile memory usage
pytest --memory-profile tests/

# Monitor peak memory usage
python -m memory_profiler run_tests.py unit
```

## Test Data Management

### Generating Test Data

Use the test data generators for consistent test scenarios:

```python
from tests.utils.generators import DocumentGenerator, PolicyGenerator

# Generate test documents
doc = DocumentGenerator.generate_document_with_pii(
    ["PHONE_NUMBER", "EMAIL_ADDRESS"],
    "test_document"
)

# Generate test policies
policy = PolicyGenerator.generate_comprehensive_policy(
    PrivacyLevel.MEDIUM
)
```

### Managing Test Fixtures

- Keep test fixtures minimal and focused
- Use parametrized tests for multiple input scenarios
- Clean up resources in fixture teardown
- Document complex fixture behavior

This comprehensive testing infrastructure ensures CloakPivot maintains high quality, performance, and reliability across all supported use cases.