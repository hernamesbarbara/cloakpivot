Contributing to CloakPivot
========================

We welcome contributions to CloakPivot! This guide will help you get started with contributing to the project.

.. contents::
   :local:
   :depth: 2

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork and Clone**

   Fork the repository on GitHub and clone your fork:

   .. code-block:: bash

       git clone https://github.com/yourusername/cloakpivot.git
       cd cloakpivot

2. **Set Up Development Environment**

   We recommend using a virtual environment:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**

   Install the package in development mode with all dependencies:

   .. code-block:: bash

       pip install -e ".[dev,docs,test]"

4. **Install Pre-commit Hooks**

   We use pre-commit hooks to ensure code quality:

   .. code-block:: bash

       pre-commit install

5. **Verify Installation**

   Run the tests to make sure everything is working:

   .. code-block:: bash

       pytest
       cloakpivot --help

Development Workflow
~~~~~~~~~~~~~~~~~~~~

1. **Create a Feature Branch**

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. **Make Changes**

   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Tests and Linting**

   .. code-block:: bash

       # Run all tests
       pytest
       
       # Run linting
       flake8 cloakpivot/
       black --check cloakpivot/
       isort --check-only cloakpivot/
       
       # Type checking
       mypy cloakpivot/

4. **Commit Changes**

   Use conventional commit messages:

   .. code-block:: bash

       git add .
       git commit -m "feat: add new masking strategy"
       git commit -m "fix: resolve anchor resolution issue"
       git commit -m "docs: update API reference"

5. **Push and Create Pull Request**

   .. code-block:: bash

       git push origin feature/your-feature-name

   Then create a pull request on GitHub.

Coding Standards
----------------

Code Style
~~~~~~~~~~

We use several tools to maintain consistent code style:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all formatting tools:

.. code-block:: bash

    black cloakpivot/
    isort cloakpivot/
    flake8 cloakpivot/
    mypy cloakpivot/

Python Guidelines
~~~~~~~~~~~~~~~~~

1. **Type Hints**

   Use type hints for all public APIs:

   .. code-block:: python

       def mask_document(
           input_path: str | Path,
           policy: MaskingPolicy,
           output_format: str = "lexical"
       ) -> MaskResult:
           """Mask a document with the specified policy."""

2. **Docstrings**

   Use Google-style docstrings:

   .. code-block:: python

       def process_entities(entities: list[RecognizerResult]) -> ProcessResult:
           """Process detected entities with policy application.
           
           Args:
               entities: List of detected PII entities from Presidio
               
           Returns:
               ProcessResult containing processed entities and statistics
               
           Raises:
               ValidationError: If entities fail validation
               PolicyError: If policy application fails
           """

3. **Error Handling**

   Use specific exception types and provide helpful error messages:

   .. code-block:: python

       from cloakpivot.core.exceptions import PolicyError, ValidationError
       
       if not policy.is_valid():
           raise PolicyError(
               f"Invalid policy configuration: {policy.validation_errors}"
           )

4. **Logging**

   Use structured logging with appropriate levels:

   .. code-block:: python

       import logging
       
       logger = logging.getLogger(__name__)
       
       def mask_entities(entities):
           logger.info(f"Processing {len(entities)} entities")
           logger.debug(f"Entity types: {[e.entity_type for e in entities]}")

Testing
-------

Test Structure
~~~~~~~~~~~~~~

Our test suite is organized as follows:

.. code-block:: text

    tests/
    ├── unit/                   # Unit tests
    │   ├── core/              # Core module tests
    │   ├── masking/           # Masking module tests
    │   └── unmasking/         # Unmasking module tests
    ├── integration/           # Integration tests
    │   ├── test_round_trip.py # End-to-end masking/unmasking
    │   └── test_formats.py    # Format handling tests
    ├── e2e/                   # End-to-end tests
    │   └── test_cli.py        # CLI command tests
    ├── fixtures/              # Test data and fixtures
    └── conftest.py            # Pytest configuration

Writing Tests
~~~~~~~~~~~~~

1. **Unit Tests**

   Test individual functions and classes:

   .. code-block:: python

       import pytest
       from cloakpivot.core import MaskingPolicy, Strategy, StrategyKind
       
       class TestMaskingPolicy:
           def test_default_policy_creation(self):
               policy = MaskingPolicy()
               assert policy.locale == "en"
               assert policy.default_strategy.kind == StrategyKind.REDACT
               
           def test_custom_strategy_override(self):
               policy = MaskingPolicy(
                   per_entity={"PERSON": Strategy(kind=StrategyKind.HASH)}
               )
               assert policy.per_entity["PERSON"].kind == StrategyKind.HASH

2. **Integration Tests**

   Test component interactions:

   .. code-block:: python

       def test_mask_unmask_round_trip(sample_document, temp_dir):
           """Test that masking and unmasking produces identical results."""
           # Mask document
           mask_result = mask_document(
               sample_document, 
               policy=MaskingPolicy(),
               output_path=temp_dir / "masked.json"
           )
           
           # Unmask document
           unmask_result = unmask_document(
               mask_result.masked_path,
               mask_result.cloakmap_path
           )
           
           # Verify round-trip accuracy
           assert_documents_equal(sample_document, unmask_result.restored_document)

3. **CLI Tests**

   Test command-line interfaces:

   .. code-block:: python

       from click.testing import CliRunner
       from cloakpivot.cli.main import cli
       
       def test_mask_command(temp_dir, sample_document_file):
           runner = CliRunner()
           result = runner.invoke(cli, [
               'mask', str(sample_document_file),
               '--out', str(temp_dir / 'masked.json'),
               '--cloakmap', str(temp_dir / 'map.json')
           ])
           assert result.exit_code == 0
           assert "Masking completed successfully" in result.output

Test Fixtures
~~~~~~~~~~~~~

Use pytest fixtures for reusable test data:

.. code-block:: python

    @pytest.fixture
    def sample_document():
        """Create a sample document for testing."""
        return {
            "name": "Test Document",
            "texts": [
                {
                    "text": "Contact John Doe at john@example.com",
                    "type": "text",
                    "node_id": "text_001"
                }
            ]
        }
    
    @pytest.fixture
    def healthcare_policy():
        """Create a healthcare-compliant policy for testing."""
        return MaskingPolicy(
            per_entity={
                "PERSON": Strategy(
                    kind=StrategyKind.TEMPLATE,
                    parameters={"template": "[PATIENT]"}
                )
            },
            thresholds={"PERSON": 0.8}
        )

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

    # Run all tests
    pytest
    
    # Run specific test file
    pytest tests/unit/core/test_policies.py
    
    # Run tests with coverage
    pytest --cov=cloakpivot --cov-report=html
    
    # Run only fast tests (exclude slow integration tests)
    pytest -m "not slow"
    
    # Run tests in parallel
    pytest -n auto

Documentation
-------------

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

1. **API Documentation**

   All public APIs should have comprehensive docstrings:

   .. code-block:: python

       class MaskingEngine:
           """Engine for applying masking policies to documents.
           
           The MaskingEngine coordinates PII detection, policy application,
           and document transformation while preserving structure.
           
           Examples:
               Basic usage with default policy:
               
               >>> engine = MaskingEngine()
               >>> result = engine.mask_document(document, policy)
               >>> print(f"Masked {result.stats.entities_masked} entities")
               
           Attributes:
               resolve_conflicts: Whether to resolve overlapping entity conflicts
               performance_mode: Optimization mode for large documents
           """

2. **User Documentation**

   Update relevant documentation sections when adding features:

   - API reference (automatically generated from docstrings)
   - User guides and tutorials
   - CLI command documentation
   - Policy examples and best practices

3. **Code Comments**

   Use comments sparingly for complex logic:

   .. code-block:: python

       # Apply conflict resolution for overlapping entities
       # This ensures deterministic masking when entities overlap
       if self.resolve_conflicts and has_overlapping_entities(entities):
           entities = self._resolve_entity_conflicts(entities)

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd docs/
    
    # Install documentation dependencies
    pip install -r requirements-docs.txt
    
    # Build HTML documentation
    make html
    
    # Serve documentation locally
    make serve  # Available at http://localhost:8000
    
    # Build with all warnings as errors (for CI)
    make html-strict
    
    # Check for broken links
    make linkcheck
    
    # Validate example policies
    make validate-policies

Issue and Pull Request Guidelines
---------------------------------

Reporting Issues
~~~~~~~~~~~~~~~~

When reporting issues, please include:

1. **Clear Description**: What you expected vs. what happened
2. **Environment**: Python version, CloakPivot version, OS
3. **Reproduction Steps**: Minimal example to reproduce the issue
4. **Logs**: Relevant error messages or log output
5. **Sample Data**: If possible, provide sample input (sanitized)

Example issue template:

.. code-block:: text

    **Bug Description**
    Masking fails when processing documents with nested tables.
    
    **Environment**
    - CloakPivot version: 0.1.0
    - Python version: 3.9.10
    - OS: macOS 13.0
    
    **Steps to Reproduce**
    1. Create document with nested table structure
    2. Apply balanced policy
    3. Run `cloakpivot mask document.json`
    
    **Expected Behavior**
    Document should be masked successfully
    
    **Actual Behavior**
    KeyError: 'nested_table' when processing table structure
    
    **Logs**
    ```
    ERROR: Failed to process nested table at node nested_table_001
    Traceback (most recent call last):
      ...
    ```

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

1. **Descriptive Titles**

   Use clear, descriptive titles:

   - ✅ "Add support for custom entity recognizers"
   - ❌ "Fix bug"

2. **Complete Descriptions**

   Include in your PR description:

   - What changes were made and why
   - Any breaking changes
   - Test coverage for new features
   - Documentation updates
   - Links to related issues

3. **Small, Focused PRs**

   Keep PRs focused on a single feature or fix:

   - Easier to review
   - Faster to merge
   - Easier to revert if needed
   - Better git history

4. **Review Checklist**

   Before submitting:

   - [ ] Tests pass locally
   - [ ] Code is properly formatted
   - [ ] Type hints are included
   - [ ] Documentation is updated
   - [ ] Commit messages follow conventions
   - [ ] No merge conflicts

Code Review Process
~~~~~~~~~~~~~~~~~~~

1. **Automated Checks**

   All PRs run automated checks:

   - Unit and integration tests
   - Code formatting (Black, isort)
   - Linting (flake8)
   - Type checking (mypy)
   - Security scanning

2. **Manual Review**

   Reviewers will check:

   - Code correctness and efficiency
   - Test coverage and quality
   - Documentation completeness
   - API design consistency
   - Security considerations

3. **Review Response**

   When responding to review feedback:

   - Address all feedback
   - Ask questions if unclear
   - Update tests and docs as needed
   - Re-request review when ready

Types of Contributions
----------------------

Code Contributions
~~~~~~~~~~~~~~~~~~

1. **Bug Fixes**

   - Fix existing functionality that's not working correctly
   - Include regression tests
   - Update documentation if behavior changes

2. **New Features**

   - Implement new masking strategies
   - Add support for new document formats
   - Enhance CLI functionality
   - Improve performance

3. **Performance Improvements**

   - Optimize slow operations
   - Reduce memory usage
   - Improve scalability
   - Add benchmarking

Documentation Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **API Documentation**

   - Improve docstring coverage
   - Add usage examples
   - Clarify complex concepts

2. **User Guides**

   - Create tutorials for new features
   - Add troubleshooting guides
   - Write best practice guides

3. **Policy Examples**

   - Create industry-specific policies
   - Add complex configuration examples
   - Document policy patterns

Testing Contributions
~~~~~~~~~~~~~~~~~~~~~

1. **Test Coverage**

   - Add tests for untested code paths
   - Create integration test scenarios
   - Write property-based tests

2. **Test Infrastructure**

   - Improve test fixtures
   - Add performance benchmarks
   - Create testing utilities

Community Contributions
~~~~~~~~~~~~~~~~~~~~~~~

1. **Issue Triage**

   - Help reproduce reported issues
   - Provide additional context
   - Suggest workarounds

2. **Support**

   - Answer questions in discussions
   - Help users with configuration
   - Share usage patterns

3. **Evangelism**

   - Write blog posts or tutorials
   - Present at conferences
   - Create video content

Recognition
-----------

Contributors are recognized in several ways:

1. **Contributors File**: All contributors are listed in the project
2. **Release Notes**: Significant contributions are highlighted
3. **GitHub Recognition**: Contributor badges and statistics
4. **Maintainer Nomination**: Active contributors may be invited as maintainers

Getting Help
------------

If you need help contributing:

1. **Documentation**: Check this guide and the main documentation
2. **Discussions**: Use GitHub Discussions for questions
3. **Discord/Slack**: Join our community channels
4. **Mentoring**: Ask for a mentor if you're new to open source

We're here to help make your contribution experience positive and productive!

Code of Conduct
---------------

Please note that this project is released with a Contributor Code of Conduct. By participating in this project, you agree to abide by its terms. See CODE_OF_CONDUCT.md for details.

Thank you for contributing to CloakPivot! 🎭✨