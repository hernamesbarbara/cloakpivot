# Changelog

All notable changes to CloakPivot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Comprehensive Refactoring (PRs 001-015)**: Major architectural improvements and code quality enhancements
  - **PR-003**: Added 38 critical unit tests for presidio_adapter internals, conflict resolution, and unmasking accuracy
  - **PR-004**: Extracted shared Presidio utilities into `core/presidio_common.py` (~150 lines deduplicated)
  - **PR-005**: Created engine factory with dependency injection for improved testability
  - **PR-006**: Consolidated document processing utilities into `document/common.py` (~45 lines deduplicated)
  - **PR-012**: Created comprehensive `BREAKING_CHANGES.md` migration guide with before/after examples
  - **PR-013**: Implemented major performance optimizations:
    - O(n²) → O(n) complexity improvement for apply_spans algorithm 
    - LRU caching for StrategyToOperatorMapper (128 entries, 2x-5x speedup)
    - Efficient list joining for document text building
    - >10% improvement for documents >10KB with >100 entities
  - **PR-014**: Updated all API documentation with comprehensive docstrings and visual workflow diagrams
  - **PR-015**: Final cleanup with zero linting warnings and CHANGELOG updates

### Changed
- **Major Module Refactoring**: Split oversized modules into focused, maintainable components
  - **PR-007-008**: Split 1,450-line `PresidioMaskingAdapter` into 5 focused modules:
    - `strategy_processors.py` (220 lines) - Strategy processing logic
    - `entity_processor.py` (350 lines) - Entity processing workflows  
    - `text_processor.py` (212 lines) - Text manipulation operations
    - `document_reconstructor.py` (281 lines) - Document rebuilding logic
    - `metadata_manager.py` (323 lines) - Metadata and statistics management
  - **PR-009**: Split 1,281-line `CloakMap` into 3 focused modules:
    - `cloakmap_validator.py` (382 lines) - Validation and integrity checking
    - `cloakmap_serializer.py` (393 lines) - Serialization and persistence
    - Core CloakMap reduced to 506 lines with delegation
  - **PR-010**: Split 1,288-line `MaskingApplicator` into 5 focused modules:
    - `conflict_resolver.py` (308 lines) - Entity conflict resolution
    - `strategy_executor.py` (387 lines) - Strategy execution engine
    - `template_helpers.py` (155 lines) - Template generation utilities
    - `format_helpers.py` (232 lines) - Format detection and processing
    - Core applicator reduced to 206 lines
- **Core Architecture Reorganization (PR-011)**: Restructured core layer into logical subpackages
  - `core/types/` - Data structures and type definitions
  - `core/policies/` - Policy definitions and loading system  
  - `core/processing/` - Analysis and processing algorithms
  - `core/utilities/` - Helper functions and validation utilities
  - Updated all imports throughout codebase (22 files moved/modified)
- **Code Quality Improvements (PR-015)**: Achieved zero linting warnings
  - Fixed all star imports with explicit imports for better maintainability
  - Resolved 254+ linting issues including nested with statements and import organization
  - Improved code readability and IDE support

### Fixed
- **PR-001**: Removed dead code including unreachable code after `raise` statements (~100 lines removed)
- **PR-002**: Fixed signal handler parameters in `presidio_mapper.py` (renamed unused params to `_signum`, `_frame`)

### Performance
- **Significant Performance Gains (PR-013)**:
  - Text replacement operations: O(n²) → O(n) algorithmic complexity
  - Strategy mapping cache: 2x-5x speedup with LRU eviction
  - Memory optimization: Efficient data structures eliminate repeated string concatenation
  - Large document processing: >10% improvement for documents >10KB with >100 entities
  - Comprehensive benchmark suite added in `tests/performance/`

### Documentation
- **Enhanced API Documentation (PR-014)**: 
  - Complete public API documentation with usage examples
  - New `docs/WORKFLOW_DIAGRAMS.md` with comprehensive mermaid diagrams
  - Visual documentation of PDF → JSON → Masked JSON → Markdown workflows
  - Updated README with enhanced code examples and CloakMap usage patterns
- **Migration Support (PR-012)**: 
  - Comprehensive breaking changes documentation
  - Before/after code examples for all API changes
  - Import path migration guides
  - Method signature change documentation

### Removed
- **Dead Code Cleanup (PR-001)**: Removed unused variables, unreachable code, and vulture-identified dead sections

### Fixed
- **CI/CD Pipeline Optimization**: Reduced GitHub Actions runtime from 20+ minutes to ~5 minutes
  - Added CPU-only PyTorch installation to eliminate 1.4GB CUDA package downloads
  - Implemented dependency caching for pip and spaCy models
  - Split workflow into parallel jobs (lint, test, test-full)
  - Configured smaller spaCy model for PR checks (en_core_web_sm)
- **Type Checking Compatibility**: Fixed mypy errors between local and CI environments
  - Created centralized `type_imports.py` module for third-party type handling
  - Replaced scattered `# type: ignore` comments with proper type fixes
  - Used `setattr()` for dynamic attribute assignment (clearer intent)
  - Fixed DoclingDocument import issues caused by missing `__all__` declaration
  - Only 4 necessary `type: ignore` comments remain (for genuinely untyped Presidio calls)
- **SURROGATE Strategy Bug**: Fixed seed parameter not being used for deterministic generation
  - SurrogateGenerator now correctly uses seed parameter when provided
  - Same seed produces consistent fake data across multiple runs
  - Added comprehensive example in `examples/surrogate_faker_strategy.py`

### Added
- **New Test Suite**: Complete rewrite of test infrastructure based on v2.0 API
  - Created clean test structure with `unit/` and `integration/` directories
  - 32 comprehensive tests covering CloakEngine and CloakEngineBuilder functionality
  - Tests use real test data from `data/json/` and `data/pdf/` directories
  - Fixtures properly configured for v2.0 API usage patterns
  - All tests passing with 31.68% code coverage
- **Table Masking Tests**: Comprehensive test suite for table cell PII masking
  - 10 unit tests covering various table masking scenarios
  - Tests for structure preservation, round-trip consistency, and different masking strategies

### Removed
- **Build Artifacts and Cache Files**:
  - Removed `.benchmarks/`, `.mypy_cache/`, `.ruff_cache/` directories
  - Removed `build/` and `cloakpivot.egg-info/` directories
  - Cleaned up all `__pycache__` directories throughout the project
- **Outdated Test Suite**:
  - Removed entire old `tests/` directory that was incompatible with v2.0 API
  - Old tests referenced non-existent modules and used incorrect API patterns

### Fixed
- **Critical Import and Code Issues**:
  - Fixed duplicate `MaskResult` import in `__init__.py`
  - Removed references to non-existent `CryptoUtils` module (removed in v2.0)
  - Fixed test suite to use `Strategy` objects instead of bare `StrategyKind` enums
  - Updated method calls to use correct names (`save_to_file`, `load_from_file`)
  - Fixed CloakEngine initialization to not accept presets directly (use CloakEngineBuilder)
- **DoclingDocument Compatibility**: Fixed version mismatch issues with Docling v1.7.0 format
  - Updated dependencies to `docling>=2.52.0` and `docling-core>=2.0.0` to match DocPivot requirements
  - Fixed masking process to preserve `origin` field in masked documents (required for v1.7.0)
  - Resolved validation errors when loading masked DoclingDocument files
- **TextItem Label Validation**: Fixed Pydantic validation errors for unsupported labels
  - Added automatic mapping of invalid labels (like 'title', 'section_header') to valid TextItem labels
  - Ensures masked documents maintain compatibility with DoclingDocument schema
- **Example Scripts**: Fixed all example scripts to work with current test data
  - Replaced non-existent `StrategyKind.KEEP` with `StrategyKind.TEMPLATE` in docling_integration.py
  - Updated examples to use existing test data in `data/json/` and `data/pdf/` directories
  - Fixed `main_text` attribute error by using `texts` attribute instead
  - Updated DoclingJsonReader import from docpivot for proper document loading

### Added
- **Advanced Builder Features Example**: New `advanced_builder_features.py` demonstrating:
  - `ConflictResolutionConfig` for controlling entity grouping behavior with `merge_threshold_chars`
  - `.with_conflict_resolution()` builder method for custom entity handling
- **Table Cell Masking**: Fixed critical bug where PII in table cells was not being masked
  - Fixed `_find_segment_for_position` to correctly map text positions to table cell segments
  - Added `_update_table_cells` method to apply masked values to table cells
  - Table cells now properly masked while preserving table structure
- **SURROGATE Strategy with Faker**: Enhanced SURROGATE strategy to generate realistic fake data
  - Fixed integration between SurrogateGenerator and Presidio adapter
  - SURROGATE entities now processed separately to ensure Faker is used
  - Produces realistic replacements (e.g., "John Doe" → "Morgan Williams") instead of asterisks
  - Fixed seed parameter usage for deterministic fake data generation
  - Same seed now produces consistent results across multiple runs
  - `.with_presidio_engine()` explicit configuration for enabling/disabling Presidio
  - Direct `DocPivotEngine` usage for format conversion (optional)
  - Combining multiple advanced features using the builder pattern
- **Project Configuration Standardization**: Adopted DocPivot's cleaner configuration approach
  - Simplified `pyproject.toml` by removing excessive comments and verbose configurations
  - Streamlined `Makefile` from 248 lines to 158 lines, removing redundant targets
  - Adopted DocPivot's minimal configuration style with selective enhancements
  - Removed redundant `PROJECT_CONFIG.md` documentation file
- **New Test Suite**: Complete rewrite of test infrastructure
  - Archived old tests to `tests_old/` for reference
  - Created clean test structure: `unit/`, `integration/`, `e2e/`
  - Leverages real test data from `data/pdf/` and `data/json/`
  - Minimal fixtures focused on actual functionality
- **Compatibility Module**: New `cloakpivot.compat` module providing:
  - `load_document()` - Direct Docling JSON loading
  - `to_lexical()` - Conversion using new DocPivotEngine
- **Migration Guide**: Added `specification/CLOAKPIVOT_MIGRATION_GUIDE.md` for DocPivot v2.0.1 migration

### Changed
- **Development Workflow**: Simplified to use Make commands exclusively
  - `make dev` - One-command development setup
  - `make all` - Single CI/CD pipeline entry point
  - `make check` - Quick pre-commit validation (format, lint, type, test-fast)
  - Removed redundant targets like `install-dev`, publishing targets, and verbose helpers
- **Configuration Philosophy**: Adopted DocPivot's cleaner approach
  - Removed 100+ lines of comments from `pyproject.toml`
  - Simplified pytest configuration to essential options only
  - Adopted moderate MyPy strictness with gradual typing
  - Kept test markers but removed verbose coverage requirements
- **DocPivot v2.0.1 Migration**: Updated to use DocPivot v2.0.1 with new `DocPivotEngine` API
  - Replaced `SerializerProvider` with `DocPivotEngine` in format registry
  - Replaced `LexicalDocSerializer` with `engine.convert_to_lexical()`
  - Replaced `load_document()` with direct JSON loading for Docling files
  - Added `cloakpivot.compat` module for backward compatibility
- **Document Loading**: Simplified to load Docling JSON files directly without DocPivot
- **Performance**: Improved performance with direct JSON loading and single engine instance

### Removed
- Helper scripts `pdf2docling.py` and `docling2cloaked.py` (functionality now in examples)
- Excessive configuration comments in `pyproject.toml` (reduced by ~60%)
- Redundant Makefile targets (90+ lines removed)
- `PROJECT_CONFIG.md` documentation file (content already in README)
- `PERFORMANCE.md` - Outdated performance documentation referencing removed features
- `run_tests.py` - Redundant test runner (functionality in Makefile)
- `coverage.xml` - Generated file that shouldn't be in version control
- `htmlcov/` directory - Generated coverage HTML reports
- `test_reports/` directory - Old performance test reports no longer needed
- `tests_old/` directory - Archived old test suite replaced by new clean structure
- `benchmarks/` directory - Old performance benchmarking configuration
- `policies/` directory - Incomplete example policy templates never fully implemented
- `scripts/` directory - Obsolete performance and setup scripts
- `docs/` directory - Misleading documentation referencing non-existent v1.x
- Publishing targets from Makefile (not needed for internal project)
- Watch target and other rarely-used development helpers

### Fixed
- Import errors with DocPivot v2.0.1 API changes
- Test compatibility with new DocPivot API
- Over-engineered configuration files now simplified

### Removed
- **Dead Code Cleanup**: Major cleanup of unused and deprecated code
  - Removed empty `cloakpivot/policies/` directory (functionality in `core/policies.py`)
  - Removed `cloakpivot/deprecated.py` with broken imports
  - Removed entire unused `cloakpivot/observability/` subsystem
  - Removed unused modules: `core/chunking.py`, `formats/registry.py`, `masking/document_masker.py`
  - Removed dead batch processing modules: `cli/batch.py`, `core/batch.py`
  - Fixed broken imports throughout codebase
  - Removed obsolete test files with non-existent imports

## [2.0.0] - 2025-09-13

### Added
- **CloakEngine API**: New simplified one-line API for PII masking/unmasking
  - `engine.mask_document(doc)` - Single method for masking with auto-detection
  - `engine.unmask_document(doc, cloakmap)` - Simple unmasking
- **CloakEngineBuilder**: Fluent builder pattern for advanced configuration
- **Smart Defaults**: Pre-configured policies (default, conservative, permissive)
- **MaskResult**: Clean result type with statistics (entities_found, entities_masked)
- **API Documentation**: Comprehensive API reference in `docs/API.md`
- **Migration Guide**: Step-by-step guide for upgrading from v1.x in `docs/MIGRATION.md`
- **CI/CD Pipeline**: GitHub Actions for testing, building, and releasing
- **Python 3.8 Support**: Extended compatibility to Python 3.8+

### Changed
- **Simplified CLI**: Reduced from 2,794 to 175 lines with focus on core mask/unmask commands
- **Unified Engine**: Replaced separate MaskingEngine/UnmaskingEngine with single CloakEngine
- **Direct Returns**: `unmask_document()` returns DoclingDocument directly (not wrapped in result)
- **Result Property**: MaskResult uses `document` property instead of `masked_document`
- **Test Suite**: Refactored all tests to use CloakEngine API (63+ new tests)
- **Examples**: Replaced all examples with simple_usage.py and advanced_usage.py
- **Dependencies**: Simplified to core dependencies only (removed unnecessary packages)

### Removed
- **Legacy Modules** (12,347 lines removed):
  - `cloakpivot.migration.*` - CloakMap migration tools
  - `cloakpivot.storage.*` - Cloud storage backends (S3, GCS, Database)
  - `cloakpivot.plugins.*` - Plugin system
  - `cloakpivot.diagnostics.*` - Diagnostic tools
  - `cloakpivot.security.*` - Complex security module (1,290 lines)
  - `cloakpivot.observability.exporters.*` - Metric exporters
  - `cloakpivot.core.parallel_analysis` - Parallel processing
  - `cloakpivot.core.performance` - Performance optimization
- **CLI Commands**: Removed migration, diagnostic, batch, and plugin commands
- **Test Files**: Removed 23 test files for deleted modules

### Deprecated
- `MaskingEngine` and `UnmaskingEngine` classes (use `CloakEngine` instead)
- Separate text extraction and entity detection steps (now handled internally)
- Complex multi-step initialization (replaced with smart defaults)

### Fixed
- CloakMap version compatibility (now supports v1.0 and v2.0)
- Import organization and circular dependencies
- Test coverage for core functionality

### Performance
- 33.6% reduction in codebase size (36,747 → 24,400 lines)
- Faster initialization with simplified architecture
- Reduced memory footprint with single engine design

## [1.8.3] - Previous Release

Last version with separate MaskingEngine/UnmaskingEngine architecture.

## Migration

See [Migration Guide](docs/MIGRATION.md) for detailed upgrade instructions from v1.x to v2.0.

### Breaking Changes in v2.0

1. **Import Changes**:
   ```python
   # Old
   from cloakpivot.masking.engine import MaskingEngine

   # New
   from cloakpivot import CloakEngine
   ```

2. **API Changes**:
   ```python
   # Old
   engine = MaskingEngine()
   result = engine.mask_document(doc, entities, policy, segments)

   # New
   engine = CloakEngine()
   result = engine.mask_document(doc)
   ```

3. **Result Access**:
   ```python
   # Old
   masked_doc = result.masked_document

   # New
   masked_doc = result.document
   ```