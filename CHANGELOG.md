# Changelog

All notable changes to CloakPivot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **DocPivot v2.0.1 Migration**: Updated to use DocPivot v2.0.1 with new `DocPivotEngine` API
  - Replaced `SerializerProvider` with `DocPivotEngine` in format registry
  - Replaced `LexicalDocSerializer` with `engine.convert_to_lexical()`
  - Replaced `load_document()` with direct JSON loading for Docling files
  - Added `cloakpivot.compat` module for backward compatibility
- **Document Loading**: Simplified to load Docling JSON files directly without DocPivot
- **Performance**: Improved performance with direct JSON loading and single engine instance

### Added
- **Compatibility Module**: New `cloakpivot.compat` module providing:
  - `load_document()` - Direct Docling JSON loading
  - `to_lexical()` - Conversion using new DocPivotEngine
- **Migration Guide**: Added `specification/CLOAKPIVOT_MIGRATION_GUIDE.md` for DocPivot v2.0.1 migration

### Fixed
- Import errors with DocPivot v2.0.1 API changes
- Test compatibility with new DocPivot API

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