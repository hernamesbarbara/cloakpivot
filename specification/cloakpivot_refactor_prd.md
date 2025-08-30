# CloakPivot Refactor — Product Requirements Document (PRD)

### 1. Overview

- **Problem**: CloakPivot leverages Presidio and Docling heavily for PII masking/unmasking while preserving document structure. Current implementation prioritizes functionality, but performance and test efficiency may not fully align with best practices for heavy ML dependencies. Presidio analyzers and Docling converters may be reinitialized repeatedly, and caching/prefetch strategies are not consistently applied across the project.
- **Solution**: Refactor CloakPivot internals to adopt best practices around singleton loaders, session-scoped fixtures, model prefetching, and lightweight defaults. This ensures faster tests, improved developer feedback loops, and better CI performance without altering external APIs or core masking/unmasking flows.
- **Outcomes**:
  - Maintain CloakPivot’s masking/unmasking fidelity while reducing redundant Presidio/Docling initialization.
  - Ensure tests run faster and CI pipelines leverage cached models/wheels.
  - Provide a sustainable architecture for continued extension (policies, plugins, storage).

### 2. Goals and Non-Goals

- **Goals**:
  - Evaluate CloakPivot’s current Presidio + Docling integration against published best practices.
  - Introduce singleton loaders for analyzers and converters to avoid repeated imports.
  - Ensure Docling pipelines can default to lightweight (fast text-only) processing with optional full OCR/tables.
  - Enhance pytest test suite with session fixtures and heavy-test markers.
  - Add CI caching instructions and validation scripts.

- **Non-Goals**:
  - Redesigning the CloakPivot CLI or API.
  - Changing the masking/unmasking data model (CloakMap).
  - Developing new entity recognition models (still use Presidio).

### 3. Users and Use Cases

- **Users**: CloakPivot maintainers, contributors, CI maintainers.
- **Use cases**:
  - Developers quickly run unit and integration tests locally without slow model initialization.
  - CI pipelines execute tests faster with cached models and prebuilt wheels.
  - Heavy OCR/table tests are isolated and only run when explicitly required.

### 4. Key Concepts

- **AnalyzerEngineWrapper**: CloakPivot’s wrapper for Presidio analyzer (currently created ad hoc in demos/examples【21†source】).
- **EntityDetectionPipeline**: Pipeline object orchestrating Presidio detection and Docling text segment integration.
- **CloakMap**: Mapping artifact for reversible unmasking【20†source】.
- **Singleton Loader**: Cached instance of heavy analyzers/converters reused across runs.

### 5. Current Architecture (Observed)

- **Analyzer Initialization**: `AnalyzerEngineWrapper` is instantiated directly within pipelines/examples【21†source】. No explicit caching or reuse across sessions.
- **Docling Converter**: Configured inside pipelines, not exposed via reusable singleton loader.
- **Testing**: CloakPivot has 142+ tests【20†source】, but not all follow session-scoped fixture patterns. Heavy and light tests are not clearly separated.
- **CI**: GitHub Actions runs standard pytest but lacks explicit caching of spaCy/Docling models【20†source】.

### 6. Refactor Requirements

- **R1: Singleton Loaders**
  - Implement `get_presidio_analyzer()` and `get_docling_converter()` with caching (using `@lru_cache`).

- **R2: Pytest Fixtures**
  - Define `scope="session"` fixtures for analyzers and converters in `conftest.py`.

- **R3: Lightweight Defaults**
  - Configure default spaCy model to `en_core_web_sm`.
  - Default Docling converter to text-only (`pypdfium2` backend), disabling OCR/tables/images.

- **R4: Prefetch Models**
  - Add scripts/README instructions for running `docling-tools models download` and spaCy model install.

- **R5: Test Tiering**
  - Introduce `@pytest.mark.heavy` for OCR/table-dependent tests.
  - Configure pytest-xdist with `--dist=loadscope` on macOS.

- **R6: CI Caching**
  - Cache `~/.cache/docling/models`, spaCy model wheels, and pip wheels.

- **R7: Documentation Updates**
  - Update developer docs and README with new instructions for running tests and setting up caches.

### 7. API Design (Refactor Impact)

- **New Loader Functions**
  - `cloakpivot.loaders.get_presidio_analyzer(lang="en")`
  - `cloakpivot.loaders.get_docling_converter(artifacts_path=None, fast_text_only=True)`

- **Fixtures**
  - `presidio_analyzer()`
  - `docling_fast_converter()`
  - `docling_full_converter()`

- **Markers**
  - `@pytest.mark.heavy`

### 8. Evaluation Criteria

- **Before Refactor**:
  - Analyzer instantiated per example run【21†source】.
  - Pipelines construct Docling converters inline.
  - No explicit caching of models in CI【20†source】.

- **After Refactor**:
  - Analyzer/Docling initialized once per session.
  - Fixtures reused across tests.
  - Clear distinction between fast vs heavy pipelines.
  - CI runs significantly faster due to model caching.

### 9. Deliverables

- `cloakpivot/loaders.py` with cached loader functions.
- `tests/conftest.py` with session fixtures and markers.
- Updated `README.md` with caching/setup instructions.
- CI workflow changes for caching.
- Internal dev docs describing fast vs full pipeline usage.

### 10. Milestones

- M1: Implement loaders and fixtures.
- M2: Add heavy test markers and pytest config.
- M3: Update CI with caching.
- M4: Documentation updates and final evaluation.

### 11. Dependencies

- Presidio Analyzer & spaCy.
- Docling + PyTorch.
- pytest, pytest-xdist.

### 12. Open Questions

- Should analyzer configuration (recognizer registry trimming) be part of loader defaults?
- Should service-boundary Presidio (REST mode) be introduced for very large workloads?

