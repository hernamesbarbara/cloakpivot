# CloakPivot Performance Optimization — Product Requirements Document (PRD)

### 1. Overview

- **Problem**: CloakPivot leverages Presidio for PII detection and DocPivot for document processing. While the current implementation has good architectural foundations (lazy initialization in AnalyzerEngineWrapper, some session-scoped fixtures), there are opportunities to optimize performance further. Key issues include: lack of singleton pattern for analyzers across the application, no CI/CD configuration for model caching, and potential for parallel processing improvements.
- **Solution**: Build upon existing performance foundations by implementing global singleton loaders with caching, enhancing test fixture reuse patterns, adding CI/CD workflows with model caching, and leveraging the existing PerformanceProfiler for metrics-driven optimization. Focus on incremental improvements that maintain backward compatibility.
- **Outcomes**:
  - Reduce analyzer initialization overhead through singleton pattern implementation
  - Accelerate test execution via enhanced fixture reuse and parallel execution strategies
  - Establish CI/CD pipeline with intelligent caching for models and dependencies
  - Provide clear performance metrics and monitoring capabilities

### 2. Goals and Non-Goals

- **Goals**:
  - Implement global singleton pattern for AnalyzerEngineWrapper with thread-safe caching
  - Enhance existing test fixtures to use session scope more broadly for expensive resources
  - Create GitHub Actions CI/CD workflows with model and dependency caching
  - Optimize spaCy model selection (default to lightweight en_core_web_sm)
  - Leverage existing parallel processing capabilities in parallel_analysis.py
  - Integrate PerformanceProfiler metrics into test suite for regression detection
  - Document performance best practices and benchmarking procedures

- **Non-Goals**:
  - Redesigning the CloakPivot CLI or API interfaces
  - Changing the masking/unmasking data model (CloakMap)
  - Modifying DocPivot integration patterns
  - Developing new entity recognition models beyond Presidio's capabilities
  - Breaking backward compatibility with existing code

### 3. Users and Use Cases

- **Users**: CloakPivot maintainers, contributors, CI maintainers.
- **Use cases**:
  - Developers quickly run unit and integration tests locally without slow model initialization.
  - CI pipelines execute tests faster with cached models and prebuilt wheels.
  - Heavy OCR/table tests are isolated and only run when explicitly required.

### 4. Key Concepts

- **AnalyzerEngineWrapper**: CloakPivot's wrapper for Presidio analyzer with lazy initialization via `_initialize_engine()` method
- **EntityDetectionPipeline**: Pipeline orchestrating Presidio detection with document text segment processing
- **DocumentProcessor**: DocPivot integration layer for loading and managing documents
- **PerformanceProfiler**: Existing performance monitoring system with timing decorators and metrics collection
- **CloakMap**: Core mapping artifact for reversible masking/unmasking operations
- **Singleton Loader Pattern**: Proposed caching mechanism for expensive resource initialization

### 5. Current Architecture (Analysis)

- **Analyzer Initialization**: 
  - `AnalyzerEngineWrapper` uses lazy initialization pattern via `_initialize_engine()` (analyzer.py:367)
  - Presidio modules imported only when needed to avoid blocking module load
  - Each pipeline instance creates its own analyzer wrapper (detection.py:121)
  - Test fixtures include `shared_analyzer` with session scope (conftest.py:429)

- **Document Processing**:
  - Uses DocPivot's `load_document` workflow, not direct Docling integration (processor.py:99)
  - DocumentProcessor handles validation and error management
  - Supports chunked processing via ChunkedDocumentProcessor

- **Testing Infrastructure**:
  - 142+ tests with comprehensive marker system (unit, integration, e2e, performance, slow, property)
  - Fast/slow mode configuration via PYTEST_FAST_MODE environment variable
  - Session and module-scoped fixtures for performance optimization
  - `masking_engine` fixture uses module scope (conftest.py:393)
  - Hypothesis profiles configured for different testing scenarios

- **Performance Monitoring**:
  - Comprehensive PerformanceProfiler class with timing decorators (performance.py)
  - Memory monitoring capabilities via MemoryMonitor integration
  - Global profiler instance pattern already established

- **CI/CD Status**:
  - No GitHub Actions workflows currently configured
  - pyproject.toml includes pytest-xdist for parallel test execution
  - Test markers and timeout configurations in place

### 6. Refactor Requirements

- **R1: Global Singleton Pattern for Analyzers**
  - Create `cloakpivot/loaders.py` module with thread-safe singleton implementations
  - Implement `get_presidio_analyzer(language="en", config=None)` with `@lru_cache` or similar caching
  - Build upon existing lazy initialization in AnalyzerEngineWrapper
  - Ensure thread safety for concurrent access in parallel processing scenarios

- **R2: Enhanced Test Fixtures**
  - Extend `shared_analyzer` fixture usage across more test modules
  - Create `shared_document_processor` fixture with session scope
  - Implement `shared_detection_pipeline` fixture to avoid repeated initialization
  - Maintain existing fast/slow mode configuration patterns

- **R3: Optimized Model Configuration**
  - Update `_get_spacy_model_name()` to default to `en_core_web_sm` (already maps correctly)
  - Add environment variable MODEL_SIZE={small|medium|large} for model selection
  - Document memory/speed tradeoffs for different model sizes
  - Implement model download verification script

- **R4: GitHub Actions CI/CD Pipeline**
  - Create `.github/workflows/ci.yml` with test matrix (Python 3.9-3.12)
  - Implement model caching strategy:
    - Cache `~/.cache/huggingface` for transformer models
    - Cache spaCy models in `~/spacy_models`
    - Cache pip wheels and dependencies
  - Add separate workflows for fast tests (PR validation) vs comprehensive tests (merge to main)

- **R5: Performance Benchmarking Integration**
  - Integrate PerformanceProfiler into critical paths
  - Add `@profile_method` decorator to expensive operations
  - Create benchmark test suite using pytest-benchmark
  - Generate performance regression reports in CI

- **R6: Parallel Processing Optimization**
  - Leverage existing parallel_analysis.py capabilities more broadly
  - Configure pytest-xdist with optimal worker count based on CPU cores
  - Implement batch processing for document collections
  - Add progress reporting for long-running operations

- **R7: Documentation and Developer Experience**
  - Create PERFORMANCE.md with optimization guidelines
  - Document fixture scoping best practices
  - Add troubleshooting guide for common performance issues
  - Include profiling instructions using PerformanceProfiler

### 7. API Design (Refactor Impact)

- **New Loader Module (`cloakpivot/loaders.py`)**:
  ```python
  def get_presidio_analyzer(
      language: str = "en",
      config: Optional[AnalyzerConfig] = None,
      cache_key: Optional[str] = None
  ) -> AnalyzerEngineWrapper
  
  def get_document_processor(
      enable_chunked: bool = True
  ) -> DocumentProcessor
  
  def get_detection_pipeline(
      policy: Optional[MaskingPolicy] = None
  ) -> EntityDetectionPipeline
  ```

- **Enhanced Test Fixtures**:
  - `shared_document_processor` (session scope)
  - `shared_detection_pipeline` (session scope)
  - `cached_analyzer_wrapper` (session scope with config parameter)
  - `performance_profiler` (session scope for test metrics)

- **Performance Decorators**:
  - `@profile_critical_path` - For operations that must be fast
  - `@cache_result(ttl=3600)` - For expensive computations
  - `@parallel_process` - For parallelizable operations

### 8. Evaluation Criteria

- **Current State**:
  - Analyzer instantiated per pipeline instance (detection.py:121)
  - DocumentProcessor created independently in each usage
  - Test fixtures partially optimized (some session/module scope)
  - No CI/CD pipeline for automated testing
  - Performance monitoring available but not integrated

- **Target State**:
  - Single analyzer instance shared across application lifecycle
  - Cached document processors with configurable TTL
  - All expensive test fixtures use session scope
  - CI pipeline with <5 minute PR validation, <15 minute full suite
  - Performance regression detection on every commit

- **Success Metrics**:
  - 50% reduction in test suite execution time
  - 80% reduction in analyzer initialization overhead
  - <100ms average entity detection time for standard documents
  - Zero performance regressions detected in production code
  - 90%+ cache hit rate for model loading in CI

### 9. Deliverables

- **Code Deliverables**:
  - `cloakpivot/loaders.py` - Singleton loader implementations with caching
  - Enhanced `tests/conftest.py` with additional session-scoped fixtures
  - `.github/workflows/ci.yml` - GitHub Actions CI/CD pipeline
  - `.github/workflows/performance.yml` - Performance regression testing workflow
  - `scripts/download_models.py` - Model download and verification script
  - `benchmarks/` directory with performance test suite

- **Documentation Deliverables**:
  - `PERFORMANCE.md` - Performance optimization guide
  - `docs/development/testing.md` - Updated testing best practices
  - `docs/development/ci-cd.md` - CI/CD pipeline documentation
  - Updated `README.md` with quick start performance tips

- **Configuration Deliverables**:
  - `.github/dependabot.yml` - Dependency update automation
  - `pytest.ini` updates for parallel execution optimization
  - Docker configuration for consistent test environments

### 10. Implementation Milestones

- **M1: Foundation (Week 1-2)**:
  - Implement singleton loaders in `cloakpivot/loaders.py`
  - Add thread-safety tests for concurrent access
  - Update existing code to use loaders where appropriate
  - Measure baseline performance metrics

- **M2: Test Optimization (Week 2-3)**:
  - Enhance test fixtures with broader session scoping
  - Implement parallel test execution optimization
  - Add performance benchmark suite
  - Document testing best practices

- **M3: CI/CD Pipeline (Week 3-4)**:
  - Create GitHub Actions workflows
  - Implement comprehensive caching strategy
  - Add performance regression detection
  - Set up automated dependency updates

- **M4: Documentation & Validation (Week 4-5)**:
  - Complete all documentation deliverables
  - Conduct performance validation testing
  - Train team on new patterns and tools
  - Create migration guide for existing code

### 11. Technical Dependencies

- **Core Dependencies** (existing):
  - Presidio Analyzer (>=2.2.0) with spaCy backend
  - DocPivot for document processing
  - pytest suite with plugins (xdist, benchmark, timeout)

- **New Dependencies** (to add):
  - GitHub Actions runners with adequate resources
  - Cache storage for CI/CD (GitHub Actions cache)
  - Performance monitoring dashboard (optional)

- **Model Dependencies**:
  - spaCy language models (en_core_web_sm as default)
  - Transformer models if using that backend
  - Storage for cached models (~500MB-2GB depending on selection)

### 12. Risk Mitigation & Open Questions

- **Risks**:
  - Thread safety issues with singleton pattern → Mitigation: Comprehensive concurrency testing
  - Cache invalidation complexity → Mitigation: TTL-based expiration and versioning
  - CI resource consumption → Mitigation: Careful cache management and resource limits

- **Open Questions**:
  - Should we implement distributed caching for multi-instance deployments?
  - What's the optimal balance between cache size and freshness?
  - Should we provide REST API mode for Presidio to enable horizontal scaling?
  - How to handle model version upgrades without breaking existing deployments?

### 13. Performance Targets

- **Test Execution**:
  - Unit tests: <30 seconds
  - Integration tests: <2 minutes
  - Full test suite: <5 minutes with parallelization

- **Runtime Performance**:
  - Analyzer initialization: <500ms (first), <10ms (cached)
  - Document processing: <100ms for typical documents
  - Entity detection: <50ms per 1000 characters

- **Resource Usage**:
  - Memory: <500MB for typical workloads
  - CPU: Efficient multi-core utilization
  - Disk: <2GB for cached models and data

### 14. Implementation Priority & Quick Wins

**Immediate Quick Wins** (can be implemented independently):
1. **Add singleton loaders** - High impact, low complexity change that will immediately improve performance
2. **Extend fixture scoping** - Simple changes to conftest.py for better test performance
3. **Create basic CI workflow** - Essential for automated testing and quality assurance

**Phase 1 - Foundation** (Prerequisites for other improvements):
1. Implement `cloakpivot/loaders.py` with caching mechanisms
2. Add thread-safety testing and validation
3. Create performance baseline measurements

**Phase 2 - Test & CI Optimization**:
1. Optimize test fixtures and parallel execution
2. Implement GitHub Actions with caching
3. Add performance regression detection

**Phase 3 - Advanced Optimizations**:
1. Integrate PerformanceProfiler throughout codebase
2. Implement batch processing optimizations
3. Add distributed caching if needed

### 15. Summary & Next Steps

This PRD outlines a pragmatic approach to optimizing CloakPivot's performance while building on its existing architectural strengths. The current codebase already has good foundations including lazy initialization, performance monitoring tools, and some optimized test fixtures. 

**Key Recommendations**:
1. **Start with singleton loaders** - This addresses the most significant performance bottleneck
2. **Leverage existing tools** - Use the PerformanceProfiler and parallel processing capabilities already in place
3. **Incremental improvements** - Each optimization can be implemented and validated independently
4. **Maintain compatibility** - All changes preserve backward compatibility and existing APIs

**Immediate Next Steps**:
1. Review and approve this PRD with stakeholders
2. Create GitHub issues for each refactor requirement
3. Establish performance baseline metrics using existing PerformanceProfiler
4. Begin implementation with R1 (singleton loaders) as the first priority

The proposed optimizations will significantly improve both developer experience and production performance while maintaining the robustness and flexibility of the current CloakPivot architecture.

