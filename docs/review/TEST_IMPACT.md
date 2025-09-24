# Test Impact & Risk Assessment Matrix
*Generated: 2025-09-24*

## Executive Summary
Analysis of 30 proposed changes across 6 categories with test impact assessment. Total test files affected: ~25 files. Overall risk: MEDIUM with proper mitigation strategies.

## Risk Assessment Scale
- **LOW**: Changes unlikely to break functionality, easy rollback
- **MEDIUM**: Changes may affect behavior, requires careful testing
- **HIGH**: Changes affect core functionality or public APIs

## Test Impact Matrix

### Category 1: Oversized Module Splits (HIGH IMPACT)

| Finding | Change | Test Files Affected | Risk | New Tests Required |
|---------|--------|-------------------|------|-------------------|
| FIND-0001 | Split presidio_adapter.py | test_presidio_adapter.py<br>test_masking_workflow.py | MEDIUM | 5 new unit test files |
| FIND-0002 | Split cloakmap.py | test_cloakmap.py<br>test_cloakmap_loader.py | LOW | 3 new unit test files |
| FIND-0003 | Split applicator.py | test_masking_workflow.py | LOW | 3 new unit test files |
| FIND-0004 | Split document_unmasker.py | test_document_processor*.py | MEDIUM | 3 new unit test files |
| FIND-0009 | Split masking/engine.py | test_masking_workflow.py<br>test_builder.py | HIGH | 2 new unit test files |

**Total Impact**: 5 existing test files, 16 new test files needed

### Category 2: DRY Violations (MEDIUM IMPACT)

| Finding | Change | Test Files Affected | Risk | Test Strategy |
|---------|--------|-------------------|------|---------------|
| FIND-0005 | Extract presidio common utilities | test_presidio_adapter.py<br>test_unmasking.py | LOW | Add integration tests |
| FIND-0011 | Extract engine patterns | test_builder.py | MEDIUM | Mock new abstractions |
| FIND-0012 | Extract document utilities | test_document_processor*.py | LOW | Unit test utilities |
| FIND-0013 | Consolidate normalization | Multiple test files | LOW | Regression tests |
| FIND-0014 | Unify error handling | All error tests | LOW | Error scenario tests |

**Total Impact**: 8-10 test files need updates

### Category 3: Boundary Violations (HIGH IMPACT)

| Finding | Change | Test Files Affected | Risk | Mitigation |
|---------|--------|-------------------|------|------------|
| FIND-0016 | Engine factory pattern | test_builder.py<br>test_engine.py | HIGH | Parallel implementation |
| FIND-0017 | Fix relative imports | All test files | MEDIUM | Automated refactoring |
| FIND-0018 | Reorganize core layer | All core tests (~10 files) | HIGH | Incremental migration |
| FIND-0019 | Dependency injection | test_builder.py | MEDIUM | Feature flag testing |

**Total Impact**: Major architectural changes affecting 15+ test files

### Category 4: Dead Code Removal (LOW IMPACT)

| Finding | Change | Test Files Affected | Risk | Validation |
|---------|--------|-------------------|------|------------|
| FIND-0021 | Remove unreachable code | None | NONE | Static analysis |
| FIND-0022 | Remove more unreachable | None | NONE | Static analysis |
| FIND-0023 | Deprecate key_version | API compatibility tests | MEDIUM | Deprecation tests |
| FIND-0024 | Rename unused params | None | NONE | Compilation check |
| FIND-0025-27 | Remove unused vars | None | LOW | Unit tests pass |

**Total Impact**: Minimal, mostly cleanup

### Category 5: Complex Refactoring (HIGH IMPACT)

| Finding | Change | Test Files Affected | Risk | Strategy |
|---------|--------|-------------------|------|----------|
| FIND-0006 | Simplify CloakEngine.__init__ | test_builder.py<br>test_engine.py | LOW | Incremental refactor |
| FIND-0007 | Split surrogate.py | test_surrogate*.py | LOW | Unit test each part |
| FIND-0008 | Separate normalization concerns | test_normalization.py | MEDIUM | Behavior preservation |
| FIND-0010 | Split analyzer.py | test_analyzer.py | MEDIUM | Mock boundaries |

**Total Impact**: 6-8 test files need comprehensive updates

### Category 6: New Functionality (MEDIUM IMPACT)

| Finding | Change | Test Files Affected | Risk | Coverage Target |
|---------|--------|-------------------|------|-----------------|
| FIND-0028 | Remove Presidio timeout workaround | All Presidio tests | MEDIUM | Stress testing |
| FIND-0029 | Deprecate compat.py | test_compat.py | LOW | Migration tests |
| FIND-0030 | Document CloakedDocument usage | test_wrappers.py | LOW | Example tests |

**Total Impact**: 3-4 test files, mostly documentation

## Test Execution Priority

### Phase 1: Pre-refactoring Tests (MUST DO FIRST)
1. **Snapshot current behavior** - Create golden files for all major operations
2. **Add missing unit tests** - Cover gaps identified in TEST_GAPS.md
3. **Performance benchmarks** - Baseline for comparison
4. **Integration test suite** - End-to-end scenarios

### Phase 2: During Refactoring
1. **Run tests continuously** - Every small change
2. **A/B testing** - Old vs new implementation
3. **Mutation testing** - Verify test effectiveness
4. **Load testing** - Ensure no performance regression

### Phase 3: Post-refactoring
1. **Full regression suite** - All tests must pass
2. **New unit tests** - For split modules
3. **Documentation tests** - Examples still work
4. **Compatibility tests** - Public API unchanged

## Risk Mitigation Strategies

### Strategy 1: Feature Flags
```python
if os.getenv("USE_NEW_PRESIDIO_ADAPTER"):
    from .presidio_adapter_v2 import PresidioMaskingAdapter
else:
    from .presidio_adapter import PresidioMaskingAdapter
```

### Strategy 2: Parallel Implementation
- Keep old code as `*_legacy.py`
- Run both implementations in tests
- Compare results for consistency

### Strategy 3: Incremental Migration
- Start with low-risk changes (dead code removal)
- Progress to medium-risk (DRY violations)
- End with high-risk (architectural changes)

### Strategy 4: Comprehensive Testing
- Unit tests: >95% coverage per module
- Integration tests: All user workflows
- Property-based tests: Edge cases
- Benchmark tests: Performance validation

## Test Coverage Requirements

| Component | Current Coverage | Target Coverage | Gap |
|-----------|-----------------|-----------------|-----|
| masking/* | ~85% | 95% | 10% |
| unmasking/* | ~80% | 95% | 15% |
| core/* | ~90% | 95% | 5% |
| document/* | ~75% | 90% | 15% |
| cli/* | ~70% | 85% | 15% |

## Critical Test Scenarios

### Must Not Break:
1. **Public API compatibility** - All existing code must work
2. **Serialization format** - CloakMap round-trip integrity
3. **Masking accuracy** - PII detection and replacement
4. **Document structure** - Table and formatting preservation
5. **Performance** - No regression >10%

### Edge Cases to Test:
1. Empty documents
2. Overlapping entities
3. Unicode and special characters
4. Large documents (>10MB)
5. Concurrent operations
6. Error recovery

## Testing Timeline

| Week | Activity | Deliverable |
|------|----------|-------------|
| 1 | Gap analysis & test creation | TEST_GAPS.md implementation |
| 2 | Snapshot & benchmark creation | Golden files & baselines |
| 3-4 | Refactoring with continuous testing | Split modules with tests |
| 5 | Integration & regression testing | Full test suite green |
| 6 | Performance & stress testing | Performance report |

## Success Criteria

- [ ] All existing tests pass without modification
- [ ] New unit tests achieve >95% coverage
- [ ] Performance within ±5% of baseline
- [ ] Zero breaking changes to public API
- [ ] All edge cases covered
- [ ] Documentation tests pass
- [ ] Integration tests unchanged

---
*Total estimated test effort: 2-3 weeks*
*Risk level: MEDIUM with proper mitigation*
*Test files affected: ~25 existing, ~30 new*