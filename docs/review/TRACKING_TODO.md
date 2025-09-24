# CloakPivot Refactoring Master Checklist
*Status: Ready for Execution*
*Last Updated: 2025-09-24*

## Quick Reference
- **Total Findings**: 30
- **Total PRs**: 15
- **Timeline**: 8 weeks
- **Effort**: 320 hours
- **Risk**: MEDIUM

## Phase 1: Foundation & Quick Wins ⏳
*Target: Week 1-2*

### PR-001: Remove Dead Code
- [ ] Remove unreachable code in `cloakmap.py:572-580` (FIND-0021)
- [ ] Remove unreachable code in `cloakmap.py:607-629` (FIND-0022)
- [ ] Remove unused `conf_hash` in `loaders.py:257` (FIND-0025)
- [ ] Remove unused `original_document` in `unmasking/engine.py:239` (FIND-0027)
- [ ] Run vulture to verify cleanup
- [ ] Run full test suite
- [ ] Submit PR for review

### PR-002: Fix Signal Handler Parameters
- [ ] Rename parameters to `_signum`, `_frame` in `presidio_mapper.py:12` (FIND-0024)
- [ ] Verify signal handler functionality
- [ ] Run tests
- [ ] Submit PR for review

### PR-003: Add Critical Missing Tests
- [ ] Create `test_presidio_adapter_internals.py` with 15 tests
- [ ] Create `test_conflict_resolution.py` with 10 tests
- [ ] Create `test_unmasking_accuracy.py` with 8 tests
- [ ] Create `test_anchor_resolution_extended.py` with 5 tests
- [ ] Verify coverage increase >10%
- [ ] Ensure all tests pass in <30 seconds
- [ ] Submit PR for review

## Phase 2: Consolidation & DRY ⏳
*Target: Week 3-4*

### PR-004: Extract Presidio Common Utilities
- [ ] Create `core/presidio_common.py` (FIND-0005)
- [ ] Extract version detection utilities
- [ ] Extract operator processing utilities
- [ ] Extract entity type mapping
- [ ] Extract statistics building
- [ ] Update `masking/presidio_adapter.py` imports
- [ ] Update `unmasking/presidio_adapter.py` imports
- [ ] Run all Presidio-related tests
- [ ] Verify ~150 lines deduplicated
- [ ] Submit PR for review

### PR-005: Create Engine Factory
- [ ] Create `engine_factory.py` (FIND-0016, FIND-0019)
- [ ] Implement `create_masking_engine()` method
- [ ] Implement `create_unmasking_engine()` method
- [ ] Update `engine.py` to use factory
- [ ] Update tests to use factory
- [ ] Verify dependency injection works
- [ ] Submit PR for review

### PR-006: Consolidate Document Utilities
- [ ] Create `document/common.py` (FIND-0012)
- [ ] Extract `TextSegmentIterator`
- [ ] Extract `BoundaryCalculator`
- [ ] Extract `DocumentValidator`
- [ ] Update document modules to use common
- [ ] Run document processing tests
- [ ] Submit PR for review

## Phase 3: Major Splits ⏳
*Target: Week 5-6*

### PR-007: Split PresidioMaskingAdapter - Part 1
- [ ] Create `masking/strategy_processors.py` (FIND-0001)
- [ ] Create `masking/entity_processor.py`
- [ ] Move strategy methods to processor
- [ ] Move entity methods to processor
- [ ] Update adapter to use processors
- [ ] Create unit tests for new modules
- [ ] Verify public API unchanged
- [ ] Run integration tests
- [ ] Submit PR for review

### PR-008: Split PresidioMaskingAdapter - Part 2
- [ ] Create `masking/text_processor.py` (FIND-0001)
- [ ] Create `masking/document_reconstructor.py`
- [ ] Create `masking/metadata_manager.py`
- [ ] Complete adapter refactoring
- [ ] Verify adapter now ~280 lines
- [ ] Run all masking tests
- [ ] Run performance benchmarks
- [ ] Submit PR for review

### PR-009: Split CloakMap
- [ ] Create `core/cloakmap_validator.py` (FIND-0002)
- [ ] Create `core/cloakmap_serializer.py`
- [ ] Move validation logic to validator
- [ ] Move serialization to serializer
- [ ] Update CloakMap to delegate
- [ ] Test serialization round-trip
- [ ] Verify format unchanged
- [ ] Submit PR for review

### PR-010: Split MaskingApplicator
- [ ] Create `masking/conflict_resolver.py` (FIND-0003)
- [ ] Create `masking/strategy_executor.py`
- [ ] Move conflict logic to resolver
- [ ] Move execution to executor
- [ ] Update applicator to delegate
- [ ] Create unit tests for modules
- [ ] Submit PR for review

## Phase 4: Architecture & Polish ⏳
*Target: Week 7-8*

### PR-011: Reorganize Core Layer
- [ ] Create `core/types/` directory (FIND-0018, FIND-0020)
- [ ] Create `core/policies/` directory
- [ ] Create `core/processing/` directory
- [ ] Create `core/utilities/` directory
- [ ] Move modules to appropriate directories
- [ ] Update all import statements
- [ ] Run full test suite
- [ ] Check for circular dependencies
- [ ] Submit PR for review

### PR-012: Add Deprecation Warnings
- [ ] Add warning to `key_version` params (FIND-0023)
- [ ] Add warning to `compat.py` (FIND-0029)
- [ ] Create migration guide
- [ ] Update documentation
- [ ] Test warning display
- [ ] Submit PR for review

### PR-013: Performance Optimizations
- [ ] Profile hot paths with cProfile
- [ ] Optimize text processing algorithms
- [ ] Add caching where beneficial
- [ ] Run performance benchmarks
- [ ] Document improvements
- [ ] Submit PR for review

### PR-014: Documentation Update
- [ ] Update all docstrings
- [ ] Create architecture diagrams
- [ ] Update README.md
- [ ] Create migration guide
- [ ] Test all examples
- [ ] Submit PR for review

### PR-015: Final Cleanup
- [ ] Remove addressed TODO comments
- [ ] Run linting and formatting
- [ ] Update CHANGELOG.md
- [ ] Prepare version bump
- [ ] Final test run
- [ ] Submit PR for review

## Validation Checklist

### Before Starting:
- [ ] Create refactoring branch
- [ ] Set up feature flags
- [ ] Baseline performance metrics
- [ ] Snapshot current behavior
- [ ] Back up current code

### During Each PR:
- [ ] All tests pass
- [ ] No breaking changes
- [ ] Performance checked
- [ ] Coverage maintained
- [ ] Documentation updated
- [ ] Review completed

### After Each Phase:
- [ ] Integration tests pass
- [ ] Performance benchmarks run
- [ ] Stakeholder update sent
- [ ] Risk assessment updated
- [ ] Next phase planned

### Final Validation:
- [ ] All 30 findings addressed
- [ ] Public API unchanged
- [ ] Performance maintained or improved
- [ ] Test coverage >95%
- [ ] Documentation complete
- [ ] Ready for release

## Risk Tracking

| Risk | Status | Mitigation |
|------|--------|------------|
| Breaking API changes | ⚠️ Monitoring | Feature flags, parallel implementation |
| Performance regression | ⚠️ Monitoring | Continuous benchmarking |
| Test coverage gaps | ✅ Addressed | PR-003 adds critical tests |
| Circular dependencies | ⚠️ Monitoring | Dependency analysis tools |
| Large PR review burden | ⚠️ Monitoring | Split into smaller PRs |

## Progress Tracking

| Week | Phase | Status | PRs Complete | Notes |
|------|-------|--------|--------------|-------|
| 1-2 | Foundation | 🔄 Not Started | 0/3 | |
| 3-4 | Consolidation | 🔄 Not Started | 0/3 | |
| 5-6 | Major Splits | 🔄 Not Started | 0/4 | |
| 7-8 | Architecture | 🔄 Not Started | 0/5 | |

## Communication Log

| Date | Update | Audience | Status |
|------|--------|----------|--------|
| 2025-09-24 | Plan Created | Team | ✅ |
| TBD | Phase 1 Complete | Stakeholders | 🔄 |
| TBD | Phase 2 Complete | Stakeholders | 🔄 |
| TBD | Phase 3 Complete | Stakeholders | 🔄 |
| TBD | Phase 4 Complete | Stakeholders | 🔄 |

## Key Metrics Dashboard

```
Current State:
- Total LOC: 17,975
- Avg File Size: 359 LOC
- Largest File: 1,310 LOC
- Duplication: ~430 lines
- Test Coverage: ~85%

Target State:
- Total LOC: ~17,500 (-475)
- Avg File Size: <250 LOC
- Largest File: <400 LOC
- Duplication: 0
- Test Coverage: >95%
```

## Contact & Resources

- **Lead Engineer**: [TBD]
- **Review Team**: [TBD]
- **Documentation**: `/docs/review/`
- **Feature Flags**: See `CHANGESET_PLAN.md`
- **Rollback Plan**: See `CHANGESET_PLAN.md`

---
*Use this checklist to track progress. Update status after each task.*
*Legend: ✅ Complete | 🔄 In Progress | ⏳ Planned | ❌ Blocked*