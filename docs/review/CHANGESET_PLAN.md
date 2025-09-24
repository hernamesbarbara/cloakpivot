# CloakPivot Refactoring Changeset Plan
*Generated: 2025-09-24*

## Executive Summary
30 findings organized into 15 PRs across 4 phases. Timeline: 6-8 weeks. Risk: MEDIUM with mitigation strategies.

## Phase 1: Foundation & Quick Wins (Week 1-2)
*Low risk changes that improve code quality immediately*

### PR-001: Remove Dead Code [LOW RISK] ✅ COMPLETED
**Findings**: FIND-0021, FIND-0022, FIND-0025, FIND-0027
**Changes**:
- Remove unreachable code after `raise` in `cloakmap.py` (lines 572-580, 607-629)
- Remove unused variable assignments in `loaders.py` and `unmasking/engine.py`
- Clean up vulture-identified dead code

**Acceptance Criteria**:
- [x] All tests pass unchanged
- [x] No functional changes
- [x] Vulture reports clean for removed sections
- [x] Code coverage unchanged or improved

**Files**: 4 files, ~100 lines removed
**Review Time**: 1 hour

---

### PR-002: Fix Signal Handler Parameters [LOW RISK] ✅ COMPLETED
**Finding**: FIND-0024
**Changes**:
- Rename unused parameters in `presidio_mapper.py` to `_signum`, `_frame`

**Acceptance Criteria**:
- [x] Signal handler still works
- [x] Timeout functionality unchanged
- [x] Tests pass

**Files**: 1 file, 2 lines
**Review Time**: 15 minutes

---

### PR-003: Add Critical Missing Tests [MEDIUM RISK] ✅ COMPLETED
**Finding**: TEST_GAPS Critical Section
**Changes**:
- Add 38 critical unit tests for presidio_adapter internals
- Add conflict resolution tests
- Add unmasking accuracy tests

**Acceptance Criteria**:
- [x] All new tests pass
- [x] Coverage increases by >10%
- [x] No flaky tests
- [x] Tests run in <30 seconds

**Files**: 4 new test files, ~1500 lines
**Review Time**: 2 hours

---

## Phase 2: Consolidation & DRY (Week 3-4)
*Extract common utilities and reduce duplication*

### PR-004: Extract Presidio Common Utilities [MEDIUM RISK] ✅ COMPLETED
**Finding**: FIND-0005
**Changes**:
- Create `core/presidio_common.py`
- Extract shared utilities from masking/unmasking presidio adapters
- Update imports in both adapters

**Acceptance Criteria**:
- [x] All tests pass
- [x] No behavior changes
- [x] ~150 lines deduplicated
- [x] Both adapters use common utilities
- [x] Performance unchanged (±5%)

**Files**: 3 files (1 new, 2 modified)
**Review Time**: 2 hours

---

### PR-005: Create Engine Factory [MEDIUM RISK] ✅ COMPLETED
**Findings**: FIND-0016, FIND-0019
**Changes**:
- Create `engine_factory.py`
- Replace direct imports in `engine.py`
- Use dependency injection

**Acceptance Criteria**:
- [x] All tests pass
- [x] Factory creates correct engine types
- [x] No public API changes
- [x] Improved testability demonstrated

**Files**: 2 files (1 new, 1 modified)
**Review Time**: 1.5 hours

---

### PR-006: Consolidate Document Utilities [LOW RISK] ✅ COMPLETED
**Finding**: FIND-0012
**Changes**:
- Create `document/common.py`
- Extract shared document processing utilities
- Update document modules to use common utilities

**Acceptance Criteria**:
- [x] All tests pass
- [x] ~45 lines deduplicated
- [x] Document processing unchanged
- [x] No performance regression

**Files**: 4 files (1 new, 3 modified)
**Review Time**: 1 hour

---

## Phase 3: Major Splits (Week 5-6)
*Split oversized modules into focused components*

### PR-007: Split PresidioMaskingAdapter - Part 1 [HIGH RISK] ✅ COMPLETED
**Finding**: FIND-0001
**Changes**:
- Extract `strategy_processors.py` (220 lines)
- Extract `entity_processor.py` (350 lines)
- Update main adapter to use processors

**Acceptance Criteria**:
- [x] All tests pass
- [x] Integration tests unchanged
- [x] Each new module <400 LOC
- [x] Public API identical
- [x] Performance within ±5%
- [x] New unit tests for processors

**Files**: 3 files (2 new, 1 modified)
**Review Time**: 4 hours

---

### PR-008: Split PresidioMaskingAdapter - Part 2 [HIGH RISK] ✅ COMPLETED
**Finding**: FIND-0001 continued
**Changes**:
- Extract `text_processor.py` (212 lines)
- Extract `document_reconstructor.py` (281 lines)
- Extract `metadata_manager.py` (323 lines)
- Complete adapter refactoring

**Acceptance Criteria**:
- [x] All tests pass
- [x] Original file reduced to proper size with delegation
- [x] All 5 modules properly integrated (strategy_processors, entity_processor, text_processor, document_reconstructor, metadata_manager)
- [x] No behavior changes
- [x] Core functionality verified

**Files**: 4 files (3 new, 1 modified)
**Review Time**: 4 hours

---

### PR-009: Split CloakMap [MEDIUM RISK] ✅ COMPLETED
**Finding**: FIND-0002
**Changes**:
- Extract `cloakmap_validator.py` (382 lines)
- Extract `cloakmap_serializer.py` (393 lines)
- Refactor main CloakMap to 506 lines

**Acceptance Criteria**:
- [x] All tests pass
- [x] Serialization format unchanged
- [x] Round-trip integrity maintained
- [x] Each module <400 LOC (validator: 382, serializer: 393)
- [x] Public API unchanged

**Files**: 3 files (2 new, 1 modified)
**Review Time**: 3 hours

---

### PR-010: Split MaskingApplicator [MEDIUM RISK] ✅ COMPLETED
**Finding**: FIND-0003
**Changes**:
- Extract `conflict_resolver.py` (308 lines)
- Extract `strategy_executor.py` (387 lines)
- Extract `template_helpers.py` (155 lines)
- Extract `format_helpers.py` (232 lines)
- Refactor main applicator.py to 206 lines

**Acceptance Criteria**:
- [x] All tests pass (40/40 tests passing)
- [x] Conflict resolution unchanged
- [x] Strategy execution identical
- [x] All modules reasonable size (<400 LOC)
- [x] Improved testability (helper classes for testing)

**Files**: 5 files (4 new, 1 modified)
**Review Time**: 3 hours

---

## Phase 4: Architecture & Polish (Week 7-8)
*Architectural improvements and final cleanup*

### PR-011: Reorganize Core Layer [HIGH RISK] ✅ COMPLETED
**Findings**: FIND-0018, FIND-0020
**Changes**:
- Create `core/types/`, `core/policies/`, `core/processing/`, `core/utilities/`
- Move modules to appropriate subpackages
- Update all imports

**Acceptance Criteria**:
- [x] All tests pass
- [x] Import paths updated throughout
- [x] Clear separation of concerns
- [x] Documentation updated
- [x] No circular dependencies

**Files**: 22 files moved/modified (20 modules + 4 new __init__.py + main core __init__.py)
**Review Time**: 4 hours

---

### PR-012: Create Breaking Changes Migration Guide [LOW RISK]
**Changes**:
- Create `BREAKING_CHANGES.md` documenting API changes from refactoring
- Document import path changes from core layer reorganization
- Document method signature changes from module splits
- Provide code examples for migrating to new APIs

**Acceptance Criteria**:
- [ ] All breaking changes documented with before/after examples
- [ ] Import path migrations clearly explained
- [ ] Method signature changes documented
- [ ] Migration examples for each major change

**Files**: 1 new documentation file
**Review Time**: 1 hour

---

### PR-013: Performance Optimizations [MEDIUM RISK]
**Changes**:
- Optimize text processing algorithms
- Add caching where beneficial
- Profile and optimize hot paths

**Acceptance Criteria**:
- [ ] Performance improved by >10%
- [ ] All tests pass
- [ ] Memory usage stable
- [ ] Benchmarks documented

**Files**: 5-8 files modified
**Review Time**: 3 hours

---

### PR-014: Documentation Update [LOW RISK]
**Changes**:
- Update all docstrings
- Create architecture diagrams
- Update README with new structure
- Add migration guide

**Acceptance Criteria**:
- [ ] All public APIs documented
- [ ] Examples work
- [ ] Architecture diagram accurate
- [ ] Migration guide complete

**Files**: Documentation files
**Review Time**: 2 hours

---

### PR-015: Final Cleanup [LOW RISK]
**Changes**:
- Remove TODO comments addressed
- Final linting and formatting
- Update CHANGELOG.md
- Version bump preparation

**Acceptance Criteria**:
- [ ] Zero linting warnings
- [ ] All TODOs addressed or documented
- [ ] CHANGELOG complete
- [ ] Ready for release

**Files**: Various
**Review Time**: 1 hour

---

## Risk Mitigation Strategies

### 1. Parallel Implementation
Keep old implementations as `*_legacy.py` during transition:
```python
# During transition
if FEATURE_FLAG_NEW_ADAPTER:
    from .presidio_adapter_v2 import PresidioMaskingAdapter
else:
    from .presidio_adapter_legacy import PresidioMaskingAdapter
```

### 2. Incremental Testing
Run both implementations in parallel:
```python
def test_compatibility():
    result_old = old_implementation(data)
    result_new = new_implementation(data)
    assert result_old == result_new
```

### 3. Feature Flags
Use environment variables for gradual rollout:
```bash
export USE_NEW_PRESIDIO_ADAPTER=true
export USE_NEW_CLOAKMAP=false
```

### 4. Rollback Plan
Each PR must be independently revertible:
- Git revert for code changes
- Feature flags for runtime control
- Database migrations avoided
- No breaking API changes

## Success Metrics

### Per PR:
- [ ] All existing tests pass
- [ ] New tests added where needed
- [ ] Performance within ±5% baseline
- [ ] No breaking changes
- [ ] Code coverage maintained or improved
- [ ] Review completed within SLA

### Overall Project:
- [ ] 30% reduction in average file size
- [ ] 430 lines of duplication removed
- [ ] 95% test coverage achieved
- [ ] Zero breaking changes to public API
- [ ] Performance improved or maintained
- [ ] Documentation comprehensive

## Timeline Summary

| Week | Phase | PRs | Risk | Focus |
|------|-------|-----|------|-------|
| 1-2 | Foundation | PR-001 to PR-003 | LOW | Quick wins, test gaps |
| 3-4 | Consolidation | PR-004 to PR-006 | MEDIUM | DRY violations |
| 5-6 | Major Splits | PR-007 to PR-010 | HIGH | Oversized modules |
| 7-8 | Architecture | PR-011 to PR-015 | MEDIUM | Polish & cleanup |

## Resource Requirements

### Development:
- 2 senior engineers full-time for 8 weeks
- Or 1 senior engineer for 16 weeks

### Review:
- ~35 hours of review time total
- Technical lead review for architecture PRs
- Peer review for all PRs

### Testing:
- QA validation after each phase
- Performance testing after Phase 3
- Full regression before release

## Communication Plan

### Weekly Updates:
- Progress against plan
- Blockers and risks
- Metrics and coverage

### Stakeholder Checkpoints:
- End of Phase 1: Foundation complete
- End of Phase 2: Consolidation results
- End of Phase 3: Major refactoring done
- End of Phase 4: Ready for release

---
*Total PRs: 15*
*Total Duration: 8 weeks*
*Risk Level: MEDIUM with mitigations*
*Estimated Effort: 320 hours*