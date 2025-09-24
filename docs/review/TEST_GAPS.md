# Test Coverage Gap Analysis
*Generated: 2025-09-24*

## Executive Summary
Critical test gaps identified in 15 areas. Priority focus on oversized modules lacking unit tests before splitting. Estimated 50-70 new test cases needed.

## Critical Coverage Gaps (MUST FIX BEFORE REFACTORING)

### 1. Presidio Adapter Internals [CRITICAL]
**File**: `cloakpivot/masking/presidio_adapter.py`
**Current Coverage**: ~60% (estimated)
**Missing Tests**:
```python
# test_presidio_adapter_internals.py - NEW FILE NEEDED
def test_filter_overlapping_entities_complex_overlaps()
def test_validate_entities_against_boundaries_edge_cases()
def test_batch_process_entities_large_batches()
def test_prepare_strategies_fallback_scenarios()
def test_apply_spans_unicode_handling()
def test_build_full_text_with_empty_segments()
def test_create_synthetic_result_various_entities()
def test_cleanup_large_results_memory_management()
```
**Priority**: HIGH - Must test before splitting into 6 modules

### 2. CloakMap Validation [HIGH]
**File**: `cloakpivot/core/cloakmap.py`
**Missing Tests**:
```python
# test_cloakmap_validation.py - NEW FILE NEEDED
def test_validate_structure_malformed_data()
def test_merge_incompatible_cloakmaps()
def test_integrity_check_against_document()
def test_position_validation_boundary_conditions()
def test_circular_reference_handling()
```
**Priority**: HIGH - Critical for data integrity

### 3. Conflict Resolution [HIGH]
**File**: `cloakpivot/masking/applicator.py`
**Missing Tests**:
```python
# test_conflict_resolution.py - NEW FILE NEEDED
def test_resolve_triple_overlap()
def test_merge_adjacent_entities_unicode_boundaries()
def test_prioritize_entities_same_confidence()
def test_resolution_strategy_custom_rules()
```
**Priority**: HIGH - Core functionality gap

### 4. Document Structure Edge Cases [MEDIUM]
**File**: `cloakpivot/document/processor.py`
**Missing Tests**:
```python
def test_nested_table_masking()
def test_document_with_no_text_segments()
def test_malformed_document_structure()
def test_extremely_large_document_performance()
def test_special_characters_in_tables()
```
**Priority**: MEDIUM - Important for robustness

### 5. Strategy Execution Failures [HIGH]
**File**: `cloakpivot/masking/engine.py`
**Missing Tests**:
```python
def test_strategy_timeout_handling()
def test_strategy_exception_recovery()
def test_fallback_chain_exhaustion()
def test_concurrent_strategy_execution()
def test_strategy_with_invalid_parameters()
```
**Priority**: HIGH - Error handling critical

### 6. Engine Builder Edge Cases [MEDIUM]
**File**: `cloakpivot/engine_builder.py`
**Missing Tests**:
```python
def test_builder_with_conflicting_configs()
def test_builder_reset_partial_state()
def test_builder_with_null_values()
def test_builder_immutability()
```
**Priority**: MEDIUM - Public API stability

### 7. Unmasking Accuracy [HIGH]
**File**: `cloakpivot/unmasking/engine.py`
**Missing Tests**:
```python
def test_unmask_with_modified_document_structure()
def test_unmask_with_corrupted_cloakmap()
def test_unmask_partial_document()
def test_unmask_with_missing_anchors()
def test_unmask_performance_large_documents()
```
**Priority**: HIGH - Core functionality

### 8. Normalization Edge Cases [LOW]
**File**: `cloakpivot/core/normalization.py`
**Missing Tests**:
```python
def test_normalize_mixed_unicode_normalization()
def test_normalize_rtl_text()
def test_normalize_with_zero_width_characters()
```
**Priority**: LOW - Edge cases

### 9. CLI Error Scenarios [MEDIUM]
**File**: `cloakpivot/cli/main.py`
**Missing Tests**:
```python
def test_cli_with_invalid_file_paths()
def test_cli_with_corrupted_input_files()
def test_cli_interrupt_handling()
def test_cli_memory_limit_handling()
```
**Priority**: MEDIUM - User experience

### 10. Anchor Resolution [HIGH]
**File**: `cloakpivot/unmasking/anchor_resolver.py`
**Missing Tests**:
```python
def test_resolve_anchor_with_text_modifications()
def test_resolve_overlapping_anchors()
def test_resolve_with_boundary_shifts()
def test_anchor_resolution_performance()
```
**Priority**: HIGH - Unmasking accuracy

## Test Categories Needed

### 1. Property-Based Tests
```python
# test_properties.py - NEW FILE
from hypothesis import given, strategies as st

@given(st.text())
def test_mask_unmask_roundtrip(text):
    """Property: mask(unmask(x)) == x"""

@given(st.lists(st.integers()))
def test_span_application_preserves_length(spans):
    """Property: applying spans maintains document structure"""
```

### 2. Performance Benchmarks
```python
# test_benchmarks.py - NEW FILE
def benchmark_large_document_masking()
def benchmark_entity_overlap_resolution()
def benchmark_cloakmap_serialization()
def benchmark_strategy_execution()
```

### 3. Integration Scenarios
```python
# test_integration_scenarios.py - NEW FILE
def test_end_to_end_pdf_workflow()
def test_multi_language_document()
def test_concurrent_document_processing()
def test_streaming_document_processing()
```

### 4. Regression Tests
```python
# test_regression.py - NEW FILE
def test_issue_001_table_masking_bug()
def test_issue_002_unicode_boundary_error()
def test_issue_003_memory_leak_large_docs()
```

## Coverage Improvement Plan

### Phase 1: Critical Gaps (Week 1)
- [ ] Add 15 tests for presidio_adapter internals
- [ ] Add 10 tests for conflict resolution
- [ ] Add 8 tests for unmasking accuracy
- [ ] Add 5 tests for anchor resolution

### Phase 2: Stability (Week 2)
- [ ] Add 10 property-based tests
- [ ] Add 5 performance benchmarks
- [ ] Add 8 integration scenarios
- [ ] Add error handling tests

### Phase 3: Edge Cases (Week 3)
- [ ] Add unicode edge cases
- [ ] Add boundary condition tests
- [ ] Add concurrent operation tests
- [ ] Add memory limit tests

## Test Data Requirements

### Missing Test Data Needed:
1. **Large documents** (>10MB) for performance testing
2. **Multilingual documents** for unicode testing
3. **Complex tables** with nested structures
4. **Corrupted files** for error handling
5. **Edge case documents**:
   - Empty documents
   - Documents with only tables
   - Documents with special characters
   - RTL text documents

### Test Data Generation Strategy:
```python
# Create test data generators
def generate_large_document(size_mb: int) -> DoclingDocument
def generate_multilingual_document() -> DoclingDocument
def generate_complex_table_document() -> DoclingDocument
def generate_corrupted_document() -> bytes
```

## Mutation Testing Targets

Priority files for mutation testing:
1. `masking/applicator.py` - Critical logic
2. `unmasking/anchor_resolver.py` - Complex algorithms
3. `core/normalization.py` - String manipulation
4. `masking/presidio_adapter.py` - Core functionality

## Success Metrics

### Coverage Targets:
- Line coverage: >95%
- Branch coverage: >90%
- Mutation coverage: >80%

### Quality Metrics:
- Zero flaky tests
- All tests run in <30 seconds (except benchmarks)
- Clear test names and documentation
- Isolated unit tests (proper mocking)

## Implementation Priority

### Must Have (Before ANY Refactoring):
1. Presidio adapter internal tests (15 tests)
2. Conflict resolution tests (10 tests)
3. Unmasking accuracy tests (8 tests)
4. Basic property tests (5 tests)

### Should Have (During Refactoring):
1. Performance benchmarks
2. Integration scenarios
3. Error handling tests
4. CLI tests

### Nice to Have (Post-Refactoring):
1. Mutation testing
2. Fuzz testing
3. Stress testing
4. Security testing

## Estimated Effort

| Task | Tests | Hours |
|------|--------|-------|
| Critical gaps | 38 | 20 |
| Property tests | 10 | 8 |
| Benchmarks | 5 | 6 |
| Integration | 8 | 10 |
| Edge cases | 15 | 12 |
| **Total** | **76** | **56** |

---
*Total gap: 76 test cases*
*Estimated effort: 7-8 developer days*
*Priority: CRITICAL before refactoring*