# DRY (Don't Repeat Yourself) Analysis Report
*Generated: 2025-09-24*

## Executive Summary
Significant code duplication found across masking and unmasking modules, with estimated **120-150 lines** that could be deduplicated through strategic refactoring.

## Major Duplication Patterns Identified

### 1. Presidio Adapter Duplication [FIND-0005]
**Files**:
- `masking/presidio_adapter.py` (1310 LOC)
- `unmasking/presidio_adapter.py` (426 LOC)

**Duplicate Patterns Found**:

| Pattern | Function/Method | Est. Lines | Risk |
|---------|----------------|------------|------|
| Version Detection | `_get_presidio_version()` | 15-20 | LOW |
| Error Handling | Error wrapping patterns | 25-30 | LOW |
| Operator Processing | `process_operator_results()` | 40-50 | MEDIUM |
| Entity Type Mapping | Entity type conversions | 20-25 | LOW |
| Statistics Building | `build_processing_stats()` | 15-20 | LOW |
| Text Processing | Segment handling | 20-25 | MEDIUM |

**Total Estimated Savings**: 135-170 lines

### 2. Engine Pattern Duplication [FIND-0011]
**Files**:
- `masking/engine.py` (611 LOC)
- `unmasking/engine.py` (458 LOC)

**Common Patterns**:
- Document validation logic (~30 lines)
- Error handling wrappers (~25 lines)
- Configuration processing (~20 lines)
- Result building (~15 lines)

**Total Estimated Savings**: 90 lines

### 3. Document Processing Duplication [FIND-0012]
**Files**:
- `document/extractor.py` (458 LOC)
- `document/mapper.py` (424 LOC)
- `document/processor.py` (various)

**Common Patterns**:
- Text segment iteration (~20 lines)
- Boundary calculation (~15 lines)
- Validation checks (~10 lines)

**Total Estimated Savings**: 45 lines

### 4. Normalization Pattern Duplication [FIND-0013]
**Files**:
- `core/normalization.py` (618 LOC)
- `core/validation.py` (515 LOC)

**Common Patterns**:
- String normalization utilities (~30 lines)
- Input validation patterns (~25 lines)
- Error formatting (~15 lines)

**Total Estimated Savings**: 70 lines

## Recommended Extraction Targets

### Priority 1: Create `core/presidio_common.py`
Extract shared Presidio utilities:
```python
# Proposed structure
core/presidio_common.py:
    - get_presidio_version()
    - PresidioErrorWrapper
    - OperatorResultProcessor
    - EntityTypeMapper
    - ProcessingStatsBuilder
    - TextSegmentHandler
```

**Impact**:
- Call sites: ~15-20 locations
- Test files affected: 6
- Risk: LOW (mostly utility functions)

### Priority 2: Create `core/engine_common.py`
Extract shared engine patterns:
```python
# Proposed structure
core/engine_common.py:
    - DocumentValidator
    - EngineErrorHandler
    - ConfigurationProcessor
    - ResultBuilder
```

**Impact**:
- Call sites: 8-10 locations
- Test files affected: 4
- Risk: MEDIUM (affects public API behavior)

### Priority 3: Create `document/common.py`
Extract shared document utilities:
```python
# Proposed structure
document/common.py:
    - TextSegmentIterator
    - BoundaryCalculator
    - DocumentValidator
```

**Impact**:
- Call sites: 5-7 locations
- Test files affected: 3
- Risk: LOW

## DRY Violation Summary Table

| Finding ID | Files | Pattern | Lines to Save | Priority | Risk |
|------------|-------|---------|---------------|----------|------|
| FIND-0005 | presidio adapters | Presidio utilities | 150 | HIGH | LOW |
| FIND-0011 | engines | Engine patterns | 90 | MEDIUM | MEDIUM |
| FIND-0012 | document/* | Document processing | 45 | LOW | LOW |
| FIND-0013 | normalization/validation | String utilities | 70 | MEDIUM | LOW |
| FIND-0014 | error_handling/exceptions | Error patterns | 35 | LOW | LOW |
| FIND-0015 | anchors/anchor_resolver | Anchor logic | 40 | LOW | MEDIUM |

**Total Potential Line Reduction**: ~430 lines (2.4% of codebase)

## Implementation Strategy

### Phase 1: Low-Risk Utilities (Week 1)
1. Extract Presidio version detection
2. Extract stats building utilities
3. Extract entity type mappings

### Phase 2: Medium-Risk Consolidation (Week 2)
1. Extract operator processing logic
2. Extract document validation
3. Extract error wrapping patterns

### Phase 3: High-Impact Refactoring (Week 3)
1. Consolidate engine patterns
2. Unify normalization utilities
3. Merge anchor processing logic

## Testing Requirements
- Add unit tests for each extracted module before moving code
- Ensure 100% backward compatibility through re-exports
- Performance benchmarks before/after extraction

## Migration Path
1. Create new common modules with extracted code
2. Add comprehensive unit tests
3. Update original modules to import from common
4. Add deprecation notices where needed
5. Update all call sites
6. Remove duplicated code

---
*End of DRY Analysis Report*