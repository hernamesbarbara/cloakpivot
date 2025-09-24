# Dead/Redundant/Legacy Code Report
*Generated: 2025-09-24*

## Executive Summary
Analysis identified **9 dead code instances** across the codebase with low overall risk. Most issues relate to removed encryption functionality and unused variables required by interfaces.

## Dead Code Inventory

### 1. Unreachable Code [HIGH PRIORITY - REMOVE]

#### FIND-0021: Unreachable code in CloakMap.save_encrypted
**Location**: `cloakpivot/core/cloakmap.py:572-580`
```python
raise NotImplementedError("Encryption has been removed in v2.0")
# Lines 572-580 are unreachable
path = Path(file_path)
path.parent.mkdir(parents=True, exist_ok=True)
# ... more dead code
```
**Action**: Delete lines 572-580
**Risk**: NONE - Code is unreachable
**Test Impact**: None

#### FIND-0022: Unreachable code in CloakMap.load_encrypted
**Location**: `cloakpivot/core/cloakmap.py:607-629`
```python
raise NotImplementedError("Encrypted loading has been removed...")
return None  # Line 607 - unreachable
# Lines 608-629 are unreachable
```
**Action**: Delete lines 607-629
**Risk**: NONE - Code is unreachable
**Test Impact**: None

### 2. Unused Variables [MEDIUM PRIORITY - DEPRECATE]

#### FIND-0023: Unused key_version parameters
**Location**: `cloakpivot/core/cloakmap.py:523,550`
**Pattern**: `key_version` parameter in encrypt/save_encrypted methods
**Reason**: Encryption functionality removed in v2.0
**Action**: Add deprecation warning, remove in v3.0
**Risk**: MEDIUM - Part of public API
**Migration**:
```python
# Add to methods:
if key_version is not None:
    warnings.warn(
        "key_version parameter is deprecated and will be removed in v3.0",
        DeprecationWarning,
        stacklevel=2
    )
```

#### FIND-0024: Unused signal handler parameters
**Location**: `cloakpivot/core/presidio_mapper.py:12`
```python
def timeout_handler(signum, frame):  # Both params unused
    raise TimeoutError("Import timed out")
```
**Action**: Rename to `_signum, _frame` to indicate intentional non-use
**Risk**: NONE - Interface requirement
**Test Impact**: None

#### FIND-0025: Unused variable in loaders.py
**Location**: `cloakpivot/loaders.py:257`
**Variable**: `conf_hash` (unused after assignment)
**Action**: Remove assignment or use for caching
**Risk**: LOW
**Test Impact**: None

#### FIND-0026: Unused variable in masking/applicator.py
**Location**: `cloakpivot/masking/applicator.py:132`
**Variable**: `primary_strategy` (assigned but never used)
**Action**: Remove or implement intended logic
**Risk**: MEDIUM - May indicate incomplete implementation
**Test Impact**: test_masking_workflow.py

#### FIND-0027: Unused variable in unmasking/engine.py
**Location**: `cloakpivot/unmasking/engine.py:239`
**Variable**: `original_document` (assigned but never used)
**Action**: Remove assignment
**Risk**: LOW
**Test Impact**: test_unmasking tests

### 3. Legacy/Compatibility Code [LOW PRIORITY - INVESTIGATE]

#### FIND-0028: Presidio import timeout workaround
**Location**: `cloakpivot/core/presidio_mapper.py:10-30`
```python
# Complex signal-based timeout for Presidio imports
if TYPE_CHECKING:
    from presidio_anonymizer.entities import OperatorConfig
else:
    import signal
    # Timeout logic...
```
**Reason**: Workaround for slow Presidio v2.2.33 imports
**Action**: Test with latest Presidio version, remove if fixed
**Risk**: MEDIUM - May be needed in some environments
**Test Impact**: All Presidio-dependent tests

#### FIND-0029: docpivot compatibility layer
**Location**: `cloakpivot/compat.py` (entire file)
**Purpose**: Migration support from docpivot v2.0.1
**Usage**:
- Used in 1 example: `examples/docling_to_lexical_workflow.py`
- Used in 1 test: `tests/unit/test_compat.py`
- Referenced in README.md
**Action**: Consider deprecation timeline
**Risk**: LOW - Limited usage
**Migration Path**:
1. Add deprecation warning in v2.1
2. Move to separate package in v3.0
3. Remove in v4.0

### 4. Wrapper Utilities [ASSESS USAGE]

#### FIND-0030: CloakedDocument wrapper
**Location**: `cloakpivot/wrappers.py:10-183`
**Purpose**: Convenience wrapper for masked documents
**Usage**:
- Used in registration.py for automatic method registration
- Has dedicated tests in test_wrappers.py
- Not widely used in examples
**Assessment**: KEEP - Active feature with tests
**Recommendation**: Add more examples to demonstrate value

## Removal Priority Queue

### Immediate (Safe to Remove Now)
1. Unreachable code in cloakmap.py (lines 572-580, 607-629)
2. Unused variable assignments (conf_hash, original_document)

### Next Minor Version (v2.1)
1. Add deprecation warnings for:
   - key_version parameters
   - compat.py module
2. Rename signal handler parameters

### Next Major Version (v3.0)
1. Remove deprecated encryption parameters
2. Extract compat.py to separate package
3. Review Presidio import workaround

## Impact Summary

| Component | Dead Code Lines | Risk | Test Files Affected |
|-----------|-----------------|------|---------------------|
| core/cloakmap.py | ~30 | NONE | None (unreachable) |
| core/presidio_mapper.py | ~20 | MEDIUM | All Presidio tests |
| compat.py | 49 | LOW | test_compat.py |
| Various unused vars | 5 | LOW | Minimal |

**Total Lines to Remove**: ~54 (immediate)
**Total Lines to Deprecate**: ~69 (gradual)

## Validation Checklist

Before removing any code:
- [ ] Run full test suite
- [ ] Check for dynamic imports/reflection
- [ ] Verify no external packages depend on removed code
- [ ] Update documentation
- [ ] Add deprecation notices where needed
- [ ] Create migration guide for deprecated features

---
*End of Dead/Redundant/Legacy Report*