# Coverage Gap TODOs - Working Document

## Current Status (2025-09-22)
- **Previous Coverage**: 32.39% (with `branch = true` enabled)
- **Session 1 Coverage**: 41.31% (after initial tests)
- **Session 2 Coverage**: 43.89% (after more tests)
- **Current Coverage**: 49.49% (after all new tests)
- **Target Coverage**: 60%
- **Gap to Close**: 10.51%
- **Total Progress Made**: +17.10%

## Work Completed
### Test Files Added (Previous Session)
1. ✅ `tests/unit/test_registration.py` - Registration system tests
2. ✅ `tests/unit/test_wrappers.py` - CloakedDocument wrapper tests (some failures need fixing)
3. ✅ `tests/unit/test_table_cell_boundary_fix.py` - Enhanced boundary validation branch coverage
4. ✅ `tests/unit/test_engine_coverage.py` - Basic engine tests (many failures need fixing)
5. ✅ `tests/unit/test_unmasking_engine_coverage.py` - Unmasking engine tests (some failures need fixing)

### Test Files Added (This Session - 2025-09-22)
6. ✅ `tests/unit/test_core_exceptions.py` - Complete exception hierarchy tests (138 tests, all passing)
7. ✅ `tests/unit/test_core_config.py` - Performance configuration tests (all passing)
8. ✅ `tests/unit/test_imports.py` - Module import tests (all passing)
9. ✅ `tests/unit/test_cli_main.py` - CLI main command tests using Click's CliRunner
10. ✅ `tests/unit/test_cli_config.py` - CLI configuration management tests
11. ✅ `tests/unit/test_formats_serialization.py` - Serialization tests (added Session 2)
12. ✅ `tests/unit/test_formats_registry.py` - Format registry tests (added Session 2)
13. ✅ `tests/unit/test_loaders.py` - Loader validation tests (added Session 2)
14. ✅ `tests/unit/test_model_info.py` - Model characteristics tests (added Session 2)
15. ✅ `tests/unit/test_compat.py` - Compatibility module tests (added Session 2)
16. ✅ `tests/unit/test_policy_loader.py` - Policy loader tests (added Session 2)
17. ✅ `tests/unit/test_masking_applicator.py` - Strategy applicator tests (added Session 2)

### Issues to Fix in Existing Tests
1. **test_wrappers.py failures**:
   - Fix AnchorEntry.create_from_detection() parameter issues
   - Remove dependencies on non-existent serializer modules

2. **test_engine_coverage.py failures**:
   - CloakEngine doesn't accept 'languages' parameter
   - Builder pattern methods don't match actual API
   - _adapter and _unmasking_adapter attributes don't exist
   - MaskingResult not exported from engine module
   - ConflictResolutionConfig parameter mismatches

3. **test_unmasking_engine_coverage.py failures**:
   - UnmaskingEngine doesn't have unmask_text method
   - Fix invalid cloakmap test expectations

## High-Impact Coverage Opportunities

### Zero Coverage Modules (Priority 1)
These modules have 0% coverage and would provide the biggest boost:

1. **CLI modules** (231 lines uncovered) ✅ ADDRESSED
   - ✅ `cloakpivot/cli/main.py` (93 lines) - Now has test coverage
   - ✅ `cloakpivot/cli/config.py` (136 lines) - Now has test coverage
   - `cloakpivot/cli/__init__.py` (2 lines)

2. **Core configuration** (419 lines uncovered) ✅ PARTIALLY ADDRESSED
   - ✅ `cloakpivot/core/config.py` (102 lines) - Now has test coverage
   - `cloakpivot/core/model_info.py` (63 lines) - Still needs tests
   - `cloakpivot/core/policy_loader.py` (254 lines) - Still needs tests

3. **Formats/Serialization** (204 lines uncovered)
   - `cloakpivot/formats/serialization.py` (124 lines)
   - `cloakpivot/formats/registry.py` (77 lines)
   - `cloakpivot/formats/__init__.py` (3 lines)

4. **Other zero coverage** (313 lines uncovered)
   - `cloakpivot/loaders.py` (163 lines)
   - `cloakpivot/unmasking/presidio_adapter.py` (135 lines)
   - `cloakpivot/compat.py` (14 lines)
   - `cloakpivot/utils/__init__.py` (1 line)

### Low Coverage Modules (Priority 2)
These have some coverage but need significant improvement:

1. **Masking applicator** - 7.33% coverage (345 lines uncovered)
   - `cloakpivot/masking/applicator.py`

2. **Unmasking modules** - 8-13% coverage
   - `cloakpivot/unmasking/document_unmasker.py` - 8.21% (231 lines uncovered)
   - `cloakpivot/unmasking/cloakmap_loader.py` - 12.99% (123 lines uncovered)

3. **Core exceptions** - ✅ NOW HAS FULL COVERAGE
   - ✅ `cloakpivot/core/exceptions.py` - Complete test coverage achieved

4. **Registration** - 17.46% coverage (34 lines uncovered)
   - `cloakpivot/registration.py`

5. **Masking engine** - 28.57% coverage (137 lines uncovered)
   - `cloakpivot/masking/engine.py`

## Strategy to Close the Gap

### Quick Wins (Est. +15% coverage)
1. **Fix all failing tests** - This alone should add ~5% coverage
2. **Add simple tests for zero-coverage utility modules**:
   - Test imports and basic functionality for `__init__.py` files
   - Test exception classes in `core/exceptions.py`
   - Test configuration loading in `core/config.py`

### Medium Effort (Est. +10% coverage)
1. **Add CLI tests using Click's testing utilities**:
   ```python
   from click.testing import CliRunner
   from cloakpivot.cli.main import cli

   def test_cli_help():
       runner = CliRunner()
       result = runner.invoke(cli, ['--help'])
       assert result.exit_code == 0
   ```

2. **Add serialization tests**:
   - Test JSON/YAML serialization of CloakMap
   - Test format registry functionality

3. **Add loader tests**:
   - Test document loading functionality
   - Test policy loading

### High Effort (Est. +5% coverage)
1. **Improve masking/unmasking engine coverage**:
   - Test the actual implementation paths
   - Add integration tests that exercise real workflows

2. **Add branch coverage for complex logic**:
   - Test all conditional branches in presidio_adapter
   - Test error handling paths

## Specific Test Files to Create

### Priority 1 - Quick fixes (Today)
1. `tests/unit/test_core_exceptions.py` - Test all exception classes
2. `tests/unit/test_core_config.py` - Test configuration loading
3. `tests/unit/test_imports.py` - Test all module imports work

### Priority 2 - CLI and utilities (Next)
1. `tests/unit/test_cli_main.py` - Test CLI commands
2. `tests/unit/test_cli_config.py` - Test CLI configuration
3. `tests/unit/test_formats_serialization.py` - Test serialization
4. `tests/unit/test_formats_registry.py` - Test format registry

### Priority 3 - Complex modules
1. `tests/unit/test_masking_applicator.py` - Test masking application
2. `tests/unit/test_loaders.py` - Test various loaders
3. `tests/unit/test_policy_loader.py` - Test policy loading

## Branch Coverage Specific Issues
With `branch = true` enabled, we need to test:
- Both True and False paths for every if statement
- All cases in if/elif/else chains
- Both success and failure paths for try/except blocks
- All iterations and early exits in loops

## Notes
- The branch coverage requirement effectively doubles the testing needed for conditional code
- Focus on high-line-count modules with zero coverage first
- Integration tests that exercise multiple modules provide better coverage ROI
- Consider using parametrized tests to cover multiple branches efficiently

## Next Immediate Steps (Updated 2025-09-22)
To reach 60% coverage (need +10.51% more):

### Priority 1 - Fix Failing Tests (~+3-5% coverage)
1. Fix test_wrappers.py failures
2. Fix test_engine_coverage.py failures
3. Fix test_unmasking_engine_coverage.py failures
4. Fix minor test_cli_config.py failures

### Priority 2 - Zero Coverage Modules (~+8-10% coverage)
1. Create `test_formats_serialization.py` - Test JSON/YAML serialization (124 lines)
2. Create `test_formats_registry.py` - Test format registry (77 lines)
3. Create `test_policy_loader.py` - Test policy loading (254 lines)
4. Create `test_loaders.py` - Test document loaders (163 lines)

### Priority 3 - Low Coverage Modules (~+5-7% coverage)
1. Create `test_masking_applicator.py` - Currently 7.33% coverage (345 lines)
2. Enhance `test_unmasking_document_unmasker.py` - Currently 8.21% (231 lines)
3. Create `test_model_info.py` - Test model characteristics (63 lines)

## Command to Check Coverage
```bash
cd /Users/hernamesbarbara/code/github/hernamesbarbara/cloakpivot
make test 2>&1 | grep -E "TOTAL|Required test coverage"
```

## Target Breakdown
To reach 60% coverage from 49.49%:
- Need to cover approximately 680 more lines
- Or reduce uncovered branches by ~170 (with branch coverage)
- Continue focusing on fixing failing tests and adding integration tests

## Coverage Gains This Session
- Added 600+ test cases across 17 new test files
- Improved coverage from 32.39% to 49.49% (+17.10%)
- Addressed high-priority CLI, core configuration, and serialization modules
- Created comprehensive exception, import, policy loader, and masking applicator tests
- Still need +10.51% to reach target 60% coverage