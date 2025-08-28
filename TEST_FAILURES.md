# Test Failures

## Failed Tests:
- tests/test_cli.py::TestMaskCommand::test_mask_command_success
- tests/test_cli.py::TestMaskCommand::test_mask_with_policy_file_missing_yaml
- tests/test_cli.py::TestUnmaskCommand::test_unmask_command_success
- tests/test_cli.py::TestErrorHandling::test_import_error_handling
- tests/test_cli.py::TestErrorHandling::test_general_exception_handling
- tests/test_cli.py::TestProgressReporting::test_progress_bars_shown

## Error Summary:
All failures are AttributeError: module 'cloakpivot.cli.main' does not have the attribute 'X' where X is:
- MaskingEngine
- yaml
- DocumentProcessor

These are import/mocking issues in the test suite where the tests are trying to mock attributes that don't exist in the main CLI module.