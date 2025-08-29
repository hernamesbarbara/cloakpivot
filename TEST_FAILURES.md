# Test Failures

The following tests are currently failing:

## tests/test_batch_cli.py::TestBatchCLI
- test_batch_mask_basic
- test_batch_mask_with_failures
- test_batch_unmask_basic
- test_batch_analyze_basic
- test_batch_keyboard_interrupt
- test_load_masking_policy_yaml
- test_load_masking_policy_enhanced_loader

## tests/test_batch_processing.py::TestBatchProcessor
- test_batch_processor_creation
- test_calculate_cloakmap_path_mask
- test_filter_existing_files

## tests/test_batch_processing.py::TestBatchProcessorIntegration
- test_process_batch_mock_operations
- test_process_batch_with_failures