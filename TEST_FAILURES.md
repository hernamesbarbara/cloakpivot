# Test Failures

## Failed Tests:
- tests/test_coverage.py::TestCoverageAnalyzer::test_analyze_coverage_with_invalid_anchor_entries
- tests/test_coverage.py::TestCoverageAnalyzer::test_analyze_coverage_with_invalid_positions
- tests/test_coverage.py::TestCoverageAnalyzer::test_analyze_coverage_with_invalid_range
- tests/test_coverage.py::TestCoverageAnalyzer::test_analyze_coverage_with_empty_entity_type
- tests/test_diagnostics.py::TestDiagnosticsCollector::test_collect_masking_statistics_with_null_stats
- tests/test_diagnostics.py::TestDiagnosticsCollector::test_collect_performance_metrics_with_missing_fields

## Error Tests:
- tests/test_diagnostics.py::TestDiagnosticsCollector::test_collect_masking_statistics_basic
- tests/test_diagnostics.py::TestDiagnosticsCollector::test_collect_masking_statistics_with_confidence_distribution
- tests/test_diagnostics.py::TestDiagnosticsCollector::test_collect_masking_statistics_strategy_breakdown
- tests/test_diagnostics.py::TestDiagnosticsCollector::test_collect_performance_metrics
- tests/test_diagnostics.py::TestDiagnosticsCollector::test_generate_comprehensive_report