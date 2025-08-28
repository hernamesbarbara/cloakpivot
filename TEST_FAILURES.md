# Test Failures

## Failed Tests:
- tests/test_analyzer.py::TestAnalyzerEngineWrapper::test_lazy_initialization
- tests/test_analyzer.py::TestAnalyzerEngineWrapper::test_configuration_from_policy
- tests/test_detection.py::TestEntityDetectionPipeline::test_empty_segment_handling
- tests/test_normalization.py::TestEntityNormalizer::test_highest_confidence_resolution
- tests/test_normalization.py::TestEntityNormalizer::test_preserve_high_confidence
- tests/test_normalization.py::TestEntityNormalizer::test_group_creation
- tests/test_normalization.py::TestEntityNormalizer::test_validation
- tests/test_normalization.py::TestEdgeCases::test_complex_overlaps

## Error Tests:
- tests/test_detection.py::TestEntityDetectionPipeline::test_analyze_document
- tests/test_detection.py::TestEntityDetectionPipeline::test_analyze_text_segments
- tests/test_detection.py::TestEntityDetectionPipeline::test_policy_filtering
- tests/test_detection.py::TestEntityDetectionPipeline::test_error_handling
- tests/test_detection.py::TestSegmentAnalysisResult::test_result_properties
- tests/test_detection.py::TestDocumentAnalysisResult::test_result_aggregation
- tests/test_detection.py::TestDocumentAnalysisResult::test_get_all_entities
- tests/test_detection.py::TestAnchorMapping::test_map_entities_to_anchors
- tests/test_detection.py::TestAnchorMapping::test_anchor_mapping_with_multiple_segments