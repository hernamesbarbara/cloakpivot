"""Processing and analysis functionality for CloakPivot core."""

# Import specific functionality for backward compatibility
from .analyzer import (
    AnalyzerConfig,
    AnalyzerEngineWrapper,
    EntityDetectionResult,
    RecognizerRegistry,
)
from .cloakmap_enhancer import CloakMapEnhancer
from .detection import DocumentAnalysisResult, EntityDetectionPipeline, SegmentAnalysisResult
from .normalization import (
    ConflictResolutionConfig,
    ConflictResolutionStrategy,
    EntityGroup,
    EntityNormalizer,
    EntityPriority,
    NormalizationResult,
)
from .presidio_common import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    ENTITY_TYPE_MAPPING,
    build_statistics,
    create_entity_mapping,
    filter_overlapping_entities,
    get_confidence_threshold,
    normalize_entity_type,
    operator_result_to_dict,
    recognizer_result_to_dict,
    validate_entity_boundaries,
    validate_presidio_version,
)
from .presidio_mapper import StrategyToOperatorMapper
from .surrogate import FormatPattern, SurrogateGenerator, SurrogateQualityMetrics

__all__ = [
    # From detection.py
    "SegmentAnalysisResult",
    "DocumentAnalysisResult",
    "EntityDetectionPipeline",

    # From analyzer.py
    "AnalyzerConfig",
    "RecognizerRegistry",
    "EntityDetectionResult",
    "AnalyzerEngineWrapper",

    # From presidio_mapper.py
    "StrategyToOperatorMapper",

    # From presidio_common.py
    "ENTITY_TYPE_MAPPING",
    "DEFAULT_CONFIDENCE_THRESHOLDS",
    "normalize_entity_type",
    "get_confidence_threshold",
    "validate_presidio_version",
    "operator_result_to_dict",
    "recognizer_result_to_dict",
    "build_statistics",
    "filter_overlapping_entities",
    "validate_entity_boundaries",
    "create_entity_mapping",

    # From normalization.py
    "ConflictResolutionStrategy",
    "EntityPriority",
    "ConflictResolutionConfig",
    "NormalizationResult",
    "EntityGroup",
    "EntityNormalizer",

    # From surrogate.py
    "FormatPattern",
    "SurrogateQualityMetrics",
    "SurrogateGenerator",

    # From cloakmap_enhancer.py
    "CloakMapEnhancer",
]
