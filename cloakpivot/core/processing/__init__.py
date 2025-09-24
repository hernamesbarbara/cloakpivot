"""Processing and analysis functionality for CloakPivot core."""

# Import all processing-related functionality for backward compatibility
from .detection import *
from .analyzer import *
from .presidio_mapper import *
from .presidio_common import *
from .normalization import *
from .surrogate import *
from .cloakmap_enhancer import *

__all__ = [
    # From detection.py
    "EntityDetector",
    "detect_entities",

    # From analyzer.py
    "TextAnalyzer",
    "analyze_document",

    # From presidio_mapper.py
    "PresidioMapper",

    # From presidio_common.py
    "PresidioCommon",

    # From normalization.py
    "TextNormalizer",
    "normalize_text",

    # From surrogate.py
    "SurrogateGenerator",

    # From cloakmap_enhancer.py
    "CloakMapEnhancer",
]