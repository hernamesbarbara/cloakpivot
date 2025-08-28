"""Document processing integration with DocPivot for CloakPivot."""

from .extractor import TextExtractor, TextSegment
from .mapper import AnchorMapper, NodeReference
from .processor import DocumentProcessor

__all__ = [
    "DocumentProcessor",
    "TextExtractor",
    "TextSegment",
    "AnchorMapper",
    "NodeReference",
]
