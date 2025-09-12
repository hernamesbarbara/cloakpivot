"""Document processing integration with DocPivot for CloakPivot.

This module provides document processing capabilities that work seamlessly
with DocPivot and DoclingDocument formats, including full support for:

- DoclingDocument v1.2.0, v1.3.0, v1.4.0 (global charspan offsets)
- DoclingDocument v1.7.0+ (segment-local charspan offsets)

The TextExtractor handles version differences transparently by building
its own segment mappings with global offsets, ensuring consistent masking
operations across all document versions.
"""

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
