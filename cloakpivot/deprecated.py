"""Deprecated APIs with migration warnings.

This module provides backward compatibility for old CloakPivot APIs.
All classes and functions here are deprecated and will be removed in v1.0.0.
Please migrate to the new CloakEngine API.
"""

import warnings
from typing import Optional, List, Dict, Any

from docling_core.types import DoclingDocument

from cloakpivot.engine import CloakEngine
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.cloakmap import CloakMap


def _deprecation_warning(old_name: str, new_name: str, version: str = "1.0.0"):
    """Issue a deprecation warning with migration guidance."""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in v{version}. "
        f"Use {new_name} instead. "
        f"See migration guide at: https://github.com/example/cloakpivot/blob/main/MIGRATION.md",
        DeprecationWarning,
        stacklevel=3
    )


class MaskingEngine:
    """Deprecated. Use CloakEngine instead.

    This class provides backward compatibility for the old MaskingEngine API.
    All methods delegate to the new CloakEngine.

    Migration example:
        # Old way (deprecated)
        engine = MaskingEngine(analyzer_config, policy)
        result = engine.mask(doc, text_result, entities)

        # New way
        engine = CloakEngine(analyzer_config, policy)
        result = engine.mask_document(doc, entities)
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning."""
        _deprecation_warning("MaskingEngine", "CloakEngine")
        self._engine = CloakEngine(*args, **kwargs)

    def mask(self, document: DoclingDocument, text_result: Any,
             entities: Optional[List[str]] = None,
             policy: Optional[MaskingPolicy] = None) -> Any:
        """Deprecated mask method."""
        _deprecation_warning("MaskingEngine.mask", "CloakEngine.mask_document")

        # Convert old API call to new API
        result = self._engine.mask_document(document, entities, policy)

        # Return in old format for compatibility
        from cloakpivot.masking.result import MaskingResult
        return MaskingResult(
            document=result.document,
            cloakmap=result.cloakmap
        )


class UnmaskingEngine:
    """Deprecated. Use CloakEngine instead.

    This class provides backward compatibility for the old UnmaskingEngine API.

    Migration example:
        # Old way (deprecated)
        engine = UnmaskingEngine()
        unmasked = engine.unmask(doc, cloakmap)

        # New way
        engine = CloakEngine()
        unmasked = engine.unmask_document(doc, cloakmap)
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning."""
        _deprecation_warning("UnmaskingEngine", "CloakEngine")
        self._engine = CloakEngine(*args, **kwargs)

    def unmask(self, document: DoclingDocument, cloakmap: CloakMap) -> DoclingDocument:
        """Deprecated unmask method."""
        _deprecation_warning("UnmaskingEngine.unmask", "CloakEngine.unmask_document")
        return self._engine.unmask_document(document, cloakmap)


class TextExtractor:
    """Deprecated. Functionality now integrated into CloakEngine.

    The TextExtractor is now automatically used internally by CloakEngine.

    Migration example:
        # Old way (deprecated)
        extractor = TextExtractor()
        text_result = extractor.extract_text(doc)
        engine = MaskingEngine()
        result = engine.mask(doc, text_result)

        # New way (extraction is automatic)
        engine = CloakEngine()
        result = engine.mask_document(doc)
    """

    def __init__(self):
        """Initialize with deprecation warning."""
        _deprecation_warning(
            "TextExtractor",
            "CloakEngine (text extraction is now automatic)"
        )
        from cloakpivot.document.extractor import TextExtractor as _TextExtractor
        self._extractor = _TextExtractor()

    def extract_text(self, document: DoclingDocument) -> Any:
        """Deprecated extract_text method."""
        _deprecation_warning(
            "TextExtractor.extract_text",
            "CloakEngine.mask_document (extraction is automatic)"
        )
        return self._extractor.extract_text(document)


class DocumentProcessor:
    """Deprecated. Use CloakEngine instead.

    Migration example:
        # Old way (deprecated)
        processor = DocumentProcessor()
        processed = processor.process(doc)

        # New way
        engine = CloakEngine()
        result = engine.mask_document(doc)
    """

    def __init__(self):
        """Initialize with deprecation warning."""
        _deprecation_warning("DocumentProcessor", "CloakEngine")
        from cloakpivot.document.processor import DocumentProcessor as _Processor
        self._processor = _Processor()

    def process(self, document: DoclingDocument) -> DoclingDocument:
        """Deprecated process method."""
        _deprecation_warning(
            "DocumentProcessor.process",
            "CloakEngine.mask_document"
        )
        return self._processor.process(document)


class BatchProcessor:
    """Deprecated. Use CloakEngine with iteration instead.

    Migration example:
        # Old way (deprecated)
        processor = BatchProcessor()
        results = processor.process_batch(documents)

        # New way
        engine = CloakEngine()
        results = [engine.mask_document(doc) for doc in documents]
    """

    def __init__(self):
        """Initialize with deprecation warning."""
        _deprecation_warning(
            "BatchProcessor",
            "CloakEngine with iteration or parallel processing"
        )

    def process_batch(self, documents: List[DoclingDocument]) -> List[Any]:
        """Deprecated batch processing."""
        _deprecation_warning(
            "BatchProcessor.process_batch",
            "[engine.mask_document(doc) for doc in documents]"
        )
        engine = CloakEngine()
        return [engine.mask_document(doc) for doc in documents]


# Deprecated configuration classes

class AnalyzerConfig:
    """Deprecated. Pass configuration as dictionary to CloakEngine.

    Migration example:
        # Old way (deprecated)
        config = AnalyzerConfig(languages=['en'], threshold=0.8)
        engine = MaskingEngine(config)

        # New way
        engine = CloakEngine(analyzer_config={
            'languages': ['en'],
            'confidence_threshold': 0.8
        })
    """

    def __init__(self, **kwargs):
        """Initialize with deprecation warning."""
        _deprecation_warning(
            "AnalyzerConfig",
            "dictionary passed to CloakEngine"
        )
        self._config = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for new API."""
        return self._config


class PolicyLoader:
    """Deprecated. Use cloakpivot.defaults functions instead.

    Migration example:
        # Old way (deprecated)
        policy = PolicyLoader.load_from_file('policy.yaml')

        # New way
        from cloakpivot.defaults import get_default_policy
        policy = get_default_policy()
    """

    @staticmethod
    def load_from_file(path: str) -> MaskingPolicy:
        """Deprecated policy loading."""
        _deprecation_warning(
            "PolicyLoader.load_from_file",
            "cloakpivot.defaults.get_policy_preset or custom loading"
        )
        from cloakpivot.core.policy_loader import PolicyLoader as _Loader
        return _Loader.load_from_file(path)

    @staticmethod
    def get_default_policy() -> MaskingPolicy:
        """Deprecated default policy getter."""
        _deprecation_warning(
            "PolicyLoader.get_default_policy",
            "cloakpivot.defaults.get_default_policy"
        )
        from cloakpivot.defaults import get_default_policy
        return get_default_policy()


# Deprecated helper functions

def mask_document(
    document: DoclingDocument,
    entities: Optional[List[str]] = None,
    policy: Optional[MaskingPolicy] = None
) -> Any:
    """Deprecated standalone masking function.

    Migration example:
        # Old way (deprecated)
        result = mask_document(doc, entities=['EMAIL'])

        # New way
        engine = CloakEngine()
        result = engine.mask_document(doc, entities=['EMAIL'])
    """
    _deprecation_warning("mask_document", "CloakEngine.mask_document")
    engine = CloakEngine()
    return engine.mask_document(document, entities, policy)


def unmask_document(
    document: DoclingDocument,
    cloakmap: CloakMap
) -> DoclingDocument:
    """Deprecated standalone unmasking function.

    Migration example:
        # Old way (deprecated)
        unmasked = unmask_document(doc, cloakmap)

        # New way
        engine = CloakEngine()
        unmasked = engine.unmask_document(doc, cloakmap)
    """
    _deprecation_warning("unmask_document", "CloakEngine.unmask_document")
    engine = CloakEngine()
    return engine.unmask_document(document, cloakmap)