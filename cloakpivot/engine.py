"""CloakEngine - Simplified high-level API for PII masking/unmasking operations."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cloakpivot.engine_builder import CloakEngineBuilder

from docling_core.types import DoclingDocument  # type: ignore[attr-defined]
from presidio_analyzer import AnalyzerEngine

from cloakpivot.core.analyzer import AnalyzerConfig
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.defaults import get_default_policy
from cloakpivot.document.extractor import TextExtractor

# CloakEngineBuilder import moved to avoid circular import
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.unmasking.engine import UnmaskingEngine


@dataclass
class MaskResult:
    """Result of a masking operation."""

    document: DoclingDocument
    cloakmap: CloakMap
    entities_found: int
    entities_masked: int


@dataclass
class TextExtractionResult:
    """Simple wrapper for text extraction results."""

    full_text: str
    segments: list[Any]  # List of TextSegment objects


class CloakEngine:
    """High-level API for PII masking/unmasking operations on DoclingDocument instances.

    Provides a Presidio-like simple interface while encapsulating:
    - TextExtractor initialization
    - AnalyzerEngine configuration
    - MaskingEngine setup
    - Policy management

    Examples:
        # Simple usage with defaults
        engine = CloakEngine()
        result = engine.mask_document(doc)
        original = engine.unmask_document(result.document, result.cloakmap)

        # Advanced configuration
        engine = CloakEngine(
            analyzer_config={'languages': ['es'], 'confidence_threshold': 0.8},
            default_policy=custom_policy
        )
    """

    def __init__(
        self,
        analyzer_config: dict[str, Any] | None = None,
        default_policy: MaskingPolicy | None = None,
        conflict_resolution_config: dict[str, Any] | None = None,
    ):
        """Initialize CloakEngine with sensible defaults for all components.

        Args:
            analyzer_config: Configuration for Presidio AnalyzerEngine
            default_policy: Default masking policy to use
            conflict_resolution_config: Configuration for entity conflict resolution
        """
        # Set up analyzer configuration
        if analyzer_config:
            # Map common parameter names to AnalyzerConfig fields
            config_dict = {}
            if "languages" in analyzer_config:
                config_dict["language"] = analyzer_config["languages"][0]  # Take first language
            elif "language" in analyzer_config:
                config_dict["language"] = analyzer_config["language"]

            if "confidence_threshold" in analyzer_config:
                config_dict["min_confidence"] = analyzer_config["confidence_threshold"]
            elif "min_confidence" in analyzer_config:
                config_dict["min_confidence"] = analyzer_config["min_confidence"]

            # Pass through other fields
            for key in [
                "enabled_recognizers",
                "disabled_recognizers",
                "custom_recognizers",
                "nlp_engine_name",
            ]:
                if key in analyzer_config:
                    config_dict[key] = analyzer_config[key]

            self._analyzer_config = AnalyzerConfig(**config_dict)
        else:
            self._analyzer_config = self._get_default_analyzer_config()

        # Set up default policy
        self._default_policy = default_policy or get_default_policy()

        # Initialize Presidio AnalyzerEngine for entity detection
        self._analyzer = AnalyzerEngine()

        # Initialize engines
        self._text_extractor = TextExtractor()

        # Convert conflict_resolution_config dict to object if needed
        from cloakpivot.core.normalization import ConflictResolutionConfig

        conflict_config_obj = None
        if conflict_resolution_config:
            if isinstance(conflict_resolution_config, dict):
                # Convert dict to ConflictResolutionConfig object
                conflict_config_obj = ConflictResolutionConfig(**conflict_resolution_config)
            else:
                conflict_config_obj = conflict_resolution_config

        self._masking_engine = MaskingEngine(
            resolve_conflicts=bool(conflict_config_obj),
            conflict_resolution_config=conflict_config_obj,
            store_original_text=True,
            use_presidio_engine=True,
        )
        self._unmasking_engine = UnmaskingEngine()

    def mask_document(
        self,
        document: DoclingDocument,
        entities: list[str] | None = None,
        policy: MaskingPolicy | None = None,
    ) -> MaskResult:
        """One-line masking with auto-detection if entities not specified.

        Args:
            document: DoclingDocument to mask
            entities: Optional list of entity types to detect (defaults to common PII types)
            policy: Optional masking policy (uses default if not provided)

        Returns:
            MaskResult containing masked document and CloakMap for reversal

        Examples:
            # Auto-detect all common PII
            result = engine.mask_document(doc)

            # Detect specific entities only
            result = engine.mask_document(doc, entities=['EMAIL_ADDRESS', 'PERSON'])

            # Use custom policy
            result = engine.mask_document(doc, policy=custom_policy)
        """
        # Use provided policy or default
        masking_policy = policy or self._default_policy

        # Extract text from document
        segments = self._text_extractor.extract_text_segments(document)
        full_text = self._text_extractor.extract_full_text(document)

        # Create a simple text result wrapper
        TextExtractionResult(full_text=full_text, segments=segments)

        # If no entities specified, use default common PII types
        entity_types = self._get_default_entities() if entities is None else entities

        # Detect entities using Presidio analyzer
        # Run Presidio analysis
        detected_entities = self._analyzer.analyze(
            text=full_text, entities=entity_types, language=self._analyzer_config.language
        )

        # Call MaskingEngine with correct parameters
        mask_result = self._masking_engine.mask_document(
            document=document,
            entities=detected_entities,
            policy=masking_policy,
            text_segments=segments,
        )

        # Count entities
        entities_found = len(mask_result.cloakmap.anchors)
        # All anchors in the cloakmap represent masked entities
        entities_masked = entities_found

        return MaskResult(
            document=mask_result.masked_document,
            cloakmap=mask_result.cloakmap,
            entities_found=entities_found,
            entities_masked=entities_masked,
        )

    def unmask_document(self, document: DoclingDocument, cloakmap: CloakMap) -> DoclingDocument:
        """Simple unmasking using stored CloakMap.

        Args:
            document: Masked DoclingDocument
            cloakmap: CloakMap containing original values and positions

        Returns:
            DoclingDocument with original PII restored

        Example:
            original = engine.unmask_document(masked_doc, cloakmap)
        """
        result = self._unmasking_engine.unmask_document(document, cloakmap)
        return result.unmasked_document

    @classmethod
    def builder(cls) -> "CloakEngineBuilder":
        """Create a builder for advanced configuration.

        Returns:
            CloakEngineBuilder for fluent configuration

        Example:
            engine = CloakEngine.builder()
                .with_languages(['en', 'es'])
                .with_confidence_threshold(0.9)
                .with_custom_policy(policy)
                .build()
        """
        from cloakpivot.engine_builder import CloakEngineBuilder

        return CloakEngineBuilder()

    def _get_default_analyzer_config(self) -> AnalyzerConfig:
        """Get optimized default analyzer configuration."""
        return AnalyzerConfig(language="en", min_confidence=0.7)

    def _get_default_entities(self) -> list[str]:
        """Get default list of common PII entity types."""
        return [
            "EMAIL_ADDRESS",
            "PERSON",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "US_SSN",
            "LOCATION",
            "DATE_TIME",
            "MEDICAL_LICENSE",
            "URL",
            "IP_ADDRESS",
        ]

    @property
    def default_policy(self) -> MaskingPolicy:
        """Access the default masking policy."""
        return self._default_policy

    @property
    def analyzer_config(self) -> AnalyzerConfig:
        """Access the analyzer configuration."""
        return self._analyzer_config
