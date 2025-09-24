"""Presidio AnalyzerEngine integration for PII detection."""

import logging
import os
import re
from dataclasses import dataclass, field
from functools import total_ordering
from typing import TYPE_CHECKING, Any, cast

# Performance profiling removed - simplified implementation
from ..policies.policies import MaskingPolicy

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_analyzer.nlp_engine import NlpEngineProvider
else:
    # Lazy import presidio to avoid blocking on module load
    # These will be imported when actually needed in _initialize_engine()
    AnalyzerEngine = None
    RecognizerResult = None
    NlpEngineProvider = None


def _import_presidio() -> None:
    """Import presidio modules with proper error handling."""
    global AnalyzerEngine, RecognizerResult, NlpEngineProvider

    if not TYPE_CHECKING and AnalyzerEngine is not None:
        return  # Already imported

    import threading

    # Track if import completed
    import_complete = threading.Event()
    import_error: Exception | None = None

    def import_with_timeout() -> None:
        """Import presidio modules in a separate thread."""
        nonlocal import_error
        global AnalyzerEngine, RecognizerResult, NlpEngineProvider

        try:
            from presidio_analyzer import (
                AnalyzerEngine,
                RecognizerResult,
            )
            from presidio_analyzer.nlp_engine import (
                NlpEngineProvider,
            )

            import_complete.set()
        except ImportError as e:
            import_error = e
            import_complete.set()
        except Exception as e:
            import_error = e
            import_complete.set()

    # Start import in a separate thread
    import_thread = threading.Thread(target=import_with_timeout, daemon=True)
    import_thread.start()

    # Wait for up to 5 seconds
    if not import_complete.wait(timeout=5.0):
        # Import timed out
        logger.warning("Presidio import timed out after 5 seconds")
        raise ImportError(
            "presidio-analyzer import timed out. There may be an issue with the installation."
        )

    # Check if there was an error during import
    if import_error is not None:
        if isinstance(import_error, ImportError):
            raise ImportError(
                "presidio-analyzer is required for PII detection. "
                "Install it with: pip install presidio-analyzer"
            ) from import_error
        raise import_error


logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """Configuration for Presidio AnalyzerEngine integration.

    Attributes:
        language: Language code for analysis (ISO 639-1 format like 'en', 'es')
        min_confidence: Minimum confidence threshold for entity detection
        enabled_recognizers: Specific recognizers to enable (None means all default)
        disabled_recognizers: Recognizers to explicitly disable
        custom_recognizers: Custom recognizers to add
        nlp_engine_name: NLP engine to use ('spacy' or 'transformers')
    """

    language: str = field(default="en")
    min_confidence: float = field(default=0.5)
    enabled_recognizers: list[str] | None = field(default=None)
    disabled_recognizers: set[str] = field(default_factory=set)
    custom_recognizers: dict[str, Any] = field(default_factory=dict)
    nlp_engine_name: str = field(default="spacy")

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_language()
        self._validate_confidence()
        self._validate_nlp_engine()

    def _validate_language(self) -> None:
        """Validate language code format."""
        if not isinstance(self.language, str):
            raise ValueError("Language must be a string")

        # Validate ISO 639-1 format (2 letters, optionally followed by country code)
        language_pattern = r"^[a-z]{2}(-[A-Z]{2})?$"
        if not re.match(language_pattern, self.language):
            raise ValueError(f"Language must be a valid ISO 639-1 code, got '{self.language}'")

    def _validate_confidence(self) -> None:
        """Validate confidence threshold."""
        if not isinstance(self.min_confidence, int | float):
            raise ValueError("min_confidence must be a number")

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be between 0.0 and 1.0, got {self.min_confidence}"
            )

    def _validate_nlp_engine(self) -> None:
        """Validate NLP engine name."""
        valid_engines = {"spacy", "transformers"}
        if self.nlp_engine_name not in valid_engines:
            raise ValueError(
                f"nlp_engine_name must be one of {valid_engines}, got '{self.nlp_engine_name}'"
            )

    @classmethod
    def from_policy(cls, policy: MaskingPolicy) -> "AnalyzerConfig":
        """Create AnalyzerConfig from MaskingPolicy.

        Args:
            policy: The masking policy to extract configuration from

        Returns:
            AnalyzerConfig instance with policy settings
        """
        # Extract language from locale (e.g., 'en-US' -> 'en')
        language = policy.locale.split("-")[0] if policy.locale else "en"

        # Use minimum threshold from policy as global confidence
        min_confidence = 0.5  # default
        if policy.thresholds:
            min_confidence = min(policy.thresholds.values())

        return cls(
            language=language,
            min_confidence=min_confidence,
            enabled_recognizers=None,  # Use defaults initially
            disabled_recognizers=set(),
            custom_recognizers={},
        )


class RecognizerRegistry:
    """Registry for managing Presidio recognizers."""

    # Default recognizers that should be enabled
    DEFAULT_RECOGNIZERS = {
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "CREDIT_CARD",
        "US_SSN",
        "PERSON",
        "URL",
        "IP_ADDRESS",
        "DATE_TIME",
        "LOCATION",
    }

    def __init__(self, enabled_recognizers: list[str] | None = None):
        """Initialize recognizer registry.

        Args:
            enabled_recognizers: Specific recognizers to enable (None for defaults)
        """
        self._enabled: set[str] = (
            set(enabled_recognizers)
            if enabled_recognizers is not None
            else self.DEFAULT_RECOGNIZERS.copy()
        )
        self._custom_recognizers: dict[str, Any] = {}

        logger.info(
            f"Initialized recognizer registry with {len(self._enabled)} enabled recognizers"
        )

    def get_enabled_recognizers(self) -> list[str]:
        """Get list of currently enabled recognizers."""
        return list(self._enabled)

    def enable_recognizer(self, recognizer_name: str) -> None:
        """Enable a recognizer.

        Args:
            recognizer_name: Name of the recognizer to enable
        """
        self._enabled.add(recognizer_name)
        logger.debug(f"Enabled recognizer: {recognizer_name}")

    def disable_recognizer(self, recognizer_name: str) -> None:
        """Disable a recognizer.

        Args:
            recognizer_name: Name of the recognizer to disable
        """
        self._enabled.discard(recognizer_name)
        logger.debug(f"Disabled recognizer: {recognizer_name}")

    def add_custom_recognizer(self, name: str, recognizer: Any) -> None:
        """Add a custom recognizer.

        Args:
            name: Name for the custom recognizer
            recognizer: The recognizer instance
        """
        self._custom_recognizers[name] = recognizer
        self._enabled.add(name)
        logger.info(f"Added custom recognizer: {name}")

    def get_custom_recognizers(self) -> dict[str, Any]:
        """Get dictionary of custom recognizers."""
        return self._custom_recognizers.copy()


@total_ordering
@dataclass
class EntityDetectionResult:
    """Result of entity detection from Presidio analysis.

    Attributes:
        entity_type: Type of entity detected (e.g., 'PHONE_NUMBER', 'EMAIL_ADDRESS')
        start: Start position in text
        end: End position in text
        confidence: Confidence score (0.0-1.0)
        text: The actual text of the detected entity
    """

    entity_type: str
    start: int
    end: int
    confidence: float
    text: str

    def __post_init__(self) -> None:
        """Validate detection result after initialization."""
        if not 0 <= self.start < self.end:
            raise ValueError(f"Invalid position range: start={self.start}, end={self.end}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    @classmethod
    def from_presidio_result(cls, result: Any, text: str) -> "EntityDetectionResult":
        """Create EntityDetectionResult from Presidio RecognizerResult.

        Args:
            result: Presidio RecognizerResult instance
            text: The detected entity text

        Returns:
            EntityDetectionResult instance
        """
        return cls(
            entity_type=result.entity_type,
            start=result.start,
            end=result.end,
            confidence=result.score,
            text=text,
        )

    def overlaps_with(self, other: "EntityDetectionResult") -> bool:
        """Check if this entity overlaps with another entity.

        Args:
            other: Another EntityDetectionResult to check overlap with

        Returns:
            True if entities overlap, False otherwise
        """
        return not (self.end <= other.start or other.end <= self.start)

    def __lt__(self, other: "EntityDetectionResult") -> bool:
        """Define ordering for deterministic sorting.

        Sort by:
        1. Start position (ascending)
        2. End position (ascending)
        3. Confidence (descending)
        4. Entity type (ascending)
        """
        if self.start != other.start:
            return self.start < other.start

        if self.end != other.end:
            return self.end < other.end

        if abs(self.confidence - other.confidence) > 1e-6:
            return self.confidence > other.confidence  # Higher confidence first

        return self.entity_type < other.entity_type

    def __eq__(self, other: object) -> bool:
        """Check equality with another EntityDetectionResult."""
        if not isinstance(other, EntityDetectionResult):
            return NotImplemented

        return (
            self.entity_type == other.entity_type
            and self.start == other.start
            and self.end == other.end
            and abs(self.confidence - other.confidence) < 1e-6
            and self.text == other.text
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "text": self.text,
        }


class AnalyzerEngineWrapper:
    """Wrapper around Presidio AnalyzerEngine with enhanced configuration."""

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        use_singleton: bool | None = None,
    ):
        """Initialize analyzer wrapper.

        Args:
            config: Configuration for the analyzer (uses defaults if None)
            use_singleton: Whether to use singleton pattern (defaults to environment
                variable or True)
        """
        self.config = config or AnalyzerConfig()

        # Determine singleton usage from parameter, environment, or default
        default_use_singleton = os.getenv("CLOAKPIVOT_USE_SINGLETON", "true").lower() == "true"
        self.use_singleton = use_singleton if use_singleton is not None else default_use_singleton

        self.registry = RecognizerRegistry(self.config.enabled_recognizers)
        self._engine: Any | None = None  # Will be AnalyzerEngine after import
        self._is_initialized = False

        # Apply disabled recognizers
        for recognizer in self.config.disabled_recognizers:
            self.registry.disable_recognizer(recognizer)

        logger.info(
            f"Created analyzer wrapper with language='{self.config.language}', "
            f"min_confidence={self.config.min_confidence}, use_singleton={self.use_singleton}"
        )

    @property
    def is_initialized(self) -> bool:
        """Check if the underlying AnalyzerEngine is initialized."""
        return self._is_initialized

    @classmethod
    def from_policy(cls, policy: MaskingPolicy) -> "AnalyzerEngineWrapper":
        """Create AnalyzerEngineWrapper from MaskingPolicy.

        Args:
            policy: MaskingPolicy to extract configuration from

        Returns:
            AnalyzerEngineWrapper configured from policy
        """
        config = AnalyzerConfig.from_policy(policy)
        return cls(config)

    @classmethod
    def create_shared(cls, config: AnalyzerConfig | None = None) -> "AnalyzerEngineWrapper":
        """Create analyzer using singleton loader from loaders module.

        This method uses the cached singleton analyzer instances from the loaders
        module to provide better performance through shared instances.

        Args:
            config: Optional AnalyzerConfig for analyzer configuration

        Returns:
            AnalyzerEngineWrapper instance from singleton cache

        Examples:
            >>> # Create shared analyzer with default config
            >>> analyzer = AnalyzerEngineWrapper.create_shared()
            >>>
            >>> # Create shared analyzer with custom config
            >>> config = AnalyzerConfig(language="es", min_confidence=0.7)
            >>> analyzer = AnalyzerEngineWrapper.create_shared(config)
        """
        from ..loaders import get_presidio_analyzer, get_presidio_analyzer_from_config

        if config is not None:
            return get_presidio_analyzer_from_config(config)
        return get_presidio_analyzer()

    def _get_spacy_model_name(self, language: str) -> str:
        """Get the appropriate spaCy model name based on language and size preference.

        Uses environment variable MODEL_SIZE to control model size selection,
        providing runtime flexibility for performance/accuracy tradeoffs.

        Environment Variables:
            MODEL_SIZE: {small|medium|large} - Controls model size/performance tradeoff
                - small: *_sm models (fast, lower memory, good accuracy)
                - medium: *_md models (balanced performance and accuracy)
                - large: *_lg models (slower, higher memory, best accuracy)

        Args:
            language: ISO 639-1 language code

        Returns:
            Full spaCy model name with appropriate size suffix
        """
        from ..types.model_info import get_model_name
        from ..utilities.config import performance_config

        # Use global performance config for model size selection
        model_size = performance_config.model_size

        logger.debug(f"Selecting {model_size} model for language '{language}'")

        return get_model_name(language, model_size)

    def _initialize_engine(self) -> None:
        """Initialize the Presidio AnalyzerEngine (lazy initialization)."""
        if self._is_initialized:
            return

        try:
            # Import presidio modules when actually needed
            _import_presidio()

            logger.info("Initializing Presidio AnalyzerEngine...")

            # Configure NLP engine with proper model names
            model_name = self._get_spacy_model_name(self.config.language)
            nlp_configuration = {
                "nlp_engine_name": self.config.nlp_engine_name,
                "models": [{"lang_code": self.config.language, "model_name": model_name}],
            }

            if not TYPE_CHECKING and (NlpEngineProvider is None or AnalyzerEngine is None):
                raise ImportError("Presidio modules not properly imported")

            nlp_engine_provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = nlp_engine_provider.create_engine()

            # Create analyzer engine
            self._engine = AnalyzerEngine(
                nlp_engine=nlp_engine, supported_languages=[self.config.language]
            )

            # Add custom recognizers if any
            for name, recognizer in self.registry.get_custom_recognizers().items():
                self._engine.registry.add_recognizer(recognizer)
                logger.debug(f"Added custom recognizer: {name}")

            self._is_initialized = True
            logger.info("Presidio AnalyzerEngine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Presidio AnalyzerEngine: {e}")
            raise RuntimeError(f"Failed to initialize Presidio AnalyzerEngine: {e}") from e

    def analyze_text(
        self,
        text: str,
        entities: list[str] | None = None,
        min_confidence: float | None = None,
    ) -> list[EntityDetectionResult]:
        """Analyze text for PII entities with performance tracking.

        Args:
            text: Text to analyze for PII entities
            entities: Specific entity types to look for (None for all enabled)
            min_confidence: Minimum confidence threshold (overrides config default)

        Returns:
            List of EntityDetectionResult instances sorted deterministically
        """
        if not text.strip():
            return []

        # Initialize engine if needed
        self._initialize_engine()

        if not self._engine:
            raise RuntimeError("AnalyzerEngine not properly initialized")

        # Determine entities to analyze
        analyze_entities = (
            entities if entities is not None else self.registry.get_enabled_recognizers()
        )

        # Determine confidence threshold
        threshold = min_confidence if min_confidence is not None else self.config.min_confidence

        try:
            logger.debug(f"Analyzing text of length {len(text)} for entities: {analyze_entities}")

            # Run Presidio analysis
            presidio_results = self._engine.analyze(
                text=text,
                entities=analyze_entities,
                language=self.config.language,
                score_threshold=threshold,
            )

            # Convert to our result format
            detection_results = []
            for result in presidio_results:
                entity_text = text[result.start : result.end]
                detection = EntityDetectionResult.from_presidio_result(result, entity_text)
                detection_results.append(detection)

            # Sort for deterministic ordering
            detection_results.sort()

            logger.info(f"Detected {len(detection_results)} entities in text")
            return detection_results

        except Exception as e:
            logger.error(f"Error during text analysis: {e}")
            raise RuntimeError(f"Error during text analysis: {e}") from e

    def get_supported_entities(self) -> list[str]:
        """Get list of supported entity types."""
        self._initialize_engine()

        if not self._engine:
            return list(self.registry.get_enabled_recognizers())

        return list(self._engine.get_supported_entities(language=self.config.language))

    def validate_configuration(self) -> dict[str, Any]:
        """Validate current configuration and return diagnostics.

        Returns:
            Dictionary with validation results and diagnostics
        """
        diagnostics = {
            "config_valid": True,
            "language": self.config.language,
            "min_confidence": self.config.min_confidence,
            "enabled_recognizers": self.registry.get_enabled_recognizers(),
            "custom_recognizers": list(self.registry.get_custom_recognizers().keys()),
            "engine_initialized": self._is_initialized,
            "errors": [],
            "warnings": [],
        }

        try:
            # Test engine initialization
            self._initialize_engine()

            # Test basic analysis
            test_results = self.analyze_text(
                "Test text with email test@example.com", min_confidence=0.1
            )
            diagnostics["test_analysis_successful"] = True
            diagnostics["test_results_count"] = len(test_results)

        except Exception as e:
            diagnostics["config_valid"] = False
            cast(list[str], diagnostics["errors"]).append(str(e))
            diagnostics["test_analysis_successful"] = False

        # Check for common issues
        if not self.registry.get_enabled_recognizers():
            cast(list[str], diagnostics["warnings"]).append("No recognizers are enabled")

        if self.config.min_confidence > 0.9:
            cast(list[str], diagnostics["warnings"]).append(
                "Very high confidence threshold may miss valid entities"
            )

        return diagnostics
