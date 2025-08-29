"""Base class for custom Presidio recognizer plugins."""

import re
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from re import Pattern
from typing import Any, Optional

from ..base import BasePlugin, PluginInfo
from ..exceptions import PluginExecutionError, PluginValidationError


@dataclass
class RecognizerPluginResult:
    """Result from a recognizer plugin execution."""

    entity_type: str
    start: int
    end: int
    confidence: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


class BaseRecognizerPlugin(BasePlugin):
    """
    Base class for custom Presidio recognizer plugins.

    Recognizer plugins implement custom PII detection logic that can be used
    alongside or in place of built-in Presidio recognizers.

    Example:
        >>> class CustomPhoneRecognizerPlugin(BaseRecognizerPlugin):
        ...     @property
        ...     def info(self) -> PluginInfo:
        ...         return PluginInfo(
        ...             name="custom_phone_recognizer",
        ...             version="1.0.0",
        ...             description="Custom phone number recognizer with format validation",
        ...             author="CloakPivot Team",
        ...             plugin_type="recognizer"
        ...         )
        ...
        ...     def analyze_text(self, text: str) -> List[RecognizerPluginResult]:
        ...         results = []
        ...         # Find phone patterns with custom logic
        ...         pattern = r'\\b(?:\\+1[-.]?)?(?:\\(?[0-9]{3}\\)?[-.]?)?[0-9]{3}[-.]?[0-9]{4}\\b'
        ...         for match in re.finditer(pattern, text):
        ...             results.append(RecognizerPluginResult(
        ...                 entity_type="PHONE_NUMBER",
        ...                 start=match.start(),
        ...                 end=match.end(),
        ...                 confidence=0.9,
        ...                 text=match.group(),
        ...                 metadata={"pattern": "custom_phone"}
        ...             ))
        ...         return results
    """

    @property
    def info(self) -> PluginInfo:
        """Get plugin information - must be implemented by subclasses."""
        return PluginInfo(
            name=self.__class__.__name__.lower(),
            version="1.0.0",
            description="Custom recognizer plugin",
            author="Unknown",
            plugin_type="recognizer"
        )

    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        Validate recognizer plugin configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if valid

        Raises:
            PluginValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise PluginValidationError(
                "Configuration must be a dictionary",
                plugin_name=self.info.name
            )

        # Validate supported languages
        supported_languages = config.get("supported_languages", ["en"])
        if not isinstance(supported_languages, list):
            raise PluginValidationError(
                "supported_languages must be a list",
                plugin_name=self.info.name
            )

        # Validate entity types
        entity_types = config.get("entity_types", [])
        if not isinstance(entity_types, list) or not entity_types:
            raise PluginValidationError(
                "entity_types must be a non-empty list",
                plugin_name=self.info.name
            )

        # Validate confidence thresholds
        min_confidence = config.get("min_confidence", 0.0)
        if not isinstance(min_confidence, (int, float)) or not 0.0 <= min_confidence <= 1.0:
            raise PluginValidationError(
                "min_confidence must be a number between 0.0 and 1.0",
                plugin_name=self.info.name
            )

        return self._validate_recognizer_config(config)

    def _validate_recognizer_config(self, config: dict[str, Any]) -> bool:
        """
        Override this method to add recognizer-specific validation.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid

        Raises:
            PluginValidationError: If configuration is invalid
        """
        return True

    def initialize(self) -> None:
        """Initialize the recognizer plugin."""
        try:
            self._initialize_recognizer()
            self.is_initialized = True
            self.logger.info(f"Recognizer plugin {self.info.name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize recognizer plugin {self.info.name}: {e}")
            raise PluginExecutionError(
                f"Recognizer plugin initialization failed: {e}",
                plugin_name=self.info.name,
                original_exception=e
            ) from e

    def _initialize_recognizer(self) -> None:
        """
        Override this method to add recognizer-specific initialization.

        This is called during initialize() and should set up any
        resources needed for the recognizer to function.
        """
        pass

    def cleanup(self) -> None:
        """Clean up recognizer plugin resources."""
        try:
            self._cleanup_recognizer()
            self.is_initialized = False
            self.logger.info(f"Recognizer plugin {self.info.name} cleaned up successfully")
        except Exception as e:
            self.logger.warning(f"Error during recognizer plugin cleanup: {e}")

    def _cleanup_recognizer(self) -> None:
        """
        Override this method to add recognizer-specific cleanup.

        This is called during cleanup() and should release any
        resources allocated during initialization.
        """
        pass

    @abstractmethod
    def analyze_text(
        self,
        text: str,
        language: str = "en",
        context: Optional[dict[str, Any]] = None
    ) -> list[RecognizerPluginResult]:
        """
        Analyze text for custom PII entities.

        Args:
            text: Text to analyze for PII entities
            language: Language code for analysis (e.g., 'en', 'es')
            context: Optional contextual information

        Returns:
            List of RecognizerPluginResult instances for detected entities

        Raises:
            PluginExecutionError: If analysis fails
        """
        pass

    def analyze_text_safe(
        self,
        text: str,
        language: str = "en",
        context: Optional[dict[str, Any]] = None
    ) -> list[RecognizerPluginResult]:
        """
        Analyze text with error handling and timing.

        This is the main entry point used by the plugin registry.
        It wraps analyze_text() with error handling and performance monitoring.

        Args:
            text: Text to analyze for PII entities
            language: Language code for analysis
            context: Optional contextual information

        Returns:
            List of RecognizerPluginResult instances, empty list on error
        """
        if not self.is_initialized:
            self.logger.warning(f"Recognizer plugin {self.info.name} not initialized")
            return []

        if not text or not text.strip():
            return []

        start_time = time.perf_counter()

        try:
            results = self.analyze_text(text, language, context)

            # Ensure results are valid
            if not isinstance(results, list):
                raise PluginExecutionError(
                    "Recognizer plugin must return a list of results",
                    plugin_name=self.info.name
                )

            # Update timing for all results
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000

            for result in results:
                if not isinstance(result, RecognizerPluginResult):
                    raise PluginExecutionError(
                        "All results must be RecognizerPluginResult instances",
                        plugin_name=self.info.name
                    )
                result.execution_time_ms = execution_time

            # Filter by minimum confidence
            min_confidence = self.get_config_value("min_confidence", 0.0)
            filtered_results = [
                result for result in results
                if result.confidence >= min_confidence
            ]

            return filtered_results

        except Exception as e:
            self.logger.error(
                f"Recognizer plugin {self.info.name} failed: {e}",
                exc_info=True
            )
            return []

    def get_supported_entity_types(self) -> list[str]:
        """
        Get list of entity types this recognizer can detect.

        Returns:
            List of supported entity types
        """
        return self.get_config_value("entity_types", []) or []

    def get_supported_languages(self) -> list[str]:
        """
        Get list of languages this recognizer supports.

        Returns:
            List of supported language codes
        """
        return self.get_config_value("supported_languages", ["en"]) or ["en"]

    def supports_language(self, language: str) -> bool:
        """
        Check if this recognizer supports a given language.

        Args:
            language: Language code to check

        Returns:
            True if language is supported
        """
        supported = self.get_supported_languages()
        return language in supported or "all" in supported

    def supports_entity_type(self, entity_type: str) -> bool:
        """
        Check if this recognizer can detect a given entity type.

        Args:
            entity_type: Entity type to check

        Returns:
            True if entity type is supported
        """
        supported = self.get_supported_entity_types()
        return entity_type in supported

    def create_presidio_recognizer(self) -> Optional[Any]:
        """
        Create a Presidio-compatible recognizer instance.

        This method can be overridden to provide integration with
        the standard Presidio analyzer engine.

        Returns:
            Presidio recognizer instance or None if not supported
        """
        return None


class PatternBasedRecognizerPlugin(BaseRecognizerPlugin):
    """
    Base class for pattern-based recognizer plugins.

    This class provides common functionality for recognizers that use
    regular expressions or other patterns to detect PII entities.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._compiled_patterns: dict[str, Pattern[str]] = {}

    def _validate_recognizer_config(self, config: dict[str, Any]) -> bool:
        """Validate pattern-based recognizer configuration."""
        patterns = config.get("patterns", {})
        if not isinstance(patterns, dict):
            raise PluginValidationError(
                "patterns must be a dictionary",
                plugin_name=self.info.name
            )

        # Validate each pattern
        for entity_type, pattern_list in patterns.items():
            if not isinstance(pattern_list, list):
                raise PluginValidationError(
                    f"patterns['{entity_type}'] must be a list",
                    plugin_name=self.info.name
                )

            for pattern in pattern_list:
                if not isinstance(pattern, str):
                    raise PluginValidationError(
                        f"Pattern for '{entity_type}' must be a string",
                        plugin_name=self.info.name
                    )

                # Test pattern compilation
                try:
                    re.compile(pattern)
                except re.error as e:
                    raise PluginValidationError(
                        f"Invalid regex pattern for '{entity_type}': {e}",
                        plugin_name=self.info.name
                    ) from e

        return True

    def _initialize_recognizer(self) -> None:
        """Initialize pattern compilation."""
        patterns = self.get_config_value("patterns", {})

        for entity_type, pattern_list in patterns.items():
            # Combine patterns with OR operator
            combined_pattern = "|".join(f"({pattern})" for pattern in pattern_list)

            try:
                compiled_pattern = re.compile(combined_pattern, re.IGNORECASE)
                self._compiled_patterns[entity_type] = compiled_pattern
                self.logger.debug(f"Compiled pattern for {entity_type}: {combined_pattern}")
            except re.error as e:
                self.logger.error(f"Failed to compile pattern for {entity_type}: {e}")
                raise

    def analyze_text(
        self,
        text: str,
        language: str = "en",
        context: Optional[dict[str, Any]] = None
    ) -> list[RecognizerPluginResult]:
        """
        Analyze text using compiled regex patterns.

        Args:
            text: Text to analyze
            language: Language code
            context: Optional context information

        Returns:
            List of detection results
        """
        if not self.supports_language(language):
            return []

        results = []

        for entity_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                # Determine confidence based on match quality
                confidence = self._calculate_confidence(match, entity_type, context)

                if confidence > 0.0:
                    results.append(RecognizerPluginResult(
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        text=match.group(),
                        metadata={
                            "pattern_based": True,
                            "recognizer": self.info.name
                        }
                    ))

        return results

    def _calculate_confidence(
        self,
        match: re.Match[str],
        entity_type: str,
        context: Optional[dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score for a pattern match.

        Args:
            match: Regex match object
            entity_type: Type of entity matched
            context: Optional context information

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Default confidence based on match length and entity type
        base_confidence = 0.7

        # Adjust confidence based on match characteristics
        match_text = match.group()

        # Longer matches generally more reliable
        if len(match_text) > 10:
            base_confidence += 0.1
        elif len(match_text) < 5:
            base_confidence -= 0.1

        # Check for common false positive patterns
        if self._is_likely_false_positive(match_text, entity_type):
            base_confidence -= 0.3

        return max(0.0, min(1.0, base_confidence))

    def _is_likely_false_positive(self, text: str, entity_type: str) -> bool:
        """
        Check if a match is likely to be a false positive.

        Args:
            text: Matched text
            entity_type: Entity type

        Returns:
            True if likely false positive
        """
        # Override in subclasses to add entity-specific validation
        return False
