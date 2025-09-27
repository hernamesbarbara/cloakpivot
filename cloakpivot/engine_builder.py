"""Builder pattern for advanced CloakEngine configuration."""

from typing import Any

from cloakpivot.core.policies.policies import MaskingPolicy
from cloakpivot.core.processing.normalization import ConflictResolutionConfig
from cloakpivot.engine import CloakEngine


class CloakEngineBuilder:
    """Fluent builder for CloakEngine with advanced configuration.

    Provides a clean API for configuring CloakEngine instances with
    custom policies, analyzer settings, and conflict resolution.

    Examples:
        # Build with custom languages and threshold
        engine = CloakEngine.builder()
            .with_languages(['en', 'es', 'fr'])
            .with_confidence_threshold(0.9)
            .build()

        # Build with custom policy and Presidio engine
        engine = CloakEngine.builder()
            .with_custom_policy(my_policy)
            .with_presidio_engine(True)
            .build()
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._languages: list[str] = ["en"]
        self._confidence_threshold: float = 0.7
        self._return_decision_process: bool = False
        self._custom_policy: MaskingPolicy | None = None
        self._conflict_resolution_config: ConflictResolutionConfig | None = None
        self._use_presidio: bool = True
        self._additional_recognizers: list[str] = []
        self._excluded_recognizers: list[str] = []

    def with_custom_policy(self, policy: MaskingPolicy) -> "CloakEngineBuilder":
        """Set a custom masking policy.

        Args:
            policy: MaskingPolicy to use for masking operations

        Returns:
            Self for method chaining
        """
        self._custom_policy = policy
        return self

    def with_languages(self, languages: list[str]) -> "CloakEngineBuilder":
        """Set the languages for entity detection.

        Args:
            languages: List of language codes (e.g., ['en', 'es', 'fr'])

        Returns:
            Self for method chaining
        """
        self._languages = languages
        return self

    def with_confidence_threshold(self, threshold: float) -> "CloakEngineBuilder":
        """Set the confidence threshold for entity detection.

        Args:
            threshold: Confidence threshold between 0.0 and 1.0

        Returns:
            Self for method chaining

        Raises:
            ValueError: If threshold is not between 0.0 and 1.0
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self._confidence_threshold = threshold
        return self

    def with_decision_process(self, enabled: bool = True) -> "CloakEngineBuilder":
        """Enable or disable returning the decision process.

        Args:
            enabled: Whether to return decision process information

        Returns:
            Self for method chaining
        """
        self._return_decision_process = enabled
        return self

    def with_analyzer_config(self, config: dict[str, Any]) -> "CloakEngineBuilder":
        """Set complete analyzer configuration from a dictionary.

        Args:
            config: Dictionary containing analyzer configuration

        Returns:
            Self for method chaining
        """
        if "languages" in config:
            self._languages = config["languages"]
        if "confidence_threshold" in config:
            self._confidence_threshold = config["confidence_threshold"]
        if "return_decision_process" in config:
            self._return_decision_process = config["return_decision_process"]
        return self

    def with_conflict_resolution(self, config: ConflictResolutionConfig) -> "CloakEngineBuilder":
        """Set conflict resolution configuration.

        Args:
            config: ConflictResolutionConfig for handling overlapping entities

        Returns:
            Self for method chaining
        """
        self._conflict_resolution_config = config
        return self

    def with_presidio_engine(self, use: bool = True) -> "CloakEngineBuilder":
        """Enable or disable use of Presidio engine.

        Args:
            use: Whether to use Presidio for entity detection

        Returns:
            Self for method chaining
        """
        self._use_presidio = use
        return self

    def with_additional_recognizers(self, recognizers: list[str]) -> "CloakEngineBuilder":
        """Add additional entity recognizers.

        Args:
            recognizers: List of additional recognizer names to include

        Returns:
            Self for method chaining
        """
        self._additional_recognizers = recognizers
        return self

    def exclude_recognizers(self, recognizers: list[str]) -> "CloakEngineBuilder":
        """Exclude specific entity recognizers.

        Args:
            recognizers: List of recognizer names to exclude

        Returns:
            Self for method chaining
        """
        self._excluded_recognizers = recognizers
        return self

    def build(self) -> CloakEngine:
        """Build and return the configured CloakEngine instance.

        Returns:
            Configured CloakEngine instance
        """
        # Build analyzer configuration
        analyzer_config = {
            "languages": self._languages,  # Will be mapped to 'language' in CloakEngine
            "confidence_threshold": self._confidence_threshold,  # Will be mapped to 'min_confidence'
            "return_decision_process": self._return_decision_process,
        }

        # Add recognizer configuration if specified
        if self._additional_recognizers:
            analyzer_config["additional_recognizers"] = self._additional_recognizers
        if self._excluded_recognizers:
            analyzer_config["excluded_recognizers"] = self._excluded_recognizers

        # Build conflict resolution config dictionary if provided
        conflict_config = None
        if self._conflict_resolution_config:
            conflict_config = {
                "strategy": self._conflict_resolution_config.strategy,
            }
            # Add custom_priority if it exists
            if hasattr(self._conflict_resolution_config, "custom_priority"):
                conflict_config["custom_priority"] = (
                    self._conflict_resolution_config.custom_priority
                )

        # Create and return CloakEngine
        return CloakEngine(
            analyzer_config=analyzer_config,
            default_policy=self._custom_policy,
            conflict_resolution_config=conflict_config,
        )

    def reset(self) -> "CloakEngineBuilder":
        """Reset builder to default values.

        Returns:
            Self for method chaining
        """
        # Reset all attributes to their default values
        self._languages = ["en"]
        self._confidence_threshold = 0.7
        self._return_decision_process = False
        self._custom_policy = None
        self._conflict_resolution_config = None
        self._use_presidio = True
        self._analyzer_config = None
        self._additional_recognizers = []
        self._resolve_conflicts = False
        return self
