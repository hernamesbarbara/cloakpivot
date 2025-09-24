"""StrategyApplicator for generating masked replacement tokens."""

import logging
import random
from typing import Any

from ..core.types.strategies import Strategy
from ..core.processing.surrogate import SurrogateGenerator
from .conflict_resolver import ConflictResolver
from .strategy_executor import StrategyExecutor

logger = logging.getLogger(__name__)


class StrategyApplicator:
    """
    Applies masking strategies to generate replacement tokens for detected PII.

    This class implements the core masking logic for different strategy types:
    - REDACT: Replace with redaction characters (*)
    - TEMPLATE: Replace with fixed templates like [PHONE], [EMAIL]
    - HASH: Replace with hashed values
    - PARTIAL: Show partial content with masking
    - SURROGATE: Generate fake data in same format
    - CUSTOM: Apply custom callback functions

    The class uses separate components for:
    - ConflictResolver: Handles fallback strategies and composition
    - StrategyExecutor: Executes individual strategy implementations

    Examples:
        >>> applicator = StrategyApplicator()
        >>>
        >>> # Template strategy
        >>> result = applicator.apply_strategy(
        ...     "555-123-4567",
        ...     "PHONE_NUMBER",
        ...     Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
        ...     0.95
        ... )
        >>> assert result == "[PHONE]"

        >>> # Partial strategy
        >>> result = applicator.apply_strategy(
        ...     "555-123-4567",
        ...     "PHONE_NUMBER",
        ...     Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}),
        ...     0.95
        ... )
        >>> assert result == "*********4567"
    """

    def __init__(self, seed: str | None = None) -> None:
        """
        Initialize the strategy applicator.

        Args:
            seed: Optional seed for deterministic random generation
        """
        self.seed = seed
        if seed:
            # Use seed for deterministic results
            self._random = random.Random(seed)
        else:
            self._random = random.Random()

        # Initialize enhanced surrogate generator
        self._surrogate_generator = SurrogateGenerator(seed=seed)

        # Initialize component managers
        self._conflict_resolver = ConflictResolver(self)
        self._strategy_executor = StrategyExecutor(self)

        logger.debug(f"StrategyApplicator initialized with seed: {seed}")

    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float,
    ) -> str:
        """
        Apply a masking strategy to generate a replacement token with fallback support.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity (e.g., 'PHONE_NUMBER', 'EMAIL_ADDRESS')
            strategy: The masking strategy to apply
            confidence: Detection confidence score

        Returns:
            str: The masked replacement text

        Raises:
            ValueError: If strategy parameters are invalid
            NotImplementedError: If strategy type is not supported
        """
        logger.debug(f"Applying {strategy.kind.value} strategy to {entity_type}")

        # Delegate to conflict resolver for fallback handling
        return self._conflict_resolver.apply_with_fallback(
            original_text, entity_type, strategy, confidence
        )

    def compose_strategies(
        self,
        original_text: str,
        entity_type: str,
        strategies: list[Strategy],
        confidence: float,
    ) -> str:
        """
        Compose multiple strategies in sequence or parallel.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity
            strategies: List of strategies to compose
            confidence: Detection confidence score

        Returns:
            str: The result of composed strategies
        """
        # Delegate to conflict resolver for composition
        return self._conflict_resolver.compose_strategies(
            original_text, entity_type, strategies, confidence
        )

    def resolve_overlapping_detections(
        self,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Resolve overlapping PII detections.

        Args:
            detections: List of detection dictionaries

        Returns:
            list: Non-overlapping detections after resolution
        """
        return self._conflict_resolver.resolve_overlapping_detections(detections)

    def prioritize_strategies(
        self,
        entity_type: str,
        confidence: float,
        available_strategies: list[Strategy],
    ) -> list[Strategy]:
        """
        Prioritize strategies based on entity type and confidence.

        Args:
            entity_type: Type of entity being masked
            confidence: Detection confidence score
            available_strategies: List of available strategies

        Returns:
            list[Strategy]: Prioritized list of strategies
        """
        return self._conflict_resolver.prioritize_strategies(
            entity_type, confidence, available_strategies
        )

    def get_surrogate_quality_metrics(self) -> Any:
        """
        Get quality metrics from the surrogate generator.

        Returns:
            Quality metrics dictionary
        """
        return self._surrogate_generator.get_quality_metrics()

    def reset_document_scope(self) -> None:
        """Reset document scope for new document processing."""
        self._surrogate_generator.reset_document_scope()

    def execute_strategy_directly(
        self,
        original_text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float,
    ) -> str:
        """
        Execute a single strategy directly without fallback handling.

        This method is exposed for testing and special use cases where
        fallback behavior is not desired.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity
            strategy: The masking strategy to apply
            confidence: Detection confidence score

        Returns:
            str: The masked replacement text

        Raises:
            ValueError: If strategy parameters are invalid
            NotImplementedError: If strategy type is not supported
        """
        return self._strategy_executor.execute_strategy(
            original_text, entity_type, strategy, confidence
        )