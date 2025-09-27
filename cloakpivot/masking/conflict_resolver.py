"""Conflict resolution for strategy application."""

import logging
from typing import TYPE_CHECKING, Any

from ..core.types.strategies import Strategy, StrategyKind

if TYPE_CHECKING:
    from .applicator import StrategyApplicator

logger = logging.getLogger(__name__)


class ConflictResolver:
    """
    Handles conflict resolution and fallback strategies for masking operations.

    This class manages:
    - Fallback strategy chains when primary strategies fail
    - Strategy composition (sequential/parallel)
    - Error recovery and graceful degradation
    """

    def __init__(self, applicator: "StrategyApplicator") -> None:
        """
        Initialize conflict resolver.

        Args:
            applicator: Parent strategy applicator instance
        """
        self.applicator = applicator

    def apply_with_fallback(
        self,
        original_text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float,
    ) -> str:
        """
        Apply a strategy with automatic fallback on failure.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity (e.g., 'PHONE_NUMBER', 'EMAIL_ADDRESS')
            strategy: The masking strategy to apply
            confidence: Detection confidence score

        Returns:
            str: The masked replacement text
        """
        logger.debug(f"Applying {strategy.kind.value} strategy to {entity_type} with fallback")

        try:
            # Try the primary strategy through the applicator
            from .strategy_executor import StrategyExecutor

            executor = StrategyExecutor(self.applicator)
            return executor.execute_strategy(original_text, entity_type, strategy, confidence)
        except Exception as e:
            logger.warning(f"Strategy {strategy.kind.value} failed for {entity_type}: {e}")

            # Apply fallback strategies
            return self._apply_fallback_chain(
                original_text, entity_type, strategy, confidence, str(e)
            )

    def _apply_fallback_chain(
        self,
        original_text: str,
        entity_type: str,
        primary_strategy: Strategy,
        confidence: float,
        error_msg: str,
    ) -> str:
        """
        Apply a chain of fallback strategies when primary strategy fails.

        Args:
            original_text: The original text to mask
            entity_type: Type of entity
            primary_strategy: The strategy that failed
            confidence: Detection confidence score
            error_msg: Error message from primary strategy failure

        Returns:
            str: Masked text from first successful fallback
        """
        # Define fallback chain based on entity type and primary strategy
        fallback_strategies = self._get_fallback_chain(entity_type, primary_strategy)

        for fallback_strategy in fallback_strategies:
            try:
                logger.info(f"Trying fallback {fallback_strategy.kind.value} for {entity_type}")

                # Use executor directly to avoid infinite recursion
                from .strategy_executor import StrategyExecutor

                executor = StrategyExecutor(self.applicator)
                return executor.execute_strategy(
                    original_text, entity_type, fallback_strategy, confidence
                )
            except Exception as fallback_error:
                logger.warning(f"Fallback {fallback_strategy.kind.value} failed: {fallback_error}")
                continue

        # Ultimate fallback - simple asterisk masking
        logger.error(f"All fallback strategies failed for {entity_type}, using ultimate fallback")
        return self._ultimate_fallback(original_text)

    def _get_fallback_chain(self, entity_type: str, primary_strategy: Strategy) -> list[Strategy]:
        """
        Get the appropriate fallback chain for given entity type and failed strategy.

        Args:
            entity_type: Type of entity being masked
            primary_strategy: The strategy that failed

        Returns:
            list[Strategy]: Ordered list of fallback strategies
        """
        # Default fallback chain
        default_chain = [
            Strategy(StrategyKind.TEMPLATE, {"template": f"[{entity_type}]"}),
            Strategy(StrategyKind.REDACT, {"redact_char": "*", "preserve_length": True}),
            Strategy(StrategyKind.REDACT, {"redact_char": "*", "preserve_length": False}),
        ]

        # Customize based on primary strategy type
        if primary_strategy.kind == StrategyKind.SURROGATE:
            # If surrogate failed, try hash before template
            return [
                Strategy(StrategyKind.HASH, {"algorithm": "sha256", "truncate": 8}),
            ] + default_chain

        if primary_strategy.kind == StrategyKind.HASH:
            # If hash failed, skip to template
            return default_chain

        if primary_strategy.kind == StrategyKind.PARTIAL:
            # If partial failed, try full redaction
            return [
                Strategy(StrategyKind.REDACT, {"redact_char": "*", "preserve_length": True}),
                Strategy(StrategyKind.TEMPLATE, {"template": f"[{entity_type}]"}),
            ]

        return default_chain

    def _ultimate_fallback(self, original_text: str) -> str:
        """
        Ultimate fallback when all strategies fail.

        Args:
            original_text: Original text to mask

        Returns:
            str: Simple asterisk masking
        """
        return "*" * max(1, len(original_text))

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

        Raises:
            ValueError: If no strategies are provided
        """
        if not strategies:
            raise ValueError("At least one strategy must be provided")

        if len(strategies) == 1:
            return self.apply_with_fallback(original_text, entity_type, strategies[0], confidence)

        # Sequential composition - each strategy transforms the result of the previous
        result = original_text
        for i, strategy in enumerate(strategies):
            try:
                result = self.apply_with_fallback(result, entity_type, strategy, confidence)
                logger.debug(f"Applied strategy {i + 1}/{len(strategies)}: {strategy.kind.value}")
            except Exception as e:
                logger.warning(
                    f"Strategy {i + 1}/{len(strategies)} ({strategy.kind.value}) failed: {e}"
                )
                # Continue with current result on failure
                continue

        return result

    def resolve_overlapping_detections(
        self,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Resolve overlapping PII detections by priority.

        Args:
            detections: List of detection dictionaries with 'start', 'end', 'entity_type', 'confidence'

        Returns:
            list[dict]: Non-overlapping detections after resolution
        """
        if not detections:
            return []

        # Sort by start position, then by confidence (higher first)
        sorted_detections = sorted(
            detections, key=lambda d: (d.get("start", 0), -d.get("confidence", 0))
        )

        resolved = []
        last_end = -1

        for detection in sorted_detections:
            start = detection.get("start", 0)
            end = detection.get("end", 0)

            # Skip if overlaps with previous detection
            if start < last_end:
                logger.debug(
                    f"Skipping overlapping detection at [{start}, {end}] "
                    f"due to previous detection ending at {last_end}"
                )
                continue

            resolved.append(detection)
            last_end = end

        logger.info(f"Resolved {len(detections)} detections to {len(resolved)} non-overlapping")
        return resolved

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
        # Define priority weights for each strategy type per entity
        priority_map = {
            "EMAIL_ADDRESS": {
                StrategyKind.SURROGATE: 3,
                StrategyKind.TEMPLATE: 2,
                StrategyKind.HASH: 1,
            },
            "PHONE_NUMBER": {
                StrategyKind.PARTIAL: 3,
                StrategyKind.TEMPLATE: 2,
                StrategyKind.SURROGATE: 1,
            },
            "CREDIT_CARD": {
                StrategyKind.PARTIAL: 3,
                StrategyKind.HASH: 2,
                StrategyKind.REDACT: 1,
            },
            "US_SSN": {
                StrategyKind.REDACT: 3,
                StrategyKind.HASH: 2,
                StrategyKind.PARTIAL: 1,
            },
        }

        # Get entity-specific priorities or use defaults
        entity_priorities = priority_map.get(entity_type, {})

        # Score each strategy
        scored_strategies = []
        for strategy in available_strategies:
            # Base score from entity-specific priority
            base_score = entity_priorities.get(strategy.kind, 0)

            # Adjust based on confidence
            if confidence > 0.9:
                # High confidence: prefer more aggressive masking
                if strategy.kind in [StrategyKind.REDACT, StrategyKind.HASH]:
                    base_score += 1
            else:
                # Low confidence: prefer reversible strategies
                if strategy.kind in [StrategyKind.PARTIAL, StrategyKind.TEMPLATE]:
                    base_score += 1

            scored_strategies.append((base_score, strategy))

        # Sort by score (descending) and return strategies
        scored_strategies.sort(key=lambda x: x[0], reverse=True)
        return [strategy for _, strategy in scored_strategies]
