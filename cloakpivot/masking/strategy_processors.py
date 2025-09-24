"""Strategy processing utilities for the PresidioMaskingAdapter.

This module contains strategy-specific processing logic extracted from the
main PresidioMaskingAdapter to improve maintainability and reduce file size.
"""

import hashlib
import logging
import uuid
from typing import Any, cast

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine, OperatorConfig

from cloakpivot.core.types.strategies import Strategy, StrategyKind
from cloakpivot.core.presidio_mapper import StrategyToOperatorMapper as OperatorMapper
from cloakpivot.core.surrogate import SurrogateGenerator

logger = logging.getLogger(__name__)

UNKNOWN_ENTITY = "PII"


class StrategyProcessor:
    """Process various masking strategies for entity anonymization."""

    def __init__(self, anonymizer: AnonymizerEngine, operator_mapper: OperatorMapper):
        """Initialize the strategy processor.

        Args:
            anonymizer: Presidio anonymizer engine
            operator_mapper: Mapper for strategy to operator conversion
        """
        self.anonymizer = anonymizer
        self.operator_mapper = operator_mapper
        self._surrogate_generator = SurrogateGenerator()
        self._fallback_char = "*"

    def apply_hash_strategy(
        self, text: str, entity_type: str, strategy: Strategy, confidence: float
    ) -> str:
        """Apply hash strategy with support for prefix and other parameters.

        Args:
            text: Text to hash
            entity_type: Type of entity being hashed
            strategy: Hash strategy configuration
            confidence: Confidence score of the entity

        Returns:
            Hashed text value
        """
        params = strategy.parameters or {}

        # Create entity for Presidio
        entity = RecognizerResult(entity_type=entity_type, start=0, end=len(text), score=confidence)

        # Use Presidio for the base hash
        operator_config = self.operator_mapper.strategy_to_operator(strategy)
        result = self.anonymizer.anonymize(
            text=text, analyzer_results=[entity], operators={entity_type: operator_config}
        )

        hashed_value = result.text

        # Add prefix if specified
        if "prefix" in params:
            hashed_value = params["prefix"] + hashed_value

        # Truncate if specified
        if "truncate" in params:
            truncate_length = params["truncate"]
            if isinstance(truncate_length, int) and truncate_length > 0:
                hashed_value = hashed_value[:truncate_length]

        return str(hashed_value)

    def apply_partial_strategy(
        self, text: str, entity_type: str, strategy: Strategy, confidence: float
    ) -> str:
        """Apply partial masking strategy with proper char count calculation.

        Args:
            text: Text to partially mask
            entity_type: Type of entity being masked
            strategy: Partial masking strategy configuration
            confidence: Confidence score of the entity

        Returns:
            Partially masked text value
        """
        params = strategy.parameters or {}
        visible_chars = max(0, int(params.get("visible_chars", 4)))
        position = params.get("position", "end")
        mask_char = params.get("mask_char", "*")

        # Create entity for Presidio
        entity = RecognizerResult(entity_type=entity_type, start=0, end=len(text), score=confidence)

        # Calculate how many chars to mask based on text length
        text_length = len(text)
        visible_chars = min(visible_chars, text_length)  # Can't show more than we have

        if position == "end":
            # Show last N chars, mask the rest
            chars_to_mask = max(0, text_length - visible_chars)
            from_end = False
        elif position == "start":
            # Show first N chars, mask the rest
            chars_to_mask = max(0, text_length - visible_chars)
            from_end = True  # Mask from the end, leaving start visible
        else:
            # Default to end behavior
            chars_to_mask = max(0, text_length - visible_chars)
            from_end = False

        # Use Presidio's mask operator
        operator_config = OperatorConfig(
            "mask",
            {"masking_char": mask_char, "chars_to_mask": chars_to_mask, "from_end": from_end},
        )

        result = self.anonymizer.anonymize(
            text=text, analyzer_results=[entity], operators={entity_type: operator_config}
        )

        return str(result.text)

    def apply_custom_strategy(self, text: str, strategy: Strategy) -> str:
        """Apply a custom strategy using the provided callback.

        Args:
            text: Text to transform
            strategy: Custom strategy with callback function

        Returns:
            Transformed text or fallback redaction
        """
        callback = strategy.parameters.get("callback") if strategy.parameters else None
        if callback and callable(callback):
            try:
                return cast(str, callback(text))
            except Exception as e:
                logger.error(f"Custom callback failed: {e}")
        return self._fallback_redaction(text)

    def apply_surrogate_strategy(self, text: str, entity_type: str, strategy: Strategy) -> str:
        """Apply surrogate strategy with high-quality fake data generation.

        Args:
            text: Original text to replace
            entity_type: Type of entity for surrogate generation
            strategy: Surrogate strategy configuration

        Returns:
            Generated surrogate value or fallback template
        """
        try:
            # Check if strategy has a seed parameter
            seed = None
            if strategy.parameters and "seed" in strategy.parameters:
                seed = strategy.parameters["seed"]

            # If seed is different from current one, recreate generator
            if seed != getattr(self._surrogate_generator, "seed", None):
                self._surrogate_generator = SurrogateGenerator(seed=seed)
                logger.debug(f"Updated SurrogateGenerator with seed: {seed}")

            # Use the surrogate generator for quality fake data
            return self._surrogate_generator.generate_surrogate(text, entity_type)
        except Exception as e:
            logger.warning(f"Surrogate generation failed: {e}")
            # Fallback to simple replacement
            return f"[{entity_type}]"

    def apply_template_strategy(self, entity_type: str, strategy: Strategy) -> str:
        """Apply template strategy for entity masking.

        Args:
            entity_type: Type of entity being masked
            strategy: Template strategy configuration

        Returns:
            Filled template string
        """
        params = strategy.parameters or {}
        template = params.get("template", f"[{entity_type}]")

        # Replace {} with a unique ID if present
        if "{}" in template:
            unique_id = str(uuid.uuid4())[:8]
            return template.replace("{}", unique_id)
        return template

    def apply_redact_strategy(self, text: str, strategy: Strategy) -> str:
        """Apply redaction strategy to text.

        Args:
            text: Text to redact
            strategy: Redaction strategy configuration

        Returns:
            Redacted text with specified character
        """
        params = strategy.parameters or {}
        char = params.get("char", "*")
        return char * len(text)

    def _fallback_redaction(self, text: str) -> str:
        """Simple fallback redaction when other strategies fail.

        Args:
            text: Text to redact

        Returns:
            Redacted text with fallback character
        """
        return self._fallback_char * len(text)