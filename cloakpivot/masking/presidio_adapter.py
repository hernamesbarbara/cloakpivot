"""Presidio-based masking adapter for CloakPivot."""

import hashlib
import logging
from typing import Any, Optional

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine, OperatorResult
from presidio_anonymizer.entities import OperatorConfig

from ..core.cloakmap import AnchorEntry, CloakMap
from ..core.cloakmap_enhancer import CloakMapEnhancer
from ..core.policies import MaskingPolicy
from ..core.presidio_mapper import StrategyToOperatorMapper
from ..core.strategies import Strategy, StrategyKind
from ..core.surrogate import SurrogateGenerator
from ..core.types import DoclingDocument
from ..document.extractor import TextSegment
from .engine import MaskingResult

logger = logging.getLogger(__name__)


class PresidioMaskingAdapter:
    """
    Adapter that uses Presidio AnonymizerEngine for masking operations.

    This adapter translates CloakPivot masking concepts to Presidio operations
    while maintaining API compatibility with the existing StrategyApplicator.
    It captures Presidio operator results for perfect reversibility and provides
    robust error handling with fallback mechanisms.

    Examples:
        >>> adapter = PresidioMaskingAdapter()
        >>> result = adapter.apply_strategy(
        ...     "555-123-4567",
        ...     "PHONE_NUMBER",
        ...     Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
        ...     0.95
        ... )
        >>> assert result == "[PHONE]"
    """

    def __init__(self, engine_config: Optional[dict[str, Any]] = None) -> None:
        """
        Initialize the Presidio masking adapter.

        Args:
            engine_config: Optional configuration for Presidio engine
        """
        self._anonymizer_instance = None  # Lazy loading
        self.operator_mapper = StrategyToOperatorMapper()
        self.cloakmap_enhancer = CloakMapEnhancer()
        self._fallback_char = "*"
        self.engine_config = engine_config or {}
        self._surrogate_generator = SurrogateGenerator()

        logger.debug(f"PresidioMaskingAdapter initialized with config: {self.engine_config}")

    @property
    def anonymizer(self) -> AnonymizerEngine:
        """Lazy-load the AnonymizerEngine on first access."""
        if self._anonymizer_instance is None:
            self._anonymizer_instance = AnonymizerEngine()
            logger.debug("AnonymizerEngine initialized")
        return self._anonymizer_instance

    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float,
    ) -> str:
        """
        Apply a masking strategy using Presidio with fallback support.

        This method maintains API compatibility with StrategyApplicator while
        delegating to Presidio's AnonymizerEngine for actual masking operations.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity (e.g., 'PHONE_NUMBER', 'EMAIL_ADDRESS')
            strategy: The masking strategy to apply
            confidence: Detection confidence score

        Returns:
            The masked replacement text
        """
        logger.debug(f"Applying {strategy.kind.value} strategy to {entity_type} via Presidio")

        try:
            # Handle CUSTOM strategy specially
            if strategy.kind == StrategyKind.CUSTOM:
                return self._apply_custom_strategy(original_text, strategy)

            # Handle SURROGATE strategy specially for better quality
            if strategy.kind == StrategyKind.SURROGATE:
                return self._apply_surrogate_strategy(original_text, entity_type, strategy)

            # Special handling for REDACT strategy since Presidio's redact operator has issues
            if strategy.kind == StrategyKind.REDACT:
                params = strategy.parameters or {}
                char = params.get("char", params.get("redact_char", "*"))
                return char * len(original_text)

            # Special handling for HASH strategy to support prefix
            if strategy.kind == StrategyKind.HASH:
                return self._apply_hash_strategy(original_text, entity_type, strategy, confidence)

            # Special handling for PARTIAL strategy
            if strategy.kind == StrategyKind.PARTIAL:
                return self._apply_partial_strategy(original_text, entity_type, strategy, confidence)

            # Map CloakPivot strategy to Presidio operator
            operator_config = self.operator_mapper.strategy_to_operator(strategy)

            # Create a single entity for this text
            entity = RecognizerResult(
                entity_type=entity_type,
                start=0,
                end=len(original_text),
                score=confidence
            )

            # Use Presidio to anonymize
            result = self.anonymizer.anonymize(
                text=original_text,
                analyzer_results=[entity],
                operators={entity_type: operator_config}
            )

            return result.text

        except Exception as e:
            logger.warning(
                f"Presidio anonymization failed for {entity_type}: {e}. "
                f"Falling back to simple redaction."
            )
            return self._fallback_redaction(original_text)

    def mask_document(
        self,
        document: DoclingDocument,
        entities: list[RecognizerResult],
        policy: MaskingPolicy,
        text_segments: list[TextSegment],
        original_format: Optional[str] = None,
    ) -> MaskingResult:
        """
        Mask PII entities in a document using Presidio.

        This method maintains compatibility with the existing MaskingEngine API
        while using Presidio for all masking operations.

        Args:
            document: The DoclingDocument to mask
            entities: List of detected PII entities
            policy: Masking policy defining strategies per entity type
            text_segments: Text segments extracted from the document
            original_format: Original document format

        Returns:
            MaskingResult containing masked document and enhanced CloakMap
        """
        logger.info(f"Masking document {document.name} with {len(entities)} entities using Presidio")

        # Prepare strategies for each entity type
        strategies = {}
        for entity in entities:
            if entity.entity_type not in strategies:
                strategies[entity.entity_type] = policy.get_strategy_for_entity(entity.entity_type)

        # Process all entities in batch
        operator_results = self._batch_process_entities(
            document._main_text, entities, strategies
        )

        # Create anchor entries for CloakMap
        anchor_entries = []
        masked_text = document._main_text

        # Apply masks and create anchors (process in reverse to maintain positions)
        for _i, (entity, op_result) in enumerate(sorted(
            zip(entities, operator_results),
            key=lambda x: x[0].start,
            reverse=True
        )):
            # Extract original text
            original = document._main_text[entity.start:entity.end]

            # Get masked value from operator result
            masked_value = op_result.text if hasattr(op_result, 'text') else self._fallback_redaction(original)

            # Apply mask to text
            masked_text = (
                masked_text[:entity.start] +
                masked_value +
                masked_text[entity.end:]
            )

            # Create anchor entry with proper fields
            import base64
            import uuid

            # Generate checksum and salt
            salt = base64.b64encode(hashlib.sha256(str(uuid.uuid4()).encode()).digest()[:8]).decode()
            checksum_hash = hashlib.sha256(f"{salt}{original}".encode()).hexdigest()

            # Ensure masked value is not empty (for anchor validation)
            if not masked_value:
                masked_value = "*"  # Use single char as fallback
                
            anchor = AnchorEntry(
                node_id="#/texts/0",  # Would need segment info for proper node_id
                start=entity.start,
                end=entity.start + len(masked_value),
                entity_type=entity.entity_type,
                masked_value=masked_value,
                confidence=entity.score,
                replacement_id=f"repl_{uuid.uuid4().hex[:12]}",
                original_checksum=checksum_hash,
                checksum_salt=salt,
                strategy_used=strategies[entity.entity_type].kind.value,
                timestamp=None,  # Will be set by CloakMap creation
                metadata={
                    "original_text": original,  # Store for reversibility in metadata
                    "presidio_operator": op_result.operator if hasattr(op_result, 'operator') else "fallback"
                }
            )
            anchor_entries.append(anchor)

        # Create masked document
        from docling_core.types.doc.document import TextItem
        
        masked_document = DoclingDocument(
            name=document.name,
            texts=[],
            tables=[],
            key_value_items=[]
        )
        
        # Add masked text as a text item if the original had texts
        if hasattr(document, 'texts') and document.texts:
            # Create a new text item with the masked text
            masked_text_item = TextItem(
                text=masked_text,
                self_ref="#/texts/0",
                label="text",
                orig=masked_text
            )
            masked_document.texts = [masked_text_item]
        
        # Set _main_text for compatibility
        masked_document._main_text = masked_text

        # Create base CloakMap
        base_cloakmap = CloakMap.create(
            doc_id=document.name,
            doc_hash=hashlib.sha256(document._main_text.encode()).hexdigest(),
            anchors=anchor_entries,
            policy=policy,
            metadata={"original_format": original_format} if original_format else {}
        )

        # Enhance with Presidio metadata
        enhanced_cloakmap = self.cloakmap_enhancer.add_presidio_metadata(
            base_cloakmap,
            operator_results=[self._operator_result_to_dict(r) for r in operator_results],
            engine_version="2.x.x",  # Would get from Presidio
            reversible_operators=self._get_reversible_operators(strategies)
        )

        # Calculate statistics
        stats = {
            "entities_masked": len(entities),
            "unique_entity_types": len(strategies),
            "presidio_engine_used": True,
            "fallback_used": any(
                a.metadata.get("presidio_operator") == "fallback"
                for a in anchor_entries
            )
        }

        return MaskingResult(
            masked_document=masked_document,
            cloakmap=enhanced_cloakmap,
            stats=stats
        )

    def _batch_process_entities(
        self,
        text: str,
        entities: list[RecognizerResult],
        strategies: dict[str, Strategy]
    ) -> list[OperatorResult]:
        """
        Process multiple entities in a single batch for efficiency.

        Args:
            text: The text containing entities
            entities: List of entities to mask
            strategies: Mapping of entity types to strategies

        Returns:
            List of OperatorResult objects
        """
        try:
            # Map strategies to operators
            operators = {}
            for entity_type, strategy in strategies.items():
                if strategy.kind == StrategyKind.CUSTOM:
                    # Handle custom strategies separately
                    operators[entity_type] = OperatorConfig("custom")
                elif strategy.kind == StrategyKind.SURROGATE:
                    # Use replace operator with fake data
                    operators[entity_type] = OperatorConfig("replace", {"new_value": "[SURROGATE]"})
                else:
                    operators[entity_type] = self.operator_mapper.strategy_to_operator(strategy)

            # Use Presidio to process all entities
            result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=entities,
                operators=operators
            )

            # Extract operator results
            if hasattr(result, 'items'):
                return result.items
            else:
                # Fallback: create synthetic results
                return [
                    self._create_synthetic_result(entity, strategies[entity.entity_type], text)
                    for entity in entities
                ]

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Create fallback results
            results = []
            for entity in entities:
                entity_type = getattr(entity, 'entity_type', None) or 'UNKNOWN'
                strategy = strategies.get(entity_type, Strategy(StrategyKind.REDACT, {"char": "*"}))
                results.append(self._create_synthetic_result(entity, strategy, text))
            return results

    def _apply_hash_strategy(
        self, 
        text: str, 
        entity_type: str, 
        strategy: Strategy,
        confidence: float
    ) -> str:
        """Apply hash strategy with support for prefix and other parameters."""
        params = strategy.parameters or {}
        
        # Create entity for Presidio
        entity = RecognizerResult(
            entity_type=entity_type,
            start=0,
            end=len(text),
            score=confidence
        )
        
        # Use Presidio for the base hash
        operator_config = self.operator_mapper.strategy_to_operator(strategy)
        result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=[entity],
            operators={entity_type: operator_config}
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
                
        return hashed_value

    def _apply_partial_strategy(
        self,
        text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float
    ) -> str:
        """Apply partial masking strategy with proper char count calculation."""
        params = strategy.parameters or {}
        visible_chars = params.get("visible_chars", 4)
        position = params.get("position", "end")
        mask_char = params.get("mask_char", "*")
        
        # Create entity for Presidio
        entity = RecognizerResult(
            entity_type=entity_type,
            start=0,
            end=len(text),
            score=confidence
        )
        
        # Calculate how many chars to mask based on text length
        text_length = len(text)
        
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
        operator_config = OperatorConfig("mask", {
            "masking_char": mask_char,
            "chars_to_mask": chars_to_mask,
            "from_end": from_end
        })
        
        result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=[entity],
            operators={entity_type: operator_config}
        )
        
        return result.text

    def _apply_custom_strategy(self, text: str, strategy: Strategy) -> str:
        """Apply a custom strategy using the provided callback."""
        callback = strategy.parameters.get("callback")
        if callback and callable(callback):
            try:
                return callback(text)
            except Exception as e:
                logger.error(f"Custom callback failed: {e}")
        return self._fallback_redaction(text)

    def _apply_surrogate_strategy(
        self,
        text: str,
        entity_type: str,
        strategy: Strategy
    ) -> str:
        """Apply surrogate strategy with high-quality fake data generation."""
        try:
            # Use the surrogate generator for quality fake data
            return self._surrogate_generator.generate_surrogate(text, entity_type)
        except Exception as e:
            logger.warning(f"Surrogate generation failed: {e}")
            # Fallback to simple replacement
            return f"[{entity_type}]"

    def _fallback_redaction(self, text: str) -> str:
        """Simple fallback redaction when Presidio fails."""
        return self._fallback_char * len(text)

    def _operator_result_to_dict(self, result: OperatorResult) -> dict[str, Any]:
        """Convert OperatorResult to dictionary for storage."""
        return {
            "entity_type": getattr(result, 'entity_type', 'UNKNOWN'),
            "start": getattr(result, 'start', 0),
            "end": getattr(result, 'end', 0),
            "operator": getattr(result, 'operator', 'unknown'),
            "text": getattr(result, 'text', '')
        }

    def _create_synthetic_result(
        self,
        entity: RecognizerResult,
        strategy: Strategy,
        text: str
    ) -> OperatorResult:
        """Create a synthetic OperatorResult for fallback scenarios."""
        # Handle invalid entity positions
        start = getattr(entity, 'start', 0) or 0
        end = getattr(entity, 'end', len(text)) or len(text)
        entity_type = getattr(entity, 'entity_type', 'UNKNOWN') or 'UNKNOWN'
        
        # Ensure valid bounds
        if start < 0:
            start = 0
        if end > len(text):
            end = len(text)
        if start >= end:
            # Invalid range, use single char
            start = min(start, len(text) - 1)
            end = start + 1
            
        original = text[start:end] if start < end else "*"

        # Apply strategy manually
        if strategy.kind == StrategyKind.REDACT:
            masked = strategy.parameters.get("char", "*") * len(original)
        elif strategy.kind == StrategyKind.TEMPLATE:
            masked = strategy.parameters.get("template", f"[{entity_type}]")
        elif strategy.kind == StrategyKind.HASH:
            algo = strategy.parameters.get("algorithm", "sha256")
            prefix = strategy.parameters.get("prefix", "")
            hash_obj = hashlib.new(algo)
            hash_obj.update(original.encode())
            masked = prefix + hash_obj.hexdigest()[:8]
        elif strategy.kind == StrategyKind.PARTIAL:
            visible = strategy.parameters.get("visible_chars", 4)
            position = strategy.parameters.get("position", "end")
            mask_char = strategy.parameters.get("mask_char", "*")
            if position == "end" and len(original) > visible:
                masked = mask_char * (len(original) - visible) + original[-visible:]
            elif position == "start" and len(original) > visible:
                masked = original[:visible] + mask_char * (len(original) - visible)
            else:
                masked = original  # Too short to partial mask
        else:
            masked = self._fallback_redaction(original)

        # Create synthetic result
        result = type('OperatorResult', (), {
            'entity_type': entity_type,
            'start': start,
            'end': end,
            'operator': strategy.kind.value,
            'text': masked
        })()

        return result

    def _get_reversible_operators(self, strategies: dict[str, Strategy]) -> list[str]:
        """Identify which operators are reversible."""
        reversible = []
        for _entity_type, strategy in strategies.items():
            if strategy.kind in [StrategyKind.HASH, StrategyKind.CUSTOM]:
                # These might be reversible depending on implementation
                reversible.append(strategy.kind.value)
        return reversible

    def _cleanup_large_results(self, results: list[OperatorResult]) -> None:
        """Clean up large result sets for memory efficiency."""
        # This is a placeholder for memory management
        # In production, might implement result pagination or streaming
        pass
