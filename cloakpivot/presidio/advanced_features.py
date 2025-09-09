"""Advanced Presidio features for encryption, operator chaining, and ad-hoc recognizers.

Note: Currently only Fernet encryption algorithm is implemented. AES-256 and ChaCha20
are planned for future releases but are not yet available.
"""

import base64
import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from queue import Queue
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .key_management import KeyProvider

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.strategies import Strategy, StrategyKind


class PresidioEncryptionManager:
    """Advanced encryption features using Presidio operators."""

    def __init__(self, key_provider: "KeyProvider"):
        """Initialize encryption manager with a key provider.

        Args:
            key_provider: Provider for encryption keys
        """
        self.key_provider = key_provider
        self.encryption_operators = self._setup_encryption_operators()

    def _setup_encryption_operators(self) -> dict[str, OperatorConfig]:
        """Set up available encryption operators."""
        return {
            "AES-256": OperatorConfig(
                "encrypt",
                {"key": None, "algorithm": "AES-256"}
            ),
            "ChaCha20": OperatorConfig(
                "encrypt",
                {"key": None, "algorithm": "ChaCha20"}
            ),
            "Fernet": OperatorConfig(
                "encrypt",
                {"key": None, "algorithm": "Fernet"}
            )
        }

    def _encryption_callback(self, original_text: str, entity_type: str, confidence: float, algorithm: str) -> str:
        """Encryption callback for custom operator.

        Args:
            original_text: Text to encrypt
            entity_type: Type of entity being encrypted
            confidence: Confidence score
            algorithm: Encryption algorithm to use

        Returns:
            Encrypted text as base64 string
        """
        # Get encryption key for this entity type
        key = self.key_provider.get_key(f"{entity_type}_{algorithm}")

        # Simple encryption simulation (in real implementation, use cryptography library)
        from cryptography.fernet import Fernet
        if algorithm == "Fernet":
            fernet = Fernet(key)
            encrypted = fernet.encrypt(original_text.encode())
            return base64.b64encode(encrypted).decode()
        else:
            # Other algorithms not yet implemented
            raise NotImplementedError(f"Encryption algorithm {algorithm} is not yet implemented. Only Fernet is currently supported.")

    def create_encryption_strategy(self, algorithm: str = "Fernet") -> Strategy:
        """Create Strategy that uses Presidio encryption operator.

        Args:
            algorithm: Encryption algorithm to use. Currently only "Fernet" is supported.
                      "AES-256" and "ChaCha20" are planned for future releases.

        Returns:
            Strategy configured for encryption
            
        Raises:
            NotImplementedError: If an unsupported algorithm is specified
        """
        def callback(text: str, entity_type: str, confidence: float) -> str:
            return self._encryption_callback(text, entity_type, confidence, algorithm)

        return Strategy(
            kind=StrategyKind.CUSTOM,
            parameters={
                "callback": callback,
                "algorithm": algorithm,
                "key_rotation": True
            }
        )

    def create_decryption_context(self, cloakmap: CloakMap) -> dict[str, dict[str, Any]]:
        """Extract decryption context from enhanced CloakMap.

        Args:
            cloakmap: CloakMap containing encryption metadata

        Returns:
            Dictionary mapping entity types to decryption contexts
        """
        presidio_metadata = getattr(cloakmap, 'presidio_metadata', {})
        decryption_context = {}

        for result in presidio_metadata.get("operator_results", []):
            if result.get("operator") == "encrypt":
                key_ref = result.get("key_reference")
                entity_type = result.get("entity_type")
                decryption_context[entity_type] = {
                    "key": self.key_provider.get_key(key_ref),
                    "algorithm": result.get("algorithm", "AES-256")
                }

        return decryption_context


class PresidioOperatorChain:
    """Support for complex operator sequences."""

    def __init__(self) -> None:
        """Initialize operator chain manager."""
        self.supported_operators = ["redact", "replace", "mask", "hash", "encrypt", "custom"]

    def _apply_single_operator(self, text: str, operator_config: dict[str, Any]) -> str:
        """Apply a single operator to text.

        Args:
            text: Input text
            operator_config: Operator configuration

        Returns:
            Transformed text
        """
        operator_type = operator_config.get("type", "replace")
        params = operator_config.get("params", {})

        if operator_type == "redact":
            char = str(params.get("char", "*"))
            return char * len(text)
        elif operator_type == "replace":
            return str(params.get("new_value", f"[{operator_type}]"))
        elif operator_type == "mask":
            char = str(params.get("char", "*"))
            visible = int(params.get("visible_chars", 4))
            if len(text) > visible:
                return text[:visible] + char * (len(text) - visible)
            return text
        elif operator_type == "hash":
            algorithm = str(params.get("algorithm", "sha256"))
            hash_obj = getattr(hashlib, algorithm)()
            hash_obj.update(text.encode())
            return hash_obj.hexdigest()[:16]
        else:
            return f"[{operator_type}:{text}]"

    def _reverse_single_operator(self, text: str, operator_metadata: dict[str, Any]) -> str:
        """Reverse a single operator.

        Args:
            text: Transformed text
            operator_metadata: Metadata about the original operation

        Returns:
            Original text if reversible, otherwise the input text
        """
        if "original" in operator_metadata:
            return str(operator_metadata["original"])
        return text

    def _is_chain_reversible(self, operator_sequence: list[dict[str, Any]]) -> bool:
        """Check if an operator chain is reversible.

        Args:
            operator_sequence: List of operator configurations

        Returns:
            True if the chain is reversible
        """
        irreversible_ops = ["hash", "redact"]
        for op in operator_sequence:
            if op.get("type") in irreversible_ops:
                return False
        return True

    def create_chained_strategy(self, operator_sequence: list[dict[str, Any]]) -> Strategy:
        """Create Strategy that applies multiple operators in sequence.

        Args:
            operator_sequence: List of operator configurations

        Returns:
            Strategy with chained operators
        """
        def chain_callback(original_text: str, entity_type: str, confidence: float) -> str:
            result = original_text
            for operator_config in operator_sequence:
                result = self._apply_single_operator(result, operator_config)
            return result

        return Strategy(
            kind=StrategyKind.CUSTOM,
            parameters={
                "callback": chain_callback,
                "operator_chain": operator_sequence,
                "reversible": self._is_chain_reversible(operator_sequence)
            }
        )

    def reverse_operator_chain(self, masked_text: str, chain_metadata: dict[str, Any]) -> str:
        """Reverse a chain of operators for deanonymization.

        Args:
            masked_text: Text with applied operator chain
            chain_metadata: Metadata about the chain operations

        Returns:
            Original text if chain is reversible
        """
        result = masked_text

        # Apply reverse operations in reverse order
        operator_chain = chain_metadata.get("operator_chain", [])
        for operator_metadata in reversed(operator_chain):
            result = self._reverse_single_operator(result, operator_metadata)

        return result


class PresidioAdHocRecognizers:
    """Dynamic recognizer creation without custom classes."""

    def __init__(self) -> None:
        """Initialize ad-hoc recognizer manager."""
        self.recognizer_cache: dict[str, PatternRecognizer] = {}

    def _build_recognizer(self, recognizer_config: dict[str, Any]) -> PatternRecognizer:
        """Build a recognizer from configuration.

        Args:
            recognizer_config: Recognizer configuration

        Returns:
            PatternRecognizer instance
        """
        entity_type = recognizer_config.get("entity_type", "CUSTOM")
        patterns = recognizer_config.get("patterns", [])

        pattern_objects = []
        for pattern_info in patterns:
            if isinstance(pattern_info, dict):
                pattern_objects.append(Pattern(
                    name=f"{entity_type}_pattern",
                    regex=pattern_info.get("pattern", ""),
                    score=pattern_info.get("score", 0.8)
                ))
            else:
                pattern_objects.append(Pattern(
                    name=f"{entity_type}_pattern",
                    regex=pattern_info,
                    score=0.8
                ))

        return PatternRecognizer(
            supported_entity=entity_type,
            patterns=pattern_objects,
            supported_language=recognizer_config.get("supported_language", "en")
        )

    def create_pattern_recognizer(
        self,
        entity_type: str,
        patterns: list[str],
        confidence: float = 0.8
    ) -> dict[str, Any]:
        """Create a pattern-based recognizer configuration.

        Args:
            entity_type: Type of entity to recognize
            patterns: List of regex patterns
            confidence: Default confidence score

        Returns:
            Recognizer configuration dictionary
        """
        return {
            "entity_type": entity_type,
            "patterns": [{"pattern": p, "score": confidence} for p in patterns],
            "supported_language": "en",
            "context": ["text"]
        }

    def create_context_recognizer(
        self,
        entity_type: str,
        context_words: list[str],
        window_size: int = 5
    ) -> dict[str, Any]:
        """Create context-aware recognizer.

        Args:
            entity_type: Type of entity to recognize
            context_words: Words that provide context
            window_size: Size of context window

        Returns:
            Recognizer configuration dictionary
        """
        # Create patterns that look for context words near potential entities
        patterns = []
        for word in context_words:
            # Pattern to find words near context
            pattern = rf"(?i)(?:{word}.{{0,{window_size * 10}}}[\w\-\.]+|[\w\-\.]+.{{0,{window_size * 10}}}{word})"
            patterns.append(pattern)

        return {
            "entity_type": entity_type,
            "patterns": [{"pattern": p, "score": 0.6} for p in patterns],
            "context_similarity_factor": 0.8,
            "min_score_with_context_similarity": 0.6,
            "context_words": context_words,
            "window_size": window_size,
            "supported_language": "en"
        }

    def apply_adhoc_recognizers(self, text: str, recognizers: list[dict[str, Any]]) -> list:
        """Apply ad-hoc recognizers to text without modifying analyzer setup.

        Args:
            text: Text to analyze
            recognizers: List of recognizer configurations

        Returns:
            List of recognition results
        """
        # Use Presidio's RecognizerRegistry for dynamic recognition
        registry = RecognizerRegistry()

        for recognizer_config in recognizers:
            recognizer = self._build_recognizer(recognizer_config)
            registry.add_recognizer(recognizer)

        analyzer = AnalyzerEngine(registry=registry)
        return analyzer.analyze(text=text, language="en")


class PresidioConnectionPool:
    """Pool management for Presidio engines."""

    def __init__(self, pool_size: int = 5):
        """Initialize connection pool.

        Args:
            pool_size: Number of engines to maintain in pool
        """
        self._analyzer_pool: Queue[AnalyzerEngine] = Queue()
        self._anonymizer_pool: Queue[AnonymizerEngine] = Queue()
        self.pool_size = pool_size

        # Pre-create engine instances
        for _ in range(pool_size):
            self._analyzer_pool.put(AnalyzerEngine())
            self._anonymizer_pool.put(AnonymizerEngine())

    @contextmanager
    def get_analyzer(self) -> Generator[AnalyzerEngine, None, None]:
        """Get an analyzer engine from the pool.

        Yields:
            AnalyzerEngine instance
        """
        engine = self._analyzer_pool.get()
        try:
            yield engine
        finally:
            self._analyzer_pool.put(engine)

    @contextmanager
    def get_anonymizer(self) -> Generator[AnonymizerEngine, None, None]:
        """Get an anonymizer engine from the pool.

        Yields:
            AnonymizerEngine instance
        """
        engine = self._anonymizer_pool.get()
        try:
            yield engine
        finally:
            self._anonymizer_pool.put(engine)
